"""hnswlib-backed semantic index.

The ``.hnsw`` file is a rebuildable cache; SQLite embedding BLOBs are the
source of truth (ADR-015). Write ordering is DB-first, index-second, so a
crash can only ever leave the index *behind* the database — a divergence
``reconcile()`` repairs at startup. Deleting or corrupting the cache file is
never a user-facing error: ``open()`` rebuilds from BLOBs.

One exclusive (re-entrant) lock serializes every hnswlib call, queries
included — a deliberate conservative superset of "writes exclusive, reads
concurrent" (stdlib has no reader-writer lock and queries are ms-scale). It
also makes label allocation atomic, so the DB-first write and the index
insert of one ``add()`` can never interleave with another thread's.

This module returns ``(entry_id, score)`` pairs — never entries, never
prose. All failures surface as ``VectorIndexError`` (D6); adapter and
storage failures are wrapped with ``__cause__`` preserved. Cross-process
safety is the advisory lock file owned by ``memory.py``; this module assumes
a single process.

Read-only mode (``read_only=True``): the index loads the cache only when it
is clean and consistent with the mapping and meta identity; it never writes
meta, labels, BLOBs, or the cache file. When the cache is missing, corrupt,
divergent, or the identity mismatches, it logs a warning and enters a
DISABLED state in which ``search()`` returns ``[]`` and ``count()`` returns
0 (semantic-unavailable degradation — the warning at ``open()`` is the loud
part). Write methods always raise ``VectorIndexError`` in read-only mode.

Known v0.1 hole (accepted): tombstone rows record a deliberate ``remove()``
and stop rebuild/reconcile from resurrecting such ids from their BLOBs, but
compaction/rebuild erases that history (``replace_labels``). If a removed
entry is still active with a BLOB after a compaction, the next
``reconcile()`` re-adds it. The durable removal signal is the entry's
archived state / cleared BLOB, owned by the store layer.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hnswlib

from tulving.adapters.embeddings import EmbeddingAdapter, EmbeddingIdentity
from tulving.adapters.storage import StorageBackend, pack_embedding, unpack_embedding
from tulving.exceptions import VectorIndexError

logger = logging.getLogger(__name__)

# Chunk sizes: embed_batch calls during rebuild(re_embed=True) and the
# storage.list() pagination that feeds them.
_EMBED_CHUNK = 256
_LIST_PAGE = 500


@dataclass(frozen=True)
class ReconcileReport:
    """What startup reconciliation did (feeds ``memory.startup()``'s report)."""

    added_from_blobs: int
    """Vectors present in the DB but missing from the index — re-added."""

    dropped_orphans: int
    """Live mapping rows with no DB embedding — tombstoned."""

    full_rebuild: bool
    """The cache file was missing/corrupt and the index was rebuilt from BLOBs."""

    compacted: bool
    """The tombstone threshold was crossed and the index was compacted."""

    tombstone_fraction: float
    """Tombstoned / total mapping rows after reconciliation."""


class SemanticIndex:
    """The hnswlib wrapper: embeds on store, answers similarity queries.

    Owns every hnswlib reality from the audit: int-label<->UUID mapping
    (persisted in the ``vector_labels`` table), ``mark_deleted`` tombstones
    with rebuild-on-threshold compaction, geometric ``max_elements`` growth,
    ``space='cosine'`` with scores clamped to ``[0, 1]``, one exclusive lock,
    and over-fetch to survive post-filtering.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        storage: StorageBackend,
        index_path: Path | None = None,
        *,
        initial_capacity: int = 10_000,
        compaction_threshold: float = 0.25,
        overfetch_multiplier: int = 4,
        ef_construction: int = 200,
        m: int = 16,
        ef_search: int = 50,
        read_only: bool = False,
    ) -> None:
        """Validate config and store references — no I/O, no adapter access (D8).

        Args:
            adapter: The active embedding adapter (identity is load-bearing).
            storage: A storage backend (BLOBs, meta identity, vector_labels).
            index_path: The ``.hnsw`` cache file, pre-validated by the caller;
                None means ephemeral (no cache file is read or written).
            initial_capacity: Starting ``max_elements`` for fresh indexes.
            compaction_threshold: Tombstoned/total fraction that triggers a
                compacting rebuild; must be strictly between 0 and 1.
            overfetch_multiplier: ``fetch_k = top_k * multiplier`` (capped at
                the live count, one 2x escalation).
            ef_construction: hnswlib build-time accuracy parameter.
            m: hnswlib graph connectivity parameter.
            ef_search: hnswlib query-time accuracy floor.
            read_only: Never write meta/labels/BLOBs/cache; on any
                inconsistency enter the disabled state (see module docstring)
                instead of rebuilding.

        Raises:
            VectorIndexError: On out-of-range configuration values.
        """
        if initial_capacity < 1:
            raise VectorIndexError("initial_capacity must be at least 1")
        if not 0.0 < compaction_threshold < 1.0:
            raise VectorIndexError("compaction_threshold must be strictly between 0 and 1")
        if overfetch_multiplier < 1:
            raise VectorIndexError("overfetch_multiplier must be at least 1")
        if ef_construction < 1 or m < 1 or ef_search < 1:
            raise VectorIndexError("ef_construction, m, and ef_search must all be at least 1")
        self._adapter = adapter
        self._storage = storage
        self._index_path = Path(index_path) if index_path is not None else None
        self._initial_capacity = initial_capacity
        self._compaction_threshold = compaction_threshold
        self._overfetch_multiplier = overfetch_multiplier
        self._ef_construction = ef_construction
        self._m = m
        self._ef_search = ef_search
        self._read_only = read_only
        self._disabled = False  # read-only only: cache unusable, semantic unavailable
        self._lock = threading.RLock()
        self._index: Any = None  # hnswlib.Index has no stubs; wrapped in typed methods
        self._opened = False
        self._dirty = False
        self._rebuilt_on_open = False
        self._orphan_logged = False
        self._dimension = 0
        self._label_to_id: dict[int, str] = {}
        self._id_to_label: dict[str, int] = {}
        self._tombstoned: set[int] = set()
        self._next_label = 0

    # ------------------------------------------------------------ lifecycle

    def open(self) -> None:
        """Check the embedding identity, load the mapping, load-or-rebuild the index.

        Fresh store (all meta identity fields absent): the adapter's identity
        is written to meta. Any identity difference — model_id, dimension, or
        metric, same-dimension model swap included — refuses to load until an
        explicit ``rebuild(re_embed=True)``. A missing or corrupt cache file
        is never an error: the index is rebuilt from BLOBs (cache semantics).

        Read-only mode never raises on mismatch/corruption and never writes:
        it enters the disabled state with a warning instead (see the module
        docstring).

        Raises:
            VectorIndexError: On identity mismatch, a non-cosine adapter, or
                storage/adapter failure.
        """
        with self._lock:
            identity = self._identity()
            if self._read_only:
                self._open_read_only(identity)
                return
            self._check_identity(identity)
            self._dimension = identity.dimension
            self._load_mapping()
            loaded = self._try_load_cache_file()
            if not loaded:
                blobs = self._load_blobs()
                if blobs:
                    self._rebuild_locked(re_embed=False)
                    self._rebuilt_on_open = True
                else:
                    self._index = self._new_index(max(self._initial_capacity, 1))
            self._opened = True

    def _open_read_only(self, identity: EmbeddingIdentity) -> None:
        """Read-only open: load the cache only if clean AND consistent; no writes.

        Any problem — non-cosine metric, identity mismatch, cache missing or
        corrupt, live mapping labels absent from the graph — logs one warning
        and enters the disabled state (search -> [], count -> 0).
        """
        self._disabled = False
        self._dimension = identity.dimension
        reason = self._read_only_inconsistency(identity)
        if reason is not None:
            logger.warning(
                "read-only semantic index is DISABLED (%s); search returns no results until a "
                "writer repairs the index (open + reconcile, or rebuild(re_embed=True))",
                reason,
            )
            self._index = None
            self._disabled = True
        self._opened = True

    def _read_only_inconsistency(self, identity: EmbeddingIdentity) -> str | None:
        """Why a read-only open cannot serve queries; None when consistent.

        Side effect on the consistent paths: loads the mapping and the index
        handle (from the cache file, or a fresh empty in-RAM index for a
        store with nothing embedded yet).
        """
        if identity.distance_metric != "cosine":
            return f"adapter metric {identity.distance_metric!r} is not 'cosine'"
        meta = self._call_storage(self._storage.get_meta)
        stored = (
            meta["embedding_model_id"],
            meta["embedding_dimension"],
            meta["distance_metric"],
        )
        self._load_mapping()
        fresh = all(value is None for value in stored)
        if fresh:
            if self._label_to_id or (self._index_path is not None and self._index_path.exists()):
                return "meta has no embedding identity but a mapping/cache exists"
            self._index = self._new_index(max(self._initial_capacity, 1))
            return None
        if stored != (identity.model_id, identity.dimension, identity.distance_metric):
            return (
                f"memory was embedded with '{stored[0]}' ({stored[1]}d, {stored[2]}) but the "
                f"active adapter is '{identity.model_id}'"
            )
        if not self._label_to_id:
            self._index = self._new_index(max(self._initial_capacity, 1))
            return None
        if self._index_path is None or not self._index_path.exists():
            return "index cache file is missing"
        if not self._try_load_cache_file():
            return "index cache file is corrupt or unreadable"
        missing = set(self._id_to_label.values()) - self._index_label_set()
        if missing:
            self._index = None
            return f"{len(missing)} mapped vector(s) are missing from the cache (stale cache)"
        return None

    def reconcile(self) -> ReconcileReport:
        """Repair DB/index divergence (startup duty, ADR-015). Idempotent.

        Vectors present in the DB but missing from the graph (crash after the
        DB write) are re-added from their BLOBs; live mapping rows with no DB
        embedding are tombstoned as orphans; finally the tombstone threshold
        is checked and compaction runs if crossed.

        Returns:
            A report of what was done; all-zero on an already-consistent index.

        Raises:
            VectorIndexError: If the index is not open, read-only, or storage
                fails.
        """
        self._ensure_open()
        self._ensure_writable()
        with self._lock:
            self._ensure_open()
            full_rebuild = self._rebuilt_on_open
            self._rebuilt_on_open = False
            blobs = self._load_blobs()
            index_labels = self._index_label_set()
            added = 0
            # Live mapping rows whose vector never reached the graph.
            for entry_id, label in list(self._id_to_label.items()):
                vector = blobs.get(entry_id)
                if label in index_labels or vector is None:
                    continue
                self._insert_vectors([vector], [label])
                added += 1
            # DB embeddings with no live mapping row at all. A tombstoned row
            # records a deliberate remove(), so those ids are skipped — but
            # this protection is best-effort: compaction/rebuild erases
            # tombstone history, after which a still-active entry with a BLOB
            # WOULD be re-added here (accepted v0.1 hole; the store's
            # archived state / cleared BLOB is the durable removal signal).
            removed = self._removed_ids()
            new_rows: list[tuple[int, str]] = []
            new_vectors: list[list[float]] = []
            for entry_id, vector in blobs.items():
                if entry_id in self._id_to_label or entry_id in removed:
                    continue
                new_rows.append((self._next_label + len(new_rows), entry_id))
                new_vectors.append(vector)
            if new_rows:
                self._call_storage(self._storage.insert_labels, new_rows)  # ONE batch (DB first)
                self._insert_vectors(new_vectors, [label for label, _ in new_rows])
                for label, entry_id in new_rows:
                    self._label_to_id[label] = entry_id
                    self._id_to_label[entry_id] = label
                self._next_label += len(new_rows)
                added += len(new_rows)
            # Live mapping rows with no DB embedding: orphans.
            orphans = [entry_id for entry_id in self._id_to_label if entry_id not in blobs]
            for entry_id in orphans:
                self._remove_locked(entry_id)
            compacted = self.maybe_compact()
            return ReconcileReport(
                added_from_blobs=added,
                dropped_orphans=len(orphans),
                full_rebuild=full_rebuild,
                compacted=compacted,
                tombstone_fraction=self.tombstone_fraction,
            )

    def flush(self) -> None:
        """Save the ``.hnsw`` cache atomically (tmp + ``os.replace``) if dirty.

        Ephemeral indexes (``index_path=None``) are a no-op.

        Raises:
            VectorIndexError: If the index is not open, read-only, or the
                save fails.
        """
        self._ensure_open()
        self._ensure_writable()
        with self._lock:
            self._ensure_open()
            if self._index_path is None or not self._dirty:
                return
            tmp_path = self._index_path.with_suffix(".hnsw.tmp")
            try:
                self._index.save_index(str(tmp_path))
                os.replace(tmp_path, self._index_path)
            except (OSError, RuntimeError) as exc:
                with suppress(OSError):
                    tmp_path.unlink()
                raise VectorIndexError(f"cannot save index cache file: {exc}") from exc
            self._dirty = False

    def close(self) -> None:
        """Flush and release the hnswlib handle. Idempotent, safe before open()."""
        with self._lock:
            if not self._opened:
                return
            if not self._read_only:  # read-only never writes the cache
                self.flush()
            self._index = None
            self._opened = False
            self._disabled = False

    # --------------------------------------------------------------- writes

    def add(self, entry_id: str, text: str) -> list[float]:
        """Embed and index one entry — DB first (BLOB, then label), index second.

        Args:
            entry_id: The entry's id; its row must already exist (the store
                creates it before indexing — that IS the DB-first ordering).
            text: The content to embed.

        Returns:
            The persisted (normalized) vector.

        Raises:
            VectorIndexError: If the index is not open or read-only, the id
                is already live in the index (supersede mints NEW ids; content
                updates go through remove + add), or the adapter/storage fails.
        """
        self._ensure_open()
        self._ensure_writable()
        vector = self._embed_one(text)
        with self._lock:
            self._ensure_open()
            self._add_vectors_locked([entry_id], [vector])
        return vector

    def add_batch(self, items: Sequence[tuple[str, str]]) -> list[list[float]]:
        """Embed and index many entries with ONE ``embed_batch`` call.

        DB rows are written first for the whole batch, then one locked
        ``add_items`` inserts all vectors (batch-parity regression).

        Args:
            items: ``(entry_id, content)`` pairs.

        Returns:
            The persisted vectors, in input order.

        Raises:
            VectorIndexError: Same contract as :meth:`add`; a duplicate id
                anywhere in the batch fails the whole batch before any write.
        """
        self._ensure_open()
        self._ensure_writable()
        if not items:
            return []
        entry_ids = [entry_id for entry_id, _ in items]
        if len(set(entry_ids)) != len(entry_ids):
            raise VectorIndexError("add_batch received duplicate entry ids")
        vectors = self._embed_many([text for _, text in items])
        with self._lock:
            self._ensure_open()
            self._add_vectors_locked(entry_ids, vectors)
        return vectors

    def remove(self, entry_id: str) -> bool:
        """Tombstone an entry's vector — DB first, then ``mark_deleted``.

        Idempotent: unknown or already-removed ids return False, never raise
        (forget paths call this freely). Space is reclaimed at compaction.

        Args:
            entry_id: The entry to remove from search.

        Returns:
            True if a live vector was tombstoned.

        Raises:
            VectorIndexError: If the index is not open, read-only, or storage
                fails.
        """
        self._ensure_open()
        self._ensure_writable()
        with self._lock:
            self._ensure_open()
            return self._remove_locked(entry_id)

    def rebuild(self, *, re_embed: bool = False) -> None:
        """Rebuild the index from truth: BLOBs, or content re-embedded.

        ``re_embed=True`` is the ONLY sanctioned adapter-migration path: it
        re-embeds all active content with the current adapter, rewrites the
        BLOBs, and stamps the adapter's identity into meta. Callable while
        the index is refused/unopened — it is the recovery path ``open()``
        names on an identity mismatch. Labels are re-issued dense ``0..n-1``.

        Args:
            re_embed: Re-embed content (adapter migration) instead of reusing
                BLOBs (compaction / cache regeneration).

        Raises:
            VectorIndexError: On a read-only index, adapter/storage failure,
                or a BLOB whose dimension no longer matches the adapter
                (``re_embed=False``).
        """
        self._ensure_writable()
        with self._lock:
            self._rebuild_locked(re_embed=re_embed)

    def maybe_compact(self) -> bool:
        """Rebuild from BLOBs if the tombstone fraction crossed the threshold.

        Returns:
            True if a compacting rebuild ran.

        Raises:
            VectorIndexError: If the index is not open or read-only.
        """
        self._ensure_open()
        self._ensure_writable()
        with self._lock:
            self._ensure_open()
            if not self.needs_compaction:
                return False
            self._rebuild_locked(re_embed=False)
            return True

    # ---------------------------------------------------------------- reads

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        """Similarity search with over-fetch and post-filtering.

        Candidates are dropped in order: unknown/stale labels (orphans),
        tombstoned labels, **archived entries** (the audit regression: an
        archived entry whose vector is still in the graph must never
        surface), then ``filter_fn`` rejections. If survivors fall short of
        ``top_k`` the fetch escalates ONCE at 2x the multiplier; short
        results are normal, never a loop.

        Args:
            query: The text to embed and search for.
            top_k: Maximum results to return; must be at least 1.
            filter_fn: Optional predicate over entry_id (the store pushes
                tag/type/importance filters through this closure).

        Returns:
            Up to ``top_k`` ``(entry_id, score)`` pairs, score descending,
            scores clamped to ``[0.0, 1.0]``.

        Raises:
            VectorIndexError: If the index is not open, ``top_k < 1``, or the
                adapter/storage fails.
        """
        self._ensure_open()
        if top_k < 1:
            raise VectorIndexError("top_k must be at least 1")
        vector = self._embed_one(query)
        with self._lock:
            self._ensure_open()
            if self._disabled:  # read-only, cache unusable: semantic-unavailable
                return []
            live = len(self._id_to_label)
            if live == 0:
                return []
            fetch_k = min(top_k * self._overfetch_multiplier, live)
            results = self._fetch_round(vector, fetch_k, filter_fn)
            if len(results) < top_k and fetch_k < live:
                fetch_k = min(top_k * self._overfetch_multiplier * 2, live)
                results = self._fetch_round(vector, fetch_k, filter_fn)
            return results[:top_k]

    def count(self) -> int:
        """Live (non-tombstoned) vector count; 0 when read-only-disabled.

        Raises:
            VectorIndexError: If the index is not open.
        """
        self._ensure_open()
        with self._lock:
            self._ensure_open()
            if self._disabled:
                return 0
            return len(self._id_to_label)

    @property
    def tombstone_fraction(self) -> float:
        """Tombstoned / total mapping rows; 0.0 on an empty index."""
        with self._lock:
            total = len(self._label_to_id)
            return len(self._tombstoned) / total if total else 0.0

    @property
    def needs_compaction(self) -> bool:
        """True when the tombstone fraction exceeds the compaction threshold."""
        return self.tombstone_fraction > self._compaction_threshold

    # ----------------------------------------------------- internal: open()

    def _ensure_open(self) -> None:
        if not self._opened:
            raise VectorIndexError("index is not open — call open() first")

    def _ensure_writable(self) -> None:
        if self._read_only:
            raise VectorIndexError(
                "index was opened read-only; writes (add/remove/rebuild/flush/reconcile) "
                "require a writer instance"
            )

    def _identity(self) -> EmbeddingIdentity:
        try:
            return EmbeddingIdentity.from_adapter(self._adapter)
        except VectorIndexError:
            raise
        except Exception as exc:
            raise VectorIndexError(f"embedding adapter failed to report identity: {exc}") from exc

    def _check_identity(self, identity: EmbeddingIdentity) -> None:
        """Compare the adapter identity against meta; write it on a fresh store."""
        if identity.distance_metric != "cosine":
            raise VectorIndexError(
                f"adapter declares distance_metric {identity.distance_metric!r}; "
                "v0.1 supports 'cosine' only"
            )
        meta = self._call_storage(self._storage.get_meta)
        stored = (
            meta["embedding_model_id"],
            meta["embedding_dimension"],
            meta["distance_metric"],
        )
        if all(value is None for value in stored):
            self._write_identity(identity)
            return
        if stored != (identity.model_id, identity.dimension, identity.distance_metric):
            raise VectorIndexError(
                f"memory was embedded with '{stored[0]}' ({stored[1]}d, {stored[2]}) but the "
                f"active adapter is '{identity.model_id}' ({identity.dimension}d, "
                f"{identity.distance_metric}). Run rebuild(re_embed=True) to re-embed, or "
                "restore the original adapter."
            )

    def _write_identity(self, identity: EmbeddingIdentity) -> None:
        self._call_storage(
            self._storage.set_meta,
            {
                "embedding_model_id": identity.model_id,
                "embedding_dimension": identity.dimension,
                "distance_metric": identity.distance_metric,
            },
        )

    def _load_mapping(self) -> None:
        """Load vector_labels into the three in-memory structures."""
        rows = self._call_storage(self._storage.load_labels)
        self._label_to_id = {label: entry_id for label, entry_id, _ in rows}
        self._id_to_label = {
            entry_id: label for label, entry_id, tombstoned in rows if not tombstoned
        }
        self._tombstoned = {label for label, _, tombstoned in rows if tombstoned}
        self._next_label = max(self._label_to_id, default=-1) + 1

    def _try_load_cache_file(self) -> bool:
        """Load the ``.hnsw`` file if present and sane; any failure means rebuild."""
        if self._index_path is None or not self._index_path.exists():
            return False
        index = self._new_index(capacity=None)
        try:
            index.load_index(
                str(self._index_path),
                max_elements=max(self._initial_capacity, len(self._label_to_id)),
            )
            if int(index.dim) != self._dimension:
                raise RuntimeError(
                    f"cache file dimension {int(index.dim)} != adapter {self._dimension}"
                )
            for label in self._tombstoned:
                with suppress(RuntimeError):
                    index.mark_deleted(label)
            index.set_ef(self._ef_search)
        except (RuntimeError, OSError) as exc:
            logger.warning("index cache file is corrupt or unreadable (%s)", exc)
            return False
        self._index = index
        self._dirty = False
        return True

    def _new_index(self, capacity: int | None) -> Any:
        """A fresh hnswlib handle; initialized when ``capacity`` is given."""
        try:
            index = hnswlib.Index(space="cosine", dim=self._dimension)
            if capacity is not None:
                index.init_index(
                    max_elements=capacity, ef_construction=self._ef_construction, M=self._m
                )
                index.set_ef(self._ef_search)
        except (RuntimeError, ValueError) as exc:
            raise VectorIndexError(f"cannot initialize hnswlib index: {exc}") from exc
        return index

    # --------------------------------------------- internal: embed & vectors

    def _embed_one(self, text: str) -> list[float]:
        try:
            vector = self._adapter.embed(text)
        except Exception as exc:
            raise VectorIndexError(f"embedding adapter failed: {exc}") from exc
        return self._canonical(vector)

    def _embed_many(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_CHUNK):
            chunk = texts[start : start + _EMBED_CHUNK]
            try:
                batch = self._adapter.embed_batch(chunk)
            except Exception as exc:
                raise VectorIndexError(f"embedding adapter failed: {exc}") from exc
            vectors.extend(batch)
        return [self._canonical(vector) for vector in vectors]

    def _canonical(self, vector: Sequence[float]) -> list[float]:
        """Validate dimension and L2-normalize (one canonical persisted form)."""
        if len(vector) != self._dimension:
            raise VectorIndexError(
                f"adapter returned a {len(vector)}-dimensional vector; expected {self._dimension}"
            )
        values = [float(value) for value in vector]
        if self._adapter.normalizes:
            return values
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0.0:
            raise VectorIndexError("adapter returned a zero vector; cannot normalize for cosine")
        return [value / norm for value in values]

    def _add_vectors_locked(self, entry_ids: list[str], vectors: list[list[float]]) -> None:
        """DB writes (BLOBs, then labels) followed by one ``add_items``."""
        for entry_id in entry_ids:
            if entry_id in self._id_to_label:
                raise VectorIndexError(
                    f"entry {entry_id!r} is already indexed; supersede creates a new id and "
                    "content updates go through remove() + add()"
                )
        for entry_id, vector in zip(entry_ids, vectors, strict=True):
            self._call_storage(self._storage.set_embedding, entry_id, pack_embedding(vector))
        labels = list(range(self._next_label, self._next_label + len(entry_ids)))
        self._call_storage(self._storage.insert_labels, list(zip(labels, entry_ids, strict=True)))
        self._insert_vectors(vectors, labels)
        for entry_id, label in zip(entry_ids, labels, strict=True):
            self._label_to_id[label] = entry_id
            self._id_to_label[entry_id] = label
        self._next_label += len(entry_ids)

    def _insert_vectors(self, vectors: list[list[float]], labels: list[int]) -> None:
        """Grow if needed and ``add_items`` — always under the lock."""
        self._grow_for(len(vectors))
        try:
            self._index.add_items(vectors, labels)
        except (RuntimeError, ValueError) as exc:
            raise VectorIndexError(f"hnswlib insert failed: {exc}") from exc
        self._dirty = True

    def _grow_for(self, n: int) -> None:
        """Geometric ``resize_index`` doubling; callers never see a capacity error."""
        capacity = int(self._index.get_max_elements())
        needed = int(self._index.get_current_count()) + n
        if needed <= capacity:
            return
        new_capacity = capacity
        while new_capacity < needed:
            new_capacity *= 2
        try:
            self._index.resize_index(new_capacity)
        except (RuntimeError, ValueError) as exc:
            raise VectorIndexError(f"hnswlib resize failed: {exc}") from exc
        logger.info("grew hnsw index capacity from %d to %d", capacity, new_capacity)

    def _remove_locked(self, entry_id: str) -> bool:
        if not self._call_storage(self._storage.tombstone_label, entry_id):
            return False
        label = self._id_to_label.pop(entry_id, None)
        if label is not None:
            with suppress(RuntimeError):  # already-deleted is fine (reconcile idempotency)
                self._index.mark_deleted(label)
            self._tombstoned.add(label)
            self._dirty = True
        return True

    # ------------------------------------------------------ internal: search

    def _fetch_round(
        self,
        vector: list[float],
        fetch_k: int,
        filter_fn: Callable[[str], bool] | None,
    ) -> list[tuple[str, float]]:
        """One knn_query + post-filter pass; order follows ascending distance."""
        self._index.set_ef(max(self._ef_search, fetch_k))
        try:
            labels, distances = self._index.knn_query([vector], k=fetch_k)
        except (RuntimeError, ValueError) as exc:
            raise VectorIndexError(f"hnswlib query failed: {exc}") from exc
        candidates: list[tuple[str, float]] = []
        for raw_label, raw_distance in zip(labels[0], distances[0], strict=True):
            label = int(raw_label)
            if label in self._tombstoned:
                continue
            entry_id = self._label_to_id.get(label)
            if entry_id is None or self._id_to_label.get(entry_id) != label:
                if not self._orphan_logged:
                    self._orphan_logged = True
                    logger.warning(
                        "search hit unknown/stale label %d; reconcile() will repair", label
                    )
                continue
            candidates.append((entry_id, float(raw_distance)))
        active = self._filter_active([entry_id for entry_id, _ in candidates])
        results: list[tuple[str, float]] = []
        for entry_id, distance in candidates:
            if entry_id not in active:
                continue
            if filter_fn is not None and not filter_fn(entry_id):
                continue
            results.append((entry_id, max(0.0, min(1.0, 1.0 - distance))))
        return results

    def _filter_active(self, entry_ids: list[str]) -> set[str]:
        """The archived post-filter: ids that exist AND are archived=0."""
        active: set[str] = set()
        for entry_id in entry_ids:
            row = self._call_storage(self._storage.read, entry_id)
            if row is not None and not row["archived"]:
                active.add(entry_id)
        return active

    # ----------------------------------------------------- internal: rebuild

    def _rebuild_locked(self, *, re_embed: bool) -> None:
        identity = self._identity()
        self._check_identity_for_rebuild(identity)
        self._dimension = identity.dimension
        # Reload the mapping from storage: rebuild may run before open()
        # succeeds (the identity-mismatch recovery path), and deliberate
        # remove()s — tombstoned rows with no live successor — are excluded
        # from this rebuild (best-effort: replace_labels below erases the
        # tombstone history itself; see _removed_ids).
        self._load_mapping()
        removed = self._removed_ids()
        if re_embed:
            items = self._re_embed_all(exclude=removed)
        else:
            items = [
                (entry_id, vector)
                for entry_id, vector in self._load_blobs().items()
                if entry_id not in removed
            ]
        count = len(items)
        index = self._new_index(max(self._initial_capacity, 2 * count))
        # CRASH-WINDOW ORDERING: discard the stale cache file BEFORE the new
        # mapping is committed. If we crashed after replace_labels with the
        # old cache still on disk, the next open() would load the OLD graph
        # under the NEW labels — silently wrong results that reconcile()
        # cannot detect. A missing cache, by contrast, always rebuilds
        # cleanly from BLOBs.
        self._discard_cache_file()
        # DB next, atomically: dense labels 0..n-1, all live.
        self._call_storage(
            self._storage.replace_labels,
            [(label, entry_id) for label, (entry_id, _) in enumerate(items)],
        )
        if count:
            try:
                index.add_items([vector for _, vector in items], list(range(count)))
            except (RuntimeError, ValueError) as exc:
                raise VectorIndexError(f"hnswlib insert failed during rebuild: {exc}") from exc
        self._index = index
        self._label_to_id = {label: entry_id for label, (entry_id, _) in enumerate(items)}
        self._id_to_label = {entry_id: label for label, (entry_id, _) in enumerate(items)}
        self._tombstoned = set()
        self._next_label = count
        self._dirty = True
        self._write_identity(identity)
        self._opened = True
        self.flush()

    def _check_identity_for_rebuild(self, identity: EmbeddingIdentity) -> None:
        if identity.distance_metric != "cosine":
            raise VectorIndexError(
                f"adapter declares distance_metric {identity.distance_metric!r}; "
                "v0.1 supports 'cosine' only"
            )

    def _discard_cache_file(self) -> None:
        """Unlink the cache (and any tmp) — called before committing a new mapping.

        Raises:
            VectorIndexError: If the stale cache cannot be removed (continuing
                would recreate the exact stale-cache/new-mapping hazard).
        """
        if self._index_path is None:
            return
        try:
            self._index_path.with_suffix(".hnsw.tmp").unlink(missing_ok=True)
            self._index_path.unlink(missing_ok=True)
        except OSError as exc:
            raise VectorIndexError(f"cannot remove the stale index cache file: {exc}") from exc

    def _removed_ids(self) -> set[str]:
        """Ids whose newest mapping row is a tombstone — deliberately removed.

        Best-effort signal only (accepted v0.1 hole): rebuild/compaction
        erases tombstone history via ``replace_labels``, after which a
        still-active entry with a BLOB would be re-added by the next
        ``reconcile()``. The durable removal signal is the entry's archived
        state / cleared BLOB, owned by the store layer.
        """
        tombstoned_ids = {self._label_to_id[label] for label in self._tombstoned}
        return tombstoned_ids - set(self._id_to_label)

    def _re_embed_all(self, exclude: set[str]) -> list[tuple[str, list[float]]]:
        """Re-embed all active content and rewrite the BLOBs (adapter migration)."""
        contents = [
            (entry_id, content)
            for entry_id, content in self._active_contents()
            if entry_id not in exclude
        ]
        vectors = self._embed_many([content for _, content in contents])
        items: list[tuple[str, list[float]]] = []
        for (entry_id, _), vector in zip(contents, vectors, strict=True):
            self._call_storage(self._storage.set_embedding, entry_id, pack_embedding(vector))
            items.append((entry_id, vector))
        return items

    def _load_blobs(self) -> dict[str, list[float]]:
        blobs: dict[str, list[float]] = {}
        for entry_id, blob in self._call_storage(self._storage.iter_embeddings):
            vector = unpack_embedding(blob)
            if len(vector) != self._dimension:
                raise VectorIndexError(
                    f"stored embedding for {entry_id!r} is {len(vector)}-dimensional; the "
                    f"active adapter expects {self._dimension}. Run rebuild(re_embed=True)."
                )
            blobs[entry_id] = vector
        return blobs

    def _active_contents(self) -> list[tuple[str, str]]:
        """ACTIVE (entry_id, content) pairs via paginated storage.list()."""
        contents: list[tuple[str, str]] = []
        offset = 0
        while True:
            page = self._call_storage(self._storage.list, {}, _LIST_PAGE, offset)
            contents.extend((row["id"], row["content"]) for row in page)
            if len(page) < _LIST_PAGE:
                return contents
            offset += _LIST_PAGE

    # ------------------------------------------------------ internal: helpers

    def _index_label_set(self) -> set[int]:
        """Labels currently present in the graph (tombstoned included)."""
        if self._index is None:
            return set()
        return {int(label) for label in self._index.get_ids_list()}

    @staticmethod
    def _call_storage(func: Callable[..., Any], *args: Any) -> Any:
        """Invoke a storage method, wrapping StorageError as VectorIndexError (D6)."""
        try:
            return func(*args)
        except VectorIndexError:
            raise
        except Exception as exc:
            raise VectorIndexError(f"storage backend failed: {exc}") from exc

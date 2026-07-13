"""Memory Store: the CRUD engine every memory flows through.

Owns identity (mints UUIDs), timestamps, the D1 supersede policy, entry
hydration (dict <-> ``MemoryEntry``), touch semantics (D3), and error
translation. Sits directly on ``StorageBackend`` and beneath everything
else. Deliberately index-agnostic: it persists embedding bytes (the ADR-015
source of truth) but never imports hnswlib or the semantic index —
``memory.py`` wires index maintenance around store calls.

Error contract (D6): policy refusals and misses the store detects are
``MemoryStoreError``; backend failures propagate as ``StorageError``
untouched; corrupt rows (hydration failures) are wrapped as ``StorageError``
per entry.py's documented contract.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta
from typing import Any, Final

from tulving.adapters.storage import StorageBackend, pack_embedding, unpack_embedding
from tulving.entry import MemoryEntry, Relationship, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import MemoryStoreError, StorageError

# Module-level aliases: the class below defines a method named ``list``, so
# ``list[...]`` annotations inside the class body would resolve against the
# method, not the builtin (mypy valid-type error).
_EntryList = list[MemoryEntry]
_StrList = list[str]
_ItemList = list[dict[str, Any]]
_RelationshipList = list[Relationship]
_TypeList = list[MemoryType]
_ReasonList = list[ArchiveReason]

# Reason-aware purge default (D3): everything EXCEPT SUMMARIZED — summarization
# sources are purged only when SUMMARIZED is explicitly listed (ADR-009).
_DEFAULT_PURGE_REASONS: tuple[ArchiveReason, ...] = (
    ArchiveReason.EVICTED,
    ArchiveReason.SUPERSEDED,
    ArchiveReason.FORGOTTEN,
    ArchiveReason.ABANDONED,
)

# Reserved lifecycle namespace (SEC-SEV-001): the public create paths refuse
# these keys so a caller can never squat — or, worse, D1-supersede — a genuine
# session marker after the session ends. Deliberately duplicates
# tulving.context.lifecycle.SESSION_KEY_PREFIX as a literal: the store sits
# BELOW lifecycle in the layer order and must not import it (tests pin the two
# constants together). Lifecycle writes markers through the backend directly,
# so this boundary never blocks the genuine writer.
_RESERVED_SESSION_KEY_PREFIX: Final[str] = "session:"


class MemoryStore:
    """CRUD engine over a ``StorageBackend``; the D1/D2/D3 policy layer."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Cheap construction (D8): stores references only.

        Args:
            backend: The persistence backend (SQLite or in-memory).
            clock: Injectable now-source; defaults to ``entry.utcnow`` so
                decay/touch tests never sleep.
        """
        self._backend = backend
        self._clock: Callable[[], datetime] = clock if clock is not None else utcnow
        # Per-thread "am I inside store.transaction()?" flag so restore() can
        # self-wrap its supersede path atomically when called standalone
        # without double-BEGIN under an ambient transaction (C4).
        self._txn_state = threading.local()

    # ----------------------------------------------------------------- create

    def create(
        self,
        *,
        content: str,
        type: MemoryType,
        source: SourceInfo,
        key: str | None = None,
        tags: _StrList | None = None,
        base_importance: float = 0.5,
        relationships: _RelationshipList | None = None,
        session_id: str | None = None,
        pinned: bool = False,
        embedding: bytes | None = None,
        _allow_summary: bool = False,
        _source_entry_ids: _StrList | None = None,
    ) -> MemoryEntry:
        """Mint, validate, and persist a new entry — superseding on live keys.

        The D1 sequence runs inside ONE backend transaction: read the live
        key-holder (no TOCTOU), archive it (``SUPERSEDED``, ``updated_at``
        bumped), insert the new entry carrying a ``supersedes`` back-link.
        A key held only by an archived row is simply free (partial unique
        index): clean insert, no back-link. **Never raises on duplicate
        keys.**

        Args:
            content: The memory text; non-empty.
            type: The memory type; ``SUMMARY`` is rejected unless
                ``_allow_summary`` (summarizer only).
            source: Provenance; ``agent_id`` must be non-empty (D7).
            key: Optional address; ``""`` is a caller bug (use None); the
                ``session:`` prefix is reserved for lifecycle markers and
                refused (SEC-SEV-001).
            tags: Deduped preserving order; each must be a non-empty str.
            base_importance: Persisted once, immutable afterwards (D2).
            relationships: Stored verbatim; metadata must be JSON-safe.
            session_id: Optional owning session.
            pinned: Eviction exemption flag.
            embedding: Opaque packed bytes persisted with the row (ADR-015).
            _allow_summary: Summarizer-only escape hatch; requires
                ``_source_entry_ids``.
            _source_entry_ids: SUMMARY back-links (D3).

        Returns:
            The new active entry (carrying the supersede back-link if any).

        Raises:
            MemoryStoreError: On any invalid input (the store is the API
                boundary for entry invariants).
            StorageError: Propagated backend failure.
        """
        entry = self._prepare_entry(
            content=content,
            type=type,
            source=source,
            key=key,
            tags=tags,
            base_importance=base_importance,
            relationships=relationships,
            session_id=session_id,
            pinned=pinned,
            _allow_summary=_allow_summary,
            _source_entry_ids=_source_entry_ids,
        )
        with self._backend.transaction():
            self._persist_with_supersede(entry, embedding)
        return entry

    def batch_create(self, items: _ItemList) -> _EntryList:
        """Bulk create with FULL parity to ``create()``, all-or-nothing.

        Each item dict holds ``create()`` kwargs (plus optional
        ``'embedding'`` bytes). ONE backend transaction wraps the whole
        batch. Supersede applies per item, including intra-batch collisions
        (deterministic input order: a later item archives an earlier item's
        row and back-links it). Embeddings are persisted with the rows, so
        batch-created entries are rebuildable from the BLOB source of truth
        immediately.

        Args:
            items: Per-entry kwargs dicts, in insertion order.

        Returns:
            The created entries, in input order.

        Raises:
            MemoryStoreError: If any item is invalid — nothing is persisted.
            StorageError: Propagated backend failure (transaction rolled back).
        """
        prepared: list[tuple[MemoryEntry, bytes | None]] = []
        for index, item in enumerate(items):
            kwargs = dict(item)
            embedding = kwargs.pop("embedding", None)
            try:
                prepared.append((self._prepare_entry(**kwargs), embedding))
            except TypeError as exc:
                raise MemoryStoreError(f"invalid batch item {index}: {exc}") from exc
        with self._backend.transaction():
            for entry, embedding in prepared:
                self._persist_with_supersede(entry, embedding)
        return [entry for entry, _ in prepared]

    def _prepare_entry(
        self,
        *,
        content: str,
        type: MemoryType,
        source: SourceInfo,
        key: str | None = None,
        tags: _StrList | None = None,
        base_importance: float = 0.5,
        relationships: _RelationshipList | None = None,
        session_id: str | None = None,
        pinned: bool = False,
        _allow_summary: bool = False,
        _source_entry_ids: _StrList | None = None,
    ) -> MemoryEntry:
        """Validate inputs and build the minted entry (no persistence)."""
        if type is MemoryType.SUMMARY and not _allow_summary:
            raise MemoryStoreError(
                "SUMMARY entries are system-generated; callers cannot store them"
            )
        if _allow_summary and not _source_entry_ids:
            raise MemoryStoreError("summaries must carry non-empty _source_entry_ids back-links")
        if key == "":
            raise MemoryStoreError("key must be None (unkeyed) or a non-empty string")
        if key is not None and key.startswith(_RESERVED_SESSION_KEY_PREFIX):
            raise MemoryStoreError(
                f"keys beginning with {_RESERVED_SESSION_KEY_PREFIX!r} are reserved for "
                "system session markers and cannot be stored by callers (SEC-SEV-001)"
            )
        clean_tags = _validated_tags(tags)
        clean_relationships = list(relationships) if relationships is not None else []
        _require_json_relationships(clean_relationships)
        now = self._clock()
        try:
            return MemoryEntry(
                # uuid4().hex: 32 lowercase hex chars — passes the
                # [a-zA-Z0-9_-] leaf-name whitelist unmodified (export
                # filenames, security req #2).
                id=uuid.uuid4().hex,
                content=content,
                type=type,
                source=source,
                key=key,
                tags=clean_tags,
                relationships=clean_relationships,
                session_id=session_id,
                base_importance=base_importance,
                created_at=now,
                updated_at=now,
                last_accessed_at=now,
                source_entry_ids=list(_source_entry_ids) if _source_entry_ids else [],
                pinned=pinned,
            )
        except ValueError as exc:
            raise MemoryStoreError(f"invalid entry: {exc}") from exc

    def _persist_with_supersede(self, entry: MemoryEntry, embedding: bytes | None) -> None:
        """The D1 write sequence; MUST run inside an ambient transaction."""
        if entry.key is not None:
            old = self._backend.get_by_key(entry.key)  # active rows only
            if old is not None:
                self._backend.update(
                    old["id"],
                    {
                        "archived": True,
                        "archive_reason": ArchiveReason.SUPERSEDED.value,
                        "updated_at": entry.created_at.isoformat(),
                    },
                )
                entry.relationships.append(
                    Relationship(target_id=old["id"], relationship_type="supersedes")
                )
        self._backend.create(entry.to_dict(), embedding=embedding)

    # ----------------------------------------------------------- import seams

    def mint_id(self) -> str:
        """Mint a fresh entry id using the same seam ``create()`` uses.

        Exposed so the importer can pre-mint the whole-file id map (references
        must be remapped before any row is written) with ids that are
        format-identical to natively created ones (uuid4 hex).
        """
        return uuid.uuid4().hex

    def get_meta(self) -> dict[str, Any]:
        """The identity/meta row (D3): schema_version + embedding identity.

        Passthrough to the backend so the exporter can stamp the envelope and
        the importer can decide vector reuse vs re-embed without reaching
        through to the backend.
        """
        return self._backend.get_meta()

    def get_embedding(self, entry_id: str) -> list[float] | None:
        """Unpack the embedding BLOB (ADR-015 source of truth) to floats.

        Reads the opaque bytes the store persisted and applies the exact
        inverse of the pack used on write. Returns None when the entry has no
        embedding.

        Raises:
            StorageError: On a missing id (propagated from the backend) or a
                corrupt BLOB.
        """
        blob = self._backend.get_embedding(entry_id)
        return None if blob is None else unpack_embedding(blob)

    def transaction(self) -> AbstractContextManager[None]:
        """Explicit all-or-nothing scope over multiple ``restore()`` calls.

        Wraps the backend transaction and marks this thread as "in a store
        transaction" so ``restore()`` skips its own self-wrap (no double-BEGIN)
        and every nested restore shares the one atomic scope — a mid-import
        failure rolls back every row (import atomicity).
        """
        return self._transaction_scope()

    @contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        """Backend transaction + a per-thread in-transaction marker."""
        with self._backend.transaction():
            previous = getattr(self._txn_state, "active", False)
            self._txn_state.active = True
            try:
                yield
            finally:
                self._txn_state.active = previous

    def restore(
        self,
        entry: MemoryEntry,
        *,
        embedding: list[float] | None = None,
        supersede_live_key: bool = False,
    ) -> MemoryEntry:
        """Persist an entry VERBATIM (the import path) — no minting, no restamp.

        Unlike ``create()``, ``restore()`` keeps the given ``id``, timestamps,
        ``access_count``, archived state, ``archive_reason``, and ``pinned``
        exactly — the caller (importer) has already minted the id and remapped
        references. The embedding is packed and written as the row's BLOB
        (ADR-015 source of truth). A LIVE keyed row lands in the KV index by
        construction (the partial unique key column IS the index); the semantic
        index is NOT touched here — the store is deliberately index-agnostic,
        and ``memory.py`` reconciles vectors from the persisted BLOBs.

        Multi-write atomicity (the supersede path) is guaranteed either way:
        under an ambient ``store.transaction()`` (the importer's batch) all
        restores share one atomic scope; called standalone, restore self-wraps
        the archive+insert pair in its own backend transaction so a failed
        insert never leaves the live holder archived-then-orphaned (C4).

        On a LIVE-key collision:
            ``supersede_live_key=True`` applies D1 — archive the live holder
            ``SUPERSEDED`` and append a ``supersedes`` back-link to ``entry``.
            ``supersede_live_key=False`` raises ``MemoryStoreError`` (a race
            guard; the importer pre-checks via ``exists()``).

        Archived imported entries never trigger supersede logic: the partial
        unique index lets any number of archived rows share a key.

        Args:
            entry: The fully-formed entry to persist verbatim.
            embedding: The vector to store as the BLOB, or None.
            supersede_live_key: Whether a live-key collision supersedes.

        Returns:
            ``entry`` (carrying the appended supersede back-link, if any).

        Raises:
            MemoryStoreError: Live-key collision without ``supersede_live_key``.
            StorageError: Propagated backend failure.
        """
        packed = pack_embedding(embedding) if embedding is not None else None
        if getattr(self._txn_state, "active", False):
            self._restore_write(entry, packed, supersede_live_key=supersede_live_key)
        else:
            with self._backend.transaction():
                self._restore_write(entry, packed, supersede_live_key=supersede_live_key)
        return entry

    def _restore_write(
        self, entry: MemoryEntry, packed: bytes | None, *, supersede_live_key: bool
    ) -> None:
        """The verbatim persist + D1 supersede; MUST run inside an atomic scope."""
        if entry.key is not None and not entry.archived:
            old = self._backend.get_by_key(entry.key)  # active rows only
            if old is not None:
                if not supersede_live_key:
                    raise MemoryStoreError(
                        f"key {entry.key!r} is held by an active entry; restore requires "
                        "supersede_live_key=True to supersede it"
                    )
                self._backend.update(
                    old["id"],
                    {
                        "archived": True,
                        "archive_reason": ArchiveReason.SUPERSEDED.value,
                        "updated_at": self._clock().isoformat(),
                    },
                )
                entry.relationships.append(
                    Relationship(target_id=old["id"], relationship_type="supersedes")
                )
        self._backend.create(entry.to_dict(), embedding=packed)

    # ------------------------------------------------------------------- read

    def get_by_id(self, entry_id: str, *, touch: bool = True) -> MemoryEntry | None:
        """Entry by id; archived entries ARE returned (back-link traversal).

        Touch applies to live entries only: the backend touch skips archived
        rows by contract, and the hydrated object mirrors via
        ``entry.touch(now)`` so DB and returned object agree without a
        second read. ``importance`` stays None (D2).
        """
        row = self._backend.read(entry_id)
        if row is None:
            return None
        entry = self._hydrate(row)
        self._touch_live(entry, touch)
        return entry

    def get_by_key(self, key: str, *, touch: bool = True) -> MemoryEntry | None:
        """Active entry holding ``key`` (partial-index path); same touch mirroring."""
        row = self._backend.get_by_key(key)
        if row is None:
            return None
        entry = self._hydrate(row)
        self._touch_live(entry, touch)
        return entry

    def _touch_live(self, entry: MemoryEntry, touch: bool) -> None:
        if touch and not entry.archived:
            now = self._clock()
            self._backend.touch([entry.id], now.isoformat())
            entry.touch(now)

    def exists(self, key: str) -> bool:
        """Active-key existence; NO entry load, NO touch (D3: not an access)."""
        return self._backend.key_exists(key)

    def list(
        self,
        *,
        tags: _StrList | None = None,
        types: _TypeList | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        created_before: datetime | None = None,
        accessed_before: datetime | None = None,
        min_base_importance: float | None = None,
        pinned: bool | None = None,
        include_archived: bool = False,
        archive_reasons: _ReasonList | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> _EntryList:
        """Typed façade over the backend filters. NO touch (listing != access).

        Raises:
            MemoryStoreError: When ``limit`` < 1.
        """
        if limit < 1:
            raise MemoryStoreError("limit must be >= 1")
        filters = self._build_filters(
            tags=tags,
            types=types,
            agent_id=agent_id,
            session_id=session_id,
            since=since,
            created_before=created_before,
            accessed_before=accessed_before,
            min_base_importance=min_base_importance,
            pinned=pinned,
            include_archived=include_archived,
            archive_reasons=archive_reasons,
        )
        return [self._hydrate(row) for row in self._backend.list(filters, limit, offset)]

    def count(self, **filters: Any) -> int:
        """Entry count under the same filter façade as ``list()``.

        Raises:
            MemoryStoreError: On an unknown filter keyword.
        """
        try:
            built = self._build_filters(**filters)
        except TypeError as exc:
            raise MemoryStoreError(f"unknown filter: {exc}") from exc
        return self._backend.count(built)

    def _build_filters(
        self,
        *,
        tags: _StrList | None = None,
        types: _TypeList | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        since: datetime | None = None,
        created_before: datetime | None = None,
        updated_before: datetime | None = None,
        accessed_before: datetime | None = None,
        min_base_importance: float | None = None,
        pinned: bool | None = None,
        include_archived: bool = False,
        archive_reasons: _ReasonList | None = None,
    ) -> dict[str, Any]:
        """Enums -> values, datetimes -> ISO; omit absent filters."""
        filters: dict[str, Any] = {"include_archived": bool(include_archived)}
        if tags is not None:
            filters["tags"] = list(tags)
        if types is not None:
            filters["types"] = [t.value for t in types]
        if agent_id is not None:
            filters["agent_id"] = agent_id
        if session_id is not None:
            filters["session_id"] = session_id
        if since is not None:
            filters["since"] = _aware_iso(since, "since")
        if created_before is not None:
            filters["created_before"] = _aware_iso(created_before, "created_before")
        if updated_before is not None:
            filters["updated_before"] = _aware_iso(updated_before, "updated_before")
        if accessed_before is not None:
            filters["accessed_before"] = _aware_iso(accessed_before, "accessed_before")
        if min_base_importance is not None:
            filters["min_base_importance"] = min_base_importance
        if pinned is not None:
            filters["pinned"] = bool(pinned)
        if archive_reasons is not None:
            filters["archive_reasons"] = [r.value for r in archive_reasons]
        return filters

    # ------------------------------------------------- key scans (kv_index)

    def scan_keys(self, prefix: str, limit: int = 100) -> _EntryList:
        """Active keyed entries with the literal prefix, key ASC. NO touch."""
        if limit < 1:
            raise MemoryStoreError("limit must be >= 1")
        return [self._hydrate(row) for row in self._backend.scan_key_prefix(prefix, limit)]

    def list_keys(self, prefix: str = "") -> _StrList:
        """Active keys, sorted ASC. NO touch, no entry load."""
        return self._backend.list_keys(prefix)

    # ----------------------------------------------------------------- update

    def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        tags: _StrList | None = None,
        relationships: _RelationshipList | None = None,
        pinned: bool | None = None,
    ) -> MemoryEntry:
        """Merge the given fields; None = leave unchanged. Bumps ``updated_at``.

        Immutable/refused fields are not even parameters: ``base_importance``
        (D2), ``id``, ``type``, ``key`` (supersede via ``create()`` is the
        key-change mechanism), ``created_at``, access fields, archived state
        (use ``archive``/``unarchive``). A content change does NOT re-embed
        here — ``memory.py`` owns re-embedding + index update around this
        call, so the persisted embedding BLOB is stale until it does.

        Raises:
            MemoryStoreError: On a missing id or invalid field values.
        """
        fields: dict[str, Any] = {"updated_at": self._clock().isoformat()}
        if content is not None:
            if not content:
                raise MemoryStoreError("content must be non-empty")
            fields["content"] = content
        if tags is not None:
            fields["tags"] = _validated_tags(tags)
        if relationships is not None:
            clean = list(relationships)
            _require_json_relationships(clean)
            fields["relationships"] = [rel.to_dict() for rel in clean]
        if pinned is not None:
            fields["pinned"] = bool(pinned)
        row = self._backend.update(entry_id, fields)
        if row is None:
            raise MemoryStoreError(f"no entry with id {entry_id!r}")
        return self._hydrate(row)

    def rebase_importance(self, entry_id: str, new_base: float, *, now: datetime) -> MemoryEntry:
        """EXPLICIT importance rebase (blueprint-memory amendment, owner-approved).

        In one backend update: sets ``base_importance = new_base`` AND resets
        the decay anchor (``last_accessed_at = now``), bumping ``updated_at``.
        Semantics: *"as of now, this memory is worth new_base"* — a fresh
        statement of importance whose decay starts when it was made. This is
        the ONLY importance mutation path (D2 stands: ``update()`` remains
        importance-free; decay/eviction/curation can never reach this).

        A rebase is NOT an access: ``access_count`` is untouched.

        Args:
            entry_id: The entry to rebase.
            new_base: The new base importance, within [0.0, 1.0].
            now: The rebase instant (tz-aware); becomes the new decay anchor.

        Returns:
            The rebased entry.

        Raises:
            MemoryStoreError: On a missing id, out-of-range value, or naive
                ``now``.
        """
        if not 0.0 <= new_base <= 1.0:
            raise MemoryStoreError(f"importance must be within [0.0, 1.0], got {new_base!r}")
        instant = _aware_iso(now, "now")
        row = self._backend.update(
            entry_id,
            {
                "base_importance": new_base,
                "last_accessed_at": instant,
                "updated_at": instant,
            },
        )
        if row is None:
            raise MemoryStoreError(f"no entry with id {entry_id!r}")
        return self._hydrate(row)

    def set_embedding(self, entry_id: str, embedding: bytes | None) -> None:
        """Passthrough for memory.py's re-embed path.

        Only the missing-id case is translated; genuine backend failures
        (closed backend, I/O errors) propagate as ``StorageError`` so they
        are never masked as CRUD misses.

        Raises:
            MemoryStoreError: When no entry with ``entry_id`` exists.
            StorageError: Propagated backend failure.
        """
        if self._backend.read(entry_id) is None:
            raise MemoryStoreError(f"no entry with id {entry_id!r}")
        self._backend.set_embedding(entry_id, embedding)

    # ---------------------------------------------------- archive lifecycle

    def archive(self, entry_id: str, reason: ArchiveReason) -> MemoryEntry:
        """Archive a live entry with a reason (D3); explicit state machine.

        Check-then-act runs inside one backend transaction so a concurrent
        archive can never overwrite an already-recorded reason (a SUMMARIZED
        source relabeled EVICTED would be purged by default — the audit
        regression this protects).

        Raises:
            MemoryStoreError: On a missing id or an already-archived entry.
        """
        with self._backend.transaction():
            updated = self._archive_checked(entry_id, reason)
        return self._hydrate(updated)

    def _archive_checked(self, entry_id: str, reason: ArchiveReason) -> dict[str, Any]:
        """Read + state check + archive update; MUST run inside an ambient
        transaction (callers: ``archive``, ``forget``)."""
        row = self._backend.read(entry_id)
        if row is None:
            raise MemoryStoreError(f"no entry with id {entry_id!r}")
        if row["archived"]:
            raise MemoryStoreError(f"entry {entry_id!r} is already archived")
        updated = self._backend.update(
            entry_id,
            {
                "archived": True,
                "archive_reason": reason.value,
                "updated_at": self._clock().isoformat(),
            },
        )
        return self._require_row(updated, entry_id)

    def unarchive(self, entry_id: str) -> MemoryEntry:
        """Restore an archived entry; refuses when its key is actively held.

        Raises:
            MemoryStoreError: On a missing id, a non-archived entry, or a key
                currently held by a live entry (checked BEFORE the unique
                index can fire).
        """
        with self._backend.transaction():
            row = self._backend.read(entry_id)
            if row is None:
                raise MemoryStoreError(f"no entry with id {entry_id!r}")
            if not row["archived"]:
                raise MemoryStoreError(f"entry {entry_id!r} is not archived")
            if row["key"] is not None and self._backend.key_exists(row["key"]):
                raise MemoryStoreError(
                    f"key {row['key']!r} is held by an active entry; cannot unarchive"
                )
            updated = self._backend.update(
                entry_id,
                {
                    "archived": False,
                    "archive_reason": None,
                    "updated_at": self._clock().isoformat(),
                },
            )
        return self._hydrate(self._require_row(updated, entry_id))

    def forget(self, key: str, *, hard: bool = False) -> bool:
        """PUBLIC forget verb (D6); archives FORGOTTEN (or hard-deletes).

        The key lookup and the archive/delete run inside one backend
        transaction (check-then-act hygiene under the many-threads model).

        Returns:
            False when no active entry holds ``key`` — never raises for a miss.
        """
        with self._backend.transaction():
            row = self._backend.get_by_key(key)
            if row is None:
                return False
            if hard:
                self._delete(row["id"])
            else:
                self._archive_checked(row["id"], ArchiveReason.FORGOTTEN)
        return True

    def forget_by_id(self, entry_id: str, *, hard: bool = False) -> bool:
        """Forget an entry by id (blueprint-memory amendment; MCP A3 requires
        reaching unkeyed entries).

        Mirrors ``forget(key)`` miss semantics: a missing id or an
        already-archived entry returns ``False``, never raises — and never
        overwrites an existing archive reason (a SUMMARIZED source relabeled
        FORGOTTEN would lose its purge protection).

        Args:
            entry_id: The entry to forget.
            hard: When True, hard-delete the row (embedding dies with it).

        Returns:
            True when a live entry was archived (or hard-deleted).
        """
        with self._backend.transaction():
            row = self._backend.read(entry_id)
            if row is None or row["archived"]:
                return False
            if hard:
                self._delete(entry_id)
            else:
                self._archive_checked(entry_id, ArchiveReason.FORGOTTEN)
        return True

    def _purge_filters(
        self, reasons: _ReasonList | None, older_than: timedelta | None
    ) -> dict[str, Any]:
        """The filter dict shared by ``purge_archived`` and ``count_archived`` (DRY).

        ``reasons=None`` defaults to everything EXCEPT ``SUMMARIZED``:
        summarization sources (ADR-009 "originals recoverable") are matched
        only when ``SUMMARIZED`` is explicitly listed. ``older_than`` filters
        on ``updated_at`` — the ``archive()`` timestamp (the schema has no
        ``archived_at`` column; the two are equivalent by construction).
        """
        selected = list(reasons) if reasons is not None else list(_DEFAULT_PURGE_REASONS)
        filters: dict[str, Any] = {"archive_reasons": [r.value for r in selected]}
        if older_than is not None:
            filters["updated_before"] = (self._clock() - older_than).isoformat()
        return filters

    def count_archived(
        self,
        *,
        reasons: _ReasonList | None = None,
        older_than: timedelta | None = None,
    ) -> int:
        """Count rows ``purge_archived`` WOULD delete under the same filter (dry-run).

        Read-only: never deletes, never touches. Backs the maintenance CLI's
        ``--dry-run`` preview and confirmation prompt.

        Returns:
            The count of matching archived rows.
        """
        return self._backend.count(self._purge_filters(reasons, older_than))

    def purge_archived(
        self,
        *,
        reasons: _ReasonList | None = None,
        older_than: timedelta | None = None,
    ) -> int:
        """Hard-delete archived rows, reason-aware (D3).

        ``reasons=None`` defaults to everything EXCEPT ``SUMMARIZED``:
        summarization sources (ADR-009 "originals recoverable") are purged
        only when ``SUMMARIZED`` is explicitly listed. ``older_than`` filters
        on ``updated_at`` — the ``archive()`` timestamp (the schema has no
        ``archived_at`` column; the two are equivalent by construction).

        Returns:
            The number of rows deleted.
        """
        filters = self._purge_filters(reasons, older_than)
        with self._backend.transaction():
            total = self._backend.count(filters)
            if total == 0:
                return 0
            rows = self._backend.list(dict(filters), limit=total)
            return self._backend.delete_many([row["id"] for row in rows])

    # --------------------------------------------------------------- internal

    def _delete(self, entry_id: str) -> bool:
        """INTERNAL hard delete (public verbs are forget/archive/purge, D6)."""
        return self._backend.delete(entry_id)

    def touch_entries(self, entry_ids: _StrList, now: datetime | None = None) -> int:
        """Batched access recording for search-hit / curate-inclusion (D3).

        One backend statement, one transaction; archived ids are skipped.

        Returns:
            The number of live entries touched.
        """
        instant = now if now is not None else self._clock()
        return self._backend.touch(list(entry_ids), instant.isoformat())

    def _hydrate(self, row: dict[str, Any]) -> MemoryEntry:
        """dict -> MemoryEntry; hydration failures are corrupt rows.

        Raises:
            StorageError: Wrapping ``ValueError``/``KeyError`` from
                ``MemoryEntry.from_dict`` (entry.py's documented contract).
        """
        try:
            return MemoryEntry.from_dict(row)
        except (ValueError, KeyError) as exc:
            raise StorageError(f"corrupt row {row.get('id')!r}: {exc}") from exc

    @staticmethod
    def _require_row(row: dict[str, Any] | None, entry_id: str) -> dict[str, Any]:
        """Defensive narrow for update-after-read paths (single-writer)."""
        if row is None:  # pragma: no cover - unreachable under the write lock
            raise MemoryStoreError(f"entry {entry_id!r} vanished mid-operation")
        return row


def _aware_iso(value: datetime, name: str) -> str:
    """Serialize a filter datetime; naive input is a caller bug at THIS boundary.

    Raises:
        MemoryStoreError: On a naive (offset-less) datetime — surfaced here
            as a store-level caller error instead of leaking the backend's
            StorageError.
    """
    if value.tzinfo is None:
        raise MemoryStoreError(f"{name} must be timezone-aware (naive datetime rejected)")
    return value.isoformat()


def _validated_tags(tags: Sequence[str] | None) -> _StrList:
    """Dedupe preserving order; every tag must be a non-empty string."""
    if not tags:
        return []
    for tag in tags:
        if not isinstance(tag, str) or not tag:
            raise MemoryStoreError(f"tags must be non-empty strings, got {tag!r}")
    return list(dict.fromkeys(tags))


def _require_json_relationships(relationships: Sequence[Relationship]) -> None:
    """Reject relationship metadata the JSON codec cannot represent (spec §8)."""
    try:
        json.dumps([rel.to_dict() for rel in relationships])
    except (TypeError, ValueError) as exc:
        raise MemoryStoreError(f"relationship metadata is not JSON-serializable: {exc}") from exc

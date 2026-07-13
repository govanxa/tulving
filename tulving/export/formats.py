"""JSON round-trip export/import (v0.1's only format — ADR-016).

Exports are an emission surface: redaction is ON by default (CLAUDE.md
security req #1) with a loud, keyword-only opt-out for trusted backups.
Imported files are untrusted input: structure is validated, vectors are
re-embedded unless provably compatible, IDs are re-minted and remapped.

Dependency rule (blueprint §Dependency Map): this module imports only
``tulving.entry``, ``tulving.store``, ``tulving.security``, ``tulving.enums``,
``tulving.exceptions`` + stdlib. It does NOT import ``memory``, ``adapters.*``
(the embedder is a local structural ``Protocol``), or ``context.*``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from re import Pattern
from typing import Any, Final, Protocol

from tulving.entry import MemoryEntry
from tulving.exceptions import MemoryStoreError, StorageError
from tulving.security import (
    contain_path,
    redact_text,
    should_mask_content,
    validate_leaf_name,
)
from tulving.store import MemoryStore

_EXPORT_FORMAT: Final[str] = "tulving-export"
_EXPORT_FORMAT_VERSION: Final[int] = 1
_VALID_CONFLICT_MODES: Final[frozenset[str]] = frozenset({"skip", "supersede"})
_JSON_SUFFIX: Final[str] = ".json"
_PAGE: Final[int] = 500


class _EmbedderLike(Protocol):
    """Structural mirror of ``adapters.embeddings.EmbeddingAdapter``.

    Defined locally so formats.py does not import ``adapters.*`` (dependency
    rule: entry, store, security only). Any ``EmbeddingAdapter`` satisfies it.
    """

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def distance_metric(self) -> str: ...


@dataclass
class ImportReport:
    """Outcome of one ``from_json()`` call. Counts per spec §2 + warnings."""

    entries_imported: int = 0
    entries_skipped: int = 0  # key-conflict skips + structurally invalid entries
    entries_reembedded: int = 0  # vectors regenerated from content
    embeddings_reused: int = 0  # exported vectors accepted as-is (exact model match)
    id_remappings: int = 0  # REFERENCE rewrites applied (rel targets + source_entry_ids)
    errors: list[str] = field(default_factory=list)  # per-entry drops
    warnings: list[str] = field(default_factory=list)  # non-fatal notices


@dataclass
class _Pending:
    """One surviving entry as it moves through the import phases."""

    entry: MemoryEntry
    old_id: str
    raw_embedding: Any  # the file's per-entry "embedding" value (list | None | absent)
    vector: list[float] | None = None
    needs_embed: bool = False


# --------------------------------------------------------------------------- #
# Exporter
# --------------------------------------------------------------------------- #


class MemoryExporter:
    """Serialize full memory state to a JSON envelope on disk (ADR-016)."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        allowed_root: str | Path | None = None,
        sensitive_key_patterns: Sequence[Pattern[str]] | None = None,
        explicit_key_patterns: Sequence[Pattern[str]] | None = None,
    ) -> None:
        """Bind the store and the redaction / path-containment policy.

        Args:
            store: The memory store to export from.
            allowed_root: Directory export files must stay under; ``None``
                defaults to ``Path.cwd()`` resolved at export time. A blank
                string is refused by containment (never the filesystem root).
            sensitive_key_patterns: Compiled sensitive-key patterns (Memory
                passes ``compile_key_patterns(sensitive_keys)``) used for the
                surgical ``redact_text`` scrub; ``None`` falls back to
                security's module defaults.
            explicit_key_patterns: ONLY the user-declared
                ``Memory(sensitive_keys=...)`` patterns (``compile_explicit_
                patterns``), threaded SEPARATELY from ``sensitive_key_patterns``
                so a user declaration masks unconditionally even on overlap
                with a built-in default (D-v02-7 Q3). ``None`` when the
                caller declared none. Do NOT reverse-derive this from the
                merged ``sensitive_key_patterns`` — compile it once at the
                ``Memory`` boundary and pass it down.
        """
        self._store = store
        self._allowed_root = allowed_root
        self._key_patterns = sensitive_key_patterns
        self._explicit_key_patterns = explicit_key_patterns

    def to_json(
        self,
        path: str | Path,
        *,
        include_embeddings: bool = False,
        include_archived: bool = False,
        include_sensitive: bool = False,
    ) -> None:
        """Write the export envelope to ``path`` atomically.

        Redaction is ON unless ``include_sensitive=True`` (keyword-only,
        loud): the envelope records ``redacted``. With embeddings enabled a
        vector is exported only when its exported content is byte-identical to
        the stored content (the vector-consistency rule closes the redaction
        side channel).

        WARNING: ``include_sensitive=True`` writes plaintext secrets to disk
        (no encryption at rest, ADR-010) — the operator's explicit choice.

        Args:
            path: Destination path; leaf-validated, directory-contained.
            include_embeddings: Emit per-entry vectors (from the BLOBs).
            include_archived: Include archived entries (lossless chains).
            include_sensitive: Opt OUT of redaction (backup mode).

        Raises:
            SecurityError: On a path violation (traversal / bad leaf / blank
                root).
            StorageError: On serialization or I/O failure.
        """
        final_path = self._validate_target_path(path)
        meta = self._store.get_meta()
        entries_out = [
            self._export_entry(entry, include_embeddings, include_sensitive)
            for entry in self._iter_entries(include_archived)
        ]
        envelope: dict[str, Any] = {
            "format": _EXPORT_FORMAT,
            "format_version": _EXPORT_FORMAT_VERSION,
            "schema_version": meta["schema_version"],
            "exported_at": datetime.now(UTC).isoformat(),
            "redacted": not include_sensitive,
            "include_archived": include_archived,
            "embeddings": (
                {
                    "model_id": meta["embedding_model_id"],
                    "dimension": meta["embedding_dimension"],
                    "distance_metric": meta["distance_metric"],
                }
                if include_embeddings
                else None
            ),
            "entry_count": len(entries_out),
            "entries": entries_out,
        }
        self._atomic_write(final_path, envelope)

    def _iter_entries(self, include_archived: bool) -> list[MemoryEntry]:
        """Paginated full listing for export (does NOT bump last_accessed_at)."""
        collected: list[MemoryEntry] = []
        offset = 0
        while True:
            page = self._store.list(include_archived=include_archived, limit=_PAGE, offset=offset)
            collected.extend(page)
            if len(page) < _PAGE:
                return collected
            offset += _PAGE

    def _export_entry(
        self, entry: MemoryEntry, include_embeddings: bool, include_sensitive: bool
    ) -> dict[str, Any]:
        """Redaction pipeline (unless opted out) + optional vector attach."""
        stored_content = entry.content
        if include_sensitive:
            data = entry.to_dict()
            content_changed = False
        else:
            verdict = should_mask_content(
                entry.key or "", entry.content, explicit_patterns=self._explicit_key_patterns
            )
            data = entry.to_safe_dict(content_is_sensitive=verdict)
            data["content"] = redact_text(data["content"], key_patterns=self._key_patterns)
            content_changed = data["content"] != stored_content
        if include_embeddings:
            # Vector-consistency rule: never export a vector for content the
            # file no longer carries verbatim (leak vector + reuse bug).
            data["embedding"] = None if content_changed else self._store.get_embedding(entry.id)
        return data

    def _validate_target_path(self, path: str | Path) -> Path:
        """Validate the leaf (whitelist) and contain the directory (realpath).

        Returns the resolved final path (validated dir / validated leaf +
        ``.json``). ``.json`` is appended by trusted code AFTER validation.
        """
        root = self._allowed_root if self._allowed_root is not None else Path.cwd()
        candidate = Path(path)
        leaf = candidate.name
        stem = leaf[: -len(_JSON_SUFFIX)] if leaf.lower().endswith(_JSON_SUFFIX) else leaf
        validate_leaf_name(stem)
        final_leaf = stem + _JSON_SUFFIX
        # A bare leaf yields ``Path(".")`` here — contain_path resolves that to
        # the cwd, which the default allowed_root also resolves to.
        resolved_dir = contain_path(candidate.parent, root)
        return resolved_dir / final_leaf

    @staticmethod
    def _atomic_write(final_path: Path, envelope: dict[str, Any]) -> None:
        """Serialize to ``<final>.tmp`` then ``os.replace`` — never a half file.

        The payload is built BEFORE any file is touched, so a serialization
        failure leaves nothing at the final path.
        """
        payload = json.dumps(envelope, ensure_ascii=False, indent=2)
        tmp_path = final_path.with_name(final_path.name + ".tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, final_path)
        except OSError as exc:
            with suppress(OSError):
                tmp_path.unlink()
            raise StorageError(f"could not write export file: {exc}") from exc


# --------------------------------------------------------------------------- #
# Importer
# --------------------------------------------------------------------------- #


class MemoryImporter:
    """Load a JSON envelope back into a store (untrusted input; all-or-nothing)."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        embedder: _EmbedderLike | None = None,
    ) -> None:
        """Bind the store and (optionally) the active embedding adapter.

        Args:
            store: The destination store.
            embedder: Active embedding adapter (Memory passes its own).
                ``None`` imports entries without vectors — invisible to
                semantic search until a rebuild — and reports a warning.
        """
        self._store = store
        self._embedder = embedder

    def from_json(self, path: str | Path, *, on_key_conflict: str = "skip") -> ImportReport:
        """Import the envelope at ``path`` and return the outcome report.

        Structurally invalid entries are skipped and recorded; the DB write is
        a single all-or-nothing transaction. Vectors are reused only under an
        exact meta match, otherwise re-embedded (or absent when no embedder).
        Never touches ``last_accessed_at``/``access_count``, never rewrites
        ``agent_id``, never adjusts ``base_importance``/tags.

        Args:
            path: Source file (deliberately NOT containment-validated —
                read-only intent).
            on_key_conflict: ``"skip"`` (default) leaves the live store
                untouched; ``"supersede"`` applies D1. Any other value is a
                ``ValueError`` raised before the file is read.

        Returns:
            The :class:`ImportReport`.

        Raises:
            ValueError: On an unknown ``on_key_conflict``.
            MemoryStoreError: On a malformed file / bad envelope / incompatible
                version.
            StorageError: On I/O failure or an embedding failure.
        """
        if on_key_conflict not in _VALID_CONFLICT_MODES:
            raise ValueError(
                f"on_key_conflict must be one of {sorted(_VALID_CONFLICT_MODES)}, "
                f"got {on_key_conflict!r}"
            )
        report = ImportReport()
        envelope = self._parse_envelope(path, report)
        pending = self._validate_entries(envelope["entries"], report)
        self._remap_references(pending, report)
        self._resolve_vectors(pending, envelope.get("embeddings"), report)
        self._write(pending, on_key_conflict, report)
        return report

    # ---------------------------------------------------------- phase 0: parse

    def _parse_envelope(self, path: str | Path, report: ImportReport) -> dict[str, Any]:
        """Read + gate the envelope (nothing written; fatal errors only)."""
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"cannot read export file: {exc}") from exc
        try:
            envelope = json.loads(text)
        except (ValueError, RecursionError) as exc:
            # JSONDecodeError is a ValueError; RecursionError (deeply nested
            # input) is a RuntimeError — both are untrusted-input parse
            # failures, never a leaked, undocumented exception type (S1).
            raise MemoryStoreError("not a Tulving export: invalid JSON") from exc
        if not isinstance(envelope, dict):
            raise MemoryStoreError("not a Tulving export: top-level value is not an object")
        if envelope.get("format") != _EXPORT_FORMAT:
            raise MemoryStoreError("not a Tulving export: unknown 'format'")
        fmt_version = envelope.get("format_version")
        if not isinstance(fmt_version, int) or fmt_version > _EXPORT_FORMAT_VERSION:
            raise MemoryStoreError(
                f"unsupported export format_version (this build supports "
                f"<= {_EXPORT_FORMAT_VERSION})"
            )
        schema_version = envelope.get("schema_version")
        store_schema = self._store.get_meta()["schema_version"]
        if not isinstance(schema_version, int) or schema_version > store_schema:
            raise MemoryStoreError(
                "export schema_version is newer than this store supports; refusing to import "
                "(no downgrades)"
            )
        entries = envelope.get("entries")
        if not isinstance(entries, list):
            raise MemoryStoreError("not a Tulving export: 'entries' is not a list")
        if envelope.get("redacted"):
            report.warnings.append(
                "source file was exported with redaction; [REDACTED] placeholders may be present "
                "in imported content"
            )
        return envelope

    # ------------------------------------------------- phase 1: entry validate

    def _validate_entries(self, entries: list[Any], report: ImportReport) -> list[_Pending]:
        """Reuse ``MemoryEntry.from_dict``; skip + record invalid entries.

        Duplicate ids within the file are skipped here (not left to a backend
        PK collision) so one crafted collision never aborts the whole import
        (C1) — the id map must be one-to-one before any row is written.
        """
        pending: list[_Pending] = []
        seen_ids: set[str] = set()
        for index, element in enumerate(entries):
            if not isinstance(element, dict):
                report.entries_skipped += 1
                report.errors.append(f"entry #{index}: not a JSON object")
                continue
            old_id = element.get("id")
            label = repr(old_id) if isinstance(old_id, str) else f"#{index}"
            try:
                entry = MemoryEntry.from_dict(element)
            except (ValueError, KeyError, TypeError) as exc:
                report.entries_skipped += 1
                report.errors.append(f"entry {label}: invalid ({redact_text(str(exc))})")
                continue
            if entry.id in seen_ids:
                report.entries_skipped += 1
                report.errors.append(f"entry {label}: duplicate id in file (skipped)")
                continue
            seen_ids.add(entry.id)
            pending.append(
                _Pending(entry=entry, old_id=entry.id, raw_embedding=element.get("embedding"))
            )
        return pending

    # --------------------------------------------------- phase 2: id remapping

    def _remap_references(self, pending: list[_Pending], report: ImportReport) -> None:
        """Mint fresh ids for every survivor and rewrite references through
        the map; dangling references are dropped and recorded."""
        id_map = {item.old_id: self._store.mint_id() for item in pending}
        for item in pending:
            entry = item.entry
            new_id = id_map[item.old_id]
            entry.id = new_id
            kept_rels = []
            for rel in entry.relationships:
                mapped = id_map.get(rel.target_id)
                if mapped is None:
                    report.errors.append(
                        f"entry {new_id!r}: dropped dangling {rel.relationship_type} "
                        f"relationship to {rel.target_id!r}"
                    )
                    continue
                rel.target_id = mapped
                report.id_remappings += 1
                kept_rels.append(rel)
            entry.relationships = kept_rels
            kept_sids = []
            for sid in entry.source_entry_ids:
                mapped = id_map.get(sid)
                if mapped is None:
                    report.errors.append(
                        f"entry {new_id!r}: dropped dangling source reference to {sid!r}"
                    )
                    continue
                kept_sids.append(mapped)
                report.id_remappings += 1
            entry.source_entry_ids = kept_sids

    # ------------------------------------------------------- phase 3: vectors

    def _resolve_vectors(
        self, pending: list[_Pending], embeddings_block: Any, report: ImportReport
    ) -> None:
        """Reuse exported vectors only on an exact meta match; else re-embed
        (before any write, so API costs precede the transaction)."""
        meta = self._store.get_meta()
        dimension = meta["embedding_dimension"]
        meta_match = (
            isinstance(embeddings_block, dict)
            and meta["embedding_model_id"] is not None
            and embeddings_block.get("model_id") == meta["embedding_model_id"]
            and embeddings_block.get("dimension") == dimension
            and embeddings_block.get("distance_metric") == meta["distance_metric"]
        )
        for item in pending:
            raw = item.raw_embedding
            reused = False
            if (
                meta_match
                and isinstance(raw, list)
                and dimension is not None
                and len(raw) == dimension
            ):
                # Defensive coercion: a crafted file can pass model_id/dimension/
                # metric (often public defaults) yet carry non-numeric values of
                # the exact declared length — float() must never escape (S2).
                try:
                    item.vector = [float(value) for value in raw]
                except (TypeError, ValueError):
                    item.vector = None
                else:
                    report.embeddings_reused += 1
                    reused = True
            item.needs_embed = not reused
        needing = [item for item in pending if item.needs_embed]
        if not needing:
            return
        if self._embedder is None:
            report.warnings.append(
                "no embedding adapter: imported entries are invisible to semantic search "
                "until rebuild()"
            )
            return
        try:
            vectors = self._embedder.embed_batch([item.entry.content for item in needing])
            # The zip consumption stays INSIDE the guard so a count mismatch
            # (strict=True) or a non-numeric vector maps to StorageError, never
            # a raw ValueError (C3).
            for item, vector in zip(needing, vectors, strict=True):
                item.vector = [float(value) for value in vector]
                report.entries_reembedded += 1
        except Exception as exc:  # provider failure: abort before any write
            raise StorageError(
                f"re-embedding failed during import: {redact_text(str(exc))}"
            ) from exc

    # --------------------------------------------------------- phase 4: write

    def _write(self, pending: list[_Pending], on_key_conflict: str, report: ImportReport) -> None:
        """Single all-or-nothing transaction; D1 dedup per entry (file order)."""
        with self._store.transaction():
            for item in pending:
                entry = item.entry
                collides = (
                    entry.key is not None and not entry.archived and self._store.exists(entry.key)
                )
                if collides and on_key_conflict == "skip":
                    report.entries_skipped += 1
                    continue
                self._store.restore(
                    entry,
                    embedding=item.vector,
                    supersede_live_key=collides,
                )
                report.entries_imported += 1

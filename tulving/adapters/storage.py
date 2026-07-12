"""Storage backends: the D3 schema, SQLite (WAL) + in-memory implementations.

No encryption at rest in v0.1 (ADR-010): content and embeddings are stored in
plain text/bytes — do not store secrets you cannot afford on disk. Redaction
is deliberately NOT applied here: it protects *outgoing* text (curator,
export, MCP); storing redacted content would destroy data.

Backends raise ``StorageError`` only (D6) and use parameterized SQL
exclusively — the two bounded exceptions are PRAGMA integer constants and
generated ``?`` placeholder lists (values still bound). Entries cross this
boundary as plain dicts in the exact ``MemoryEntry.to_dict()`` shape: nested
``source`` dict, ISO timestamps, enum ``str`` values, no ``embedding`` and no
``importance`` key. Backends never construct ``MemoryEntry`` objects and
never mint IDs.

The vector index (hnswlib) is not this module: embedding bytes are persisted
as opaque BLOBs (the ADR-015 source of truth); ``semantic_index.py`` owns the
``.hnsw`` cache. The advisory single-writer lock file is ``memory.py``'s job.
"""

from __future__ import annotations

import copy
import json
import os
import sqlite3
import struct
import sys
import threading
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

from tulving.enums import ArchiveReason, MemoryType, SessionStatus
from tulving.exceptions import StorageError

SCHEMA_VERSION: Final[int] = 2

# Type aliases (module level so annotations inside classes that define a
# method named ``list`` never resolve against the class scope).
_Row = dict[str, Any]
_Rows = list[dict[str, Any]]
_Keys = list[str]
_LabelRows = list[tuple[int, str, bool]]

_TIMESTAMP_FIELDS: Final[frozenset[str]] = frozenset(
    {"created_at", "updated_at", "last_accessed_at"}
)

_UPDATABLE_ENTRY_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "content",
        "type",
        "source",
        "key",
        "tags",
        "relationships",
        "session_id",
        "base_importance",
        "created_at",
        "updated_at",
        "last_accessed_at",
        "access_count",
        "archived",
        "archive_reason",
        "source_entry_ids",
        "pinned",
    }
)

_FILTER_KEYS: Final[frozenset[str]] = frozenset(
    {
        "types",
        "tags",
        "agent_id",
        "session_id",
        "key_prefix",
        "pinned",
        "min_base_importance",
        "include_archived",
        "archive_reasons",
        "since",
        "created_before",
        "updated_before",
        "accessed_before",
    }
)

_TEMPORAL_FILTER_KEYS: Final[frozenset[str]] = frozenset(
    {"since", "created_before", "updated_before", "accessed_before"}
)

_UPDATABLE_SESSION_FIELDS: Final[frozenset[str]] = frozenset(
    {"goal", "agent_id", "started_at", "last_activity_at", "ended_at", "status"}
)

_SESSION_TIMESTAMP_FIELDS: Final[frozenset[str]] = frozenset(
    {"started_at", "last_activity_at", "ended_at"}
)

_SETTABLE_META_FIELDS: Final[frozenset[str]] = frozenset(
    {"embedding_model_id", "embedding_dimension", "distance_metric"}
)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def pack_embedding(vector: Sequence[float]) -> bytes:
    """Pack floats into little-endian float32 bytes.

    Deterministic across platforms; dimension = ``len(blob) // 4``. The
    backend never interprets these bytes; ``semantic_index`` validates the
    dimension against ``meta.embedding_dimension``.

    Args:
        vector: The embedding values; must be non-empty.

    Returns:
        The packed bytes.

    Raises:
        StorageError: On empty or non-numeric input.
    """
    if not vector:
        raise StorageError("cannot pack an empty embedding vector")
    try:
        return struct.pack(f"<{len(vector)}f", *vector)
    except struct.error as exc:
        raise StorageError(f"embedding vector is not packable as float32: {exc}") from exc


def unpack_embedding(blob: bytes) -> list[float]:
    """Inverse of :func:`pack_embedding`.

    Args:
        blob: Little-endian float32 bytes.

    Returns:
        The unpacked floats.

    Raises:
        StorageError: If ``blob`` is empty or its length is not a multiple
            of 4 (corrupt row).
    """
    if not blob or len(blob) % 4 != 0:
        raise StorageError(
            f"corrupt embedding blob: length {len(blob)} is not a positive multiple of 4"
        )
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


_SYNC_SEGMENT_PREFIXES: Final[tuple[str, ...]] = (
    "onedrive",
    "dropbox",
    "google drive",
    "googledrive",
    "icloud",
)
_ONEDRIVE_ENV_VARS: Final[tuple[str, ...]] = (
    "OneDrive",
    "OneDriveConsumer",
    "OneDriveCommercial",
)
_DRIVE_REMOTE: Final[int] = 4


def cloud_sync_risk(path: str | Path) -> str | None:
    """Best-effort detection of cloud-synced / network storage (ADR-015 #4).

    Pure detector — the caller decides to warn. Heuristics only, all stdlib,
    no filesystem I/O beyond a drive-type query; NEVER raises.

    Args:
        path: The candidate database path.

    Returns:
        A short human-readable reason, or None when no risk is detected.
    """
    try:
        resolved = os.path.abspath(os.fspath(path))
        for segment in Path(resolved).parts:
            lowered_segment = segment.casefold()
            if any(lowered_segment.startswith(p) for p in _SYNC_SEGMENT_PREFIXES):
                return f"path segment {segment!r} looks cloud-synced"
        lowered = resolved.casefold()
        for var in _ONEDRIVE_ENV_VARS:
            root = os.environ.get(var)
            if root and lowered.startswith(os.path.abspath(root).casefold()):
                return f"path lies under the {var} environment root"
        if resolved.startswith("\\\\"):
            return "UNC network path"
        if sys.platform == "win32":
            import ctypes

            drive = os.path.splitdrive(resolved)[0]
            if drive.endswith(":"):
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive + "\\")
                if drive_type == _DRIVE_REMOTE:
                    return "network-mapped drive"
        return None
    except Exception:  # best-effort detector must never raise
        return None


def _normalize_ts(value: str, field_name: str) -> str:
    """Canonicalize an ISO-8601 timestamp for storage.

    The result is fixed-width ``YYYY-MM-DDTHH:MM:SS.ffffff+00:00``, which
    makes lexicographic SQL comparison equal to chronological comparison.
    Round-trips preserve the instant, not the original offset.

    Args:
        value: An ISO-8601 timestamp string carrying a UTC offset.
        field_name: For error messages.

    Returns:
        The normalized UTC timestamp string.

    Raises:
        StorageError: On unparseable input or a naive (offset-less) value.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"{field_name}: invalid ISO-8601 timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise StorageError(f"{field_name} must carry a UTC offset (naive timestamp rejected)")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _require_json(value: Any, field_name: str) -> None:
    """Reject values the JSON codec cannot represent (StorageError, D6)."""
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise StorageError(f"{field_name} is not JSON-serializable: {exc}") from exc


def _canonical_source(value: Any, field_name: str = "source") -> dict[str, Any]:
    """Canonicalize a nested SourceInfo dict; agent_id is required (D7)."""
    if not isinstance(value, dict) or "agent_id" not in value:
        raise StorageError(f"{field_name} must be a dict carrying 'agent_id'")
    return {
        "agent_id": value["agent_id"],
        "step_id": value.get("step_id"),
        "run_id": value.get("run_id"),
        "workflow_name": value.get("workflow_name"),
    }


def _canonical_entry_value(name: str, value: Any) -> Any:
    """Validate and deep-copy one entry field into its canonical form."""
    try:
        if name in _TIMESTAMP_FIELDS:
            return _normalize_ts(value, name)
        if name == "type":
            return MemoryType(value).value
        if name == "archive_reason":
            return None if value is None else ArchiveReason(value).value
        if name in ("archived", "pinned"):
            return bool(value)
        if name == "source":
            return _canonical_source(value)
        if name in ("tags", "relationships", "source_entry_ids"):
            copied = copy.deepcopy(list(value))
            _require_json(copied, name)
            return copied
        if name == "access_count":
            return int(value)
        return value
    except (ValueError, TypeError) as exc:
        # StorageError is NOT a ValueError/TypeError subclass — errors raised
        # by _normalize_ts/_require_json/_canonical_source propagate as-is.
        raise StorageError(f"invalid value for {name!r}: {value!r}") from exc


def _validated_entry(entry: _Row) -> _Row:
    """Validate a full to_dict()-shaped entry dict into canonical form.

    Unknown keys are ignored (forward-compat); missing required keys, unknown
    enum values, naive timestamps, and non-JSON payloads raise StorageError.
    Asymmetry by design: a derived ``importance`` key on CREATE is silently
    dropped here (``to_dict()`` may carry it on decay-populated entries; it is
    never persisted, D2), while ``update`` REJECTS it — naming it in an update
    is an attempt to persist a derived value.
    """
    try:
        required = {
            "id": entry["id"],
            "content": entry["content"],
            "type": entry["type"],
            "source": entry["source"],
            "created_at": entry["created_at"],
            "updated_at": entry["updated_at"],
            "last_accessed_at": entry["last_accessed_at"],
        }
    except KeyError as exc:
        raise StorageError(f"entry is missing required field {exc.args[0]!r}") from exc
    raw: _Row = {
        **required,
        "key": entry.get("key"),
        "tags": entry.get("tags", []),
        "relationships": entry.get("relationships", []),
        "session_id": entry.get("session_id"),
        "base_importance": entry.get("base_importance", 0.5),
        "access_count": entry.get("access_count", 0),
        "archived": entry.get("archived", False),
        "archive_reason": entry.get("archive_reason"),
        "source_entry_ids": entry.get("source_entry_ids", []),
        "pinned": entry.get("pinned", False),
    }
    return {name: _canonical_entry_value(name, value) for name, value in raw.items()}


def _validated_entry_update(fields: _Row) -> _Row:
    """Validate a partial update dict; unknown field names raise StorageError."""
    unknown = set(fields) - _UPDATABLE_ENTRY_FIELDS
    if unknown:
        raise StorageError(f"unknown entry field(s) in update: {sorted(unknown)}")
    return {name: _canonical_entry_value(name, value) for name, value in fields.items()}


def _normalized_filters(filters: _Row) -> _Row:
    """Validate the closed filter-key set; normalize temporal/enum values."""
    unknown = set(filters) - _FILTER_KEYS
    if unknown:
        raise StorageError(f"unknown filter key(s): {sorted(unknown)}")
    normalized = dict(filters)
    for key in _TEMPORAL_FILTER_KEYS & set(normalized):
        normalized[key] = _normalize_ts(normalized[key], key)
    try:
        if "types" in normalized:
            normalized["types"] = [MemoryType(v).value for v in normalized["types"]]
        if "archive_reasons" in normalized:
            normalized["archive_reasons"] = [
                ArchiveReason(v).value for v in normalized["archive_reasons"]
            ]
    except ValueError as exc:
        raise StorageError(f"unknown enum value in filter: {exc}") from exc
    return normalized


def _validated_session(session: _Row) -> _Row:
    """Validate a full session dict into canonical form."""
    try:
        required = {
            "id": session["id"],
            "agent_id": session["agent_id"],
            "started_at": session["started_at"],
        }
    except KeyError as exc:
        raise StorageError(f"session is missing required field {exc.args[0]!r}") from exc
    raw: _Row = {
        **required,
        "goal": session.get("goal"),
        "last_activity_at": session.get("last_activity_at"),
        "ended_at": session.get("ended_at"),
        "status": session.get("status", SessionStatus.ACTIVE.value),
    }
    return {name: _canonical_session_value(name, value) for name, value in raw.items()}


def _canonical_session_value(name: str, value: Any) -> Any:
    """Validate and canonicalize one session field."""
    try:
        if name in _SESSION_TIMESTAMP_FIELDS:
            return None if value is None else _normalize_ts(value, name)
        if name == "status":
            return SessionStatus(value).value
        return value
    except ValueError as exc:
        # StorageError is NOT a ValueError subclass — _normalize_ts errors
        # propagate as-is.
        raise StorageError(f"invalid value for session {name!r}: {value!r}") from exc


def _validated_session_update(fields: _Row) -> _Row:
    """Validate a partial session update dict."""
    unknown = set(fields) - _UPDATABLE_SESSION_FIELDS
    if unknown:
        raise StorageError(f"unknown session field(s) in update: {sorted(unknown)}")
    return {name: _canonical_session_value(name, value) for name, value in fields.items()}


def _validated_meta(fields: _Row) -> _Row:
    """Validate a set_meta payload; schema_version is never settable."""
    if "schema_version" in fields:
        raise StorageError("meta.schema_version is owned by the migration runner")
    unknown = set(fields) - _SETTABLE_META_FIELDS
    if unknown:
        raise StorageError(f"unknown meta field(s): {sorted(unknown)}")
    return dict(fields)


def _escape_like_prefix(prefix: str) -> str:
    """Escape LIKE wildcards so the prefix is matched as literal text."""
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return escaped + "%"


def _validate_page(limit: int, offset: int = 0) -> None:
    """Reject negative pagination — SQLite's LIMIT -1 means "all rows" while
    Python slicing wraps, so the two backends would silently diverge."""
    if limit < 0 or offset < 0:
        raise StorageError(
            f"limit and offset must be non-negative (got limit={limit}, offset={offset})"
        )


# Chunk size for generated IN (...) placeholder lists: comfortably below
# SQLITE_MAX_VARIABLE_NUMBER (32766 since 3.32) so huge id batches cannot
# blow the statement variable limit. All chunks share one transaction.
_IN_CLAUSE_CHUNK: Final[int] = 500


def _chunked(items: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield consecutive slices of at most ``size`` items."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# The Protocol
# ---------------------------------------------------------------------------


class StorageBackend(Protocol):
    """Persistence contract shared by SQLiteBackend and InMemoryBackend.

    The two implementations are behaviorally identical across this entire
    surface — same errors, same ordering, same constraint enforcement. All
    failures raise ``StorageError`` (D6).
    """

    def close(self) -> None:
        """Idempotent shutdown; any later call raises StorageError."""
        ...

    def create(self, entry: _Row, *, embedding: bytes | None = None) -> None:
        """Persist a to_dict()-shaped entry (id verbatim) atomically."""
        ...

    def create_batch(self, items: Sequence[tuple[_Row, bytes | None]]) -> None:
        """All-or-nothing bulk create in a single transaction."""
        ...

    def read(self, entry_id: str) -> _Row | None:
        """Return the row dict (archived included) or None; never touches."""
        ...

    def get_by_key(self, key: str) -> _Row | None:
        """Return the ACTIVE row holding ``key``, or None."""
        ...

    def update(self, entry_id: str, fields: _Row) -> _Row | None:
        """Merge a to_dict()-shaped subset; None when the id is missing."""
        ...

    def delete(self, entry_id: str) -> bool:
        """Hard delete (junction cascades, embedding dies with the row)."""
        ...

    def delete_many(self, entry_ids: Sequence[str]) -> int:
        """Hard delete many in one transaction; returns rows deleted."""
        ...

    def list(self, filters: _Row, limit: int = 100, offset: int = 0) -> _Rows:
        """Closed filter-key set; ordering ``created_at DESC, id DESC``."""
        ...

    def count(self, filters: _Row) -> int:
        """Count rows matching the same filter contract as ``list``."""
        ...

    def touch(self, entry_ids: Sequence[str], accessed_at: str) -> int:
        """Batched access recording (D3); archived rows skipped."""
        ...

    def scan_key_prefix(self, prefix: str, limit: int = 100) -> _Rows:
        """Active rows whose key starts with the literal prefix, key ASC."""
        ...

    def list_keys(self, prefix: str = "") -> _Keys:
        """Active keys only, sorted ASC; no entry load, no touch."""
        ...

    def key_exists(self, key: str) -> bool:
        """Active-key existence probe; no touch."""
        ...

    def get_embedding(self, entry_id: str) -> bytes | None:
        """Return the opaque embedding blob; StorageError on a missing id."""
        ...

    def set_embedding(self, entry_id: str, embedding: bytes | None) -> None:
        """Set or clear (None) the blob; StorageError on a missing id."""
        ...

    def iter_embeddings(self, *, include_archived: bool = False) -> Iterator[tuple[str, bytes]]:
        """(entry_id, blob) for non-NULL embeddings, id ASC (ADR-015 #3)."""
        ...

    def create_session(self, session: _Row) -> None:
        """Persist a session dict; duplicate id raises StorageError."""
        ...

    def get_session(self, session_id: str) -> _Row | None:
        """Return the session dict or None."""
        ...

    def update_session(self, session_id: str, fields: _Row) -> _Row | None:
        """Merge session fields; None when the id is missing."""
        ...

    def list_sessions(self, *, agent_id: str | None = None, status: str | None = None) -> _Rows:
        """Sessions ordered ``started_at DESC, id DESC``; per-agent filter."""
        ...

    def get_meta(self) -> _Row:
        """The one-row identity table (D3)."""
        ...

    def set_meta(self, fields: _Row) -> None:
        """Set the three embedding fields; schema_version is refused."""
        ...

    def load_labels(self) -> _LabelRows:
        """All (label, entry_id, tombstoned) mapping rows, label ASC."""
        ...

    def insert_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Insert live mapping rows atomically; constraint hit sinks the batch."""
        ...

    def tombstone_label(self, entry_id: str) -> bool:
        """Tombstone the LIVE row for ``entry_id``; False when none exists."""
        ...

    def replace_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Atomic truncate + rewrite (rebuild path); all rows live."""
        ...

    def transaction(self) -> AbstractContextManager[None]:
        """Explicit atomic scope; non-reentrant (nesting raises)."""
        ...


# ---------------------------------------------------------------------------
# Schema DDL & migration runner (SQLite)
# ---------------------------------------------------------------------------

_V1_DDL: Final[tuple[str, ...]] = (
    """
    CREATE TABLE memories (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        type              TEXT NOT NULL,
        key               TEXT,
        tags              TEXT NOT NULL DEFAULT '[]',
        base_importance   REAL NOT NULL DEFAULT 0.5,
        agent_id          TEXT NOT NULL,
        step_id           TEXT,
        run_id            TEXT,
        workflow_name     TEXT,
        session_id        TEXT,
        relationships     TEXT NOT NULL DEFAULT '[]',
        source_entry_ids  TEXT NOT NULL DEFAULT '[]',
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL,
        last_accessed_at  TEXT NOT NULL,
        access_count      INTEGER NOT NULL DEFAULT 0,
        archived          INTEGER NOT NULL DEFAULT 0,
        archive_reason    TEXT,
        pinned            INTEGER NOT NULL DEFAULT 0,
        embedding         BLOB
    )
    """,
    """
    CREATE UNIQUE INDEX idx_key_active ON memories(key)
        WHERE archived = 0 AND key IS NOT NULL
    """,
    "CREATE INDEX idx_type          ON memories(type)",
    "CREATE INDEX idx_agent         ON memories(agent_id)",
    "CREATE INDEX idx_importance    ON memories(base_importance)",
    "CREATE INDEX idx_created       ON memories(created_at)",
    "CREATE INDEX idx_last_accessed ON memories(last_accessed_at)",
    "CREATE INDEX idx_archived      ON memories(archived, archive_reason)",
    "CREATE INDEX idx_session       ON memories(session_id)",
    """
    CREATE TABLE memory_tags (
        entry_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
        tag      TEXT NOT NULL,
        PRIMARY KEY (entry_id, tag)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX idx_tag ON memory_tags(tag, entry_id)",
    """
    CREATE TABLE sessions (
        id               TEXT PRIMARY KEY,
        goal             TEXT,
        agent_id         TEXT NOT NULL,
        started_at       TEXT NOT NULL,
        last_activity_at TEXT,
        ended_at         TEXT,
        status           TEXT NOT NULL DEFAULT 'active'
    )
    """,
    "CREATE INDEX idx_session_agent_status ON sessions(agent_id, status)",
    """
    CREATE TABLE meta (
        id                  INTEGER PRIMARY KEY CHECK (id = 1),
        schema_version      INTEGER NOT NULL,
        embedding_model_id  TEXT,
        embedding_dimension INTEGER,
        distance_metric     TEXT
    )
    """,
    """
    INSERT INTO meta (id, schema_version, embedding_model_id, embedding_dimension,
                      distance_metric) VALUES (1, 1, NULL, NULL, NULL)
    """,
)


def _migrate_v1(conn: sqlite3.Connection) -> None:
    """Apply the v1 schema (D3) to a fresh database."""
    for statement in _V1_DDL:
        conn.execute(statement)


# v2: the semantic index's label<->UUID mapping (blueprint-semantic-index §4).
# One LIVE row per entry_id via a partial unique index (mirrors idx_key_active)
# so tombstoned history rows can coexist with a later live row — this is what
# lets the semantic index re-add an entry after remove(). No FK to memories:
# the mapping is reconciled against it at startup, not cascade-coupled.
_V2_DDL: Final[tuple[str, ...]] = (
    """
    CREATE TABLE vector_labels (
        label      INTEGER PRIMARY KEY,
        entry_id   TEXT NOT NULL,
        tombstoned INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE UNIQUE INDEX idx_vector_label_active ON vector_labels(entry_id)
        WHERE tombstoned = 0
    """,
)


def _migrate_v2(conn: sqlite3.Connection) -> None:
    """Add the vector_labels mapping table (schema v2)."""
    for statement in _V2_DDL:
        conn.execute(statement)


_MIGRATIONS: Final[dict[int, Callable[[sqlite3.Connection], None]]] = {
    1: _migrate_v1,
    2: _migrate_v2,
}


def _run_migrations(conn: sqlite3.Connection) -> None:
    """Forward-only migration runner keyed on ``PRAGMA user_version``.

    Each step is one transaction: DDL, ``user_version``, and
    ``meta.schema_version`` advance together — a crashed migration leaves the
    previous version fully intact. Re-opening at the current version is a
    no-op. A database newer than this build is refused (no downgrades).
    """
    try:
        current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    except sqlite3.Error as exc:
        raise StorageError(f"cannot read database schema version: {exc}") from exc
    if current > SCHEMA_VERSION:
        raise StorageError(
            f"database schema version {current} is newer than this build supports "
            f"({SCHEMA_VERSION}); refusing to open (no downgrades)"
        )
    if current == SCHEMA_VERSION:
        return
    # auto_vacuum is set in _connection() BEFORE any header-writing pragma —
    # setting it here would be too late (journal_mode already initialized
    # the header).
    for version in range(current + 1, SCHEMA_VERSION + 1):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as exc:
            raise StorageError(f"cannot begin migration to schema v{version}: {exc}") from exc
        try:
            _MIGRATIONS[version](conn)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.execute("UPDATE meta SET schema_version = ? WHERE id = 1", (version,))
            conn.execute("COMMIT")
        except Exception as exc:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            if isinstance(exc, StorageError):
                raise
            raise StorageError(f"migration to schema v{version} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# SQLiteBackend
# ---------------------------------------------------------------------------

_MEMORY_COLUMNS: Final[str] = (
    "id, content, type, key, tags, base_importance, agent_id, step_id, run_id, "
    "workflow_name, session_id, relationships, source_entry_ids, created_at, "
    "updated_at, last_accessed_at, access_count, archived, archive_reason, pinned"
)

_SELECT_MEMORY: Final[str] = f"SELECT {_MEMORY_COLUMNS} FROM memories"

_INSERT_MEMORY: Final[str] = """
    INSERT INTO memories (id, content, type, key, tags, base_importance, agent_id,
        step_id, run_id, workflow_name, session_id, relationships, source_entry_ids,
        created_at, updated_at, last_accessed_at, access_count, archived,
        archive_reason, pinned, embedding)
    VALUES (:id, :content, :type, :key, :tags, :base_importance, :agent_id,
        :step_id, :run_id, :workflow_name, :session_id, :relationships,
        :source_entry_ids, :created_at, :updated_at, :last_accessed_at,
        :access_count, :archived, :archive_reason, :pinned, :embedding)
"""

_UPDATE_MEMORY: Final[str] = """
    UPDATE memories SET content = :content, type = :type, key = :key, tags = :tags,
        base_importance = :base_importance, agent_id = :agent_id, step_id = :step_id,
        run_id = :run_id, workflow_name = :workflow_name, session_id = :session_id,
        relationships = :relationships, source_entry_ids = :source_entry_ids,
        created_at = :created_at, updated_at = :updated_at,
        last_accessed_at = :last_accessed_at, access_count = :access_count,
        archived = :archived, archive_reason = :archive_reason, pinned = :pinned
    WHERE id = :id
"""

_SELECT_SESSION: Final[str] = (
    "SELECT id, goal, agent_id, started_at, last_activity_at, ended_at, status FROM sessions"
)

_INSERT_SESSION: Final[str] = """
    INSERT INTO sessions (id, goal, agent_id, started_at, last_activity_at, ended_at, status)
    VALUES (:id, :goal, :agent_id, :started_at, :last_activity_at, :ended_at, :status)
"""

_UPDATE_SESSION: Final[str] = """
    UPDATE sessions SET goal = :goal, agent_id = :agent_id, started_at = :started_at,
        last_activity_at = :last_activity_at, ended_at = :ended_at, status = :status
    WHERE id = :id
"""


def _flatten_entry(row: _Row) -> _Row:
    """Canonical nested dict -> flat column mapping (source flattened, JSON dumped)."""
    source = row["source"]
    return {
        "id": row["id"],
        "content": row["content"],
        "type": row["type"],
        "key": row["key"],
        "tags": json.dumps(row["tags"]),
        "base_importance": row["base_importance"],
        "agent_id": source["agent_id"],
        "step_id": source["step_id"],
        "run_id": source["run_id"],
        "workflow_name": source["workflow_name"],
        "session_id": row["session_id"],
        "relationships": json.dumps(row["relationships"]),
        "source_entry_ids": json.dumps(row["source_entry_ids"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_accessed_at": row["last_accessed_at"],
        "access_count": row["access_count"],
        "archived": int(row["archived"]),
        "archive_reason": row["archive_reason"],
        "pinned": int(row["pinned"]),
    }


def _row_to_dict(row: sqlite3.Row) -> _Row:
    """Flat column row -> nested to_dict() shape; malformed JSON is a corrupt row."""
    try:
        tags = json.loads(row["tags"])
        relationships = json.loads(row["relationships"])
        source_entry_ids = json.loads(row["source_entry_ids"])
    except (json.JSONDecodeError, TypeError) as exc:
        raise StorageError(f"corrupt row {row['id']!r}: malformed JSON column") from exc
    return {
        "id": row["id"],
        "content": row["content"],
        "type": row["type"],
        "source": {
            "agent_id": row["agent_id"],
            "step_id": row["step_id"],
            "run_id": row["run_id"],
            "workflow_name": row["workflow_name"],
        },
        "key": row["key"],
        "tags": tags,
        "relationships": relationships,
        "session_id": row["session_id"],
        "base_importance": row["base_importance"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_accessed_at": row["last_accessed_at"],
        "access_count": row["access_count"],
        "archived": bool(row["archived"]),
        "archive_reason": row["archive_reason"],
        "source_entry_ids": source_entry_ids,
        "pinned": bool(row["pinned"]),
    }


def _session_row_to_dict(row: sqlite3.Row) -> _Row:
    """Flat session row -> session dict."""
    return {
        "id": row["id"],
        "goal": row["goal"],
        "agent_id": row["agent_id"],
        "started_at": row["started_at"],
        "last_activity_at": row["last_activity_at"],
        "ended_at": row["ended_at"],
        "status": row["status"],
    }


def _filter_sql(filters: _Row) -> tuple[str, list[Any]]:
    """Normalized filters -> (WHERE clause, bound params).

    SQL text is assembled from fixed clause literals plus generated ``?``
    placeholder lists only; every value travels as a bound parameter.
    """
    clauses: list[str] = []
    params: list[Any] = []

    def in_clause(column_sql: str, values: Sequence[Any]) -> None:
        if not values:
            clauses.append("1 = 0")
            return
        placeholders = ",".join("?" * len(values))
        clauses.append(column_sql.replace("<PH>", placeholders))
        params.extend(values)

    reasons = filters.get("archive_reasons")
    if reasons is not None:
        clauses.append("archived = 1")
        in_clause("archive_reason IN (<PH>)", reasons)
    elif not filters.get("include_archived", False):
        clauses.append("archived = 0")
    if "types" in filters:
        in_clause("type IN (<PH>)", filters["types"])
    if "tags" in filters:
        in_clause("id IN (SELECT entry_id FROM memory_tags WHERE tag IN (<PH>))", filters["tags"])
    if "agent_id" in filters:
        clauses.append("agent_id = ?")
        params.append(filters["agent_id"])
    if "session_id" in filters:
        clauses.append("session_id = ?")
        params.append(filters["session_id"])
    if "key_prefix" in filters:
        clauses.append("key IS NOT NULL AND key LIKE ? ESCAPE '\\'")
        params.append(_escape_like_prefix(filters["key_prefix"]))
    if "pinned" in filters:
        clauses.append("pinned = ?")
        params.append(int(bool(filters["pinned"])))
    if "min_base_importance" in filters:
        clauses.append("base_importance >= ?")
        params.append(filters["min_base_importance"])
    if "since" in filters:
        clauses.append("created_at >= ?")
        params.append(filters["since"])
    if "created_before" in filters:
        clauses.append("created_at < ?")
        params.append(filters["created_before"])
    if "updated_before" in filters:
        clauses.append("updated_at < ?")
        params.append(filters["updated_before"])
    if "accessed_before" in filters:
        clauses.append("last_accessed_at < ?")
        params.append(filters["accessed_before"])
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params


class SQLiteBackend:
    """Default zero-config backend: WAL mode, thread-local connections.

    One process, many threads (ADR-015 tier 1): each thread gets its own
    connection lazily; WAL allows concurrent readers alongside the single
    writer; ``busy_timeout`` arbitrates writer contention. All transactional
    scopes issue explicit ``BEGIN IMMEDIATE``.
    """

    def __init__(
        self,
        db_path: str | Path = "./tulving_memory/tulving.db",
        *,
        busy_timeout_ms: int = 5000,
    ) -> None:
        """Open (creating if needed) the database and run migrations.

        Bounded work only (D8, sanctioned by architecture §3): no LLM calls,
        no full-table scans, no decay pass. Warns (never fails) when the path
        looks cloud-synced or network-mounted (ADR-015 #4).

        Args:
            db_path: The SQLite file path; parent directories are created.
            busy_timeout_ms: Writer-contention timeout per connection.

        Raises:
            StorageError: On open/migration failure or a newer-schema file.
        """
        if sqlite3.threadsafety < 3:
            raise StorageError(
                f"sqlite3 threadsafety is {sqlite3.threadsafety}; SQLiteBackend requires "
                "serialized mode (3) — close() reaches other threads' connections"
            )
        # Reject embedded NUL uniformly across platforms (never echo the raw
        # path). On Windows a NUL in the leaf trips .parent.mkdir below; on
        # POSIX (backslashes are literal) it would otherwise slip through to
        # sqlite3.connect and escape as a raw ValueError. Guard once, up front.
        if "\x00" in os.fspath(db_path):
            raise StorageError("database path contains an embedded null byte")
        self._db_path = Path(db_path)
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._closed = False
        self._local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._conn_lock = threading.Lock()
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            # Embedded NUL is already rejected up front; this still guards other
            # malformed/unusable paths. Do not echo the raw path in the message.
            detail = getattr(exc, "strerror", None) or "invalid or unusable path"
            raise StorageError(f"cannot create database parent directory: {detail}") from exc
        reason = cloud_sync_risk(self._db_path)
        if reason is not None:
            warnings.warn(
                f"{reason}; SQLite WAL is unsafe on synced/network storage (ADR-015)",
                UserWarning,
                stacklevel=2,
            )
        try:
            _run_migrations(self._connection())
        except BaseException:
            self.close()
            raise

    @property
    def db_path(self) -> Path:
        """The database file path this backend was opened on."""
        return self._db_path

    # ------------------------------------------------------------ plumbing

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("backend is closed")

    def _connection(self) -> sqlite3.Connection:
        """The calling thread's connection, created lazily with pragmas."""
        self._ensure_open()
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # A non-empty file already exists: auto_vacuum is a documented no-op
            # there (changing the mode on an existing database requires a full
            # VACUUM; merely restating it does not take effect) -- but SQLite
            # still increments the header's file-change-counter every time the
            # PRAGMA is re-issued, even at an unchanged value. Skip it entirely
            # in that case so opening a new connection to an EXISTING store
            # never dirties a single header byte just from being opened (DB fix:
            # this was the sole source of the read-only "phantom write").
            is_existing_nonempty = self._db_path.exists() and self._db_path.stat().st_size > 0
            try:
                # check_same_thread=False so close() can reach every thread's
                # connection; thread-locality is guaranteed by this method.
                conn = sqlite3.connect(
                    str(self._db_path), isolation_level=None, check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                # busy_timeout FIRST, before any other pragma: auto_vacuum and
                # journal_mode are header-writing pragmas that need the write
                # lock, and SQLite's per-connection default wait is 0ms until
                # busy_timeout is set. Under a live writer, a new connection
                # that hits auto_vacuum/journal_mode before busy_timeout is set
                # fails immediately with "database is locked" instead of
                # retrying (DB fix: reproduced 16/16 at the auto_vacuum line).
                conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
                if not is_existing_nonempty:
                    # auto_vacuum MUST precede any OTHER header-writing pragma
                    # on a brand-new (or still-empty) database (journal_mode
                    # initializes the header) -- see the skip note above for
                    # why an existing non-empty database never runs this.
                    conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA case_sensitive_like = ON")
            except sqlite3.Error as exc:
                raise StorageError(f"cannot open SQLite database: {exc}") from exc
            self._local.conn = conn
            self._local.in_txn = False
            with self._conn_lock:
                self._connections.add(conn)
        return conn

    def _execute(
        self, conn: sqlite3.Connection, sql: str, params: Sequence[Any] | _Row = ()
    ) -> sqlite3.Cursor:
        """Execute with error translation; retries once on a busy writer."""
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc):
                try:
                    return conn.execute(sql, params)
                except sqlite3.Error as retry_exc:
                    raise StorageError(
                        f"database is locked after busy timeout and one retry: {retry_exc}"
                    ) from retry_exc
            raise StorageError(f"SQLite error: {exc}") from exc
        except sqlite3.IntegrityError as exc:
            raise StorageError(f"constraint violation: {exc}") from exc
        except sqlite3.Error as exc:
            raise StorageError(f"SQLite error: {exc}") from exc

    @contextmanager
    def _atomic(self) -> Iterator[sqlite3.Connection]:
        """Ambient-or-own atomic scope for multi-statement mutations."""
        conn = self._connection()
        if getattr(self._local, "in_txn", False):
            yield conn
            return
        self._begin(conn)
        self._local.in_txn = True
        try:
            yield conn
        except Exception:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        else:
            self._commit(conn)
        finally:
            self._local.in_txn = False

    def _commit(self, conn: sqlite3.Connection) -> None:
        """COMMIT with D6 translation; attempts ROLLBACK on commit failure.

        Raises:
            StorageError: When COMMIT fails (the transaction is rolled back
                best-effort so no open transaction is left behind).
        """
        try:
            self._raw_commit(conn)
        except sqlite3.Error as exc:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise StorageError(f"commit failed: {exc}") from exc

    @staticmethod
    def _raw_commit(conn: sqlite3.Connection) -> None:
        """Seam for commit-failure injection in tests."""
        conn.execute("COMMIT")

    def _begin(self, conn: sqlite3.Connection) -> None:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if "database is locked" in str(exc):
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    return
                except sqlite3.Error as retry_exc:
                    raise StorageError(
                        f"cannot acquire write lock after busy timeout and one retry: {retry_exc}"
                    ) from retry_exc
            raise StorageError(f"cannot begin transaction: {exc}") from exc

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Idempotent: checkpoint WAL best-effort, close all connections."""
        if self._closed:
            return
        self._closed = True
        with self._conn_lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn in connections:
            with suppress(sqlite3.Error):
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            with suppress(sqlite3.Error):
                conn.close()

    # ------------------------------------------------------------- entries

    def create(self, entry: _Row, *, embedding: bytes | None = None) -> None:
        """Persist the entry (id verbatim) + tag junction rows atomically."""
        row = _validated_entry(entry)
        with self._atomic() as conn:
            self._insert_row(conn, row, embedding)

    def create_batch(self, items: Sequence[tuple[_Row, bytes | None]]) -> None:
        """All-or-nothing bulk create: one transaction for the whole batch."""
        validated = [(_validated_entry(entry), emb) for entry, emb in items]
        with self._atomic() as conn:
            for row, emb in validated:
                self._insert_row(conn, row, emb)

    def _insert_row(self, conn: sqlite3.Connection, row: _Row, embedding: bytes | None) -> None:
        params = _flatten_entry(row)
        params["embedding"] = embedding
        self._execute(conn, _INSERT_MEMORY, params)
        self._insert_tags(conn, row["id"], row["tags"])

    def _insert_tags(self, conn: sqlite3.Connection, entry_id: str, tags: list[str]) -> None:
        for tag in dict.fromkeys(tags):
            self._execute(
                conn, "INSERT INTO memory_tags (entry_id, tag) VALUES (?, ?)", (entry_id, tag)
            )

    def read(self, entry_id: str) -> _Row | None:
        """Row dict by id, archived included; never touches."""
        conn = self._connection()
        row = self._execute(conn, _SELECT_MEMORY + " WHERE id = ?", (entry_id,)).fetchone()
        return None if row is None else _row_to_dict(row)

    def get_by_key(self, key: str) -> _Row | None:
        """ACTIVE row holding ``key`` (partial unique index path)."""
        conn = self._connection()
        row = self._execute(
            conn, _SELECT_MEMORY + " WHERE key = ? AND archived = 0", (key,)
        ).fetchone()
        return None if row is None else _row_to_dict(row)

    def update(self, entry_id: str, fields: _Row) -> _Row | None:
        """Merge the validated subset into the row; rewrite tags junction."""
        validated = _validated_entry_update(fields)
        with self._atomic() as conn:
            row = self._execute(conn, _SELECT_MEMORY + " WHERE id = ?", (entry_id,)).fetchone()
            if row is None:
                return None
            merged = {**_row_to_dict(row), **validated}
            self._execute(conn, _UPDATE_MEMORY, _flatten_entry(merged))
            if "tags" in validated:
                self._execute(conn, "DELETE FROM memory_tags WHERE entry_id = ?", (entry_id,))
                self._insert_tags(conn, entry_id, merged["tags"])
            return merged

    def delete(self, entry_id: str) -> bool:
        """Hard delete; junction rows cascade, the embedding dies with the row."""
        with self._atomic() as conn:
            cursor = self._execute(conn, "DELETE FROM memories WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def delete_many(self, entry_ids: Sequence[str]) -> int:
        """Hard delete many in ONE transaction (chunked statements); returns
        rows actually deleted."""
        ids = list(dict.fromkeys(entry_ids))
        if not ids:
            return 0
        deleted = 0
        with self._atomic() as conn:
            for chunk in _chunked(ids, _IN_CLAUSE_CHUNK):
                placeholders = ",".join("?" * len(chunk))
                cursor = self._execute(
                    conn, f"DELETE FROM memories WHERE id IN ({placeholders})", chunk
                )
                deleted += cursor.rowcount
        return deleted

    def list(self, filters: _Row, limit: int = 100, offset: int = 0) -> _Rows:
        """Filtered rows, ``created_at DESC, id DESC``, paginated."""
        _validate_page(limit, offset)
        normalized = _normalized_filters(filters)
        where, params = _filter_sql(normalized)
        conn = self._connection()
        sql = _SELECT_MEMORY + where + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"
        rows = self._execute(conn, sql, [*params, limit, offset]).fetchall()
        return [_row_to_dict(row) for row in rows]

    def count(self, filters: _Row) -> int:
        """Row count under the same filter contract as ``list``."""
        normalized = _normalized_filters(filters)
        where, params = _filter_sql(normalized)
        conn = self._connection()
        result = self._execute(conn, "SELECT COUNT(*) FROM memories" + where, params).fetchone()
        return int(result[0])

    def touch(self, entry_ids: Sequence[str], accessed_at: str) -> int:
        """Batched access recording (D3) in ONE transaction (chunked
        statements, one shared instant); skips archived rows."""
        ts_value = _normalize_ts(accessed_at, "accessed_at")
        ids = list(dict.fromkeys(entry_ids))
        if not ids:
            return 0
        touched = 0
        with self._atomic() as conn:
            for chunk in _chunked(ids, _IN_CLAUSE_CHUNK):
                placeholders = ",".join("?" * len(chunk))
                cursor = self._execute(
                    conn,
                    f"UPDATE memories SET last_accessed_at = ?, access_count = access_count + 1 "
                    f"WHERE id IN ({placeholders}) AND archived = 0",
                    [ts_value, *chunk],
                )
                touched += cursor.rowcount
        return touched

    # ------------------------------------------------------- key retrieval

    def scan_key_prefix(self, prefix: str, limit: int = 100) -> _Rows:
        """Active rows with a literal key prefix, ORDER BY key ASC."""
        _validate_page(limit)
        conn = self._connection()
        rows = self._execute(
            conn,
            _SELECT_MEMORY
            + " WHERE archived = 0 AND key IS NOT NULL AND key LIKE ? ESCAPE '\\'"
            + " ORDER BY key ASC LIMIT ?",
            (_escape_like_prefix(prefix), limit),
        ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_keys(self, prefix: str = "") -> _Keys:
        """Active keys only, sorted ASC; no entry load."""
        conn = self._connection()
        rows = self._execute(
            conn,
            "SELECT key FROM memories WHERE archived = 0 AND key IS NOT NULL "
            "AND key LIKE ? ESCAPE '\\' ORDER BY key ASC",
            (_escape_like_prefix(prefix),),
        ).fetchall()
        return [row[0] for row in rows]

    def key_exists(self, key: str) -> bool:
        """EXISTS probe on the partial unique index; no touch."""
        conn = self._connection()
        row = self._execute(
            conn, "SELECT 1 FROM memories WHERE key = ? AND archived = 0 LIMIT 1", (key,)
        ).fetchone()
        return row is not None

    # ---------------------------------------------------------- embeddings

    def get_embedding(self, entry_id: str) -> bytes | None:
        """The opaque blob, or None when unset; missing id is an error."""
        conn = self._connection()
        row = self._execute(
            conn, "SELECT embedding FROM memories WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            raise StorageError(f"no entry with id {entry_id!r}")
        blob: bytes | None = row[0]
        return blob

    def set_embedding(self, entry_id: str, embedding: bytes | None) -> None:
        """Set or clear (None) the blob; missing id is an error."""
        with self._atomic() as conn:
            cursor = self._execute(
                conn, "UPDATE memories SET embedding = ? WHERE id = ?", (embedding, entry_id)
            )
            if cursor.rowcount == 0:
                raise StorageError(f"no entry with id {entry_id!r}")

    def iter_embeddings(self, *, include_archived: bool = False) -> Iterator[tuple[str, bytes]]:
        """(entry_id, blob) pairs, id ASC — the rebuild-from-truth path."""
        conn = self._connection()
        sql = "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
        if not include_archived:
            sql += " AND archived = 0"
        sql += " ORDER BY id ASC"
        rows = self._execute(conn, sql).fetchall()
        return iter([(row[0], row[1]) for row in rows])

    # ------------------------------------------------------------ sessions

    def create_session(self, session: _Row) -> None:
        """Persist a session dict; duplicate id raises StorageError."""
        validated = _validated_session(session)
        with self._atomic() as conn:
            self._execute(conn, _INSERT_SESSION, validated)

    def get_session(self, session_id: str) -> _Row | None:
        """Session dict by id, or None."""
        conn = self._connection()
        row = self._execute(conn, _SELECT_SESSION + " WHERE id = ?", (session_id,)).fetchone()
        return None if row is None else _session_row_to_dict(row)

    def update_session(self, session_id: str, fields: _Row) -> _Row | None:
        """Merge validated session fields; None when the id is missing."""
        validated = _validated_session_update(fields)
        with self._atomic() as conn:
            row = self._execute(conn, _SELECT_SESSION + " WHERE id = ?", (session_id,)).fetchone()
            if row is None:
                return None
            merged = {**_session_row_to_dict(row), **validated}
            self._execute(conn, _UPDATE_SESSION, merged)
            return merged

    def list_sessions(self, *, agent_id: str | None = None, status: str | None = None) -> _Rows:
        """Sessions ordered ``started_at DESC, id DESC``, optionally filtered."""
        conn = self._connection()
        clauses: list[str] = []
        params: list[Any] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        sql = _SELECT_SESSION
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC, id DESC"
        rows = self._execute(conn, sql, params).fetchall()
        return [_session_row_to_dict(row) for row in rows]

    # ---------------------------------------------------------------- meta

    def get_meta(self) -> _Row:
        """The one-row identity table (D3)."""
        conn = self._connection()
        row = self._execute(
            conn,
            "SELECT schema_version, embedding_model_id, embedding_dimension, "
            "distance_metric FROM meta WHERE id = 1",
        ).fetchone()
        if row is None:
            raise StorageError("corrupt database: meta row is missing")
        return {
            "schema_version": row["schema_version"],
            "embedding_model_id": row["embedding_model_id"],
            "embedding_dimension": row["embedding_dimension"],
            "distance_metric": row["distance_metric"],
        }

    def set_meta(self, fields: _Row) -> None:
        """Set the embedding identity fields; schema_version is refused."""
        validated = _validated_meta(fields)
        with self._atomic() as conn:
            merged = {**self.get_meta(), **validated}
            self._execute(
                conn,
                "UPDATE meta SET embedding_model_id = :embedding_model_id, "
                "embedding_dimension = :embedding_dimension, "
                "distance_metric = :distance_metric WHERE id = 1",
                merged,
            )

    # ------------------------------------------------------- vector labels

    def load_labels(self) -> _LabelRows:
        """All (label, entry_id, tombstoned) mapping rows, label ASC."""
        conn = self._connection()
        rows = self._execute(
            conn, "SELECT label, entry_id, tombstoned FROM vector_labels ORDER BY label ASC"
        ).fetchall()
        return [(int(row[0]), row[1], bool(row[2])) for row in rows]

    def insert_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Insert live mapping rows atomically.

        A duplicate label or a second LIVE row for the same entry_id (the
        partial unique index) sinks the whole batch with ``StorageError``.
        Tombstoned history rows for the same entry_id never collide.
        """
        if not rows:
            return
        with self._atomic() as conn:
            for label, entry_id in rows:
                self._execute(
                    conn,
                    "INSERT INTO vector_labels (label, entry_id, tombstoned) VALUES (?, ?, 0)",
                    (label, entry_id),
                )

    def tombstone_label(self, entry_id: str) -> bool:
        """Tombstone the LIVE row for ``entry_id``; False when none exists."""
        with self._atomic() as conn:
            cursor = self._execute(
                conn,
                "UPDATE vector_labels SET tombstoned = 1 WHERE entry_id = ? AND tombstoned = 0",
                (entry_id,),
            )
            return cursor.rowcount > 0

    def replace_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Atomic truncate + rewrite (rebuild path); all rows live."""
        with self._atomic() as conn:
            self._execute(conn, "DELETE FROM vector_labels")
            for label, entry_id in rows:
                self._execute(
                    conn,
                    "INSERT INTO vector_labels (label, entry_id, tombstoned) VALUES (?, ?, 0)",
                    (label, entry_id),
                )

    # ----------------------------------------------------------- atomicity

    @contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        conn = self._connection()
        if getattr(self._local, "in_txn", False):
            raise StorageError(
                "nested transaction() is not supported (programming error; fail loudly)"
            )
        self._begin(conn)
        self._local.in_txn = True
        try:
            yield
        except Exception:
            with suppress(sqlite3.Error):
                conn.execute("ROLLBACK")
            raise
        else:
            self._commit(conn)
        finally:
            self._local.in_txn = False

    def transaction(self) -> AbstractContextManager[None]:
        """Explicit atomic scope (D1 supersede); non-reentrant per thread."""
        return self._transaction_scope()


# ---------------------------------------------------------------------------
# InMemoryBackend
# ---------------------------------------------------------------------------


class InMemoryBackend:
    """Dict-backed backend, behaviorally identical to SQLiteBackend.

    The default test double for every higher module. Deep-copies on ingest
    and egress so callers can never mutate stored state through a returned
    dict; enforces the same constraints raising the same ``StorageError``s
    (duplicate id, active-key uniqueness, unknown filter keys/enum values,
    naive timestamps, nested transactions, settable-meta discipline).
    """

    def __init__(self) -> None:
        """Cheap construction: empty state, no I/O."""
        self._entries: dict[str, _Row] = {}
        self._embeddings: dict[str, bytes] = {}
        self._sessions: dict[str, _Row] = {}
        self._labels: dict[int, tuple[str, bool]] = {}  # label -> (entry_id, tombstoned)
        self._meta: _Row = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model_id": None,
            "embedding_dimension": None,
            "distance_metric": None,
        }
        self._lock = threading.RLock()
        self._local = threading.local()
        self._closed = False

    @property
    def db_path(self) -> None:
        """No file backs this backend."""
        return None

    # ------------------------------------------------------------ plumbing

    def _ensure_open(self) -> None:
        if self._closed:
            raise StorageError("backend is closed")

    def _check_new_row(self, row: _Row) -> None:
        """Emulate the PRIMARY KEY and the partial unique index (D3)."""
        if row["id"] in self._entries:
            raise StorageError(f"duplicate id {row['id']!r}")
        self._check_active_key(row["id"], row)

    def _check_active_key(self, entry_id: str, row: _Row) -> None:
        """Partial-unique-index emulation: one ACTIVE holder per non-NULL key."""
        if row["archived"] or row["key"] is None:
            return
        for other_id, other in self._entries.items():
            if other_id != entry_id and not other["archived"] and other["key"] == row["key"]:
                raise StorageError(
                    f"constraint violation: active key {row['key']!r} is already held "
                    f"(partial unique index)"
                )

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Idempotent; any later call raises StorageError."""
        self._closed = True

    # ------------------------------------------------------------- entries

    def create(self, entry: _Row, *, embedding: bytes | None = None) -> None:
        """Persist the entry (id verbatim); same constraints as SQLite."""
        self._ensure_open()
        row = _validated_entry(entry)
        with self._lock:
            self._check_new_row(row)
            self._entries[row["id"]] = row
            if embedding is not None:
                self._embeddings[row["id"]] = embedding

    def create_batch(self, items: Sequence[tuple[_Row, bytes | None]]) -> None:
        """All-or-nothing bulk create (snapshot rollback on any failure)."""
        self._ensure_open()
        validated = [(_validated_entry(entry), emb) for entry, emb in items]
        with self._lock:
            entries_snapshot = copy.deepcopy(self._entries)
            embeddings_snapshot = dict(self._embeddings)
            try:
                for row, emb in validated:
                    self._check_new_row(row)
                    self._entries[row["id"]] = row
                    if emb is not None:
                        self._embeddings[row["id"]] = emb
            except Exception:
                self._entries = entries_snapshot
                self._embeddings = embeddings_snapshot
                raise

    def read(self, entry_id: str) -> _Row | None:
        """Row dict by id, archived included; never touches."""
        self._ensure_open()
        with self._lock:
            entry = self._entries.get(entry_id)
            return None if entry is None else copy.deepcopy(entry)

    def get_by_key(self, key: str) -> _Row | None:
        """ACTIVE row holding ``key``, or None."""
        self._ensure_open()
        with self._lock:
            for entry in self._entries.values():
                if not entry["archived"] and entry["key"] == key:
                    return copy.deepcopy(entry)
        return None

    def update(self, entry_id: str, fields: _Row) -> _Row | None:
        """Merge the validated subset; None when the id is missing."""
        self._ensure_open()
        validated = _validated_entry_update(fields)
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return None
            merged = {**copy.deepcopy(entry), **validated}
            self._check_active_key(entry_id, merged)
            self._entries[entry_id] = merged
            return copy.deepcopy(merged)

    def delete(self, entry_id: str) -> bool:
        """Hard delete row + embedding; junction is implicit in tags."""
        self._ensure_open()
        with self._lock:
            existed = self._entries.pop(entry_id, None) is not None
            self._embeddings.pop(entry_id, None)
            return existed

    def delete_many(self, entry_ids: Sequence[str]) -> int:
        """Hard delete many; returns rows actually deleted."""
        self._ensure_open()
        with self._lock:
            return sum(1 for entry_id in dict.fromkeys(entry_ids) if self.delete(entry_id))

    def list(self, filters: _Row, limit: int = 100, offset: int = 0) -> _Rows:
        """Filtered rows, ``created_at DESC, id DESC``, paginated."""
        self._ensure_open()
        _validate_page(limit, offset)
        normalized = _normalized_filters(filters)
        with self._lock:
            matching = [e for e in self._entries.values() if _entry_matches(e, normalized)]
            matching.sort(key=lambda e: (e["created_at"], e["id"]), reverse=True)
            return [copy.deepcopy(e) for e in matching[offset : offset + limit]]

    def count(self, filters: _Row) -> int:
        """Row count under the same filter contract as ``list``."""
        self._ensure_open()
        normalized = _normalized_filters(filters)
        with self._lock:
            return sum(1 for e in self._entries.values() if _entry_matches(e, normalized))

    def touch(self, entry_ids: Sequence[str], accessed_at: str) -> int:
        """Batched access recording (D3); archived rows skipped."""
        self._ensure_open()
        ts_value = _normalize_ts(accessed_at, "accessed_at")
        touched = 0
        with self._lock:
            for entry_id in dict.fromkeys(entry_ids):
                entry = self._entries.get(entry_id)
                if entry is not None and not entry["archived"]:
                    entry["last_accessed_at"] = ts_value
                    entry["access_count"] += 1
                    touched += 1
        return touched

    # ------------------------------------------------------- key retrieval

    def scan_key_prefix(self, prefix: str, limit: int = 100) -> _Rows:
        """Active rows with a literal key prefix, ordered key ASC."""
        self._ensure_open()
        _validate_page(limit)
        with self._lock:
            matching = [
                e
                for e in self._entries.values()
                if not e["archived"] and e["key"] is not None and e["key"].startswith(prefix)
            ]
            matching.sort(key=lambda e: str(e["key"]))
            return [copy.deepcopy(e) for e in matching[:limit]]

    def list_keys(self, prefix: str = "") -> _Keys:
        """Active keys only, sorted ASC; no entry load."""
        self._ensure_open()
        with self._lock:
            return sorted(
                e["key"]
                for e in self._entries.values()
                if not e["archived"] and e["key"] is not None and e["key"].startswith(prefix)
            )

    def key_exists(self, key: str) -> bool:
        """Active-key existence probe; no touch."""
        self._ensure_open()
        with self._lock:
            return any(not e["archived"] and e["key"] == key for e in self._entries.values())

    # ---------------------------------------------------------- embeddings

    def get_embedding(self, entry_id: str) -> bytes | None:
        """The opaque blob, or None when unset; missing id is an error."""
        self._ensure_open()
        with self._lock:
            if entry_id not in self._entries:
                raise StorageError(f"no entry with id {entry_id!r}")
            return self._embeddings.get(entry_id)

    def set_embedding(self, entry_id: str, embedding: bytes | None) -> None:
        """Set or clear (None) the blob; missing id is an error."""
        self._ensure_open()
        with self._lock:
            if entry_id not in self._entries:
                raise StorageError(f"no entry with id {entry_id!r}")
            if embedding is None:
                self._embeddings.pop(entry_id, None)
            else:
                self._embeddings[entry_id] = embedding

    def iter_embeddings(self, *, include_archived: bool = False) -> Iterator[tuple[str, bytes]]:
        """(entry_id, blob) pairs, id ASC — the rebuild-from-truth path."""
        self._ensure_open()
        with self._lock:
            pairs = [
                (entry_id, blob)
                for entry_id, blob in sorted(self._embeddings.items())
                if include_archived or not self._entries[entry_id]["archived"]
            ]
        return iter(pairs)

    # ------------------------------------------------------------ sessions

    def create_session(self, session: _Row) -> None:
        """Persist a session dict; duplicate id raises StorageError."""
        self._ensure_open()
        validated = _validated_session(session)
        with self._lock:
            if validated["id"] in self._sessions:
                raise StorageError(f"duplicate session id {validated['id']!r}")
            self._sessions[validated["id"]] = validated

    def get_session(self, session_id: str) -> _Row | None:
        """Session dict by id, or None."""
        self._ensure_open()
        with self._lock:
            session = self._sessions.get(session_id)
            return None if session is None else copy.deepcopy(session)

    def update_session(self, session_id: str, fields: _Row) -> _Row | None:
        """Merge validated session fields; None when the id is missing."""
        self._ensure_open()
        validated = _validated_session_update(fields)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            merged = {**copy.deepcopy(session), **validated}
            self._sessions[session_id] = merged
            return copy.deepcopy(merged)

    def list_sessions(self, *, agent_id: str | None = None, status: str | None = None) -> _Rows:
        """Sessions ordered ``started_at DESC, id DESC``, optionally filtered."""
        self._ensure_open()
        with self._lock:
            matching = [
                s
                for s in self._sessions.values()
                if (agent_id is None or s["agent_id"] == agent_id)
                and (status is None or s["status"] == status)
            ]
            matching.sort(key=lambda s: (s["started_at"], s["id"]), reverse=True)
            return [copy.deepcopy(s) for s in matching]

    # ---------------------------------------------------------------- meta

    def get_meta(self) -> _Row:
        """The identity mapping (schema_version + embedding fields)."""
        self._ensure_open()
        with self._lock:
            return dict(self._meta)

    def set_meta(self, fields: _Row) -> None:
        """Set the embedding identity fields; schema_version is refused."""
        self._ensure_open()
        validated = _validated_meta(fields)
        with self._lock:
            self._meta.update(validated)

    # ------------------------------------------------------- vector labels

    @staticmethod
    def _check_label_rows(
        existing: dict[int, tuple[str, bool]], rows: Sequence[tuple[int, str]]
    ) -> None:
        """Emulate the label PRIMARY KEY and the partial unique index."""
        live_ids = {entry_id for entry_id, tombstoned in existing.values() if not tombstoned}
        seen_labels = set(existing)
        for label, entry_id in rows:
            if label in seen_labels:
                raise StorageError(f"constraint violation: duplicate label {label}")
            if entry_id in live_ids:
                raise StorageError(
                    f"constraint violation: entry {entry_id!r} already holds a live label "
                    f"(partial unique index)"
                )
            seen_labels.add(label)
            live_ids.add(entry_id)

    def load_labels(self) -> _LabelRows:
        """All (label, entry_id, tombstoned) mapping rows, label ASC."""
        self._ensure_open()
        with self._lock:
            return [
                (label, entry_id, tombstoned)
                for label, (entry_id, tombstoned) in sorted(self._labels.items())
            ]

    def insert_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Insert live mapping rows atomically; constraint hit sinks the batch."""
        self._ensure_open()
        with self._lock:
            self._check_label_rows(self._labels, rows)
            for label, entry_id in rows:
                self._labels[label] = (entry_id, False)

    def tombstone_label(self, entry_id: str) -> bool:
        """Tombstone the LIVE row for ``entry_id``; False when none exists."""
        self._ensure_open()
        with self._lock:
            for label, (row_entry_id, tombstoned) in self._labels.items():
                if row_entry_id == entry_id and not tombstoned:
                    self._labels[label] = (entry_id, True)
                    return True
            return False

    def replace_labels(self, rows: Sequence[tuple[int, str]]) -> None:
        """Atomic truncate + rewrite (rebuild path); all rows live."""
        self._ensure_open()
        with self._lock:
            self._check_label_rows({}, rows)
            self._labels = {label: (entry_id, False) for label, entry_id in rows}

    # ----------------------------------------------------------- atomicity

    @contextmanager
    def _transaction_scope(self) -> Iterator[None]:
        self._ensure_open()
        if getattr(self._local, "in_txn", False):
            raise StorageError(
                "nested transaction() is not supported (programming error; fail loudly)"
            )
        with self._lock:
            snapshot = (
                copy.deepcopy(self._entries),
                dict(self._embeddings),
                copy.deepcopy(self._sessions),
                dict(self._meta),
                dict(self._labels),
            )
            self._local.in_txn = True
            try:
                yield
            except Exception:
                self._entries, self._embeddings, self._sessions, self._meta, self._labels = (
                    copy.deepcopy(snapshot[0]),
                    dict(snapshot[1]),
                    copy.deepcopy(snapshot[2]),
                    dict(snapshot[3]),
                    dict(snapshot[4]),
                )
                raise
            finally:
                self._local.in_txn = False

    def transaction(self) -> AbstractContextManager[None]:
        """Explicit atomic scope with real snapshot rollback; non-reentrant."""
        return self._transaction_scope()


def _entry_matches(entry: _Row, filters: _Row) -> bool:
    """InMemory filter evaluation — semantics mirror :func:`_filter_sql`."""
    reasons = filters.get("archive_reasons")
    if reasons is not None:
        if not entry["archived"] or entry["archive_reason"] not in reasons:
            return False
    elif not filters.get("include_archived", False) and entry["archived"]:
        return False
    if "types" in filters and entry["type"] not in filters["types"]:
        return False
    if "tags" in filters and not set(filters["tags"]) & set(entry["tags"]):
        return False
    if "agent_id" in filters and entry["source"]["agent_id"] != filters["agent_id"]:
        return False
    if "session_id" in filters and entry["session_id"] != filters["session_id"]:
        return False
    if "key_prefix" in filters and (
        entry["key"] is None or not entry["key"].startswith(filters["key_prefix"])
    ):
        return False
    if "pinned" in filters and entry["pinned"] is not bool(filters["pinned"]):
        return False
    if (
        "min_base_importance" in filters
        and entry["base_importance"] < filters["min_base_importance"]
    ):
        return False
    if "since" in filters and entry["created_at"] < filters["since"]:
        return False
    if "created_before" in filters and entry["created_at"] >= filters["created_before"]:
        return False
    if "updated_before" in filters and entry["updated_at"] >= filters["updated_before"]:
        return False
    return not (
        "accessed_before" in filters and entry["last_accessed_at"] >= filters["accessed_before"]
    )


if TYPE_CHECKING:  # pragma: no cover - static Protocol-conformance check only

    def _conformance(sqlite: SQLiteBackend, memory: InMemoryBackend) -> None:
        _a: StorageBackend = sqlite
        _b: StorageBackend = memory

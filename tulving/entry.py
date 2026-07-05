"""Canonical memory-entry data model. Imports only tulving.enums + stdlib.

Invariant violations raise stdlib ``ValueError`` (programming errors at the
dataclass level); boundary layers translate upward to Tulving exceptions.
This module may not import ``tulving.exceptions`` or ``tulving.security``
(dependency rule — see blueprint-entry).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from tulving.enums import ArchiveReason, MemoryType

# Kept in sync with tulving.security.REDACTED (entry.py may not import
# security); tests/test_entry.py pins the two values together.
_REDACTED_PLACEHOLDER: Final[str] = "[REDACTED]"


def utcnow() -> datetime:
    """Timezone-aware UTC now — the only clock this module knows.

    Exposed so tests can monkeypatch one seam.

    Returns:
        The current UTC time with ``tzinfo`` set.
    """
    return datetime.now(UTC)


def _require_aware(value: datetime, field_name: str) -> None:
    """Reject naive datetimes — silent coercion hides caller bugs (D2 math)."""
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (naive datetime rejected)")


def _parse_aware_iso(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 string that must carry a UTC offset."""
    parsed = datetime.fromisoformat(value)
    _require_aware(parsed, field_name)
    return parsed


@dataclass
class SourceInfo:
    """Provenance: who stored this, and from where."""

    agent_id: str  # required (D7); filled from Memory's bound agent_id
    step_id: str | None = None  # kairos-ai step id, if applicable (v0.2 hook)
    run_id: str | None = None
    workflow_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict."""
        return {
            "agent_id": self.agent_id,
            "step_id": self.step_id,
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceInfo:
        """Inverse of ``to_dict()``. Raises KeyError on missing ``agent_id``."""
        return cls(
            agent_id=data["agent_id"],
            step_id=data.get("step_id"),
            run_id=data.get("run_id"),
            workflow_name=data.get("workflow_name"),
        )


@dataclass
class Relationship:
    """Directed link to another entry (agent-supplied in v0.1; includes supersede back-links)."""

    target_id: str
    relationship_type: str  # "relates_to", "supersedes", "supports", ...
    metadata: dict[str, Any] | None = None  # must be JSON-serializable; enforced at storage

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict.

        ``metadata`` is shallow-copied (consistent with ``tags`` /
        ``source_entry_ids``) so the exported dict never aliases live state.
        """
        return {
            "target_id": self.target_id,
            "relationship_type": self.relationship_type,
            "metadata": dict(self.metadata) if self.metadata is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        """Inverse of ``to_dict()``. Raises KeyError on missing required keys.

        ``metadata`` is shallow-copied on ingest so restored entries never
        alias the input dict.
        """
        raw_metadata = data.get("metadata")
        return cls(
            target_id=data["target_id"],
            relationship_type=data["relationship_type"],
            metadata=dict(raw_metadata) if raw_metadata is not None else None,
        )


@dataclass
class MemoryEntry:
    """A single memory. Field inventory is architecture.md §3 — do not extend casually."""

    # --- identity & content (required, no defaults) ---
    id: str  # UUID string; minted by the Memory Store, NOT here
    content: str
    type: MemoryType
    source: SourceInfo

    # --- addressing & organization ---
    key: str | None = None
    tags: list[str] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    session_id: str | None = None

    # --- importance (D2: base stored; effective derived elsewhere) ---
    base_importance: float = 0.5  # 0.0-1.0, immutable after store time
    importance: float | None = field(default=None, compare=False)
    # ^ DERIVED slot. None until a read path populates it via the decay
    #   module. Never persisted, never restored (from_dict discards it).

    # --- lifecycle timestamps (tz-aware UTC, always) ---
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    last_accessed_at: datetime | None = None
    # ^ normalized in __post_init__: None -> created_at (never-accessed
    #   entries decay from creation). After __post_init__ it is always a
    #   datetime; the type stays `datetime | None` only for the default.
    access_count: int = 0

    # --- archival (D3) ---
    archived: bool = False
    archive_reason: ArchiveReason | None = None

    # --- summaries & exemptions ---
    source_entry_ids: list[str] = field(default_factory=list)  # SUMMARY back-links
    pinned: bool = False

    def __post_init__(self) -> None:
        """Enforce invariants; raise ValueError on violation."""
        if not self.id:
            raise ValueError("id must be non-empty (minted by the store)")
        if not self.content:
            raise ValueError("content must be non-empty")
        if not self.source.agent_id:
            raise ValueError("source.agent_id must be non-empty (D7)")
        if not 0.0 <= self.base_importance <= 1.0:
            raise ValueError("base_importance must be within [0.0, 1.0]")
        if self.access_count < 0:
            raise ValueError("access_count must be >= 0")
        if self.archived and self.archive_reason is None:
            raise ValueError("archived entries must carry an archive_reason (D3)")
        if not self.archived and self.archive_reason is not None:
            raise ValueError("non-archived entries must not carry an archive_reason (D3)")
        if self.source_entry_ids and self.type is not MemoryType.SUMMARY:
            raise ValueError("source_entry_ids is only valid on SUMMARY entries")
        _require_aware(self.created_at, "created_at")
        _require_aware(self.updated_at, "updated_at")
        if self.last_accessed_at is None:
            self.last_accessed_at = self.created_at
        else:
            _require_aware(self.last_accessed_at, "last_accessed_at")

    def touch(self, now: datetime | None = None) -> None:
        """Record an access: bump ``access_count``, stamp ``last_accessed_at``.

        Called by the get / search-hit / curate-inclusion paths in store.py
        and the curator (D3) — never by users. Does NOT touch ``updated_at``.

        Args:
            now: The access instant; defaults to ``utcnow()``. Injected by
                callers so decay math stays testable without sleeping.
        """
        self.last_accessed_at = now if now is not None else utcnow()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize stored fields to a JSON-safe plain dict.

        Enums become their str values; datetimes become ISO-8601 strings with
        offset; nested dataclasses become nested dicts. The derived
        ``importance`` is included ONLY when populated (``from_dict`` never
        reads it back — D2). No embedding field exists (ADR-015).

        Returns:
            A plain dict safe for ``json.dumps`` and storage mapping.
        """
        last_accessed = self.last_accessed_at
        if last_accessed is None:  # pragma: no cover - normalized in __post_init__
            raise ValueError("last_accessed_at is unset; __post_init__ was bypassed")
        data: dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "type": self.type.value,
            "source": self.source.to_dict(),
            "key": self.key,
            "tags": list(self.tags),
            "relationships": [rel.to_dict() for rel in self.relationships],
            "session_id": self.session_id,
            "base_importance": self.base_importance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed_at": last_accessed.isoformat(),
            "access_count": self.access_count,
            "archived": self.archived,
            "archive_reason": self.archive_reason.value if self.archive_reason else None,
            "source_entry_ids": list(self.source_entry_ids),
            "pinned": self.pinned,
        }
        if self.importance is not None:
            data["importance"] = self.importance
        return data

    def to_safe_dict(self, *, content_is_sensitive: bool) -> dict[str, Any]:
        """``to_dict()`` with content masked when the caller judged it sensitive.

        The JUDGMENT (``security.is_sensitive_key``) lives in security.py,
        which this module may not import — so the caller passes the verdict::

            e.to_safe_dict(content_is_sensitive=is_sensitive_key(e.key or ""))

        Content-level secret scanning of outgoing text is applied later by the
        emitting module (curator/export/MCP) via ``security.redact_text``.

        Args:
            content_is_sensitive: The caller's verdict; no pattern matching
                happens here.

        Returns:
            A plain dict with ``content`` masked when sensitive.
        """
        data = self.to_dict()
        if content_is_sensitive:
            data["content"] = _REDACTED_PLACEHOLDER
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Inverse of ``to_dict()``.

        Parses ISO datetimes via ``datetime.fromisoformat`` (they must carry
        an offset — naive strings raise ValueError); reconstructs enums;
        IGNORES any ``importance`` key (re-derived on read, D2); ignores
        unknown keys (forward-compat for v0.2 columns).

        Args:
            data: A dict produced by ``to_dict()`` (or a storage row mapping).

        Returns:
            The reconstructed entry.

        Raises:
            ValueError: On unknown enum values or naive datetime strings.
            KeyError: On missing required keys. The storage layer translates
                both to ``StorageError``.
        """
        raw_reason = data.get("archive_reason")
        raw_last_accessed = data.get("last_accessed_at")
        return cls(
            id=data["id"],
            content=data["content"],
            type=MemoryType(data["type"]),
            source=SourceInfo.from_dict(data["source"]),
            key=data.get("key"),
            tags=list(data.get("tags", [])),
            relationships=[Relationship.from_dict(rel) for rel in data.get("relationships", [])],
            session_id=data.get("session_id"),
            base_importance=data.get("base_importance", 0.5),
            created_at=_parse_aware_iso(data["created_at"], "created_at"),
            updated_at=_parse_aware_iso(data["updated_at"], "updated_at"),
            last_accessed_at=(
                _parse_aware_iso(raw_last_accessed, "last_accessed_at")
                if raw_last_accessed is not None
                else None
            ),
            access_count=data.get("access_count", 0),
            archived=data.get("archived", False),
            archive_reason=ArchiveReason(raw_reason) if raw_reason is not None else None,
            source_entry_ids=list(data.get("source_entry_ids", [])),
            pinned=data.get("pinned", False),
        )

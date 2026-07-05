"""Enumerations for the Tulving memory model. Zero dependencies.

All values are persisted into SQLite rows and JSON exports; they are a
compatibility contract from the first release. Renaming or removing a
member is a schema migration, not a refactor.
"""

from enum import StrEnum


class MemoryType(StrEnum):
    """Type of a memory entry. Values are persisted; never rename."""

    FACT = "fact"  # discrete piece of information
    DECISION = "decision"  # a choice made and the reasoning — never decays (ADR-006)
    OBSERVATION = "observation"  # something noticed or analyzed
    PLAN = "plan"  # intended future action
    SUMMARY = "summary"  # system-generated digest (store() rejects it from callers)


class MatchType(StrEnum):
    """How a search result matched the query (D6: enum, not bare strings)."""

    SEMANTIC = "semantic"
    KEY = "key"
    TEMPORAL = "temporal"


class ArchiveReason(StrEnum):
    """Why an entry was archived (D3). An archived entry always carries one."""

    EVICTED = "evicted"  # effective importance fell below threshold
    SUMMARIZED = "summarized"  # rolled into a SUMMARY entry
    SUPERSEDED = "superseded"  # replaced by a newer entry with the same key (D1)
    FORGOTTEN = "forgotten"  # explicit forget()
    ABANDONED = "abandoned"  # belonged to an abandoned session


class SessionStatus(StrEnum):
    """Lifecycle state of a per-agent session (D6 inventory addition)."""

    ACTIVE = "active"
    ENDED = "ended"
    ABANDONED = "abandoned"

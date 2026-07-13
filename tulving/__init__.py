"""Tulving — the context-budget engine for AI agents.

Persistent, typed, searchable working memory for a single AI agent, with
token-budget context curation (``curate(query, token_budget)``) as the
headline primitive. Named after Endel Tulving, the psychologist who
established that memory has types.
"""

from tulving.adapters.embeddings import (
    EmbeddingAdapter,
    HashEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from tulving.context.config import LifecycleConfig
from tulving.context.curator import CuratedContext
from tulving.context.lifecycle import Session, SessionSummary
from tulving.entry import MemoryEntry, Relationship, SourceInfo
from tulving.enums import ArchiveReason, MatchType, MemoryType, SessionStatus
from tulving.exceptions import (
    ConfigError,
    MemoryStoreError,
    ScopeError,
    SecurityError,
    StorageError,
    TulvingError,
    VectorIndexError,
)
from tulving.export import ImportReport, MemoryExporter, MemoryImporter
from tulving.memory import Memory, SearchResult, StartupReport, StoreStats, VacuumResult

__version__ = "0.1.2"

__all__ = [
    "ArchiveReason",
    "ConfigError",
    "CuratedContext",
    "EmbeddingAdapter",
    "HashEmbedder",
    "ImportReport",
    "LifecycleConfig",
    "LocalEmbedder",
    "MatchType",
    "Memory",
    "MemoryEntry",
    "MemoryExporter",
    "MemoryImporter",
    "MemoryStoreError",
    "MemoryType",
    "OpenAIEmbedder",
    "Relationship",
    "ScopeError",
    "SearchResult",
    "SecurityError",
    "Session",
    "SessionStatus",
    "SessionSummary",
    "SourceInfo",
    "StartupReport",
    "StorageError",
    "StoreStats",
    "TulvingError",
    "VacuumResult",
    "VectorIndexError",
    "__version__",
]

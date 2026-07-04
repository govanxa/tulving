"""Tulving — the context-budget engine for AI agents.

Persistent, typed, searchable working memory for a single AI agent, with
token-budget context curation (``curate(query, token_budget)``) as the
headline primitive. Named after Endel Tulving, the psychologist who
established that memory has types.

This is a name-holding pre-release (0.0.1). The v0.1 implementation is in
active development.
"""

from tulving.adapters.embeddings import (
    EmbeddingAdapter,
    HashEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
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

__version__ = "0.0.1"

__all__ = [
    "ArchiveReason",
    "ConfigError",
    "EmbeddingAdapter",
    "HashEmbedder",
    "LocalEmbedder",
    "MatchType",
    "MemoryEntry",
    "MemoryStoreError",
    "MemoryType",
    "OpenAIEmbedder",
    "Relationship",
    "ScopeError",
    "SecurityError",
    "SessionStatus",
    "SourceInfo",
    "StorageError",
    "TulvingError",
    "VectorIndexError",
    "__version__",
]

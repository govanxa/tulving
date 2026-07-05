"""Tulving exception hierarchy (revision-plan D6). Zero dependencies.

NEVER define exceptions named ``MemoryError`` or ``IndexError`` — they
shadow Python builtins.
"""


class TulvingError(Exception):
    """Base class for all Tulving errors. Catch this to catch everything Tulving raises."""


class MemoryStoreError(TulvingError):
    """CRUD-level failure: entry not found, invalid type (e.g. caller-supplied
    SUMMARY), malformed entry data at the store boundary."""


class StorageError(TulvingError):
    """Backend persistence failure: SQLite errors, schema/migration problems,
    corrupt rows, unknown persisted enum values."""


class VectorIndexError(TulvingError):
    """Vector-index failure: hnswlib load/save errors, label-mapping
    inconsistency, dimension/model mismatch against the meta table."""


class ScopeError(TulvingError):
    """Cooperative-isolation violation (reserved: v0.2 multi-agent scoping,
    ADR-016). NOT a security boundary — see SecurityError."""


class SecurityError(TulvingError):
    """Genuine security boundary: path traversal, inline credentials,
    invalid leaf names, containment escape."""


class ConfigError(TulvingError):
    """Invalid configuration: missing environment credential, invalid adapter,
    bad LifecycleConfig values."""

"""Key-value retrieval view: exact match + prefix scan over LIVE keyed entries.

Read-only — the write path (upsert/supersede, D1) is ``MemoryStore.create()``;
forgetting is ``Memory.forget()``. No namespaces in v0.1 (revision-plan §4):
keys are global per memory path and the ``category:identifier`` convention
covers the practical need. There is no separate index structure to maintain:
the D3 partial unique index ``UNIQUE(key) WHERE archived = 0 AND key IS NOT
NULL`` *is* the index; this module is the disciplined way to query it.

Touch discipline (D3): ``get`` counts as access and touches via the store;
``scan``/``keys``/``exists`` are listings/probes and never touch — if the
curator later includes a scanned entry, the CURATOR's inclusion-touch covers
it (no double counting).
"""

from __future__ import annotations

from tulving.entry import MemoryEntry
from tulving.exceptions import MemoryStoreError
from tulving.store import MemoryStore


class KVIndex:
    """The "I know exactly what I'm looking for" retrieval path.

    Sits on ``MemoryStore`` (not the backend) so touch semantics, hydration,
    and corrupt-row error translation exist in exactly one place. Stateless:
    no caching, no writes, no crash windows.
    """

    def __init__(self, store: MemoryStore) -> None:
        """Cheap: holds the reference. No scans, no I/O (D8 discipline)."""
        self._store = store

    def get(self, key: str) -> MemoryEntry | None:
        """Exact match against ACTIVE entries; None on miss.

        A superseded or forgotten entry's key is invisible. COUNTS AS ACCESS
        (D3): delegates ``store.get_by_key(key, touch=True)``.

        Args:
            key: The exact key; must be non-empty.

        Returns:
            The live entry, or None when no active entry holds ``key``.

        Raises:
            MemoryStoreError: On an empty key (caller bug).
        """
        if not key:
            raise MemoryStoreError("key must be non-empty")
        return self._store.get_by_key(key, touch=True)

    def scan(self, prefix: str, limit: int = 100) -> list[MemoryEntry]:
        """All ACTIVE entries whose key starts with ``prefix``, key ASC.

        The prefix is literal text — ``%``/``_`` have no special meaning
        (escaping is the backend's contract). ``prefix=""`` scans every keyed
        live entry (up to ``limit``). Does NOT touch: a scan is a listing,
        not a targeted access (D3).

        Args:
            prefix: Literal key prefix; empty means "all keyed entries".
            limit: Maximum entries returned; must be >= 1.

        Returns:
            Matching live entries ordered by key ascending.

        Raises:
            MemoryStoreError: When ``limit`` < 1.
        """
        if limit < 1:
            raise MemoryStoreError("limit must be >= 1")
        return self._store.scan_keys(prefix, limit)

    def keys(self, prefix: str = "") -> list[str]:
        """All active keys matching ``prefix``, sorted ASC.

        No entry loads, no touch. Backs the MCP ``list_keys`` tool.

        Args:
            prefix: Literal key prefix; empty means "all keys".

        Returns:
            Sorted active keys.
        """
        return self._store.list_keys(prefix)

    def exists(self, key: str) -> bool:
        """Active-key probe; no load, therefore NO touch (D3).

        Args:
            key: The exact key.

        Returns:
            True when a live entry holds ``key``.
        """
        return self._store.exists(key)

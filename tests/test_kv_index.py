"""Tests for tulving.kv_index — written BEFORE implementation.

The KV index is a read-only view over MemoryStore: exact match (touches,
D3) + prefix scan / key listing / existence probe (never touch). Entries are
seeded through store.create() — the only write path (D1).
"""

from typing import Any

import pytest

from tulving.adapters.storage import InMemoryBackend
from tulving.entry import MemoryEntry, SourceInfo
from tulving.enums import MemoryType
from tulving.exceptions import MemoryStoreError, StorageError
from tulving.kv_index import KVIndex
from tulving.store import MemoryStore


@pytest.fixture
def store(fake_clock: Any) -> MemoryStore:
    return MemoryStore(InMemoryBackend(), clock=fake_clock)


@pytest.fixture
def kv(store: MemoryStore) -> KVIndex:
    return KVIndex(store)


def seed(store: MemoryStore, key: str | None, content: str = "seeded") -> MemoryEntry:
    return store.create(
        content=content,
        type=MemoryType.FACT,
        source=SourceInfo(agent_id="agent-1"),
        key=key,
    )


class _ExplodingStore:
    """Duck-typed store stub: backend errors must pass through untranslated."""

    def get_by_key(self, key: str, *, touch: bool = True) -> MemoryEntry | None:
        raise StorageError("backend exploded")

    def scan_keys(self, prefix: str, limit: int = 100) -> list[MemoryEntry]:
        raise StorageError("backend exploded")

    def list_keys(self, prefix: str = "") -> list[str]:
        raise StorageError("backend exploded")

    def exists(self, key: str) -> bool:
        raise StorageError("backend exploded")


# ---------------------------------------------------------------------------
# Failure & boundary paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_get_empty_key_rejected(self, kv: KVIndex) -> None:
        with pytest.raises(MemoryStoreError, match="non-empty"):
            kv.get("")

    def test_scan_bad_limit_rejected(self, kv: KVIndex) -> None:
        with pytest.raises(MemoryStoreError):
            kv.scan("x", limit=0)
        with pytest.raises(MemoryStoreError):
            kv.scan("x", limit=-1)

    def test_storage_errors_propagate_untranslated(self) -> None:
        kv = KVIndex(_ExplodingStore())  # type: ignore[arg-type]
        with pytest.raises(StorageError, match="backend exploded"):
            kv.get("k")
        with pytest.raises(StorageError, match="backend exploded"):
            kv.scan("k")
        with pytest.raises(StorageError, match="backend exploded"):
            kv.keys()
        with pytest.raises(StorageError, match="backend exploded"):
            kv.exists("k")


class TestBoundaryConditions:
    def test_get_unknown_key_returns_none(self, kv: KVIndex) -> None:
        assert kv.get("nope") is None

    def test_keys_no_match_returns_empty(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "a:1")
        assert kv.keys("zzz") == []

    def test_scan_no_match_returns_empty(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "a:1")
        assert kv.scan("zzz") == []

    def test_exists_unknown_key_false(self, kv: KVIndex) -> None:
        assert kv.exists("nope") is False


# ---------------------------------------------------------------------------
# Live-only visibility (audit-derived)
# ---------------------------------------------------------------------------


class TestLiveOnlyVisibility:
    def test_archived_keys_invisible_everywhere(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "k")
        assert store.forget("k") is True
        assert kv.get("k") is None
        assert kv.exists("k") is False
        all_keys = kv.keys()
        assert "k" not in all_keys
        assert all(e.key != "k" for e in kv.scan(""))

    def test_supersede_visibility(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "k", content="A")
        seed(store, "k", content="B")
        got = kv.get("k")
        assert got is not None
        assert got.content == "B"  # A's archived row never surfaces
        assert kv.keys() == ["k"]  # exactly once

        store.forget("k")
        seed(store, "k", content="C")
        got = kv.get("k")
        assert got is not None and got.content == "C"
        assert kv.keys() == ["k"]  # still exactly one live k

    def test_unkeyed_entries_never_appear(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, None)
        seed(store, "keyed:1")
        assert kv.keys("") == ["keyed:1"]
        assert [e.key for e in kv.scan("")] == ["keyed:1"]


# ---------------------------------------------------------------------------
# Touch semantics (D3)
# ---------------------------------------------------------------------------


class TestTouchSemantics:
    def test_get_counts_as_access(self, kv: KVIndex, store: MemoryStore, fake_clock: Any) -> None:
        entry = seed(store, "k")
        fake_clock.advance(hours=2)
        got = kv.get("k")
        assert got is not None
        assert got.access_count == 1
        assert got.last_accessed_at == fake_clock.current
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.access_count == 1
        assert reread.last_accessed_at == fake_clock.current

    def test_repeated_gets_keep_counting(self, kv: KVIndex, store: MemoryStore) -> None:
        entry = seed(store, "k")
        kv.get("k")
        kv.get("k")
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None and reread.access_count == 2

    def test_exists_keys_scan_never_touch(
        self, kv: KVIndex, store: MemoryStore, fake_clock: Any
    ) -> None:
        entry = seed(store, "fact:one")
        fake_clock.advance(hours=1)
        kv.exists("fact:one")
        kv.keys()
        kv.scan("fact:")
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.access_count == 0
        assert reread.last_accessed_at == entry.created_at


# ---------------------------------------------------------------------------
# Prefix scan
# ---------------------------------------------------------------------------


class TestPrefixScan:
    def seed_keys(self, store: MemoryStore) -> None:
        for key in ("decision:db", "decision:api", "decisions_old:x", "fact:one"):
            seed(store, key)

    def test_true_prefixes_only(self, kv: KVIndex, store: MemoryStore) -> None:
        self.seed_keys(store)
        assert [e.key for e in kv.scan("decision:")] == ["decision:api", "decision:db"]
        assert kv.keys("decision:") == ["decision:api", "decision:db"]

    def test_ordering_and_limit(self, kv: KVIndex, store: MemoryStore) -> None:
        self.seed_keys(store)
        assert [e.key for e in kv.scan("decision:", limit=1)] == ["decision:api"]
        assert kv.keys() == sorted(kv.keys())

    def test_literal_wildcards(self, kv: KVIndex, store: MemoryStore) -> None:
        for key in ("pct%x", "pctYx", "under_a", "underXa"):
            seed(store, key)
        assert [e.key for e in kv.scan("pct%")] == ["pct%x"]
        assert [e.key for e in kv.scan("under_")] == ["under_a"]

    def test_empty_prefix_scans_all_keyed_up_to_limit(
        self, kv: KVIndex, store: MemoryStore
    ) -> None:
        self.seed_keys(store)
        seed(store, None)
        assert [e.key for e in kv.scan("")] == [
            "decision:api",
            "decision:db",
            "decisions_old:x",
            "fact:one",
        ]
        assert [e.key for e in kv.scan("", limit=2)] == ["decision:api", "decision:db"]

    def test_prefix_matching_is_case_sensitive(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "Case:a")
        seed(store, "case:b")
        assert [e.key for e in kv.scan("case:")] == ["case:b"]
        assert kv.keys("Case:") == ["Case:a"]


# ---------------------------------------------------------------------------
# Returned-object hygiene
# ---------------------------------------------------------------------------


class TestReturnedObjectHygiene:
    def test_importance_is_none(self, kv: KVIndex, store: MemoryStore) -> None:
        seed(store, "k")
        got = kv.get("k")
        assert got is not None and got.importance is None
        assert all(e.importance is None for e in kv.scan(""))

    def test_mutating_returned_entry_does_not_corrupt_store(
        self, kv: KVIndex, store: MemoryStore
    ) -> None:
        entry = store.create(
            content="c",
            type=MemoryType.FACT,
            source=SourceInfo(agent_id="agent-1"),
            key="k",
            tags=["clean"],
        )
        got = kv.get("k")
        assert got is not None
        got.tags.append("dirty")
        got.content = "dirty"
        fresh = store.get_by_id(entry.id, touch=False)
        assert fresh is not None
        assert fresh.tags == ["clean"]
        assert fresh.content == "c"
        scanned = kv.scan("k")
        assert scanned[0].tags == ["clean"]

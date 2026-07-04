"""Tests for tulving.store — written BEFORE implementation.

Runs against InMemoryBackend (licensed by the parity suite in
tests/test_storage.py) plus a thin SQLite smoke subset. Covers the D1
supersede algorithm, D2/D3 touch semantics, the batch path, and the
mandatory audit regressions (supersede-vs-archived-key, batch searchability
at stub level, purge refuses SUMMARIZED sources unless explicit).
"""

import inspect
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tulving.adapters.storage import InMemoryBackend, SQLiteBackend, pack_embedding
from tulving.entry import MemoryEntry, Relationship, SourceInfo
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import MemoryStoreError, StorageError
from tulving.security import validate_leaf_name
from tulving.store import MemoryStore


def src(agent_id: str = "agent-1") -> SourceInfo:
    return SourceInfo(agent_id=agent_id)


@pytest.fixture
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture
def store(backend: InMemoryBackend, fake_clock: Any) -> MemoryStore:
    return MemoryStore(backend, clock=fake_clock)


def create_fact(store: MemoryStore, content: str = "the sky is blue", **kwargs: Any) -> MemoryEntry:
    kwargs.setdefault("type", MemoryType.FACT)
    kwargs.setdefault("source", src())
    return store.create(content=content, **kwargs)


class _CorruptTypeBackend(InMemoryBackend):
    """Returns rows whose persisted type no longer parses (corrupt-row path)."""

    def read(self, entry_id: str) -> dict[str, Any] | None:
        row = super().read(entry_id)
        if row is not None:
            row["type"] = "bogus"
        return row


class _ExplodingListBackend(InMemoryBackend):
    def list(
        self, filters: dict[str, Any], limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        raise StorageError("backend exploded")


class _TxnAssertingBackend(InMemoryBackend):
    """Fails any entry update issued OUTSIDE an ambient transaction —
    pins that archive()/forget() run their check-then-act atomically."""

    def update(self, entry_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        assert getattr(self._local, "in_txn", False), "update ran outside a transaction"
        return super().update(entry_id, fields)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_caller_summary_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.create(content="digest", type=MemoryType.SUMMARY, source=src())

    def test_allow_summary_requires_source_entry_ids(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.create(
                content="digest", type=MemoryType.SUMMARY, source=src(), _allow_summary=True
            )
        with pytest.raises(MemoryStoreError):
            store.create(
                content="digest",
                type=MemoryType.SUMMARY,
                source=src(),
                _allow_summary=True,
                _source_entry_ids=[],
            )

    def test_summarizer_path_succeeds(self, store: MemoryStore) -> None:
        entry = store.create(
            content="digest",
            type=MemoryType.SUMMARY,
            source=src(),
            _allow_summary=True,
            _source_entry_ids=["a", "b"],
        )
        assert entry.type is MemoryType.SUMMARY
        assert entry.source_entry_ids == ["a", "b"]

    def test_empty_content_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.create(content="", type=MemoryType.FACT, source=src())

    def test_out_of_range_importance_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            create_fact(store, base_importance=1.5)

    def test_empty_agent_id_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.create(content="x", type=MemoryType.FACT, source=SourceInfo(agent_id=""))

    def test_empty_key_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            create_fact(store, key="")

    def test_empty_tag_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            create_fact(store, tags=["ok", ""])

    def test_non_str_tag_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            create_fact(store, tags=["ok", 42])  # type: ignore[list-item]

    def test_non_json_relationship_metadata_rejected(self, store: MemoryStore) -> None:
        bad = Relationship(target_id="t", relationship_type="relates_to", metadata={"x": object()})
        with pytest.raises(MemoryStoreError):
            create_fact(store, relationships=[bad])

    def test_list_bad_limit_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.list(limit=0)
        with pytest.raises(MemoryStoreError):
            store.list(limit=-5)

    def test_scan_keys_bad_limit_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.scan_keys("x", limit=0)

    def test_not_found_update_archive_unarchive(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.update("missing", content="x")
        with pytest.raises(MemoryStoreError):
            store.archive("missing", ArchiveReason.EVICTED)
        with pytest.raises(MemoryStoreError):
            store.unarchive("missing")

    def test_get_by_id_miss_returns_none(self, store: MemoryStore) -> None:
        assert store.get_by_id("missing") is None

    def test_get_by_key_miss_returns_none(self, store: MemoryStore) -> None:
        assert store.get_by_key("missing") is None

    def test_forget_miss_returns_false(self, store: MemoryStore) -> None:
        assert store.forget("nope") is False

    def test_set_embedding_missing_id(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError):
            store.set_embedding("missing", b"\x00\x00\x80?")

    def test_set_embedding_backend_failure_stays_storage_error(
        self, store: MemoryStore, backend: InMemoryBackend
    ) -> None:
        """Only the missing-id case is translated; real backend failures
        (here: closed backend) must NOT be masked as CRUD misses."""
        entry = create_fact(store)
        backend.close()
        with pytest.raises(StorageError):
            store.set_embedding(entry.id, b"\x00\x00\x80?")

    def test_update_empty_content_rejected(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        with pytest.raises(MemoryStoreError, match="non-empty"):
            store.update(entry.id, content="")
        unchanged = store.get_by_id(entry.id, touch=False)
        assert unchanged is not None and unchanged.content == "the sky is blue"

    def test_batch_item_unknown_kwarg_rejected(self, store: MemoryStore) -> None:
        items = [{"content": "x", "type": MemoryType.FACT, "source": src(), "bogus_kwarg": 1}]
        with pytest.raises(MemoryStoreError, match="batch item"):
            store.batch_create(items)

    def test_count_unknown_filter_rejected(self, store: MemoryStore) -> None:
        with pytest.raises(MemoryStoreError, match="unknown filter"):
            store.count(bogus=1)

    def test_naive_filter_datetime_rejected(self, store: MemoryStore) -> None:
        naive = datetime(2026, 7, 3)
        with pytest.raises(MemoryStoreError, match="timezone-aware"):
            store.list(since=naive)
        with pytest.raises(MemoryStoreError, match="timezone-aware"):
            store.list(accessed_before=naive)

    def test_archive_already_archived_rejected(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        store.archive(entry.id, ArchiveReason.EVICTED)
        with pytest.raises(MemoryStoreError):
            store.archive(entry.id, ArchiveReason.FORGOTTEN)

    def test_unarchive_live_entry_rejected(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        with pytest.raises(MemoryStoreError):
            store.unarchive(entry.id)

    def test_update_has_no_immutable_params(self) -> None:
        params = inspect.signature(MemoryStore.update).parameters
        for forbidden in ("base_importance", "importance", "type", "key", "created_at"):
            assert forbidden not in params

    def test_update_archived_entry_allowed_metadata_repair(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        store.archive(entry.id, ArchiveReason.EVICTED)
        updated = store.update(entry.id, content="repaired")
        assert updated.content == "repaired"
        assert updated.archived is True
        assert updated.archive_reason is ArchiveReason.EVICTED

    def test_corrupt_row_raises_storage_error(self, fake_clock: Any) -> None:
        backend = _CorruptTypeBackend()
        store = MemoryStore(backend, clock=fake_clock)
        entry = create_fact(store)
        with pytest.raises(StorageError):
            store.get_by_id(entry.id)

    def test_storage_error_passes_through_untranslated(self, fake_clock: Any) -> None:
        store = MemoryStore(_ExplodingListBackend(), clock=fake_clock)
        with pytest.raises(StorageError, match="backend exploded"):
            store.list()


# ---------------------------------------------------------------------------
# Supersede (D1 — mandatory audit regressions)
# ---------------------------------------------------------------------------


class TestSupersede:
    def test_round_trip_never_raises(
        self, store: MemoryStore, backend: InMemoryBackend, fake_clock: Any
    ) -> None:
        old = create_fact(store, content="A", key="k")
        fake_clock.advance(hours=1)
        new = create_fact(store, content="B", key="k")  # never raises (D1)

        old_after = store.get_by_id(old.id, touch=False)
        assert old_after is not None
        assert old_after.archived is True
        assert old_after.archive_reason is ArchiveReason.SUPERSEDED
        assert old_after.updated_at == fake_clock.current  # bumped at supersede time

        assert new.archived is False
        links = [r for r in new.relationships if r.relationship_type == "supersedes"]
        assert [r.target_id for r in links] == [old.id]

        current = store.get_by_key("k", touch=False)
        assert current is not None and current.id == new.id
        assert store.count(include_archived=True) == 2

    def test_supersede_against_archived_key_holder(self, store: MemoryStore) -> None:
        """Mandatory: a key held only by an ARCHIVED row is simply free."""
        first = create_fact(store, content="A", key="k")
        assert store.forget("k") is True  # archived FORGOTTEN
        second = create_fact(store, content="B", key="k")  # clean insert, no exception

        assert not any(r.relationship_type == "supersedes" for r in second.relationships)
        first_after = store.get_by_id(first.id, touch=False)
        assert first_after is not None
        assert first_after.archive_reason is ArchiveReason.FORGOTTEN  # untouched

    def test_three_generation_chain(self, store: MemoryStore) -> None:
        gen1 = create_fact(store, content="v1", key="k")
        gen2 = create_fact(store, content="v2", key="k")
        gen3 = create_fact(store, content="v3", key="k")

        superseded = store.list(archive_reasons=[ArchiveReason.SUPERSEDED])
        assert {e.id for e in superseded} == {gen1.id, gen2.id}

        def supersedes_target(entry: MemoryEntry) -> str:
            (link,) = [r for r in entry.relationships if r.relationship_type == "supersedes"]
            return link.target_id

        assert supersedes_target(gen3) == gen2.id
        gen2_row = store.get_by_id(gen2.id, touch=False)
        assert gen2_row is not None
        assert supersedes_target(gen2_row) == gen1.id

    def test_supersede_atomicity_rolls_back(
        self, store: MemoryStore, backend: InMemoryBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        old = create_fact(store, content="A", key="k")

        def boom(*args: Any, **kwargs: Any) -> None:
            raise StorageError("insert failed")

        monkeypatch.setattr(backend, "create", boom)
        with pytest.raises(StorageError):
            create_fact(store, content="B", key="k")
        monkeypatch.undo()

        old_after = store.get_by_key("k", touch=False)
        assert old_after is not None and old_after.id == old.id
        assert old_after.archived is False  # rollback restored the old entry
        assert store.count(include_archived=True) == 1

    def test_unkeyed_entries_never_collide(self, store: MemoryStore) -> None:
        first = create_fact(store, content="A")
        second = create_fact(store, content="B")
        assert first.id != second.id
        assert store.count() == 2


# ---------------------------------------------------------------------------
# Batch (mandatory audit regression: batch-created entries are searchable)
# ---------------------------------------------------------------------------


class TestBatchCreate:
    def test_batch_rows_and_blobs_present(
        self, store: MemoryStore, backend: InMemoryBackend
    ) -> None:
        """Stub-level searchability: rows + embedding BLOBs + iter_embeddings."""
        items = [
            {
                "content": f"item {i}",
                "type": MemoryType.FACT,
                "source": src(),
                "key": f"batch:{i}",
                "embedding": pack_embedding([float(i + 1)]),
            }
            for i in range(3)
        ]
        entries = store.batch_create(items)
        assert len(entries) == 3
        for i, entry in enumerate(entries):
            fetched = store.get_by_id(entry.id, touch=False)
            assert fetched is not None
            by_key = store.get_by_key(f"batch:{i}", touch=False)
            assert by_key is not None and by_key.id == entry.id
            assert backend.get_embedding(entry.id) == pack_embedding([float(i + 1)])
        assert {i for i, _ in backend.iter_embeddings()} == {e.id for e in entries}

    def test_intra_batch_supersede(self, store: MemoryStore) -> None:
        entries = store.batch_create(
            [
                {"content": "A", "type": MemoryType.FACT, "source": src(), "key": "k"},
                {"content": "other", "type": MemoryType.FACT, "source": src()},
                {"content": "B", "type": MemoryType.FACT, "source": src(), "key": "k"},
            ]
        )
        first, _, third = entries
        first_after = store.get_by_id(first.id, touch=False)
        assert first_after is not None
        assert first_after.archived is True
        assert first_after.archive_reason is ArchiveReason.SUPERSEDED
        links = [r for r in third.relationships if r.relationship_type == "supersedes"]
        assert [r.target_id for r in links] == [first.id]
        current = store.get_by_key("k", touch=False)
        assert current is not None and current.id == third.id

    def test_batch_supersedes_pre_existing_key(self, store: MemoryStore) -> None:
        old = create_fact(store, content="old", key="k")
        (new,) = store.batch_create(
            [{"content": "new", "type": MemoryType.FACT, "source": src(), "key": "k"}]
        )
        old_after = store.get_by_id(old.id, touch=False)
        assert old_after is not None and old_after.archive_reason is ArchiveReason.SUPERSEDED
        links = [r for r in new.relationships if r.relationship_type == "supersedes"]
        assert [r.target_id for r in links] == [old.id]

    def test_batch_all_or_nothing(self, store: MemoryStore, backend: InMemoryBackend) -> None:
        pre_existing = create_fact(store, content="keeper", key="k")
        items = [
            {
                "content": "ok",
                "type": MemoryType.FACT,
                "source": src(),
                "embedding": pack_embedding([1.0]),
            },
            {"content": "k-stealer", "type": MemoryType.FACT, "source": src(), "key": "k"},
            {"content": "", "type": MemoryType.FACT, "source": src()},  # invalid
        ]
        with pytest.raises(MemoryStoreError):
            store.batch_create(items)
        assert store.count(include_archived=True) == 1
        keeper = store.get_by_key("k", touch=False)
        assert keeper is not None and keeper.id == pre_existing.id
        assert keeper.archived is False
        assert list(backend.iter_embeddings(include_archived=True)) == []

    def test_batch_order_and_parity_with_single_create(self, fake_clock: Any) -> None:
        single_store = MemoryStore(InMemoryBackend(), clock=fake_clock)
        batch_store = MemoryStore(InMemoryBackend(), clock=fake_clock)

        single = single_store.create(
            content="same", type=MemoryType.FACT, source=src(), tags=["t"], pinned=True
        )
        (batched,) = batch_store.batch_create(
            [
                {
                    "content": "same",
                    "type": MemoryType.FACT,
                    "source": src(),
                    "tags": ["t"],
                    "pinned": True,
                }
            ]
        )
        single_dict = single.to_dict()
        batched_dict = batched.to_dict()
        single_dict.pop("id")
        batched_dict.pop("id")
        assert single_dict == batched_dict

        entries = batch_store.batch_create(
            [
                {"content": "one", "type": MemoryType.FACT, "source": src()},
                {"content": "two", "type": MemoryType.PLAN, "source": src()},
            ]
        )
        assert [e.content for e in entries] == ["one", "two"]


# ---------------------------------------------------------------------------
# Touch semantics (D2/D3)
# ---------------------------------------------------------------------------


class TestTouchSemantics:
    def test_get_by_id_touches_by_default(self, store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(store)
        created_at = entry.created_at
        fake_clock.advance(hours=2)
        got = store.get_by_id(entry.id)
        assert got is not None
        assert got.access_count == 1
        assert got.last_accessed_at == fake_clock.current
        assert got.updated_at == created_at  # access is not an edit
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.access_count == 1
        assert reread.last_accessed_at == fake_clock.current

    def test_get_by_key_touches_by_default(self, store: MemoryStore, fake_clock: Any) -> None:
        create_fact(store, key="k")
        fake_clock.advance(hours=1)
        got = store.get_by_key("k")
        assert got is not None and got.access_count == 1
        reread = store.get_by_key("k", touch=False)
        assert reread is not None and reread.access_count == 1

    def test_touch_false_changes_nothing(self, store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(store)
        fake_clock.advance(hours=1)
        store.get_by_id(entry.id, touch=False)
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.access_count == 0
        assert reread.last_accessed_at == entry.created_at

    def test_repeated_gets_keep_counting(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        store.get_by_id(entry.id)
        store.get_by_id(entry.id)
        got = store.get_by_id(entry.id, touch=False)
        assert got is not None and got.access_count == 2

    def test_listings_never_touch(self, store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(store, key="fact:k", tags=["t"])
        fake_clock.advance(hours=1)
        store.exists("fact:k")
        store.list()
        store.list(tags=["t"])
        store.scan_keys("fact:")
        store.list_keys()
        store.count()
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.access_count == 0
        assert reread.last_accessed_at == entry.created_at

    def test_get_archived_returns_but_never_touches(
        self, store: MemoryStore, fake_clock: Any
    ) -> None:
        entry = create_fact(store)
        store.archive(entry.id, ArchiveReason.EVICTED)
        fake_clock.advance(hours=1)
        got = store.get_by_id(entry.id)  # touch=True, but archived
        assert got is not None
        assert got.archived is True
        assert got.access_count == 0
        assert got.last_accessed_at == entry.created_at

    def test_touch_entries_batched(self, store: MemoryStore, fake_clock: Any) -> None:
        a = create_fact(store, content="a")
        b = create_fact(store, content="b")
        c = create_fact(store, content="c")
        store.archive(c.id, ArchiveReason.EVICTED)
        fake_clock.advance(hours=3)
        touched = store.touch_entries([a.id, b.id, c.id])
        assert touched == 2
        for entry_id in (a.id, b.id):
            got = store.get_by_id(entry_id, touch=False)
            assert got is not None
            assert got.access_count == 1
            assert got.last_accessed_at == fake_clock.current
        archived = store.get_by_id(c.id, touch=False)
        assert archived is not None and archived.access_count == 0

    def test_decay_hygiene_importance_stays_none(self, store: MemoryStore) -> None:
        entry = create_fact(store, base_importance=0.7, key="k")
        assert entry.importance is None
        got = store.get_by_id(entry.id)
        assert got is not None and got.importance is None
        assert got.base_importance == 0.7
        listed = store.list()
        assert all(e.importance is None for e in listed)
        by_key = store.get_by_key("k", touch=False)
        assert by_key is not None and by_key.base_importance == 0.7


# ---------------------------------------------------------------------------
# Archive / forget transactionality (DBE-H1: check-then-act under threads)
# ---------------------------------------------------------------------------


class TestArchiveTransactionality:
    def test_archive_check_and_update_share_one_transaction(self, fake_clock: Any) -> None:
        backend = _TxnAssertingBackend()
        store = MemoryStore(backend, clock=fake_clock)
        entry = store.create(content="source doc", type=MemoryType.FACT, source=src())
        archived = store.archive(entry.id, ArchiveReason.SUMMARIZED)
        assert archived.archive_reason is ArchiveReason.SUMMARIZED

    def test_forget_lookup_and_archive_share_one_transaction(self, fake_clock: Any) -> None:
        backend = _TxnAssertingBackend()
        store = MemoryStore(backend, clock=fake_clock)
        store.create(content="x", type=MemoryType.FACT, source=src(), key="k")
        assert store.forget("k") is True
        got = store.get_by_id(store.list(include_archived=True)[0].id, touch=False)
        assert got is not None and got.archive_reason is ArchiveReason.FORGOTTEN

    def test_recorded_reason_cannot_be_overwritten(self, store: MemoryStore) -> None:
        """A SUMMARIZED source relabeled EVICTED would be purged by default —
        the already-archived check inside the transaction prevents it."""
        entry = create_fact(store, content="summarization source")
        store.archive(entry.id, ArchiveReason.SUMMARIZED)
        with pytest.raises(MemoryStoreError, match="already archived"):
            store.archive(entry.id, ArchiveReason.EVICTED)
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.archive_reason is ArchiveReason.SUMMARIZED
        assert store.purge_archived() == 0  # default purge still spares it
        assert store.get_by_id(entry.id, touch=False) is not None


# ---------------------------------------------------------------------------
# Purge (mandatory audit regression: refuses SUMMARIZED unless explicit)
# ---------------------------------------------------------------------------


class TestPurge:
    def _archive_one_per_reason(self, store: MemoryStore) -> dict[ArchiveReason, str]:
        ids: dict[ArchiveReason, str] = {}
        for reason in ArchiveReason:
            entry = create_fact(store, content=f"for {reason.value}")
            store.archive(entry.id, reason)
            ids[reason] = entry.id
        return ids

    def test_default_purge_spares_summarized(
        self, store: MemoryStore, backend: InMemoryBackend
    ) -> None:
        ids = self._archive_one_per_reason(store)
        survivor = create_fact(store, content="active")
        deleted = store.purge_archived()
        assert deleted == 4  # EVICTED, SUPERSEDED, FORGOTTEN, ABANDONED
        assert backend.read(ids[ArchiveReason.SUMMARIZED]) is not None
        for reason in (
            ArchiveReason.EVICTED,
            ArchiveReason.SUPERSEDED,
            ArchiveReason.FORGOTTEN,
            ArchiveReason.ABANDONED,
        ):
            assert backend.read(ids[reason]) is None
        assert backend.read(survivor.id) is not None

    def test_explicit_summarized_purge(self, store: MemoryStore, backend: InMemoryBackend) -> None:
        ids = self._archive_one_per_reason(store)
        deleted = store.purge_archived(reasons=[ArchiveReason.SUMMARIZED])
        assert deleted == 1
        assert backend.read(ids[ArchiveReason.SUMMARIZED]) is None
        assert backend.read(ids[ArchiveReason.EVICTED]) is not None

    def test_older_than_cutoff(
        self, store: MemoryStore, backend: InMemoryBackend, fake_clock: Any
    ) -> None:
        early = create_fact(store, content="early")
        late = create_fact(store, content="late")
        store.archive(early.id, ArchiveReason.EVICTED)  # archived at t0
        fake_clock.advance(hours=10)
        store.archive(late.id, ArchiveReason.EVICTED)  # archived at t0+10h
        deleted = store.purge_archived(older_than=timedelta(hours=5))
        assert deleted == 1
        assert backend.read(early.id) is None
        assert backend.read(late.id) is not None

    def test_active_rows_never_purged(self, store: MemoryStore) -> None:
        create_fact(store, content="live one")
        create_fact(store, content="live two")
        assert store.purge_archived(older_than=timedelta(0)) == 0
        assert store.count() == 2


# ---------------------------------------------------------------------------
# Remaining CRUD
# ---------------------------------------------------------------------------


class TestCrud:
    def test_update_merges_and_bumps_updated_at(self, store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(store, tags=["old"], pinned=False)
        fake_clock.advance(hours=1)
        rel = Relationship(target_id="t1", relationship_type="relates_to")
        updated = store.update(
            entry.id, content="new", tags=["new-tag"], relationships=[rel], pinned=True
        )
        assert updated.content == "new"
        assert updated.tags == ["new-tag"]
        assert updated.relationships == [rel]
        assert updated.pinned is True
        assert updated.updated_at == fake_clock.current
        assert updated.created_at == entry.created_at
        assert [e.id for e in store.list(tags=["new-tag"])] == [entry.id]
        assert store.list(tags=["old"]) == []

    def test_update_without_content_leaves_content(self, store: MemoryStore) -> None:
        entry = create_fact(store, tags=["t"])
        updated = store.update(entry.id, pinned=True)
        assert updated.content == "the sky is blue"
        assert updated.pinned is True
        assert updated.tags == ["t"]

    def test_tag_dedupe_on_create(self, store: MemoryStore) -> None:
        entry = create_fact(store, tags=["a", "b", "a"])
        assert entry.tags == ["a", "b"]

    def test_unarchive_restores(self, store: MemoryStore) -> None:
        entry = create_fact(store, key="k")
        store.archive(entry.id, ArchiveReason.FORGOTTEN)
        assert store.exists("k") is False
        restored = store.unarchive(entry.id)
        assert restored.archived is False
        assert restored.archive_reason is None
        assert store.exists("k") is True

    def test_unarchive_key_conflict_rejected(self, store: MemoryStore) -> None:
        old = create_fact(store, content="old", key="k")
        create_fact(store, content="new", key="k")  # supersedes old
        with pytest.raises(MemoryStoreError, match="key"):
            store.unarchive(old.id)
        still_archived = store.get_by_id(old.id, touch=False)
        assert still_archived is not None and still_archived.archived is True

    def test_forget_soft(self, store: MemoryStore) -> None:
        entry = create_fact(store, key="k")
        assert store.forget("k") is True
        assert store.exists("k") is False
        assert "k" not in store.list_keys()
        archived = store.get_by_id(entry.id, touch=False)
        assert archived is not None
        assert archived.archive_reason is ArchiveReason.FORGOTTEN

    def test_forget_hard_removes_row_and_embedding(
        self, store: MemoryStore, backend: InMemoryBackend
    ) -> None:
        entry = create_fact(store, key="k", embedding=pack_embedding([1.0]))
        assert store.forget("k", hard=True) is True
        assert store.get_by_id(entry.id, touch=False) is None
        assert list(backend.iter_embeddings(include_archived=True)) == []

    def test_set_embedding_passthrough(self, store: MemoryStore, backend: InMemoryBackend) -> None:
        entry = create_fact(store)
        blob = pack_embedding([1.0, 2.0])
        store.set_embedding(entry.id, blob)
        assert backend.get_embedding(entry.id) == blob
        store.set_embedding(entry.id, None)
        assert backend.get_embedding(entry.id) is None

    def test_list_filters_facade(self, store: MemoryStore, fake_clock: Any) -> None:
        e1 = create_fact(store, content="one", tags=["alpha"], session_id="s1")
        fake_clock.advance(hours=1)
        t1 = fake_clock.current
        e2 = store.create(
            content="two",
            type=MemoryType.DECISION,
            source=src("agent-2"),
            base_importance=0.9,
            pinned=True,
        )
        fake_clock.advance(hours=1)
        e3 = create_fact(store, content="three")
        store.archive(e3.id, ArchiveReason.EVICTED)

        assert [e.id for e in store.list()] == [e2.id, e1.id]
        assert [e.id for e in store.list(types=[MemoryType.DECISION])] == [e2.id]
        assert [e.id for e in store.list(tags=["alpha"])] == [e1.id]
        assert [e.id for e in store.list(agent_id="agent-2")] == [e2.id]
        assert [e.id for e in store.list(session_id="s1")] == [e1.id]
        assert [e.id for e in store.list(since=t1)] == [e2.id]
        assert [e.id for e in store.list(created_before=t1)] == [e1.id]
        assert [e.id for e in store.list(pinned=True)] == [e2.id]
        assert [e.id for e in store.list(min_base_importance=0.8)] == [e2.id]
        included = store.list(include_archived=True)
        assert [e.id for e in included] == [e3.id, e2.id, e1.id]
        assert [e.id for e in store.list(archive_reasons=[ArchiveReason.EVICTED])] == [e3.id]
        assert store.count() == 2
        assert store.count(include_archived=True) == 3

    def test_accessed_before_filter(self, store: MemoryStore, fake_clock: Any) -> None:
        stale = create_fact(store, content="stale")
        fake_clock.advance(hours=5)
        fresh = create_fact(store, content="fresh")
        cutoff = fake_clock.current
        fake_clock.advance(hours=1)
        store.get_by_id(fresh.id)  # touch fresh past the cutoff
        result = store.list(accessed_before=cutoff)
        assert [e.id for e in result] == [stale.id]

    def test_scan_keys_and_list_keys(self, store: MemoryStore) -> None:
        create_fact(store, content="a", key="decision:db")
        create_fact(store, content="b", key="decision:api")
        create_fact(store, content="c", key="fact:x")
        create_fact(store, content="d")  # unkeyed
        scanned = store.scan_keys("decision:")
        assert [e.key for e in scanned] == ["decision:api", "decision:db"]
        assert store.list_keys() == ["decision:api", "decision:db", "fact:x"]
        assert store.list_keys("fact:") == ["fact:x"]

    def test_exists(self, store: MemoryStore) -> None:
        create_fact(store, key="k")
        assert store.exists("k") is True
        assert store.exists("nope") is False

    def test_count_updated_before_facade(self, store: MemoryStore, fake_clock: Any) -> None:
        """updated_before flows through the filter facade (count path) — the
        cutoff purge_archived relies on for older_than."""
        stale = create_fact(store, content="stale")
        fake_clock.advance(hours=2)
        cutoff = fake_clock.current
        fake_clock.advance(hours=1)
        fresh = create_fact(store, content="fresh")
        assert store.count(updated_before=cutoff) == 1
        store.update(stale.id, content="bumped past the cutoff")
        assert store.count(updated_before=cutoff) == 0
        assert store.count() == 2
        assert fresh.updated_at > cutoff


# ---------------------------------------------------------------------------
# Identity (ID minting)
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_ids_are_32_char_lowercase_hex_and_unique(self, store: MemoryStore) -> None:
        ids = {create_fact(store, content=f"c{i}").id for i in range(20)}
        assert len(ids) == 20
        for entry_id in ids:
            assert len(entry_id) == 32
            assert all(c in "0123456789abcdef" for c in entry_id)

    def test_ids_pass_leaf_name_whitelist(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        assert validate_leaf_name(entry.id) == entry.id  # export forward-compat


# ---------------------------------------------------------------------------
# SQLite smoke subset (same behavior on the real backend)
# ---------------------------------------------------------------------------


class TestSQLiteSmoke:
    @pytest.fixture
    def sqlite_store(self, tmp_path: Path, fake_clock: Any) -> Any:
        backend = SQLiteBackend(tmp_path / "t.db")
        yield MemoryStore(backend, clock=fake_clock)
        backend.close()

    def test_supersede_round_trip_on_sqlite(self, sqlite_store: MemoryStore) -> None:
        old = create_fact(sqlite_store, content="A", key="k")
        new = create_fact(sqlite_store, content="B", key="k")
        old_after = sqlite_store.get_by_id(old.id, touch=False)
        assert old_after is not None
        assert old_after.archive_reason is ArchiveReason.SUPERSEDED
        current = sqlite_store.get_by_key("k", touch=False)
        assert current is not None and current.id == new.id

    def test_intra_batch_supersede_on_sqlite(self, sqlite_store: MemoryStore) -> None:
        """DBE-L4: the intra-batch supersede path exercised against the real
        partial unique index, not just the InMemory emulation."""
        first, second = sqlite_store.batch_create(
            [
                {"content": "A", "type": MemoryType.FACT, "source": src(), "key": "k"},
                {"content": "B", "type": MemoryType.FACT, "source": src(), "key": "k"},
            ]
        )
        first_after = sqlite_store.get_by_id(first.id, touch=False)
        assert first_after is not None
        assert first_after.archived is True
        assert first_after.archive_reason is ArchiveReason.SUPERSEDED
        links = [r for r in second.relationships if r.relationship_type == "supersedes"]
        assert [r.target_id for r in links] == [first.id]
        current = sqlite_store.get_by_key("k", touch=False)
        assert current is not None and current.id == second.id

    def test_touch_and_purge_on_sqlite(self, sqlite_store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(sqlite_store, key="k")
        fake_clock.advance(hours=1)
        got = sqlite_store.get_by_id(entry.id)
        assert got is not None and got.access_count == 1
        sqlite_store.forget("k")
        assert sqlite_store.purge_archived() == 1
        assert sqlite_store.get_by_id(entry.id, touch=False) is None


# ---------------------------------------------------------------------------
# blueprint-memory amendments: rebase_importance + forget_by_id
# ---------------------------------------------------------------------------


class TestRebaseImportance:
    """The owner-visible D2 resolution: update(importance=) is an explicit
    REBASE — new base_importance AND a fresh decay anchor, in one update."""

    def test_rebase_sets_base_anchor_and_updated_at(
        self, store: MemoryStore, fake_clock: Any
    ) -> None:
        entry = create_fact(store, base_importance=0.8)
        fake_clock.advance(hours=100)
        now = fake_clock.current
        rebased = store.rebase_importance(entry.id, 0.6, now=now)
        assert rebased.base_importance == 0.6
        assert rebased.last_accessed_at == now
        assert rebased.updated_at == now
        assert rebased.created_at == entry.created_at
        # A rebase is NOT an access: access_count is untouched.
        assert rebased.access_count == entry.access_count

    def test_rebase_persists(self, store: MemoryStore, fake_clock: Any) -> None:
        entry = create_fact(store, base_importance=0.8)
        fake_clock.advance(hours=1)
        store.rebase_importance(entry.id, 0.3, now=fake_clock.current)
        reread = store.get_by_id(entry.id, touch=False)
        assert reread is not None
        assert reread.base_importance == 0.3
        assert reread.last_accessed_at == fake_clock.current

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_rebase_out_of_range_rejected(
        self, store: MemoryStore, fake_clock: Any, bad: float
    ) -> None:
        entry = create_fact(store)
        with pytest.raises(MemoryStoreError):
            store.rebase_importance(entry.id, bad, now=fake_clock.current)

    def test_rebase_missing_id(self, store: MemoryStore, fake_clock: Any) -> None:
        with pytest.raises(MemoryStoreError, match="missing"):
            store.rebase_importance("missing", 0.5, now=fake_clock.current)

    def test_rebase_naive_now_rejected(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        with pytest.raises(MemoryStoreError):
            store.rebase_importance(entry.id, 0.5, now=datetime(2026, 1, 1))

    def test_ordinary_update_still_importance_free(self, store: MemoryStore) -> None:
        # The blueprint-store invariant stands: no importance parameter on update().
        assert "importance" not in inspect.signature(store.update).parameters
        assert "base_importance" not in inspect.signature(store.update).parameters


class TestForgetById:
    """MCP A3: unkeyed entries must be forgettable by id."""

    def test_forget_by_id_soft(self, store: MemoryStore) -> None:
        entry = create_fact(store)  # unkeyed
        assert store.forget_by_id(entry.id) is True
        archived = store.get_by_id(entry.id, touch=False)
        assert archived is not None
        assert archived.archived is True
        assert archived.archive_reason is ArchiveReason.FORGOTTEN

    def test_forget_by_id_hard(self, store: MemoryStore, backend: InMemoryBackend) -> None:
        entry = create_fact(store, embedding=pack_embedding([1.0]))
        assert store.forget_by_id(entry.id, hard=True) is True
        assert store.get_by_id(entry.id, touch=False) is None
        assert list(backend.iter_embeddings(include_archived=True)) == []

    def test_forget_by_id_missing_returns_false(self, store: MemoryStore) -> None:
        assert store.forget_by_id("missing") is False

    def test_forget_by_id_already_archived_returns_false(self, store: MemoryStore) -> None:
        entry = create_fact(store)
        store.archive(entry.id, ArchiveReason.SUMMARIZED)
        assert store.forget_by_id(entry.id) is False
        # The recorded reason is never overwritten (purge-protection regression).
        after = store.get_by_id(entry.id, touch=False)
        assert after is not None
        assert after.archive_reason is ArchiveReason.SUMMARIZED

    def test_forget_by_id_runs_transactionally(self, fake_clock: Any) -> None:
        backend = _TxnAssertingBackend()
        store = MemoryStore(backend, clock=fake_clock)
        entry = create_fact(store)
        assert store.forget_by_id(entry.id) is True

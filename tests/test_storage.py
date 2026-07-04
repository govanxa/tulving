"""Tests for tulving.adapters.storage — written BEFORE implementation.

One parity suite parametrized over SQLiteBackend (tmp_path) and
InMemoryBackend — behavioral identity across the whole Protocol surface is
the point (blueprint-storage-backends). SQLite-only suites cover migrations,
pragmas, persistence, the synced-path warning, and the parameterized-SQL
source guard.
"""

import ast
import os
import re
import sqlite3
import struct
import threading
from collections.abc import Iterator
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

import tulving.adapters.storage as storage_module
from tulving.adapters.storage import (
    SCHEMA_VERSION,
    InMemoryBackend,
    SQLiteBackend,
    cloud_sync_risk,
    pack_embedding,
    unpack_embedding,
)
from tulving.exceptions import StorageError

TS = "2026-07-03T12:00:00+00:00"
TS_NORM = "2026-07-03T12:00:00.000000+00:00"
TS_LATER = "2026-07-03T13:00:00+00:00"


def ts(hour: int, minute: int = 0) -> str:
    """A distinct UTC ISO timestamp within the test day."""
    return f"2026-07-03T{hour:02d}:{minute:02d}:00+00:00"


def make_row(entry_id: str = "e1", **overrides: Any) -> dict[str, Any]:
    """A minimal valid MemoryEntry.to_dict()-shaped dict."""
    row: dict[str, Any] = {
        "id": entry_id,
        "content": "the sky is blue",
        "type": "fact",
        "source": {
            "agent_id": "agent-1",
            "step_id": None,
            "run_id": None,
            "workflow_name": None,
        },
        "key": None,
        "tags": [],
        "relationships": [],
        "session_id": None,
        "base_importance": 0.5,
        "created_at": TS,
        "updated_at": TS,
        "last_accessed_at": TS,
        "access_count": 0,
        "archived": False,
        "archive_reason": None,
        "source_entry_ids": [],
        "pinned": False,
    }
    row.update(overrides)
    return row


def full_fat_row() -> dict[str, Any]:
    """Every field populated (SUMMARY carrying source back-links)."""
    return make_row(
        "full-1",
        content="all fields set",
        type="summary",
        source={
            "agent_id": "agent-1",
            "step_id": "step-9",
            "run_id": "run-4",
            "workflow_name": "wf",
        },
        key="topic:summary",
        tags=["alpha", "beta"],
        relationships=[
            {"target_id": "other-1", "relationship_type": "supersedes", "metadata": None},
            {
                "target_id": "other-2",
                "relationship_type": "relates_to",
                "metadata": {"weight": 0.7},
            },
        ],
        session_id="sess-1",
        base_importance=0.9,
        created_at="2026-07-03T17:30:00+05:30",  # == 12:00 UTC; offset must normalize
        updated_at=TS,
        last_accessed_at=TS_LATER,
        access_count=3,
        source_entry_ids=["src-1", "src-2"],
        pinned=True,
    )


def make_session(session_id: str = "s1", **overrides: Any) -> dict[str, Any]:
    """A minimal valid session dict."""
    session: dict[str, Any] = {
        "id": session_id,
        "goal": "test goal",
        "agent_id": "agent-1",
        "started_at": TS,
        "last_activity_at": TS,
        "ended_at": None,
        "status": "active",
    }
    session.update(overrides)
    return session


@pytest.fixture(params=["sqlite", "memory"])
def backend(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[SQLiteBackend | InMemoryBackend]:
    """One fixture, both backends — every parity test runs against each."""
    b: SQLiteBackend | InMemoryBackend = (
        SQLiteBackend(tmp_path / "tulving.db") if request.param == "sqlite" else InMemoryBackend()
    )
    yield b
    b.close()


@pytest.fixture
def sqlite_backend(tmp_path: Path) -> Iterator[SQLiteBackend]:
    b = SQLiteBackend(tmp_path / "tulving.db")
    yield b
    b.close()


# ---------------------------------------------------------------------------
# Failure paths (parity)
# ---------------------------------------------------------------------------


class TestFailurePaths:
    """Constraint violations and misuse raise StorageError (D6) — never other names."""

    def test_duplicate_id_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.create(make_row("e1"))

    def test_second_active_key_holder_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        with pytest.raises(StorageError):
            backend.create(make_row("e2", key="k"))

    def test_unknown_type_value_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create(make_row("e1", type="bogus"))

    def test_unknown_archive_reason_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create(make_row("e1", archived=True, archive_reason="bogus"))

    def test_missing_required_field_rejected(self, backend: Any) -> None:
        row = make_row("e1")
        del row["content"]
        with pytest.raises(StorageError):
            backend.create(row)

    def test_missing_source_agent_id_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create(make_row("e1", source={"step_id": "s"}))

    def test_naive_timestamp_in_create_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create(make_row("e1", created_at="2026-07-03T12:00:00"))

    def test_garbage_timestamp_in_create_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create(make_row("e1", created_at="not-a-timestamp"))

    def test_naive_timestamp_in_update_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.update("e1", {"updated_at": "2026-07-03T12:00:00"})

    def test_unknown_filter_key_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.list({"bogus_filter": 1})

    def test_unknown_filter_key_rejected_in_count(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.count({"bogus_filter": 1})

    def test_unknown_type_in_types_filter_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.list({"types": ["bogus"]})

    def test_unknown_reason_in_archive_reasons_filter_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.list({"archive_reasons": ["bogus"]})

    def test_naive_temporal_filter_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.list({"since": "2026-07-03T12:00:00"})

    def test_update_unknown_field_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.update("e1", {"bogus_field": 1})

    def test_update_id_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.update("e1", {"id": "e2"})

    def test_update_importance_rejected(self, backend: Any) -> None:
        """'importance' is derived (D2) — never a persistable field."""
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.update("e1", {"importance": 0.9})

    def test_update_unknown_enum_values_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.update("e1", {"type": "bogus"})
        with pytest.raises(StorageError):
            backend.update("e1", {"archived": True, "archive_reason": "bogus"})

    def test_set_embedding_missing_id_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.set_embedding("missing", b"\x00\x00\x80?")

    def test_get_embedding_missing_id_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.get_embedding("missing")

    def test_duplicate_session_id_rejected(self, backend: Any) -> None:
        backend.create_session(make_session("s1"))
        with pytest.raises(StorageError):
            backend.create_session(make_session("s1"))

    def test_unknown_session_status_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create_session(make_session("s1", status="bogus"))

    def test_set_meta_schema_version_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.set_meta({"schema_version": 2})

    def test_set_meta_unknown_field_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.set_meta({"bogus": 1})

    def test_nested_transaction_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            with backend.transaction():
                with backend.transaction():
                    pass  # pragma: no cover

    def test_operations_after_close_rejected(self, backend: Any) -> None:
        backend.close()
        with pytest.raises(StorageError):
            backend.read("e1")
        with pytest.raises(StorageError):
            backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.get_meta()

    def test_close_is_idempotent(self, backend: Any) -> None:
        backend.close()
        backend.close()  # no exception

    def test_negative_limit_offset_rejected(self, backend: Any) -> None:
        """SQLite's LIMIT -1 means 'all rows' while Python slicing wraps —
        negative pagination must fail identically on both backends."""
        with pytest.raises(StorageError):
            backend.list({}, limit=-1)
        with pytest.raises(StorageError):
            backend.list({}, offset=-1)
        with pytest.raises(StorageError):
            backend.scan_key_prefix("x", limit=-1)

    def test_zero_limit_returns_empty(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        assert backend.list({}, limit=0) == []
        assert backend.scan_key_prefix("", limit=0) == []

    def test_non_json_tag_value_rejected(self, backend: Any) -> None:
        """Backend-level JSON discipline: a non-serializable tag payload is a
        StorageError, not a TypeError leaking from json.dumps."""
        with pytest.raises(StorageError):
            backend.create(make_row("e1", tags=[object()]))

    def test_non_json_relationship_metadata_rejected(self, backend: Any) -> None:
        rel = {"target_id": "t", "relationship_type": "relates_to", "metadata": {"x": object()}}
        with pytest.raises(StorageError):
            backend.create(make_row("e1", relationships=[rel]))

    def test_session_missing_required_field_rejected(self, backend: Any) -> None:
        for field in ("id", "agent_id", "started_at"):
            session = make_session("s1")
            del session[field]
            with pytest.raises(StorageError):
                backend.create_session(session)


# ---------------------------------------------------------------------------
# pack/unpack helpers
# ---------------------------------------------------------------------------


class TestEmbeddingCodec:
    def test_round_trip_float32_precision(self) -> None:
        vector = [0.1, -2.5, 3.14159, 0.0]
        blob = pack_embedding(vector)
        assert blob == struct.pack("<4f", *vector)
        restored = unpack_embedding(blob)
        assert restored == pytest.approx(vector, abs=1e-6)

    def test_dimension_is_len_over_four(self) -> None:
        assert len(pack_embedding([1.0] * 7)) // 4 == 7

    def test_pack_empty_rejected(self) -> None:
        with pytest.raises(StorageError):
            pack_embedding([])

    def test_pack_non_numeric_rejected(self) -> None:
        with pytest.raises(StorageError):
            pack_embedding(["x"])  # type: ignore[list-item]

    def test_unpack_empty_rejected(self) -> None:
        with pytest.raises(StorageError):
            unpack_embedding(b"")

    def test_unpack_bad_length_rejected(self) -> None:
        with pytest.raises(StorageError):
            unpack_embedding(b"\x00\x00\x00")


# ---------------------------------------------------------------------------
# Create / read round-trip (parity)
# ---------------------------------------------------------------------------


class TestCreateRead:
    def test_full_fat_round_trip(self, backend: Any) -> None:
        row = full_fat_row()
        backend.create(row)
        got = backend.read("full-1")
        assert got is not None
        for field in (
            "id",
            "content",
            "type",
            "source",
            "key",
            "tags",
            "relationships",
            "session_id",
            "base_importance",
            "access_count",
            "archived",
            "archive_reason",
            "source_entry_ids",
            "pinned",
        ):
            assert got[field] == row[field], field
        # Timestamps: same instant, normalized to +00:00 fixed-width form.
        for field in ("created_at", "updated_at", "last_accessed_at"):
            assert datetime.fromisoformat(got[field]) == datetime.fromisoformat(row[field])
            assert got[field].endswith("+00:00")
            assert len(got[field]) == len(TS_NORM)

    def test_offset_input_normalizes_to_utc(self, backend: Any) -> None:
        backend.create(make_row("e1", created_at="2026-07-03T17:30:00+05:30"))
        got = backend.read("e1")
        assert got["created_at"] == "2026-07-03T12:00:00.000000+00:00"

    def test_id_persisted_verbatim(self, backend: Any) -> None:
        backend.create(make_row("weird-ID_42"))
        assert backend.read("weird-ID_42")["id"] == "weird-ID_42"

    def test_read_missing_returns_none(self, backend: Any) -> None:
        assert backend.read("missing") is None

    def test_read_includes_archived_rows(self, backend: Any) -> None:
        backend.create(make_row("e1", archived=True, archive_reason="forgotten"))
        got = backend.read("e1")
        assert got is not None
        assert got["archived"] is True
        assert got["archive_reason"] == "forgotten"

    def test_read_has_no_embedding_or_importance_key(self, backend: Any) -> None:
        backend.create(make_row("e1"), embedding=b"\x00\x00\x80?")
        got = backend.read("e1")
        assert "embedding" not in got
        assert "importance" not in got

    def test_returned_dict_is_isolated_from_storage(self, backend: Any) -> None:
        backend.create(make_row("e1", tags=["a"]))
        got = backend.read("e1")
        got["tags"].append("mutated")
        got["source"]["agent_id"] = "mutated"
        fresh = backend.read("e1")
        assert fresh["tags"] == ["a"]
        assert fresh["source"]["agent_id"] == "agent-1"

    def test_input_dict_is_isolated_from_storage(self, backend: Any) -> None:
        row = make_row("e1", tags=["a"])
        backend.create(row)
        row["tags"].append("mutated")
        row["content"] = "mutated"
        assert backend.read("e1")["tags"] == ["a"]
        assert backend.read("e1")["content"] == "the sky is blue"

    def test_create_with_embedding_stores_blob(self, backend: Any) -> None:
        blob = pack_embedding([1.0, 2.0])
        backend.create(make_row("e1"), embedding=blob)
        assert backend.get_embedding("e1") == blob

    def test_failed_create_leaves_no_partial_state(self, backend: Any) -> None:
        """Atomicity of create across row + junction: a failed create leaves
        no junction rows discoverable through the tags filter."""
        backend.create(make_row("e1", key="k", tags=["old"]))
        with pytest.raises(StorageError):
            backend.create(make_row("e2", key="k", tags=["newtag"]))
        assert backend.list({"tags": ["newtag"], "include_archived": True}) == []
        assert backend.read("e2") is None


# ---------------------------------------------------------------------------
# Partial unique index / active-key uniqueness (parity — audit regression)
# ---------------------------------------------------------------------------


class TestActiveKeyUniqueness:
    def test_archiving_frees_the_key(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        with pytest.raises(StorageError):
            backend.create(make_row("e2", key="k"))
        backend.update(
            "e1", {"archived": True, "archive_reason": "superseded", "updated_at": TS_LATER}
        )
        backend.create(make_row("e3", key="k"))  # succeeds now
        got = backend.get_by_key("k")
        assert got["id"] == "e3"

    def test_archived_rows_and_one_active_coexist_on_same_key(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        backend.update("e1", {"archived": True, "archive_reason": "superseded"})
        backend.create(make_row("e2", key="k"))
        backend.update("e2", {"archived": True, "archive_reason": "superseded"})
        backend.create(make_row("e3", key="k"))
        rows = backend.list({"include_archived": True})
        assert sorted(r["id"] for r in rows) == ["e1", "e2", "e3"]
        assert all(r["key"] == "k" for r in rows)

    def test_unarchive_into_held_key_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        backend.update("e1", {"archived": True, "archive_reason": "forgotten"})
        backend.create(make_row("e2", key="k"))
        with pytest.raises(StorageError):
            backend.update("e1", {"archived": False, "archive_reason": None})

    def test_unkeyed_rows_never_collide(self, backend: Any) -> None:
        backend.create(make_row("e1", key=None))
        backend.create(make_row("e2", key=None))
        assert backend.count({}) == 2


# ---------------------------------------------------------------------------
# Key retrieval (parity)
# ---------------------------------------------------------------------------


class TestKeyRetrieval:
    def seed(self, backend: Any) -> None:
        backend.create(make_row("e1", key="decision:db"))
        backend.create(make_row("e2", key="decision:api"))
        backend.create(make_row("e3", key="decisions_old:x"))
        backend.create(make_row("e4", key="fact:one"))
        backend.create(make_row("e5", key=None))

    def test_get_by_key_active_only(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        assert backend.get_by_key("k")["id"] == "e1"
        backend.update("e1", {"archived": True, "archive_reason": "forgotten"})
        assert backend.get_by_key("k") is None

    def test_key_exists(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        assert backend.key_exists("k") is True
        assert backend.key_exists("nope") is False
        backend.update("e1", {"archived": True, "archive_reason": "forgotten"})
        assert backend.key_exists("k") is False

    def test_list_keys_sorted_active_only(self, backend: Any) -> None:
        self.seed(backend)
        backend.update("e4", {"archived": True, "archive_reason": "forgotten"})
        assert backend.list_keys() == ["decision:api", "decision:db", "decisions_old:x"]
        assert backend.list_keys("decision:") == ["decision:api", "decision:db"]

    def test_scan_key_prefix_true_prefixes_only(self, backend: Any) -> None:
        self.seed(backend)
        rows = backend.scan_key_prefix("decision:")
        assert [r["key"] for r in rows] == ["decision:api", "decision:db"]

    def test_scan_key_prefix_orders_and_limits(self, backend: Any) -> None:
        self.seed(backend)
        rows = backend.scan_key_prefix("decision:", limit=1)
        assert [r["key"] for r in rows] == ["decision:api"]

    def test_scan_key_prefix_skips_archived_and_unkeyed(self, backend: Any) -> None:
        self.seed(backend)
        backend.update("e1", {"archived": True, "archive_reason": "forgotten"})
        rows = backend.scan_key_prefix("")
        assert [r["key"] for r in rows] == ["decision:api", "decisions_old:x", "fact:one"]

    def test_scan_key_prefix_literal_wildcards(self, backend: Any) -> None:
        backend.create(make_row("p1", key="pct%x"))
        backend.create(make_row("p2", key="pctYx"))
        backend.create(make_row("u1", key="under_a"))
        backend.create(make_row("u2", key="underXa"))
        backend.create(make_row("b1", key="back\\x"))
        assert [r["key"] for r in backend.scan_key_prefix("pct%")] == ["pct%x"]
        assert [r["key"] for r in backend.scan_key_prefix("under_")] == ["under_a"]
        assert [r["key"] for r in backend.scan_key_prefix("back\\")] == ["back\\x"]

    def test_scan_key_prefix_case_sensitive(self, backend: Any) -> None:
        backend.create(make_row("c1", key="Case:a"))
        backend.create(make_row("c2", key="case:b"))
        assert [r["key"] for r in backend.scan_key_prefix("case:")] == ["case:b"]
        assert backend.list_keys("Case:") == ["Case:a"]


# ---------------------------------------------------------------------------
# list / count filter contract (parity)
# ---------------------------------------------------------------------------


class TestListFilters:
    def seed(self, backend: Any) -> None:
        backend.create(
            make_row(
                "e1",
                type="fact",
                tags=["alpha"],
                key="fact:1",
                created_at=ts(1),
                updated_at=ts(1),
                last_accessed_at=ts(1),
            )
        )
        backend.create(
            make_row(
                "e2",
                type="decision",
                tags=["alpha", "beta"],
                base_importance=0.9,
                pinned=True,
                session_id="sess-1",
                created_at=ts(2),
                updated_at=ts(2),
                last_accessed_at=ts(2),
            )
        )
        backend.create(
            make_row(
                "e3",
                type="plan",
                source={
                    "agent_id": "agent-2",
                    "step_id": None,
                    "run_id": None,
                    "workflow_name": None,
                },
                created_at=ts(3),
                updated_at=ts(3),
                last_accessed_at=ts(3),
            )
        )
        backend.create(
            make_row(
                "e4",
                type="fact",
                archived=True,
                archive_reason="superseded",
                created_at=ts(4),
                updated_at=ts(4),
                last_accessed_at=ts(4),
            )
        )
        backend.create(
            make_row(
                "e5",
                type="fact",
                archived=True,
                archive_reason="summarized",
                created_at=ts(5),
                updated_at=ts(5),
                last_accessed_at=ts(5),
            )
        )

    def test_default_excludes_archived(self, backend: Any) -> None:
        self.seed(backend)
        ids = [r["id"] for r in backend.list({})]
        assert ids == ["e3", "e2", "e1"]  # created_at DESC

    def test_include_archived(self, backend: Any) -> None:
        self.seed(backend)
        ids = [r["id"] for r in backend.list({"include_archived": True})]
        assert ids == ["e5", "e4", "e3", "e2", "e1"]

    def test_types_filter(self, backend: Any) -> None:
        self.seed(backend)
        ids = [r["id"] for r in backend.list({"types": ["decision", "plan"]})]
        assert ids == ["e3", "e2"]

    def test_tags_any_of(self, backend: Any) -> None:
        self.seed(backend)
        ids = [r["id"] for r in backend.list({"tags": ["beta", "nope"]})]
        assert ids == ["e2"]
        ids = [r["id"] for r in backend.list({"tags": ["alpha"]})]
        assert ids == ["e2", "e1"]

    def test_agent_id_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"agent_id": "agent-2"})] == ["e3"]

    def test_session_id_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"session_id": "sess-1"})] == ["e2"]

    def test_key_prefix_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"key_prefix": "fact:"})] == ["e1"]

    def test_pinned_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"pinned": True})] == ["e2"]
        assert [r["id"] for r in backend.list({"pinned": False})] == ["e3", "e1"]

    def test_min_base_importance(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"min_base_importance": 0.6})] == ["e2"]

    def test_archive_reasons_implies_archived_only(self, backend: Any) -> None:
        self.seed(backend)
        rows = backend.list({"archive_reasons": ["summarized"]})
        assert [r["id"] for r in rows] == ["e5"]
        rows = backend.list({"archive_reasons": ["superseded", "summarized"]})
        assert [r["id"] for r in rows] == ["e5", "e4"]

    def test_since_filter(self, backend: Any) -> None:
        self.seed(backend)
        ids = [r["id"] for r in backend.list({"since": ts(3), "include_archived": True})]
        assert ids == ["e5", "e4", "e3"]

    def test_created_before_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"created_before": ts(2)})] == ["e1"]

    def test_updated_before_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"updated_before": ts(2)})] == ["e1"]

    def test_accessed_before_filter(self, backend: Any) -> None:
        self.seed(backend)
        assert [r["id"] for r in backend.list({"accessed_before": ts(3)})] == ["e2", "e1"]

    def test_temporal_filters_compare_across_offsets(self, backend: Any) -> None:
        backend.create(make_row("e1", created_at="2026-07-03T12:00:00+00:00"))
        backend.create(make_row("e2", created_at="2026-07-03T18:00:00+05:30"))  # 12:30 UTC
        ids = [r["id"] for r in backend.list({"since": "2026-07-03T12:15:00+00:00"})]
        assert ids == ["e2"]

    def test_combined_filters(self, backend: Any) -> None:
        self.seed(backend)
        rows = backend.list({"types": ["fact", "decision"], "tags": ["alpha"], "pinned": True})
        assert [r["id"] for r in rows] == ["e2"]

    def test_ordering_tiebreak_id_desc(self, backend: Any) -> None:
        backend.create(make_row("a", created_at=TS))
        backend.create(make_row("b", created_at=TS))
        backend.create(make_row("c", created_at=TS))
        assert [r["id"] for r in backend.list({})] == ["c", "b", "a"]

    def test_limit_offset_pagination(self, backend: Any) -> None:
        self.seed(backend)
        page1 = backend.list({"include_archived": True}, limit=2, offset=0)
        page2 = backend.list({"include_archived": True}, limit=2, offset=2)
        page3 = backend.list({"include_archived": True}, limit=2, offset=4)
        assert [r["id"] for r in page1] == ["e5", "e4"]
        assert [r["id"] for r in page2] == ["e3", "e2"]
        assert [r["id"] for r in page3] == ["e1"]

    def test_count_agrees_with_list(self, backend: Any) -> None:
        self.seed(backend)
        for filters in (
            {},
            {"include_archived": True},
            {"types": ["fact"]},
            {"tags": ["alpha"]},
            {"archive_reasons": ["summarized"]},
        ):
            assert backend.count(filters) == len(backend.list(dict(filters), limit=1000))

    def test_empty_types_filter_matches_nothing(self, backend: Any) -> None:
        """Empty IN-lists must mean 'match nothing' identically on both
        backends (SQLite generates a 1 = 0 clause; slicing must not diverge)."""
        self.seed(backend)
        assert backend.list({"types": []}) == []
        assert backend.count({"types": []}) == 0

    def test_empty_tags_filter_matches_nothing(self, backend: Any) -> None:
        self.seed(backend)
        assert backend.list({"tags": []}) == []
        assert backend.count({"tags": []}) == 0

    def test_empty_archive_reasons_filter_matches_nothing(self, backend: Any) -> None:
        self.seed(backend)
        assert backend.list({"archive_reasons": []}) == []
        assert backend.count({"archive_reasons": []}) == 0


# ---------------------------------------------------------------------------
# touch (parity — the batched access primitive, D3)
# ---------------------------------------------------------------------------


class TestTouch:
    def test_touch_batches_and_skips_archived(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        backend.create(make_row("e2"))
        backend.create(make_row("e3", archived=True, archive_reason="forgotten"))
        touched = backend.touch(["e1", "e2", "e3"], TS_LATER)
        assert touched == 2
        for entry_id in ("e1", "e2"):
            got = backend.read(entry_id)
            assert got["access_count"] == 1
            assert datetime.fromisoformat(got["last_accessed_at"]) == datetime.fromisoformat(
                TS_LATER
            )
        archived = backend.read("e3")
        assert archived["access_count"] == 0
        assert datetime.fromisoformat(archived["last_accessed_at"]) == datetime.fromisoformat(TS)

    def test_touch_does_not_bump_updated_at(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        backend.touch(["e1"], TS_LATER)
        got = backend.read("e1")
        assert datetime.fromisoformat(got["updated_at"]) == datetime.fromisoformat(TS)

    def test_touch_increments_repeatedly(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        backend.touch(["e1"], TS_LATER)
        backend.touch(["e1"], TS_LATER)
        assert backend.read("e1")["access_count"] == 2

    def test_touch_naive_timestamp_rejected(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        with pytest.raises(StorageError):
            backend.touch(["e1"], "2026-07-03T13:00:00")

    def test_touch_empty_and_missing_ids(self, backend: Any) -> None:
        assert backend.touch([], TS_LATER) == 0
        assert backend.touch(["missing"], TS_LATER) == 0


# ---------------------------------------------------------------------------
# update (parity)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_merge_updates_only_given_fields(self, backend: Any) -> None:
        backend.create(make_row("e1", tags=["a"]))
        got = backend.update("e1", {"content": "new content", "updated_at": TS_LATER})
        assert got["content"] == "new content"
        assert got["tags"] == ["a"]
        assert datetime.fromisoformat(got["updated_at"]) == datetime.fromisoformat(TS_LATER)
        assert got == backend.read("e1")

    def test_update_missing_id_returns_none(self, backend: Any) -> None:
        assert backend.update("missing", {"content": "x"}) is None

    def test_tags_update_rewrites_junction(self, backend: Any) -> None:
        backend.create(make_row("e1", tags=["old"]))
        backend.update("e1", {"tags": ["new"]})
        assert backend.list({"tags": ["old"]}) == []
        assert [r["id"] for r in backend.list({"tags": ["new"]})] == ["e1"]
        assert backend.read("e1")["tags"] == ["new"]

    def test_source_update_replaces_all_four_columns(self, backend: Any) -> None:
        backend.create(
            make_row(
                "e1",
                source={
                    "agent_id": "agent-1",
                    "step_id": "step-1",
                    "run_id": "run-1",
                    "workflow_name": "wf",
                },
            )
        )
        backend.update("e1", {"source": {"agent_id": "agent-2"}})
        assert backend.read("e1")["source"] == {
            "agent_id": "agent-2",
            "step_id": None,
            "run_id": None,
            "workflow_name": None,
        }

    def test_archive_via_update(self, backend: Any) -> None:
        backend.create(make_row("e1", key="k"))
        backend.update("e1", {"archived": True, "archive_reason": "superseded"})
        got = backend.read("e1")
        assert got["archived"] is True
        assert got["archive_reason"] == "superseded"
        assert backend.get_by_key("k") is None

    def test_failed_update_changes_nothing(self, backend: Any) -> None:
        backend.create(make_row("e1", tags=["old"], content="original"))
        with pytest.raises(StorageError):
            backend.update("e1", {"tags": ["new"], "bogus_field": 1})
        got = backend.read("e1")
        assert got["tags"] == ["old"]
        assert got["content"] == "original"
        assert [r["id"] for r in backend.list({"tags": ["old"]})] == ["e1"]


# ---------------------------------------------------------------------------
# delete / delete_many (parity)
# ---------------------------------------------------------------------------


class TestDelete:
    def test_hard_delete_removes_row_junction_and_embedding(self, backend: Any) -> None:
        backend.create(make_row("e1", tags=["t"]), embedding=pack_embedding([1.0]))
        assert backend.delete("e1") is True
        assert backend.read("e1") is None
        assert backend.list({"tags": ["t"], "include_archived": True}) == []
        with pytest.raises(StorageError):
            backend.get_embedding("e1")
        assert list(backend.iter_embeddings(include_archived=True)) == []

    def test_delete_missing_returns_false(self, backend: Any) -> None:
        assert backend.delete("missing") is False

    def test_delete_many_counts_actual_deletions(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        backend.create(make_row("e2"))
        assert backend.delete_many(["e1", "e2", "missing"]) == 2
        assert backend.count({"include_archived": True}) == 0

    def test_delete_many_empty(self, backend: Any) -> None:
        assert backend.delete_many([]) == 0


class TestSQLiteCascade:
    def test_junction_rows_cascade_on_delete(self, sqlite_backend: SQLiteBackend) -> None:
        sqlite_backend.create(make_row("e1", tags=["t1", "t2"]))
        sqlite_backend.delete("e1")
        with closing(sqlite3.connect(sqlite_backend.db_path)) as conn:
            count = conn.execute("SELECT COUNT(*) FROM memory_tags").fetchone()[0]
        assert count == 0


class TestCorruptDatabase:
    """SQLite-only: rows damaged out-of-band surface as StorageError (D6)."""

    def _raw_execute(self, db_path: Path, sql: str) -> None:
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(sql)

    def test_malformed_json_column_is_corrupt_row(self, sqlite_backend: SQLiteBackend) -> None:
        sqlite_backend.create(make_row("e1"))
        self._raw_execute(
            sqlite_backend.db_path, "UPDATE memories SET tags = 'not-json' WHERE id = 'e1'"
        )
        with pytest.raises(StorageError, match="corrupt row"):
            sqlite_backend.read("e1")

    def test_missing_meta_row_is_corrupt_database(self, sqlite_backend: SQLiteBackend) -> None:
        self._raw_execute(sqlite_backend.db_path, "DELETE FROM meta WHERE id = 1")
        with pytest.raises(StorageError, match="meta row is missing"):
            sqlite_backend.get_meta()


# ---------------------------------------------------------------------------
# Embeddings (parity — ADR-015 source of truth)
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_set_get_round_trip_exact_bytes(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        blob = pack_embedding([0.25, -1.0, 3.5])
        backend.set_embedding("e1", blob)
        assert backend.get_embedding("e1") == blob

    def test_none_clears(self, backend: Any) -> None:
        backend.create(make_row("e1"), embedding=pack_embedding([1.0]))
        backend.set_embedding("e1", None)
        assert backend.get_embedding("e1") is None

    def test_get_embedding_none_when_never_set(self, backend: Any) -> None:
        backend.create(make_row("e1"))
        assert backend.get_embedding("e1") is None

    def test_iter_embeddings_id_asc_skips_null(self, backend: Any) -> None:
        backend.create(make_row("b"), embedding=b"bb")
        backend.create(make_row("a"), embedding=b"aa")
        backend.create(make_row("c"))  # no embedding
        assert list(backend.iter_embeddings()) == [("a", b"aa"), ("b", b"bb")]

    def test_iter_embeddings_excludes_archived_unless_flagged(self, backend: Any) -> None:
        backend.create(make_row("a"), embedding=b"aa")
        backend.create(make_row("b", archived=True, archive_reason="superseded"), embedding=b"bb")
        assert list(backend.iter_embeddings()) == [("a", b"aa")]
        assert list(backend.iter_embeddings(include_archived=True)) == [
            ("a", b"aa"),
            ("b", b"bb"),
        ]


# ---------------------------------------------------------------------------
# Transactions (parity — D1 atomicity substrate)
# ---------------------------------------------------------------------------


class TestTransactions:
    def test_supersede_shape_commits_together(self, backend: Any) -> None:
        backend.create(make_row("old", key="k"))
        with backend.transaction():
            backend.update("old", {"archived": True, "archive_reason": "superseded"})
            backend.create(make_row("new", key="k"))
        assert backend.get_by_key("k")["id"] == "new"
        assert backend.read("old")["archived"] is True

    def test_exception_rolls_back_everything(self, backend: Any) -> None:
        backend.create(make_row("old", key="k"))
        with pytest.raises(RuntimeError):
            with backend.transaction():
                backend.update("old", {"archived": True, "archive_reason": "superseded"})
                backend.create(make_row("new", key="k"), embedding=b"nn")
                raise RuntimeError("boom")
        got = backend.get_by_key("k")
        assert got is not None and got["id"] == "old"
        assert got["archived"] is False
        assert backend.read("new") is None
        assert list(backend.iter_embeddings(include_archived=True)) == []

    def test_transaction_usable_again_after_rollback(self, backend: Any) -> None:
        with pytest.raises(RuntimeError):
            with backend.transaction():
                backend.create(make_row("x"))
                raise RuntimeError("boom")
        with backend.transaction():
            backend.create(make_row("y"))
        assert backend.read("y") is not None
        assert backend.read("x") is None

    def test_sessions_and_meta_roll_back_too(self, backend: Any) -> None:
        with pytest.raises(RuntimeError):
            with backend.transaction():
                backend.create_session(make_session("s1"))
                backend.set_meta({"embedding_model_id": "m1"})
                raise RuntimeError("boom")
        assert backend.get_session("s1") is None
        assert backend.get_meta()["embedding_model_id"] is None


# ---------------------------------------------------------------------------
# Chunked IN (...) lists (parity — SQLITE_MAX_VARIABLE_NUMBER hygiene)
# ---------------------------------------------------------------------------


class TestChunkedInLists:
    def test_touch_and_delete_many_beyond_one_chunk(self, backend: Any) -> None:
        """1200 ids > the 500-id chunk: results must be identical to one
        statement (single instant, one transaction, exact counts)."""
        n = 1200
        ids = [f"e{i:04d}" for i in range(n)]
        backend.create_batch([(make_row(entry_id), None) for entry_id in ids])
        assert backend.touch(ids, TS_LATER) == n
        for probe in (ids[0], ids[600], ids[-1]):  # one row per chunk
            got = backend.read(probe)
            assert got["access_count"] == 1
            assert datetime.fromisoformat(got["last_accessed_at"]) == datetime.fromisoformat(
                TS_LATER
            )
        assert backend.delete_many([*ids, "missing"]) == n
        assert backend.count({"include_archived": True}) == 0


# ---------------------------------------------------------------------------
# Commit failure (SQLite-only — D6 translation + rollback)
# ---------------------------------------------------------------------------


class TestCommitFailure:
    @staticmethod
    def _boom(conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    def test_transaction_commit_failure_translated_and_rolled_back(
        self, sqlite_backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqlite_backend, "_raw_commit", self._boom)
        with pytest.raises(StorageError, match="commit failed"):
            with sqlite_backend.transaction():
                sqlite_backend.create(make_row("e1"))
        monkeypatch.undo()
        assert sqlite_backend.read("e1") is None  # rolled back, no open txn left
        with sqlite_backend.transaction():
            sqlite_backend.create(make_row("e2"))  # backend fully usable again
        assert sqlite_backend.read("e2") is not None

    def test_atomic_commit_failure_translated_and_rolled_back(
        self, sqlite_backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sqlite_backend, "_raw_commit", self._boom)
        with pytest.raises(StorageError, match="commit failed"):
            sqlite_backend.create(make_row("e1"))
        monkeypatch.undo()
        assert sqlite_backend.read("e1") is None
        sqlite_backend.create(make_row("e1"))  # usable again
        assert sqlite_backend.read("e1") is not None


# ---------------------------------------------------------------------------
# Sessions (parity — D3 per-agent sessions)
# ---------------------------------------------------------------------------


class TestSessions:
    def test_create_get_round_trip(self, backend: Any) -> None:
        session = make_session("s1")
        backend.create_session(session)
        got = backend.get_session("s1")
        assert got is not None
        assert got["id"] == "s1"
        assert got["goal"] == "test goal"
        assert got["agent_id"] == "agent-1"
        assert got["status"] == "active"
        assert got["ended_at"] is None
        assert datetime.fromisoformat(got["started_at"]) == datetime.fromisoformat(TS)

    def test_get_missing_returns_none(self, backend: Any) -> None:
        assert backend.get_session("missing") is None

    def test_update_session_merges(self, backend: Any) -> None:
        backend.create_session(make_session("s1"))
        got = backend.update_session("s1", {"status": "ended", "ended_at": TS_LATER})
        assert got["status"] == "ended"
        assert datetime.fromisoformat(got["ended_at"]) == datetime.fromisoformat(TS_LATER)
        assert got["goal"] == "test goal"
        assert got == backend.get_session("s1")

    def test_update_session_missing_returns_none(self, backend: Any) -> None:
        assert backend.update_session("missing", {"status": "ended"}) is None

    def test_update_session_unknown_field_rejected(self, backend: Any) -> None:
        backend.create_session(make_session("s1"))
        with pytest.raises(StorageError):
            backend.update_session("s1", {"bogus": 1})

    def test_list_sessions_filters_by_agent_and_status(self, backend: Any) -> None:
        backend.create_session(make_session("s1", agent_id="agent-1", status="active"))
        backend.create_session(make_session("s2", agent_id="agent-1", status="ended"))
        backend.create_session(make_session("s3", agent_id="agent-2", status="active"))
        assert {s["id"] for s in backend.list_sessions(agent_id="agent-1")} == {"s1", "s2"}
        assert {s["id"] for s in backend.list_sessions(status="active")} == {"s1", "s3"}
        only = backend.list_sessions(agent_id="agent-1", status="active")
        assert [s["id"] for s in only] == ["s1"]

    def test_list_sessions_ordering(self, backend: Any) -> None:
        backend.create_session(make_session("s1", started_at=ts(1)))
        backend.create_session(make_session("s2", started_at=ts(3)))
        backend.create_session(make_session("s3", started_at=ts(2)))
        assert [s["id"] for s in backend.list_sessions()] == ["s2", "s3", "s1"]

    def test_list_sessions_ordering_tiebreak_id_desc(self, backend: Any) -> None:
        backend.create_session(make_session("sa", started_at=TS))
        backend.create_session(make_session("sb", started_at=TS))
        assert [s["id"] for s in backend.list_sessions()] == ["sb", "sa"]

    def test_naive_session_timestamp_rejected(self, backend: Any) -> None:
        with pytest.raises(StorageError):
            backend.create_session(make_session("s1", started_at="2026-07-03T12:00:00"))


# ---------------------------------------------------------------------------
# Meta (parity — D3 identity row)
# ---------------------------------------------------------------------------


class TestMeta:
    def test_fresh_meta_defaults(self, backend: Any) -> None:
        meta = backend.get_meta()
        assert meta == {
            "schema_version": SCHEMA_VERSION,
            "embedding_model_id": None,
            "embedding_dimension": None,
            "distance_metric": None,
        }

    def test_set_meta_round_trips_embedding_fields(self, backend: Any) -> None:
        backend.set_meta(
            {
                "embedding_model_id": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "distance_metric": "cosine",
            }
        )
        meta = backend.get_meta()
        assert meta["embedding_model_id"] == "all-MiniLM-L6-v2"
        assert meta["embedding_dimension"] == 384
        assert meta["distance_metric"] == "cosine"
        assert meta["schema_version"] == SCHEMA_VERSION

    def test_set_meta_partial(self, backend: Any) -> None:
        backend.set_meta({"embedding_model_id": "m1"})
        assert backend.get_meta()["embedding_model_id"] == "m1"
        assert backend.get_meta()["embedding_dimension"] is None


# ---------------------------------------------------------------------------
# Batch create (parity)
# ---------------------------------------------------------------------------


class TestCreateBatch:
    def test_batch_persists_rows_and_embeddings(self, backend: Any) -> None:
        items = [
            (make_row("e1", key="k1"), pack_embedding([1.0])),
            (make_row("e2", key="k2"), pack_embedding([2.0])),
            (make_row("e3"), None),
        ]
        backend.create_batch(items)
        assert backend.count({}) == 3
        assert backend.get_embedding("e1") == pack_embedding([1.0])
        assert backend.get_embedding("e2") == pack_embedding([2.0])
        assert backend.get_embedding("e3") is None
        assert [i for i, _ in backend.iter_embeddings()] == ["e1", "e2"]

    def test_batch_is_all_or_nothing(self, backend: Any) -> None:
        items = [
            (make_row("e1"), pack_embedding([1.0])),
            (make_row("e1"), None),  # duplicate id — must sink the whole batch
        ]
        with pytest.raises(StorageError):
            backend.create_batch(items)
        assert backend.read("e1") is None
        assert backend.count({"include_archived": True}) == 0
        assert list(backend.iter_embeddings(include_archived=True)) == []


# ---------------------------------------------------------------------------
# SQLite-only: migrations
# ---------------------------------------------------------------------------


class TestMigrations:
    def test_fresh_db_reaches_schema_version(self, tmp_path: Path) -> None:
        b = SQLiteBackend(tmp_path / "t.db")
        try:
            assert b.get_meta()["schema_version"] == SCHEMA_VERSION
        finally:
            b.close()
        with closing(sqlite3.connect(tmp_path / "t.db")) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    def test_reopen_is_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "t.db"
        for _ in range(3):
            b = SQLiteBackend(db)
            assert b.get_meta()["schema_version"] == SCHEMA_VERSION
            b.close()

    def test_newer_schema_refused(self, tmp_path: Path) -> None:
        db = tmp_path / "t.db"
        with closing(sqlite3.connect(db)) as conn, conn:
            conn.execute("PRAGMA user_version = 99")
        with pytest.raises(StorageError, match="99"):
            SQLiteBackend(db)

    def test_failed_migration_leaves_version_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exploding_migration(conn: sqlite3.Connection) -> None:
            conn.execute("CREATE TABLE half_done (id TEXT)")
            raise RuntimeError("boom mid-migration")

        monkeypatch.setattr(storage_module, "_MIGRATIONS", {1: exploding_migration})
        db = tmp_path / "t.db"
        with pytest.raises(StorageError):
            SQLiteBackend(db)
        with closing(sqlite3.connect(db)) as conn:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'half_done'"
            ).fetchall()
        assert tables == []

    def test_migration_storage_error_passes_through_unwrapped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A migration step that already raises StorageError keeps its message
        (never double-wrapped as 'migration to schema vN failed')."""

        def storage_error_migration(conn: sqlite3.Connection) -> None:
            raise StorageError("specific migration diagnosis")

        monkeypatch.setattr(storage_module, "_MIGRATIONS", {1: storage_error_migration})
        with pytest.raises(StorageError, match="specific migration diagnosis"):
            SQLiteBackend(tmp_path / "t.db")


# ---------------------------------------------------------------------------
# SQLite-only: pragmas & threading
# ---------------------------------------------------------------------------


class TestPragmas:
    def _pragmas(self, conn: sqlite3.Connection) -> dict[str, Any]:
        return {
            "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
            "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
            "busy_timeout": conn.execute("PRAGMA busy_timeout").fetchone()[0],
        }

    def test_fresh_connection_pragmas(self, sqlite_backend: SQLiteBackend) -> None:
        pragmas = self._pragmas(sqlite_backend._connection())
        assert pragmas["journal_mode"] == "wal"
        assert pragmas["foreign_keys"] == 1
        assert pragmas["busy_timeout"] == 5000

    def test_auto_vacuum_incremental_on_fresh_db(self, sqlite_backend: SQLiteBackend) -> None:
        """DBE-M1: auto_vacuum must land in the header BEFORE journal_mode
        initializes it — 2 == INCREMENTAL."""
        conn = sqlite_backend._connection()
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2

    def test_threadsafety_below_serialized_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(storage_module.sqlite3, "threadsafety", 1)
        with pytest.raises(StorageError, match="threadsafety"):
            SQLiteBackend(tmp_path / "t.db")

    def test_embedded_null_path_rejected_without_echo(self, tmp_path: Path) -> None:
        bad_path = str(tmp_path) + "\\bad\x00dir\\t.db"
        with pytest.raises(StorageError) as excinfo:
            SQLiteBackend(bad_path)
        assert "\x00" not in str(excinfo.value)  # raw path never echoed

    def test_db_path_pointing_at_directory_rejected(self, tmp_path: Path) -> None:
        """An unopenable database file (here: an existing directory) is a
        StorageError from the constructor, not a raw sqlite3 error."""
        directory = tmp_path / "iam_a_dir"
        directory.mkdir()
        with pytest.raises(StorageError, match="cannot open SQLite database"):
            SQLiteBackend(directory)

    def test_busy_timeout_configurable(self, tmp_path: Path) -> None:
        b = SQLiteBackend(tmp_path / "t.db", busy_timeout_ms=250)
        try:
            assert b._connection().execute("PRAGMA busy_timeout").fetchone()[0] == 250
        finally:
            b.close()

    def test_second_thread_gets_own_connection_same_pragmas(
        self, sqlite_backend: SQLiteBackend
    ) -> None:
        results: dict[str, Any] = {}

        def worker() -> None:
            conn = sqlite_backend._connection()
            results["conn"] = conn
            results.update(self._pragmas(conn))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        main_conn = sqlite_backend._connection()
        assert results["conn"] is not main_conn
        assert results["journal_mode"] == "wal"
        assert results["foreign_keys"] == 1
        assert results["busy_timeout"] == 5000

    def test_cross_thread_writes_visible(self, sqlite_backend: SQLiteBackend) -> None:
        def worker() -> None:
            sqlite_backend.create(make_row("from-thread"))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert sqlite_backend.read("from-thread") is not None

    def test_held_write_lock_raises_storage_error_after_retry(self, tmp_path: Path) -> None:
        """A competing writer holding BEGIN IMMEDIATE past the busy timeout
        surfaces as StorageError (D6), and the backend recovers afterwards."""
        db = tmp_path / "t.db"
        b = SQLiteBackend(db, busy_timeout_ms=1)
        blocker = sqlite3.connect(db)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            with pytest.raises(StorageError, match="write lock"):
                b.create(make_row("e1"))
            blocker.rollback()
            b.create(make_row("e1"))  # usable again once the lock is gone
            assert b.read("e1") is not None
        finally:
            blocker.close()
            b.close()


# ---------------------------------------------------------------------------
# SQLite-only: persistence across close/reopen
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_everything_survives_reopen(self, tmp_path: Path) -> None:
        db = tmp_path / "t.db"
        b = SQLiteBackend(db)
        b.create(make_row("e1", key="k", tags=["t"]), embedding=pack_embedding([1.0, 2.0]))
        b.create_session(make_session("s1"))
        b.set_meta(
            {"embedding_model_id": "m1", "embedding_dimension": 2, "distance_metric": "cosine"}
        )
        b.close()

        reopened = SQLiteBackend(db)
        try:
            got = reopened.read("e1")
            assert got is not None
            assert got["key"] == "k"
            assert got["tags"] == ["t"]
            assert [r["id"] for r in reopened.list({"tags": ["t"]})] == ["e1"]
            assert reopened.get_embedding("e1") == pack_embedding([1.0, 2.0])
            assert reopened.get_session("s1") is not None
            meta = reopened.get_meta()
            assert meta["embedding_model_id"] == "m1"
            assert meta["embedding_dimension"] == 2
            assert meta["distance_metric"] == "cosine"
        finally:
            reopened.close()

    def test_db_path_property(self, tmp_path: Path) -> None:
        db = tmp_path / "t.db"
        b = SQLiteBackend(db)
        try:
            assert b.db_path == db
        finally:
            b.close()

    def test_in_memory_db_path_is_none(self) -> None:
        b = InMemoryBackend()
        try:
            assert b.db_path is None
        finally:
            b.close()


# ---------------------------------------------------------------------------
# SQLite-only: synced-path warning (ADR-015 #4)
# ---------------------------------------------------------------------------


class TestCloudSyncRisk:
    def test_onedrive_segment_detected(self, tmp_path: Path) -> None:
        assert cloud_sync_risk(tmp_path / "OneDrive" / "db.sqlite") is not None

    def test_dropbox_segment_detected(self, tmp_path: Path) -> None:
        assert cloud_sync_risk(tmp_path / "Dropbox" / "db.sqlite") is not None

    def test_google_drive_segment_detected(self, tmp_path: Path) -> None:
        assert cloud_sync_risk(tmp_path / "Google Drive" / "db.sqlite") is not None

    def test_unc_path_detected(self) -> None:
        assert cloud_sync_risk("\\\\server\\share\\db.sqlite") is not None

    def test_onedrive_env_prefix_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synced_root = tmp_path / "synced"
        monkeypatch.setenv("OneDrive", str(synced_root))
        assert cloud_sync_risk(synced_root / "db.sqlite") is not None

    def test_clean_path_is_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            monkeypatch.delenv(var, raising=False)
        assert cloud_sync_risk(tmp_path / "plain" / "db.sqlite") is None

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_network_mapped_drive_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ctypes

        for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", lambda _drive: 4)
        assert cloud_sync_risk(tmp_path / "plain" / "db.sqlite") == "network-mapped drive"

    @pytest.mark.skipif(os.name != "nt", reason="GetDriveTypeW is Windows-only")
    def test_drive_type_query_failure_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import ctypes

        for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            monkeypatch.delenv(var, raising=False)

        def boom(_drive: str) -> int:
            raise OSError("drive query exploded")

        monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", boom)
        assert cloud_sync_risk(tmp_path / "plain" / "db.sqlite") is None

    def test_pathological_input_never_raises(self) -> None:
        # Contract: NEVER raises. (Relative junk resolves against the cwd, so
        # the verdict may legitimately be non-None on a synced dev machine.)
        for junk in ("bad\x00path\x00", "\\\\\x00" * 10, ""):
            result = cloud_sync_risk(junk)
            assert result is None or isinstance(result, str)

    def test_constructor_warns_once_and_still_works(self, tmp_path: Path) -> None:
        risky = tmp_path / "OneDrive" / "t.db"
        with pytest.warns(UserWarning, match="synced/network") as record:
            b = SQLiteBackend(risky)
        try:
            assert len([w for w in record if w.category is UserWarning]) == 1
            b.create(make_row("e1"))
            assert b.read("e1") is not None
        finally:
            b.close()

    def test_constructor_silent_on_clean_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
            monkeypatch.delenv(var, raising=False)
        import warnings as warnings_module

        with warnings_module.catch_warnings():
            warnings_module.simplefilter("error")
            b = SQLiteBackend(tmp_path / "t.db")
            b.close()


# ---------------------------------------------------------------------------
# SQLite-only: parameterized-SQL source guard
# ---------------------------------------------------------------------------


class TestParameterizationGuard:
    """Source-level guard: user data never lands in SQL text.

    Scans EVERY f-string in the module via the AST — any quote style, any
    argument position, including SQL assembled outside execute() calls (the
    ``self._execute(conn, sql)`` indirection and clause-fragment lists).
    SQL-shaped f-strings must interpolate only allowlisted expressions:
    PRAGMA int constants, generated ``?`` placeholder lists (values still
    bound), and the fixed-clause WHERE assembly (literal clauses, bound
    ``?`` params).
    """

    _ALLOWED_INTERPOLATIONS = frozenset(
        {
            "version",  # PRAGMA user_version = {int loop constant}
            "self._busy_timeout_ms",  # PRAGMA busy_timeout = {int() ctor arg}
            "placeholders",  # generated '?,?,...' lists — values still bound
            "_MEMORY_COLUMNS",  # fixed column-name module constant
            "' AND '.join(clauses)",  # fixed clause literals with bound '?'
        }
    )

    _SQL_SHAPE = re.compile(
        r"\b(SELECT|INSERT|UPDATE|DELETE|WHERE|LIKE|FROM|SET|PRAGMA|VALUES)\b"
        r"| = | >= | < | IN \(|\bIS NOT NULL\b"
    )

    def _sql_violations(self, code: str) -> list[str]:
        """All SQL-shaped f-strings in ``code`` interpolating non-allowlisted
        expressions."""
        violations: list[str] = []
        for node in ast.walk(ast.parse(code)):
            if not isinstance(node, ast.JoinedStr):
                continue
            literal = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if not self._SQL_SHAPE.search(literal):
                continue
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    expr = ast.unparse(part.value)
                    if expr not in self._ALLOWED_INTERPOLATIONS:
                        violations.append(f"{{{expr}}} in {literal!r}")
        return violations

    def test_no_interpolated_sql_anywhere(self) -> None:
        source = Path(storage_module.__file__).read_text(encoding="utf-8")
        assert ".format(" not in source
        assert not re.search(r"execute(?:many)?\([^)]*%[sd]", source)
        assert self._sql_violations(source) == []

    def test_guard_bites_on_injected_interpolation(self) -> None:
        """Self-test: the exact injection shape the guard must catch — a value
        interpolated into a clause fragment — IS flagged, while the documented
        exceptions are not."""
        injected = "clauses.append(f\"agent_id = '{filters['agent_id']}'\")"
        assert self._sql_violations(injected) != []
        injected_like = "sql = f\"SELECT * FROM memories WHERE key LIKE '{prefix}%'\""
        assert self._sql_violations(injected_like) != []
        pragma_ok = 'conn.execute(f"PRAGMA user_version = {version}")'
        assert self._sql_violations(pragma_ok) == []
        placeholders_ok = 'sql = f"DELETE FROM memories WHERE id IN ({placeholders})"'
        assert self._sql_violations(placeholders_ok) == []

"""Tests for tulving.entry — written BEFORE implementation."""

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from tulving.entry import MemoryEntry, Relationship, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MemoryType

AWARE = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)


def make_entry(**overrides: Any) -> MemoryEntry:
    """Minimal valid entry; override any field."""
    kwargs: dict[str, Any] = {
        "id": "abc123",
        "content": "the sky is blue",
        "type": MemoryType.FACT,
        "source": SourceInfo(agent_id="agent-1"),
    }
    kwargs.update(overrides)
    return MemoryEntry(**kwargs)


def full_fat_entry() -> MemoryEntry:
    """Every field populated (non-archived variant)."""
    return MemoryEntry(
        id="full-1",
        content="all fields set",
        type=MemoryType.SUMMARY,
        source=SourceInfo(
            agent_id="agent-1",
            step_id="step-9",
            run_id="run-4",
            workflow_name="wf",
        ),
        key="topic/summary",
        tags=["alpha", "beta"],
        relationships=[
            Relationship(target_id="other-1", relationship_type="supersedes"),
            Relationship(
                target_id="other-2",
                relationship_type="relates_to",
                metadata={"weight": 0.7},
            ),
        ],
        session_id="sess-1",
        base_importance=0.9,
        created_at=AWARE,
        updated_at=AWARE,
        last_accessed_at=AWARE + timedelta(hours=1),
        access_count=3,
        source_entry_ids=["src-1", "src-2"],
        pinned=True,
    )


class TestFailurePaths:
    """Invariant violations raise stdlib ValueError (D6: no Tulving exceptions here)."""

    def test_archived_without_reason_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(archived=True, archive_reason=None)

    def test_reason_without_archived_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(archived=False, archive_reason=ArchiveReason.EVICTED)

    def test_naive_created_at_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(created_at=datetime(2026, 7, 3))

    def test_naive_updated_at_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(updated_at=datetime(2026, 7, 3))

    def test_naive_last_accessed_at_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(last_accessed_at=datetime(2026, 7, 3))

    def test_base_importance_below_zero_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(base_importance=-0.1)

    def test_base_importance_above_one_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(base_importance=1.1)

    def test_empty_agent_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(source=SourceInfo(agent_id=""))

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(id="")

    def test_empty_content_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(content="")

    def test_negative_access_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(access_count=-1)

    def test_source_entry_ids_on_non_summary_rejected(self) -> None:
        with pytest.raises(ValueError):
            make_entry(type=MemoryType.FACT, source_entry_ids=["x"])

    def test_from_dict_empty_raises(self) -> None:
        with pytest.raises((ValueError, KeyError)):
            MemoryEntry.from_dict({})

    def test_from_dict_bogus_type_raises(self) -> None:
        data = make_entry().to_dict()
        data["type"] = "bogus"
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)

    def test_from_dict_naive_datetime_string_raises(self) -> None:
        data = make_entry().to_dict()
        data["created_at"] = "2026-07-03T12:00:00"  # no offset
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)

    def test_from_dict_naive_updated_at_string_raises(self) -> None:
        data = make_entry().to_dict()
        data["updated_at"] = "2026-07-03T12:00:00"  # no offset
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)

    def test_from_dict_naive_last_accessed_at_string_raises(self) -> None:
        data = make_entry().to_dict()
        data["last_accessed_at"] = "2026-07-03T12:00:00"  # no offset
        with pytest.raises(ValueError):
            MemoryEntry.from_dict(data)

    def test_source_info_from_dict_missing_agent_id_raises(self) -> None:
        """Documented contract: KeyError on missing agent_id (QA addition)."""
        with pytest.raises(KeyError):
            SourceInfo.from_dict({"step_id": "s"})

    def test_relationship_from_dict_missing_target_id_raises(self) -> None:
        with pytest.raises(KeyError):
            Relationship.from_dict({"relationship_type": "relates_to"})

    def test_relationship_from_dict_missing_relationship_type_raises(self) -> None:
        with pytest.raises(KeyError):
            Relationship.from_dict({"target_id": "t"})


class TestBoundaryConditions:
    def test_base_importance_zero_and_one_legal(self) -> None:
        assert make_entry(base_importance=0.0).base_importance == 0.0
        assert make_entry(base_importance=1.0).base_importance == 1.0

    def test_never_accessed_defaults_to_created_at(self) -> None:
        entry = make_entry(created_at=AWARE, updated_at=AWARE)
        assert entry.last_accessed_at == AWARE

    def test_from_dict_missing_last_accessed_at_normalizes_to_created_at(self) -> None:
        data = make_entry(created_at=AWARE, updated_at=AWARE).to_dict()
        del data["last_accessed_at"]
        restored = MemoryEntry.from_dict(data)
        assert restored.last_accessed_at == AWARE

    def test_from_dict_none_last_accessed_at_normalizes_to_created_at(self) -> None:
        data = make_entry(created_at=AWARE, updated_at=AWARE).to_dict()
        data["last_accessed_at"] = None
        restored = MemoryEntry.from_dict(data)
        assert restored.last_accessed_at == AWARE

    def test_summary_with_empty_source_entry_ids_is_legal(self) -> None:
        entry = make_entry(type=MemoryType.SUMMARY, source_entry_ids=[])
        assert entry.source_entry_ids == []

    def test_archived_with_reason_is_legal(self) -> None:
        entry = make_entry(archived=True, archive_reason=ArchiveReason.SUPERSEDED)
        assert entry.archive_reason is ArchiveReason.SUPERSEDED

    def test_from_dict_minimal_dict_applies_documented_defaults(self) -> None:
        """A storage-row mapping with only required keys restores with defaults
        (QA addition — exercises every .get() default in from_dict)."""
        minimal = {
            "id": "m-1",
            "content": "bare minimum",
            "type": "fact",
            "source": {"agent_id": "agent-1"},
            "created_at": AWARE.isoformat(),
            "updated_at": AWARE.isoformat(),
        }
        restored = MemoryEntry.from_dict(minimal)
        assert restored.key is None
        assert restored.tags == []
        assert restored.relationships == []
        assert restored.session_id is None
        assert restored.base_importance == 0.5
        assert restored.last_accessed_at == AWARE  # normalized to created_at
        assert restored.access_count == 0
        assert restored.archived is False
        assert restored.archive_reason is None
        assert restored.source_entry_ids == []
        assert restored.pinned is False
        assert restored.importance is None

    def test_utcnow_is_timezone_aware_utc(self) -> None:
        now = utcnow()
        assert now.tzinfo is not None
        assert now.utcoffset() == timedelta(0)


class TestBasicBehavior:
    def test_touch_bumps_count_and_sets_last_accessed(self) -> None:
        entry = make_entry(created_at=AWARE, updated_at=AWARE)
        now = AWARE + timedelta(hours=5)
        entry.touch(now)
        assert entry.access_count == 1
        assert entry.last_accessed_at == now
        assert entry.updated_at == AWARE
        assert entry.created_at == AWARE

    def test_two_touches_same_instant(self) -> None:
        entry = make_entry()
        now = AWARE + timedelta(hours=1)
        entry.touch(now)
        entry.touch(now)
        assert entry.access_count == 2
        assert entry.last_accessed_at == now

    def test_touch_defaults_to_utcnow(self) -> None:
        entry = make_entry()
        before = utcnow()
        entry.touch()
        assert entry.last_accessed_at is not None
        assert entry.last_accessed_at >= before
        assert entry.access_count == 1

    def test_replace_leaves_original_untouched(self) -> None:
        entry = make_entry()
        clone = dataclasses.replace(entry, pinned=True)
        assert clone.pinned is True
        assert entry.pinned is False

    def test_mutating_round_tripped_tags_does_not_affect_source(self) -> None:
        entry = make_entry(tags=["one"])
        restored = MemoryEntry.from_dict(entry.to_dict())
        restored.tags.append("two")
        assert entry.tags == ["one"]

    def test_mutating_round_tripped_metadata_does_not_affect_source(self) -> None:
        rel = Relationship(target_id="t", relationship_type="relates_to", metadata={"k": 1})
        entry = make_entry(relationships=[rel])
        restored = MemoryEntry.from_dict(entry.to_dict())
        restored_meta = restored.relationships[0].metadata
        assert restored_meta is not None
        restored_meta["k"] = 999
        assert entry.relationships[0].metadata == {"k": 1}

    def test_to_dict_metadata_does_not_alias_live_entry(self) -> None:
        rel = Relationship(target_id="t", relationship_type="relates_to", metadata={"k": 1})
        entry = make_entry(relationships=[rel])
        exported = entry.to_dict()
        exported["relationships"][0]["metadata"]["k"] = 999
        assert entry.relationships[0].metadata == {"k": 1}


class TestSerialization:
    """Round-trip is a mandatory audit regression: preserves all STORED fields."""

    def test_full_fat_round_trip(self) -> None:
        entry = full_fat_entry()
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored == entry

    def test_archived_variant_round_trip(self) -> None:
        entry = make_entry(archived=True, archive_reason=ArchiveReason.FORGOTTEN)
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored == entry
        assert restored.archive_reason is ArchiveReason.FORGOTTEN

    def test_importance_is_never_state(self) -> None:
        """D2: derived importance must not survive a round-trip."""
        entry = make_entry()
        entry.importance = 0.42
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored.importance is None

    def test_importance_excluded_from_equality(self) -> None:
        a = make_entry(created_at=AWARE, updated_at=AWARE)
        b = make_entry(created_at=AWARE, updated_at=AWARE)
        a.importance = 0.9
        b.importance = None
        assert a == b

    def test_to_dict_omits_importance_when_unpopulated(self) -> None:
        assert "importance" not in make_entry().to_dict()

    def test_to_dict_includes_importance_when_populated(self) -> None:
        entry = make_entry()
        entry.importance = 0.42
        assert entry.to_dict()["importance"] == 0.42

    def test_json_safe_and_pickle_free(self) -> None:
        entry = full_fat_entry()
        data = entry.to_dict()
        json.dumps(data)  # must not raise
        assert "embedding" not in data
        assert isinstance(data["created_at"], str)
        assert "+" in data["created_at"] or data["created_at"].endswith("Z")
        assert data["type"] == "summary"

        def only_plain(value: Any) -> bool:
            if isinstance(value, dict):
                return all(isinstance(k, str) and only_plain(v) for k, v in value.items())
            if isinstance(value, list):
                return all(only_plain(v) for v in value)
            return value is None or type(value) in (str, int, float, bool)

        assert only_plain(data)

    def test_forward_compat_unknown_keys_ignored(self) -> None:
        data = make_entry().to_dict()
        data["scope"] = "x"
        restored = MemoryEntry.from_dict(data)
        assert not hasattr(restored, "scope")

    def test_non_utc_timezone_round_trips_same_instant(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        created = datetime(2026, 7, 3, 17, 30, tzinfo=ist)
        entry = make_entry(created_at=created, updated_at=created)
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored.created_at == created

    def test_source_info_round_trip(self) -> None:
        source = SourceInfo(agent_id="a", step_id="s", run_id="r", workflow_name="w")
        assert SourceInfo.from_dict(source.to_dict()) == source

    def test_relationship_round_trip(self) -> None:
        rel = Relationship(target_id="t", relationship_type="supports", metadata={"k": 1})
        assert Relationship.from_dict(rel.to_dict()) == rel

    def test_relationship_round_trip_without_metadata(self) -> None:
        rel = Relationship(target_id="t", relationship_type="relates_to")
        assert Relationship.from_dict(rel.to_dict()) == rel


class TestSecurity:
    """to_safe_dict masks mechanically; the verdict is injected (D10 seam)."""

    def test_safe_dict_masks_content_when_sensitive(self) -> None:
        entry = make_entry(key="api_key", content="sk-live-value")
        data = entry.to_safe_dict(content_is_sensitive=True)
        assert data["content"] != "sk-live-value"
        assert "sk-live-value" not in json.dumps(data)
        json.dumps(data)  # still JSON-safe

    def test_safe_dict_placeholder_matches_security_constant(self) -> None:
        """entry.py may not import security; this test pins the two in sync."""
        from tulving.security import REDACTED

        data = make_entry().to_safe_dict(content_is_sensitive=True)
        assert data["content"] == REDACTED

    def test_safe_dict_leaves_other_fields_intact(self) -> None:
        entry = full_fat_entry()
        data = entry.to_safe_dict(content_is_sensitive=True)
        expected = entry.to_dict()
        expected.pop("content")
        data_content = data.pop("content")
        assert data == expected
        assert data_content != "all fields set"

    def test_safe_dict_verbatim_when_not_sensitive(self) -> None:
        entry = make_entry(content="plain text")
        data = entry.to_safe_dict(content_is_sensitive=False)
        assert data["content"] == "plain text"

    def test_safe_dict_does_no_pattern_matching_of_its_own(self) -> None:
        """An entry keyed 'monkey_facts' passed False stays verbatim: the mask
        decision is purely the parameter, never inferred from the key."""
        entry = make_entry(key="monkey_facts", content="password = hunter2secret")
        data = entry.to_safe_dict(content_is_sensitive=False)
        assert data["content"] == "password = hunter2secret"

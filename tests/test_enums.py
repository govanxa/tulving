"""Tests for tulving.enums — written BEFORE implementation."""

import json

import pytest

from tulving.enums import ArchiveReason, MatchType, MemoryType, SessionStatus


class TestFailurePaths:
    """Unknown values must fail loudly — storage translation relies on this."""

    def test_memory_type_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            MemoryType("nonsense")

    def test_match_type_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            MatchType("fuzzy")

    def test_archive_reason_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ArchiveReason("deleted")

    def test_session_status_unknown_value_raises(self) -> None:
        with pytest.raises(ValueError):
            SessionStatus("paused")

    def test_case_sensitive_lookup_fails(self) -> None:
        """Persisted values are lowercase; uppercase lookups must not resolve."""
        with pytest.raises(ValueError):
            MemoryType("FACT")


class TestBoundaryConditions:
    """Exact member inventory — no more, no fewer (persistence contract)."""

    def test_memory_type_exact_inventory(self) -> None:
        assert {m.value for m in MemoryType} == {
            "fact",
            "decision",
            "observation",
            "plan",
            "summary",
        }

    def test_match_type_exact_inventory(self) -> None:
        assert {m.value for m in MatchType} == {"semantic", "key", "temporal"}

    def test_archive_reason_exact_inventory(self) -> None:
        assert {m.value for m in ArchiveReason} == {
            "evicted",
            "summarized",
            "superseded",
            "forgotten",
            "abandoned",
        }

    def test_session_status_exact_inventory(self) -> None:
        assert {m.value for m in SessionStatus} == {"active", "ended", "abandoned"}


class TestBasicBehavior:
    """Persisted-value contract and StrEnum semantics."""

    def test_memory_type_member_values(self) -> None:
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.DECISION.value == "decision"
        assert MemoryType.OBSERVATION.value == "observation"
        assert MemoryType.PLAN.value == "plan"
        assert MemoryType.SUMMARY.value == "summary"

    def test_match_type_member_values(self) -> None:
        assert MatchType.SEMANTIC.value == "semantic"
        assert MatchType.KEY.value == "key"
        assert MatchType.TEMPORAL.value == "temporal"

    def test_archive_reason_member_values(self) -> None:
        assert ArchiveReason.EVICTED.value == "evicted"
        assert ArchiveReason.SUMMARIZED.value == "summarized"
        assert ArchiveReason.SUPERSEDED.value == "superseded"
        assert ArchiveReason.FORGOTTEN.value == "forgotten"
        assert ArchiveReason.ABANDONED.value == "abandoned"

    def test_session_status_member_values(self) -> None:
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.ENDED.value == "ended"
        assert SessionStatus.ABANDONED.value == "abandoned"

    def test_members_are_str_instances(self) -> None:
        assert isinstance(MemoryType.FACT, str)
        assert isinstance(MatchType.SEMANTIC, str)
        assert isinstance(ArchiveReason.EVICTED, str)
        assert isinstance(SessionStatus.ACTIVE, str)

    def test_str_yields_bare_value(self) -> None:
        assert str(MemoryType.FACT) == "fact"
        assert str(ArchiveReason.SUPERSEDED) == "superseded"

    def test_json_dumps_without_custom_encoder(self) -> None:
        assert json.dumps({"t": MemoryType.FACT}) == '{"t": "fact"}'

    def test_round_trip_identity(self) -> None:
        assert MemoryType("fact") is MemoryType.FACT
        assert MatchType("semantic") is MatchType.SEMANTIC
        assert ArchiveReason("superseded") is ArchiveReason.SUPERSEDED
        assert SessionStatus("active") is SessionStatus.ACTIVE


class TestSerialization:
    """Hashability for D2 half-life lookup and dict keying."""

    def test_memory_type_dict_keying_for_half_life(self) -> None:
        half_life: dict[MemoryType, float] = {
            MemoryType.FACT: 168.0,
            MemoryType.DECISION: float("inf"),
            MemoryType.OBSERVATION: 24.0,
            MemoryType.PLAN: 72.0,
            MemoryType.SUMMARY: 336.0,
        }
        assert len(half_life) == len(MemoryType)
        assert half_life[MemoryType("fact")] == 168.0

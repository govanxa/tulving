"""Tests for tulving.context.curator — written BEFORE implementation.

Pure-unit level per blueprint-curator: hand-rolled fakes for the
``RetrievalPort`` and ``ImportanceEvaluator`` protocols (no store, no
index, no hnswlib, no LLM), an injected ``FakeClock`` (no sleeps), and a
deterministic ``WordEstimator`` so budget arithmetic is exact.

Owns mandatory audit regression #4: oversize / tiny / empty budgets.
"""

import sys
import types
from datetime import datetime, timedelta

import pytest
from conftest import CLOCK_START, FakeClock

from tulving.context import curator as curator_mod
from tulving.context.curator import (
    REDACTED_CONTENT,
    ContextCurator,
    CuratedContext,
    HeuristicEstimator,
    TiktokenEstimator,
    resolve_estimator,
)
from tulving.context.decay import effective_importance as decay_effective_importance
from tulving.entry import MemoryEntry, SourceInfo
from tulving.enums import MemoryType
from tulving.exceptions import ConfigError
from tulving.security import REDACTED, compile_explicit_patterns, compile_key_patterns

STALE_TAG = "potentially_stale"

# Frame costs under WordEstimator (whitespace-split word counts).
QUERY_FRAME_WORDS = 12  # 8-word header + 4-word footer
ORIENT_FRAME_WORDS = 11  # 5-word header + 6-word footer


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class WordEstimator:
    """Deterministic estimator: token count == whitespace word count."""

    def estimate(self, text: str) -> int:
        return len(text.split())


class CharEstimator:
    """Deterministic estimator: token count == character count (exact budget
    control for character-level truncation math)."""

    def estimate(self, text: str) -> int:
        return len(text)


class FakeImportance:
    """Scripted ImportanceEvaluator; defaults to base_importance."""

    def __init__(self, values: dict[str, float] | None = None) -> None:
        self.values = values or {}

    def effective_importance(self, entry: MemoryEntry, now: datetime | None = None) -> float:
        return self.values.get(entry.id, entry.base_importance)


class DecayImportance:
    """Real decay formula (imported, never re-derived — D12) for loop tests."""

    def __init__(self, half_life_hours: dict[MemoryType, float]) -> None:
        self.half_life_hours = half_life_hours

    def effective_importance(self, entry: MemoryEntry, now: datetime | None = None) -> float:
        assert now is not None
        return decay_effective_importance(entry, now, self.half_life_hours)


class FakeRetrieval:
    """Dict-backed RetrievalPort: access-neutral reads, recorded writes."""

    def __init__(self) -> None:
        self.entries: dict[str, MemoryEntry] = {}
        self.semantic: list[tuple[str, float]] = []  # (entry_id, score), pre-ordered
        self.lookup_calls: list[str] = []
        self.semantic_calls: list[tuple[str, int]] = []
        self.recent_calls: list[int] = []
        self.list_by_calls: list[dict[str, object]] = []
        self.record_access_calls: list[tuple[list[str], datetime]] = []
        self.reflect_touches = False

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        self.entries[entry.id] = entry
        return entry

    # -- access-neutral reads (never touch) --

    def lookup_key(self, key: str) -> MemoryEntry | None:
        self.lookup_calls.append(key)
        for entry in self.entries.values():
            if entry.key == key:
                return entry
        return None

    def semantic_candidates(self, query: str, *, top_k: int) -> list[tuple[MemoryEntry, float]]:
        self.semantic_calls.append((query, top_k))
        return [(self.entries[i], score) for i, score in self.semantic[:top_k]]

    def recent_entries(self, *, limit: int) -> list[MemoryEntry]:
        self.recent_calls.append(limit)
        ordered = sorted(self.entries.values(), key=lambda e: (-e.created_at.timestamp(), e.id))
        return ordered[:limit]

    def list_by(
        self,
        *,
        types: list[MemoryType] | None = None,
        tags: list[str] | None = None,
        pinned_only: bool = False,
        limit: int,
    ) -> list[MemoryEntry]:
        self.list_by_calls.append(
            {"types": types, "tags": tags, "pinned_only": pinned_only, "limit": limit}
        )
        result = []
        for entry in sorted(self.entries.values(), key=lambda e: (-e.created_at.timestamp(), e.id)):
            if types is not None and entry.type not in types:
                continue
            if tags is not None and not set(tags) & set(entry.tags):
                continue
            if pinned_only and not entry.pinned:
                continue
            result.append(entry)
        return result[:limit]

    # -- THE single write --

    def record_access(self, entry_ids: list[str], *, now: datetime) -> None:
        self.record_access_calls.append((list(entry_ids), now))
        for entry_id in entry_ids:
            entry = self.entries[entry_id]
            if self.reflect_touches:
                entry.touch(now)
            if STALE_TAG in entry.tags:
                entry.tags = [t for t in entry.tags if t != STALE_TAG]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_entry(
    entry_id: str,
    content: str,
    *,
    key: str | None = None,
    type_: MemoryType = MemoryType.FACT,
    tags: list[str] | None = None,
    base_importance: float = 0.5,
    created_at: datetime | None = None,
    last_accessed_at: datetime | None = None,
    pinned: bool = False,
) -> MemoryEntry:
    created = created_at if created_at is not None else CLOCK_START - timedelta(hours=1)
    return MemoryEntry(
        id=entry_id,
        content=content,
        type=type_,
        source=SourceInfo(agent_id="agent-x"),
        key=key,
        tags=list(tags or []),
        base_importance=base_importance,
        created_at=created,
        updated_at=created,
        last_accessed_at=last_accessed_at,
        pinned=pinned,
    )


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def make_curator(
    retrieval: FakeRetrieval,
    importance: object | None = None,
    clock: FakeClock | None = None,
    **kwargs: object,
) -> ContextCurator:
    kwargs.setdefault("estimator", WordEstimator())
    kwargs.setdefault("token_safety_margin", 0.0)
    return ContextCurator(
        retrieval,  # type: ignore[arg-type]
        importance if importance is not None else FakeImportance(),  # type: ignore[arg-type]
        now_fn=clock if clock is not None else FakeClock(),
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def retrieval() -> FakeRetrieval:
    return FakeRetrieval()


# ---------------------------------------------------------------------------
# Failure paths — first, and most numerous
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_zero_budget_raises_config_error(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="token_budget"):
            curator.curate("q", token_budget=0)

    def test_negative_budget_raises_config_error(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="token_budget"):
            curator.curate("q", token_budget=-100)

    def test_bad_budget_in_orient_mode_raises(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="token_budget"):
            curator.curate("", token_budget=0, mode="orient")

    def test_unknown_mode_raises_config_error(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="mode"):
            curator.curate("q", mode="oreint")

    def test_validation_happens_before_any_port_call(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError):
            curator.curate("q", mode="oreint")
        with pytest.raises(ConfigError):
            curator.curate("q", token_budget=0)
        assert retrieval.lookup_calls == []
        assert retrieval.semantic_calls == []
        assert retrieval.recent_calls == []
        assert retrieval.list_by_calls == []

    def test_negative_weight_raises_config_error(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="weight"):
            curator.curate("q", recency_weight=-0.1)
        with pytest.raises(ConfigError, match="weight"):
            curator.curate("q", importance_weight=-1.0)
        with pytest.raises(ConfigError, match="weight"):
            curator.curate("q", relevance_weight=-0.5)

    def test_all_zero_weights_raise_config_error(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        with pytest.raises(ConfigError, match="weight"):
            curator.curate("q", recency_weight=0.0, importance_weight=0.0, relevance_weight=0.0)

    def test_constructor_rejects_bad_margin(self, retrieval: FakeRetrieval) -> None:
        with pytest.raises(ConfigError, match="token_safety_margin"):
            make_curator(retrieval, token_safety_margin=1.0)
        with pytest.raises(ConfigError, match="token_safety_margin"):
            make_curator(retrieval, token_safety_margin=-0.1)

    def test_constructor_rejects_bad_half_life(self, retrieval: FakeRetrieval) -> None:
        with pytest.raises(ConfigError, match="recency_half_life_hours"):
            make_curator(retrieval, recency_half_life_hours=0.0)
        with pytest.raises(ConfigError, match="recency_half_life_hours"):
            make_curator(retrieval, recency_half_life_hours=-1.0)

    def test_constructor_rejects_bad_limits(self, retrieval: FakeRetrieval) -> None:
        with pytest.raises(ConfigError, match="semantic_top_k"):
            make_curator(retrieval, semantic_top_k=0)
        with pytest.raises(ConfigError, match="recent_limit"):
            make_curator(retrieval, recent_limit=0)


# ---------------------------------------------------------------------------
# Mandatory audit regression #4: oversize / tiny / empty budgets
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_oversize_single_entry_truncated_to_fit(self, retrieval: FakeRetrieval) -> None:
        entry = retrieval.add(make_entry("e1", words(100)))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=30)
        assert [e.id for e in result.entries] == ["e1"]
        assert "truncated by curator:" in result.content
        assert "tokens elided]" in result.content
        assert "w99" not in result.content  # tail actually elided
        assert result.token_count <= 30
        # Truncated inclusion IS inclusion: the entry is access-touched.
        assert retrieval.record_access_calls == [(["e1"], CLOCK_START)]
        assert result.entries[0] is entry

    def test_oversize_rule_only_when_nothing_fits(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e1", words(50), base_importance=0.9))
        retrieval.add(make_entry("e2", words(3, "b"), base_importance=0.8))
        retrieval.add(make_entry("e3", words(2, "c"), base_importance=0.7))
        curator = make_curator(retrieval)
        # content_budget = 30 - 12 = 18: e1 (block 55) skipped whole,
        # e2 (block 8) and e3 (block 7) fit.
        result = curator.curate("", token_budget=30)
        assert [e.id for e in result.entries] == ["e2", "e3"]
        assert "truncated by curator" not in result.content
        assert "w0" not in result.content

    def test_truncation_impossible_returns_empty(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e1", words(100)))
        curator = make_curator(retrieval)
        # content_budget = 18 - 12 = 6 < header(5) + marker(7): nothing can ship.
        result = curator.curate("", token_budget=18)
        assert result.content == ""
        assert result.entries == []
        assert result.token_count == 0
        assert result.budget_remaining == 18
        assert retrieval.record_access_calls == []

    def test_tiny_budget_graceful_empty_no_side_effects(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e1", "small entry"))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=5)  # frame alone needs 12
        assert isinstance(result, CuratedContext)
        assert result.content == ""
        assert result.entries == []
        assert result.token_count == 0
        assert result.budget_remaining == 5
        assert result.sources_consulted == 1  # evaluated, never touched
        assert retrieval.record_access_calls == []

    def test_tiny_budget_orient_graceful_empty(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("d1", "big decision", type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=3, mode="orient")
        assert result.content == ""
        assert result.entries == []
        assert result.budget_remaining == 3
        assert retrieval.record_access_calls == []

    def test_tiny_budget_orient_reports_sources_consulted(self, retrieval: FakeRetrieval) -> None:
        # MINOR-2: orient gathers BEFORE the content_budget<1 early-return, so a
        # tiny-budget empty result reports the real evaluated count (parity with
        # query mode's test_tiny_budget_graceful_empty).
        retrieval.add(make_entry("d1", "big decision", type_=MemoryType.DECISION))
        retrieval.add(make_entry("d2", "another decision", type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=3, mode="orient")
        assert result.content == ""
        assert result.entries == []
        assert result.sources_consulted == 2
        assert retrieval.record_access_calls == []

    def test_empty_store_header_only(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        result = curator.curate("anything", token_budget=100)
        assert result.entries == []
        assert result.sources_consulted == 0
        assert "0 entries" in result.content
        assert result.token_count == WordEstimator().estimate(result.content)
        assert result.budget_remaining == 100 - result.token_count
        assert retrieval.record_access_calls == []

    def test_empty_store_orient_never_raises(self, retrieval: FakeRetrieval) -> None:
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=100, mode="orient")
        assert result.entries == []
        assert result.sources_consulted == 0
        assert "=== Project Memory (orient) ===" in result.content
        assert retrieval.record_access_calls == []

    def test_oversize_budget_returns_everything_without_padding(
        self, retrieval: FakeRetrieval
    ) -> None:
        retrieval.add(make_entry("e1", words(3, "a")))
        retrieval.add(make_entry("e2", words(3, "b")))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=1_000_000)
        assert {e.id for e in result.entries} == {"e1", "e2"}
        assert "truncated by curator" not in result.content
        # No padding: token_count is the real content size, budget mostly left.
        assert result.token_count == WordEstimator().estimate(result.content)
        assert result.budget_remaining == 1_000_000 - result.token_count


# ---------------------------------------------------------------------------
# Selection, scoring, deterministic ordering
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_per_source_relevance_ordering(self, retrieval: FakeRetrieval) -> None:
        created = CLOCK_START - timedelta(hours=1)
        kv = retrieval.add(make_entry("e_kv", "kv content", key="the-key", created_at=created))
        sem = retrieval.add(make_entry("e_sem", "sem content", created_at=created))
        rec = retrieval.add(make_entry("e_rec", "rec content", created_at=created))
        retrieval.semantic = [("e_sem", 0.6)]
        curator = make_curator(retrieval)
        result = curator.curate("the-key", token_budget=500)
        assert [e.id for e in result.entries] == [kv.id, sem.id, rec.id]

    def test_dedup_keeps_max_relevance(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e_kv", "shared entry", key="the-key"))
        retrieval.add(make_entry("e_other", "other entry"))
        retrieval.semantic = [("e_kv", 0.6), ("e_other", 0.4)]
        curator = make_curator(retrieval)
        result = curator.curate("the-key", token_budget=500)
        assert [e.id for e in result.entries].count("e_kv") == 1
        # KV(1.0) beat the 0.6 semantic score for the same entry.
        assert result.entries[0].id == "e_kv"
        assert result.sources_consulted == 2

    def test_greedy_skip_if_doesnt_fit(self, retrieval: FakeRetrieval) -> None:
        created = CLOCK_START - timedelta(hours=1)
        retrieval.add(make_entry("e1", words(3, "a"), base_importance=0.9, created_at=created))
        retrieval.add(make_entry("e2", words(20, "b"), base_importance=0.8, created_at=created))
        retrieval.add(make_entry("e3", words(2, "c"), base_importance=0.7, created_at=created))
        curator = make_curator(retrieval)
        # content_budget = 28 - 12 = 16: e1 (8) fits, e2 (25) skipped, e3 (7) fits.
        result = curator.curate("", token_budget=28)
        assert [e.id for e in result.entries] == ["e1", "e3"]
        assert retrieval.record_access_calls == [(["e1", "e3"], CLOCK_START)]
        assert retrieval.entries["e2"].access_count == 0

    def test_score_tie_falls_to_newest_then_id(self, retrieval: FakeRetrieval) -> None:
        anchor = CLOCK_START - timedelta(hours=5)
        retrieval.add(
            make_entry("b", "older by creation", created_at=anchor, last_accessed_at=anchor)
        )
        retrieval.add(
            make_entry(
                "a",
                "newer by creation",
                created_at=anchor + timedelta(hours=1),
                last_accessed_at=anchor,
            )
        )
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500)
        # Equal scores (same importance/recency/relevance): newest created first.
        assert [e.id for e in result.entries] == ["a", "b"]

    def test_score_tie_same_created_id_ascending(self, retrieval: FakeRetrieval) -> None:
        created = CLOCK_START - timedelta(hours=2)
        retrieval.add(make_entry("z2", "same instant z", created_at=created))
        retrieval.add(make_entry("a1", "same instant a", created_at=created))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500)
        assert [e.id for e in result.entries] == ["a1", "z2"]

    def test_repeat_curate_is_byte_identical(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e1", "alpha beta gamma"))
        retrieval.add(make_entry("e2", "delta epsilon"))
        curator = make_curator(retrieval)
        first = curator.curate("", token_budget=200)
        second = curator.curate("", token_budget=200)
        assert first.content == second.content
        assert first.token_count == second.token_count

    def test_exclude_wins_over_include(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e_both", "tagged both ways", tags=["x", "y"]))
        retrieval.add(make_entry("e_keep", "tagged keep", tags=["x"]))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500, include_tags=["x"], exclude_tags=["y"])
        assert [e.id for e in result.entries] == ["e_keep"]
        # Filtered entries still count as evaluated, never touched.
        assert result.sources_consulted == 2
        assert retrieval.record_access_calls == [(["e_keep"], CLOCK_START)]
        assert retrieval.entries["e_both"].access_count == 0

    def test_include_tags_drops_untagged_candidate(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e_tagged", "has the tag", tags=["keep"]))
        retrieval.add(make_entry("e_bare", "no tags at all"))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500, include_tags=["keep"])
        assert [e.id for e in result.entries] == ["e_tagged"]
        assert result.sources_consulted == 2

    def test_include_types_filter(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("f1", "a fact"))
        retrieval.add(make_entry("d1", "a decision", type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500, include_types=[MemoryType.DECISION])
        assert [e.id for e in result.entries] == ["d1"]
        assert result.sources_consulted == 2

    def test_empty_query_recency_only_no_kv_or_semantic_calls(
        self, retrieval: FakeRetrieval
    ) -> None:
        retrieval.add(make_entry("e1", "recent thing"))
        curator = make_curator(retrieval)
        result = curator.curate("   ", token_budget=200)
        assert retrieval.lookup_calls == []
        assert retrieval.semantic_calls == []
        assert retrieval.recent_calls == [10]  # default recent_limit
        assert [e.id for e in result.entries] == ["e1"]

    def test_relevance_weight_alone_ranks_kv_first(self, retrieval: FakeRetrieval) -> None:
        old = CLOCK_START - timedelta(hours=500)
        retrieval.add(
            make_entry(
                "e_kv",
                "old unimportant keyed",
                key="q",
                base_importance=0.05,
                created_at=old,
                last_accessed_at=old,
            )
        )
        retrieval.add(
            make_entry(
                "e_new",
                "fresh important",
                base_importance=0.9,
                created_at=CLOCK_START,
                last_accessed_at=CLOCK_START,
            )
        )
        curator = make_curator(retrieval)
        result = curator.curate(
            "q",
            token_budget=500,
            relevance_weight=1.0,
            importance_weight=0.0,
            recency_weight=0.0,
        )
        assert result.entries[0].id == "e_kv"

    def test_recency_weight_alone_ranks_newest_accessed_first(
        self, retrieval: FakeRetrieval
    ) -> None:
        old = CLOCK_START - timedelta(hours=500)
        retrieval.add(make_entry("e_old", "stale", created_at=old, last_accessed_at=old))
        retrieval.add(make_entry("e_new", "fresh", created_at=old, last_accessed_at=CLOCK_START))
        curator = make_curator(retrieval)
        result = curator.curate(
            "",
            token_budget=500,
            relevance_weight=0.0,
            importance_weight=0.0,
            recency_weight=1.0,
        )
        assert result.entries[0].id == "e_new"

    def test_weights_are_normalized_by_sum(self, retrieval: FakeRetrieval) -> None:
        created = CLOCK_START - timedelta(hours=1)
        retrieval.add(make_entry("e_kv", "keyed", key="q", created_at=created))
        retrieval.add(make_entry("e_imp", "important", base_importance=0.95, created_at=created))
        retrieval.add(make_entry("e_plain", "plain", base_importance=0.2, created_at=created))
        curator = make_curator(retrieval)
        scaled = curator.curate(
            "q",
            token_budget=500,
            relevance_weight=2.0,
            importance_weight=1.0,
            recency_weight=1.0,
        )
        normalized = curator.curate(
            "q",
            token_budget=500,
            relevance_weight=0.5,
            importance_weight=0.25,
            recency_weight=0.25,
        )
        assert [e.id for e in scaled.entries] == [e.id for e in normalized.entries]


# ---------------------------------------------------------------------------
# Access side effects (D3) & the feedback loop
# ---------------------------------------------------------------------------


class TestAccessSideEffects:
    def test_inclusion_touches_evaluation_does_not(self, retrieval: FakeRetrieval) -> None:
        created = CLOCK_START - timedelta(hours=1)
        retrieval.add(make_entry("e1", words(3, "a"), base_importance=0.9, created_at=created))
        retrieval.add(make_entry("e2", words(30, "b"), base_importance=0.8, created_at=created))
        retrieval.add(make_entry("e3", "excluded by tag", tags=["drop"], created_at=created))
        clock = FakeClock()
        curator = make_curator(retrieval, clock=clock)
        result = curator.curate("", token_budget=25, exclude_tags=["drop"])
        assert [e.id for e in result.entries] == ["e1"]
        assert retrieval.record_access_calls == [(["e1"], clock.current)]
        # Evaluated-but-excluded and filtered entries: zero touches.
        assert retrieval.entries["e2"].access_count == 0
        assert retrieval.entries["e3"].access_count == 0

    def test_feedback_loop_never_exceeds_base_importance(self, retrieval: FakeRetrieval) -> None:
        half_lives = {t: 336.0 for t in MemoryType}
        half_lives[MemoryType.DECISION] = float("inf")
        entry = retrieval.add(
            make_entry("e1", "repeatedly curated", base_importance=0.4, created_at=CLOCK_START)
        )
        retrieval.reflect_touches = True
        clock = FakeClock()
        curator = make_curator(retrieval, importance=DecayImportance(half_lives), clock=clock)
        for _ in range(5):
            clock.advance(hours=100)
            result = curator.curate("", token_budget=300)
            assert [e.id for e in result.entries] == ["e1"]
            assert entry.importance is not None
            assert entry.importance <= 0.4 + 1e-9

    def test_derived_importance_slot_populated_for_included_only(
        self, retrieval: FakeRetrieval
    ) -> None:
        created = CLOCK_START - timedelta(hours=1)
        included = retrieval.add(
            make_entry("e1", words(2, "a"), base_importance=0.9, created_at=created)
        )
        excluded = retrieval.add(
            make_entry("e2", words(40, "b"), base_importance=0.1, created_at=created)
        )
        curator = make_curator(retrieval, importance=FakeImportance({"e1": 0.7}))
        result = curator.curate("", token_budget=20)
        assert [e.id for e in result.entries] == ["e1"]
        assert included.importance == 0.7
        assert excluded.importance is None

    def test_stale_annotation_rendered_and_cleared_via_record_access(
        self, retrieval: FakeRetrieval
    ) -> None:
        created = CLOCK_START - timedelta(days=50)
        entry = retrieval.add(
            make_entry("e1", "maybe outdated", tags=[STALE_TAG], created_at=created)
        )
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=200)
        assert "potentially stale" in result.content
        assert "stored 50 days ago" in result.content
        assert retrieval.record_access_calls == [(["e1"], CLOCK_START)]
        assert STALE_TAG not in entry.tags  # the fake models the store-side clear


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    def test_heuristic_estimator_values(self) -> None:
        estimator = HeuristicEstimator()
        assert estimator.estimate("") == 0
        assert estimator.estimate("abc") == 1
        assert estimator.estimate(400 * "a") == 100

    def test_injected_estimator_wins(self, retrieval: FakeRetrieval) -> None:
        injected = WordEstimator()
        assert resolve_estimator(injected) is injected
        curator = make_curator(retrieval, estimator=injected)
        # No margin applied to raw estimates — delegation is verbatim.
        assert curator.estimate_tokens("one two three") == 3

    def test_tiktoken_import_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(curator_mod, "_default_estimator", None)
        monkeypatch.setitem(sys.modules, "tiktoken", None)  # forces ImportError
        estimator = resolve_estimator()
        assert isinstance(estimator, HeuristicEstimator)

    def test_tiktoken_success_path_and_memoization(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeEncoding:
            def encode(self, text: str) -> list[str]:
                return text.split()

        requested: list[str] = []

        def get_encoding(name: str) -> _FakeEncoding:
            requested.append(name)
            return _FakeEncoding()

        monkeypatch.setattr(curator_mod, "_default_estimator", None)
        monkeypatch.setitem(
            sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=get_encoding)
        )
        estimator = resolve_estimator()
        assert isinstance(estimator, TiktokenEstimator)
        assert estimator.estimate("a b c") == 3
        assert requested == ["cl100k_base"]
        assert resolve_estimator() is estimator  # memoized: probe ran once

    def test_get_encoding_failure_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def get_encoding(name: str) -> object:
            raise RuntimeError("BPE fetch failed")

        monkeypatch.setattr(curator_mod, "_default_estimator", None)
        monkeypatch.setitem(
            sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=get_encoding)
        )
        assert isinstance(resolve_estimator(), HeuristicEstimator)

    def test_safety_margin_bounds_filled_tokens(self, retrieval: FakeRetrieval) -> None:
        for i in range(30):
            retrieval.add(make_entry(f"e{i:02d}", f"c{i}"))
        # recent_limit raised above the default 10 so the BUDGET (not the
        # recency window) is the binding constraint this test measures.
        with_margin = make_curator(retrieval, token_safety_margin=0.15, recent_limit=40).curate(
            "", token_budget=100
        )
        assert with_margin.token_count <= 85  # floor(100 * (1 - 0.15))
        no_margin = make_curator(retrieval, token_safety_margin=0.0, recent_limit=40).curate(
            "", token_budget=100
        )
        assert no_margin.token_count <= 100
        assert len(no_margin.entries) > len(with_margin.entries)


# ---------------------------------------------------------------------------
# Redaction (security requirement #1) — curated context is an egress surface
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_content_secret_shapes_redacted(self, retrieval: FakeRetrieval) -> None:
        secret = "sk-" + "a" * 24
        retrieval.add(make_entry("e1", f"the token is {secret} beware"))
        retrieval.add(make_entry("e2", "login with password = hunter22 please"))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500)
        assert secret not in result.content
        assert "hunter22" not in result.content
        assert REDACTED in result.content

    def test_sensitive_key_masks_content(self, retrieval: FakeRetrieval) -> None:
        """Default-sensitive key + secret-SHAPED content -> whole-masked
        (v0.2 softening, D-v02-7: the key alone is no longer enough).

        The secret is embedded inside an unrelated sentence and the
        SURROUNDING sentence is asserted absent too — surgical redact_text
        alone would strip only the secret substring and leave the sentence
        text intact, so this can only pass via true whole-body masking (test
        review MAJOR: distinguishes whole-mask from content-shape-only
        scrubbing, which a byte-identical secret-only fixture cannot)."""
        secret = "sk-" + "x" * 24
        content = f"the rotation doc says {secret} expires monthly"
        retrieval.add(make_entry("e1", content, key="api_key:acme"))
        retrieval.add(make_entry("e2", "bananas are curved", key="monkey_facts"))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500)
        assert secret not in result.content
        assert "rotation doc says" not in result.content
        assert "expires monthly" not in result.content
        assert REDACTED_CONTENT in result.content
        assert "bananas are curved" in result.content  # word-boundary negative control

    def test_whitespace_broken_secret_still_masks(self, retrieval: FakeRetrieval) -> None:
        """SEV-001 end-to-end repro: a real secret broken by ONE embedded
        whitespace char (newline/tab/double-space) must not leak through
        curate() under a sensitive key."""
        variants = {
            "\n": "deploy secret is aB3dE6fG9hJ2kL5m\nN8pQ1rS4tU7vW0xY3zA6",
            "\t": "deploy secret is aB3dE6fG9hJ2kL5m\tN8pQ1rS4tU7vW0xY3zA6",
            "  ": "deploy secret is aB3dE6fG9hJ2kL5m  N8pQ1rS4tU7vW0xY3zA6",
        }
        for label, content in variants.items():
            r = FakeRetrieval()
            r.add(make_entry("e1", content, key="secret:deploy"))
            result = make_curator(r).curate("", token_budget=500)
            assert "aB3dE6fG9hJ2kL5m" not in result.content, label
            assert REDACTED_CONTENT in result.content, label

    def test_sensitive_key_prose_passes_through(self, retrieval: FakeRetrieval) -> None:
        """The reported bug fix: a default-sensitive-named key holding plain
        prose is no longer whole-masked (v0.2 softening, D-v02-7)."""
        retrieval.add(make_entry("e1", "auth token TTL is 15 min", key="fact:auth-ttl"))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500)
        assert "auth token TTL is 15 min" in result.content
        assert REDACTED_CONTENT not in result.content

    def test_custom_key_patterns_mask_camel_case(self, retrieval: FakeRetrieval) -> None:
        """A user-declared pattern threaded as ``explicit_key_patterns`` masks
        unconditionally (D-v02-7 Q3) even for camelCase defaults miss."""
        retrieval.add(make_entry("e1", "jwt-value-here", key="authToken"))
        patterns = compile_key_patterns(["authToken"])
        explicit = compile_explicit_patterns(["authToken"])
        curator = make_curator(retrieval, key_patterns=patterns, explicit_key_patterns=explicit)
        result = curator.curate("", token_budget=500)
        assert "jwt-value-here" not in result.content
        # Defaults miss camelCase: without the extras the content leaks through.
        default_result = make_curator(retrieval).curate("", token_budget=500)
        assert "jwt-value-here" in default_result.content

    def test_user_declared_key_overrides_default_overlap(self, retrieval: FakeRetrieval) -> None:
        """D-v02-7 Q3 (mandatory): a user-declared pattern masks unconditionally
        even when the key ALSO matches a built-in default, and even for prose."""
        retrieval.add(make_entry("e1", "rotate quarterly, no value here", key="auth-prod-token"))
        explicit = compile_explicit_patterns(["auth-prod"])
        curator = make_curator(retrieval, explicit_key_patterns=explicit)
        result = curator.curate("", token_budget=500)
        assert "rotate quarterly" not in result.content
        assert REDACTED_CONTENT in result.content

    def test_truncation_cannot_resurrect_a_secret(self, retrieval: FakeRetrieval) -> None:
        content = words(30) + " password = hunter22 " + words(30, "z")
        retrieval.add(make_entry("e1", content))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=40)  # forces the oversize truncation
        assert len(result.entries) == 1
        assert "truncated by curator" in result.content
        assert "hunter22" not in result.content

    def test_redacted_placeholder_pinned_to_security_module(self) -> None:
        assert REDACTED_CONTENT == REDACTED

    def test_budget_measured_post_redaction(self, retrieval: FakeRetrieval) -> None:
        secret = "sk-" + "a" * 400  # ~100 heuristic tokens raw, ~3 once redacted
        retrieval.add(make_entry("e1", f"key {secret}", base_importance=0.9))
        retrieval.add(make_entry("e2", "tiny extra entry", base_importance=0.8))
        curator = make_curator(retrieval, estimator=HeuristicEstimator(), token_safety_margin=0.0)
        result = curator.curate("", token_budget=60)
        # Both fit ONLY if the giant secret was redacted before estimation.
        assert {e.id for e in result.entries} == {"e1", "e2"}
        assert secret not in result.content


# ---------------------------------------------------------------------------
# Orient mode
# ---------------------------------------------------------------------------


class TestOrientMode:
    def test_section_order_with_abundant_budget(self, retrieval: FakeRetrieval) -> None:
        base = CLOCK_START - timedelta(hours=2)
        retrieval.add(
            make_entry(
                "sm1",
                "session marker one",
                type_=MemoryType.SUMMARY,
                tags=["session_marker"],
                created_at=base,
            )
        )
        retrieval.add(make_entry("d1", "decision one", type_=MemoryType.DECISION, created_at=base))
        retrieval.add(make_entry("s1", "summary one", type_=MemoryType.SUMMARY, created_at=base))
        retrieval.add(make_entry("p1", "pinned one", pinned=True, created_at=base))
        goal = retrieval.add(make_entry("g1", "goal relevant", created_at=base))
        retrieval.semantic = [("g1", 0.9)]
        curator = make_curator(retrieval)
        result = curator.curate("the goal", token_budget=5000, mode="orient")
        content = result.content
        assert content.startswith("=== Project Memory (orient) ===")
        positions = [
            content.index("--- Session History ---"),
            content.index("--- Key Decisions ---"),
            content.index("--- Recent Knowledge ---"),
            content.index("--- Pinned ---"),
            content.index("--- Goal-Relevant ---"),
        ]
        assert positions == sorted(positions)
        assert goal.id in [e.id for e in result.entries]
        assert "=== End Memory (" in content

    def test_multiple_entries_in_one_section(self, retrieval: FakeRetrieval) -> None:
        base = CLOCK_START - timedelta(hours=2)
        retrieval.add(
            make_entry("d1", "first decision", type_=MemoryType.DECISION, created_at=base)
        )
        retrieval.add(
            make_entry(
                "d2",
                "second decision",
                type_=MemoryType.DECISION,
                created_at=base - timedelta(hours=1),
            )
        )
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=5000, mode="orient")
        assert result.content.count("--- Key Decisions ---") == 1  # header once
        assert {e.id for e in result.entries} == {"d1", "d2"}

    def test_unused_share_rolls_forward(self, retrieval: FakeRetrieval) -> None:
        # No session markers: the session-marker share rolls into decisions,
        # whose own (renormalized) share alone could NOT fit the 19-token first
        # inclusion — only the rollover makes it fit.
        retrieval.add(make_entry("d1", words(10, "d"), type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=51, mode="orient")
        assert "--- Key Decisions ---" in result.content
        assert "--- Session History ---" not in result.content
        assert "d1" in [e.id for e in result.entries]

    def test_empty_query_orient_uses_full_budget(self, retrieval: FakeRetrieval) -> None:
        # MINOR-1: empty-query orient (category 5 absent) renormalizes shares so
        # 100% of content_budget is usable. Only pinned entries exist, so the
        # pinned category (last active) receives all rollover PLUS the reclaimed
        # 10% that category 5 would otherwise strand. content_budget = 100.
        for i in range(20):
            retrieval.add(make_entry(f"p{i:02d}", f"v{i}", pinned=True))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=111, mode="orient")
        assert result.sources_consulted == 20
        # 15 entries ship (token_count 104). The old stranded-10% behavior
        # capped the pinned category at 90 tokens ⇒ only 14 entries / token 98.
        assert len(result.entries) == 15
        assert result.token_count == 104
        assert result.token_count > 98  # strictly beats the old 90%-cap ceiling

    def test_cross_category_dedup_first_category_wins(self, retrieval: FakeRetrieval) -> None:
        pinned_decision = retrieval.add(
            make_entry("pd1", "pinned decision text", type_=MemoryType.DECISION, pinned=True)
        )
        retrieval.add(
            make_entry(
                "sm1",
                "marker summary text",
                type_=MemoryType.SUMMARY,
                tags=["session_marker"],
            )
        )
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=5000, mode="orient")
        assert result.content.count("pinned decision text") == 1
        assert result.content.count("marker summary text") == 1
        # Charged to the first category that selected each.
        assert "--- Key Decisions ---" in result.content
        assert "--- Pinned ---" not in result.content  # nothing left for it
        assert "--- Session History ---" in result.content
        assert "--- Recent Knowledge ---" not in result.content
        assert [e.id for e in result.entries].count(pinned_decision.id) == 1

    def test_empty_goal_skips_semantic_category(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("d1", "a decision", type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        curator.curate("", token_budget=500, mode="orient")
        assert retrieval.semantic_calls == []
        curator.curate("find the goal", token_budget=500, mode="orient")
        assert retrieval.semantic_calls == [("find the goal", 20)]

    def test_orient_redaction_and_access_match_query_mode(self, retrieval: FakeRetrieval) -> None:
        decision = retrieval.add(
            make_entry(
                "d1",
                "we chose password = hunter22 for the demo",
                type_=MemoryType.DECISION,
            )
        )
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500, mode="orient")
        assert "hunter22" not in result.content
        assert REDACTED in result.content
        assert retrieval.record_access_calls == [([decision.id], CLOCK_START)]
        assert decision.importance is not None

    def test_orient_oversize_single_candidate_truncated(self, retrieval: FakeRetrieval) -> None:
        # One decision too large for any category share, but the whole
        # content_budget can hold a truncated version (blueprint orient
        # oversize fallback: single best candidate, truncated).
        retrieval.add(make_entry("d1", words(50, "d"), type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=70, mode="orient")
        assert [e.id for e in result.entries] == ["d1"]
        assert "--- Key Decisions ---" in result.content
        assert "truncated by curator:" in result.content
        assert "d49" not in result.content  # tail elided
        assert retrieval.record_access_calls == [(["d1"], CLOCK_START)]

    def test_orient_truncation_impossible_returns_empty(self, retrieval: FakeRetrieval) -> None:
        # content_budget >= 1 but too small to hold even the section header
        # plus the truncation marker: graceful empty, no side effects.
        retrieval.add(make_entry("d1", words(50, "d"), type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=20, mode="orient")
        assert result.content == ""
        assert result.entries == []
        assert result.budget_remaining == 20
        assert retrieval.record_access_calls == []

    def test_orient_filters_apply(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(
            make_entry("d1", "obsolete decision", type_=MemoryType.DECISION, tags=["obsolete"])
        )
        retrieval.add(make_entry("d2", "current decision", type_=MemoryType.DECISION))
        curator = make_curator(retrieval)
        result = curator.curate("", token_budget=500, mode="orient", exclude_tags=["obsolete"])
        assert [e.id for e in result.entries] == ["d2"]
        assert "obsolete decision" not in result.content
        assert result.sources_consulted == 2


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


class TestPublicSurface:
    def test_curated_context_exported_from_package_root(self) -> None:
        # CuratedContext is the public return type of Memory.curate; it must be
        # reachable from the package root and be the same object as the
        # curator-module symbol (pinned in tests/test_public_api.py too).
        import tulving

        assert tulving.CuratedContext is CuratedContext
        assert curator_mod.CuratedContext is CuratedContext

    def test_estimator_satisfies_summarizer_protocol_shape(self) -> None:
        # Structural contract shared with the summarizer: .estimate(str) -> int.
        estimator = resolve_estimator(WordEstimator())
        assert estimator.estimate("two words") == 2

    def test_query_frame_shape(self, retrieval: FakeRetrieval) -> None:
        retrieval.add(make_entry("e1", "hello world", key="greeting"))
        curator = make_curator(retrieval)
        result = curator.curate("greeting", token_budget=200)
        lines = result.content.splitlines()
        assert lines[0].startswith("=== Agent Memory (1 entries,")
        assert lines[-1] == "=== End Memory ==="
        assert "[FACT] greeting (importance: 0.5," in result.content
        assert "hello world" in result.content


# ---------------------------------------------------------------------------
# QA mutation-hardening — close escapes found during the QA gate
# ---------------------------------------------------------------------------


class TestQaMutationHardening:
    """Tests added by QA to bite mutations the pre-existing suite let escape."""

    def test_ranking_uses_effective_not_raw_importance(self, retrieval: FakeRetrieval) -> None:
        # D12: query ranking must sort on the ImportanceEvaluator's EFFECTIVE
        # (lazily decayed / reinforced) value, never the raw stored
        # base_importance. Here the effective values INVERT the base ordering,
        # so a curator that scored on base_importance would flip the result.
        created = CLOCK_START - timedelta(hours=1)
        retrieval.add(
            make_entry(
                "hi_base", "high base, low effective", base_importance=0.9, created_at=created
            )
        )
        retrieval.add(
            make_entry(
                "lo_base", "low base, high effective", base_importance=0.2, created_at=created
            )
        )
        importance = FakeImportance({"hi_base": 0.1, "lo_base": 0.9})
        curator = make_curator(retrieval, importance=importance)
        result = curator.curate(
            "",
            token_budget=500,
            importance_weight=1.0,
            relevance_weight=0.0,
            recency_weight=0.0,
        )
        # Effective wins: lo_base (0.9) outranks hi_base (0.1) — the reverse of
        # base_importance. Raw-value scoring would yield ["hi_base", "lo_base"].
        assert [e.id for e in result.entries] == ["lo_base", "hi_base"]

    def test_truncation_operates_on_redacted_body_no_fragment_leak(
        self, retrieval: FakeRetrieval
    ) -> None:
        # Redact-BEFORE-truncate ordering (SEC): a splittable secret sits at the
        # FRONT of the content (the KEPT region, not the elided tail). The body
        # is redacted to [REDACTED] before any cut, so no secret fragment can
        # survive. A truncate-BEFORE-redact curator would cut the raw
        # "sk-zzz..." mid-token, leaving a <20-char fragment that the frame-level
        # redactor's `sk-...{20,}` shape no longer matches — a resurrected secret.
        secret = "sk-" + "z" * 20  # exactly matches sk-{20,}; any 1-char cut breaks the match
        filler = " ".join(f"w{i}" for i in range(100))
        retrieval.add(make_entry("e1", f"{secret} {filler}"))
        curator = make_curator(retrieval, estimator=CharEstimator())
        result = curator.curate("", token_budget=150)  # forces a cut INSIDE the secret word
        assert [e.id for e in result.entries] == ["e1"]
        assert "truncated by curator" in result.content
        assert REDACTED in result.content
        assert "sk-z" not in result.content  # no secret fragment resurrected past the redactor

"""Tests for tulving.context.summarizer — written BEFORE implementation.

All LLM calls are mocked (``FakeLLM`` records prompts, returns canned
digests, and can be told to raise); storage is ``InMemoryBackend`` behind a
real ``MemoryStore``; every ``now`` comes from the injected ``FakeClock`` —
zero ``sleep()`` calls, zero network.

As-built deviation pinned here (see module docstring of summarizer.py):
the no-LLM fallback digest is stored as ``MemoryType.OBSERVATION`` (not
SUMMARY as blueprint-summarizer sketched) because the as-built store
enforces "SUMMARY entries always carry non-empty ``_source_entry_ids``"
(blueprint-store, ``test_store.py::test_allow_summary_requires_source_entry_ids``)
and the fallback digest rolled up nothing.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import pytest
from conftest import CLOCK_START, FakeClock

from tulving.adapters.storage import InMemoryBackend
from tulving.context.config import LifecycleConfig
from tulving.context.summarizer import (
    SUMMARIZE_PROMPT,
    TRUNCATION_MARKER,
    MemorySummarizer,
)
from tulving.entry import MemoryEntry, SourceInfo
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import ConfigError, MemoryStoreError
from tulving.security import compile_explicit_patterns, compile_key_patterns
from tulving.store import MemoryStore

AGENT = "agent-1"
T0 = CLOCK_START


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Protocol-conforming mock: records prompts, canned/failing responses."""

    def __init__(
        self,
        *,
        fail_on: set[int] | None = None,
        empty_on: set[int] | None = None,
        max_input_tokens: int = 10**9,
    ) -> None:
        self.prompts: list[str] = []
        self._fail_on = fail_on or set()
        self._empty_on = empty_on or set()
        self._max_input_tokens = max_input_tokens

    def complete(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        call_number = len(self.prompts)
        if call_number in self._fail_on:
            raise RuntimeError("provider exploded")
        if call_number in self._empty_on:
            return "   "
        return f"digest {call_number}"

    async def complete_async(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    @property
    def max_input_tokens(self) -> int:
        return self._max_input_tokens


class CharEstimator:
    """1 char == 1 token: makes every cap computation exact in tests."""

    def estimate(self, text: str) -> int:
        return len(text)


class QuarterEstimator:
    """The curator's fallback shape: len // 4."""

    def estimate(self, text: str) -> int:
        return len(text) // 4


class StalenessDecay:
    """Effective importance halves every hour since last access."""

    def effective_importance(self, entry: MemoryEntry, now: Any) -> float:
        anchor = entry.last_accessed_at or entry.created_at
        hours = max((now - anchor).total_seconds() / 3600.0, 0.0)
        result: float = entry.base_importance * 0.5**hours
        return result


class SpyStore(MemoryStore):
    """Records create/archive call order for the write-ordering test."""

    def __init__(self, backend: InMemoryBackend, *, clock: FakeClock) -> None:
        super().__init__(backend, clock=clock)
        self.calls: list[tuple[str, str]] = []

    def create(self, **kwargs: Any) -> MemoryEntry:
        entry = super().create(**kwargs)
        self.calls.append(("create", entry.id))
        return entry

    def archive(self, entry_id: str, reason: ArchiveReason) -> MemoryEntry:
        self.calls.append(("archive", entry_id))
        return super().archive(entry_id, reason)


class RacingStore(MemoryStore):
    """Archives ``race_target`` as EVICTED right after a SUMMARY is created —
    simulating a concurrent archive landing between the summary write and
    the summarizer's source-archive loop."""

    def __init__(self, backend: InMemoryBackend, *, clock: FakeClock) -> None:
        super().__init__(backend, clock=clock)
        self.race_target: str | None = None

    def create(self, **kwargs: Any) -> MemoryEntry:
        entry = super().create(**kwargs)
        if entry.type is MemoryType.SUMMARY and self.race_target is not None:
            target, self.race_target = self.race_target, None
            super().archive(target, ArchiveReason.EVICTED)
        return entry


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0)


@pytest.fixture
def store(clock: FakeClock) -> MemoryStore:
    return MemoryStore(InMemoryBackend(), clock=clock)


def put(
    store: MemoryStore,
    content: str,
    *,
    type_: MemoryType = MemoryType.FACT,
    key: str | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    importance: float = 0.5,
    pinned: bool = False,
) -> MemoryEntry:
    return store.create(
        content=content,
        type=type_,
        source=SourceInfo(agent_id=AGENT),
        key=key,
        tags=tags,
        session_id=session_id,
        base_importance=importance,
        pinned=pinned,
    )


def make(
    store: MemoryStore,
    clock: FakeClock,
    *,
    llm: FakeLLM | None,
    config: LifecycleConfig | None = None,
    estimator: Any = None,
    agent_id: str = AGENT,
    key_patterns: Any = None,
    explicit_key_patterns: Any = None,
) -> MemorySummarizer:
    return MemorySummarizer(
        store=store,
        llm=llm,
        config=config if config is not None else LifecycleConfig(),
        token_estimator=estimator if estimator is not None else QuarterEstimator(),
        decay=StalenessDecay(),
        agent_id=agent_id,
        clock=clock,
        key_patterns=key_patterns,
        explicit_key_patterns=explicit_key_patterns,
    )


def live_ids(store: MemoryStore) -> set[str]:
    return {e.id for e in store.list(limit=500)}


def summaries(store: MemoryStore) -> list[MemoryEntry]:
    return [e for e in store.list(limit=500) if e.type is MemoryType.SUMMARY]


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_summarize_group_without_llm_raises_config_error(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        entry = put(store, "some fact")
        summarizer = make(store, clock, llm=None)
        with pytest.raises(ConfigError):
            summarizer.summarize_group([entry])

    def test_summarize_group_all_exempt_raises(self, store: MemoryStore, clock: FakeClock) -> None:
        decision = put(store, "we chose sqlite", type_=MemoryType.DECISION)
        pinned = put(store, "pinned fact", pinned=True)
        summarizer = make(store, clock, llm=FakeLLM())
        with pytest.raises(MemoryStoreError):
            summarizer.summarize_group([decision, pinned])
        # Nothing was archived by the refusal.
        assert store.get_by_id(decision.id, touch=False) is not None
        assert not store.get_by_id(decision.id, touch=False).archived  # type: ignore[union-attr]
        assert not store.get_by_id(pinned.id, touch=False).archived  # type: ignore[union-attr]

    def test_summarize_group_empty_list_raises(self, store: MemoryStore, clock: FakeClock) -> None:
        summarizer = make(store, clock, llm=FakeLLM())
        with pytest.raises(MemoryStoreError):
            summarizer.summarize_group([])

    def test_adapter_raises_every_call_spends_budget_then_stops_cleanly(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 7: counter increments BEFORE the call — a raising adapter
        still consumes its slot; the pass returns [] without raising."""
        for i in range(5):
            put(store, f"fact {i}", tags=[f"t{i}"])  # five distinct groups
        llm = FakeLLM(fail_on={1, 2, 3, 4, 5})
        summarizer = make(store, clock, llm=llm, config=LifecycleConfig(llm_call_budget=3))
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert result == []
        assert len(llm.prompts) == 3  # exactly budget attempts, then clean stop
        assert summaries(store) == []
        assert len(live_ids(store)) == 5  # nothing archived

    def test_adapter_failure_leaves_group_live_and_continues(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 8: raise on call 1 of 3, succeed on 2 and 3 -> 2 summaries."""
        e1 = put(store, "fact one", tags=["t1"])
        put(store, "fact two", tags=["t2"])
        put(store, "fact three", tags=["t3"])
        llm = FakeLLM(fail_on={1})
        summarizer = make(store, clock, llm=llm)
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(result) == 2
        assert len(llm.prompts) == 3
        failed = store.get_by_id(e1.id, touch=False)
        assert failed is not None and not failed.archived
        assert any("failed" in record.message for record in caplog.records)

    def test_empty_llm_response_leaves_chunk_live(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        entry = put(store, "only fact")
        summarizer = make(store, clock, llm=FakeLLM(empty_on={1}))
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert result == []
        assert summaries(store) == []
        refreshed = store.get_by_id(entry.id, touch=False)
        assert refreshed is not None and not refreshed.archived

    def test_max_group_size_below_one_raises(self, store: MemoryStore, clock: FakeClock) -> None:
        summarizer = make(store, clock, llm=FakeLLM())
        with pytest.raises(MemoryStoreError):
            summarizer.summarize(max_group_size=0)

    def test_empty_agent_id_raises_config_error(self, store: MemoryStore, clock: FakeClock) -> None:
        with pytest.raises(ConfigError):
            make(store, clock, llm=None, agent_id="")

    def test_summarize_group_single_chunk_failure_raises(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Review MAJOR (i): the only chunk's call fails -> MemoryStoreError,
        nothing archived, no summary stored."""
        fact = put(store, "plain fact")
        summarizer = make(store, clock, llm=FakeLLM(fail_on={1}))
        with pytest.raises(MemoryStoreError, match="no summary"):
            summarizer.summarize_group([fact])
        refreshed = store.get_by_id(fact.id, touch=False)
        assert refreshed is not None and not refreshed.archived
        assert summaries(store) == []

    def test_summarize_group_budget_exhaustion_mid_group(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Review MAJOR (ii): budget 1 over a 2-chunk group -> first chunk
        summarized, remainder left live, exhaustion warning logged."""
        e1 = put(store, "FIRST " + "x" * 594)
        clock.advance(seconds=1)
        e2 = put(store, "SECOND " + "y" * 593)
        llm = FakeLLM()
        config = LifecycleConfig(llm_call_budget=1, max_input_tokens=1000, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            summary = summarizer.summarize_group([e1, e2])
        assert len(llm.prompts) == 1
        assert summary.source_entry_ids == [e1.id]
        archived_first = store.get_by_id(e1.id, touch=False)
        assert archived_first is not None and archived_first.archived
        live_second = store.get_by_id(e2.id, touch=False)
        assert live_second is not None and not live_second.archived
        exhausted = [r for r in caplog.records if "budget exhausted" in r.message]
        assert len(exhausted) == 1
        assert "1 chunk(s)" in exhausted[0].getMessage()

    def test_summarize_group_all_chunks_fail_raises_within_budget(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Review MAJOR (iii): every call fails -> MemoryStoreError, and the
        number of attempts never exceeds the budget."""
        entries = []
        for i in range(3):
            entries.append(put(store, f"CHUNK{i} " + "z" * 592))
            clock.advance(seconds=1)
        llm = FakeLLM(fail_on={1, 2, 3})
        config = LifecycleConfig(llm_call_budget=2, max_input_tokens=1000, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        with (
            caplog.at_level(logging.WARNING, logger="tulving.summarizer"),
            pytest.raises(MemoryStoreError, match="2 LLM call"),
        ):
            summarizer.summarize_group(entries)
        assert len(llm.prompts) == 2  # budget never exceeded
        for entry in entries:
            refreshed = store.get_by_id(entry.id, touch=False)
            assert refreshed is not None and not refreshed.archived
        assert summaries(store) == []


# ---------------------------------------------------------------------------
# llm=None — loud degradation (never silent)
# ---------------------------------------------------------------------------


class TestNoLLMDegradation:
    def test_fallback_digest_stored_and_loud(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 1: exactly one warning; one digest; nothing archived."""
        put(store, "fact alpha", key="alpha")
        put(store, "fact beta", key="beta")
        summarizer = make(store, clock, llm=None)
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        warnings = [r for r in caplog.records if "no LLM adapter configured" in r.message]
        assert len(warnings) == 1
        assert len(result) == 1
        digest = result[0]
        # OBSERVATION, not SUMMARY: the as-built store forbids SUMMARY rows
        # with empty back-links (see module docstring deviation note).
        assert digest.type is MemoryType.OBSERVATION
        assert digest.tags == ["summarize_skipped"]
        assert digest.source_entry_ids == []
        assert digest.key is None
        assert digest.base_importance == pytest.approx(0.3)
        assert digest.source.agent_id == AGENT
        assert store.get_by_id(digest.id, touch=False) is not None  # persisted
        # Zero entries archived.
        archived = store.list(include_archived=True, limit=500)
        assert all(not e.archived for e in archived)

    def test_no_candidates_stores_nothing(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 2: warning still logged, nothing stored, returns []."""
        summarizer = make(store, clock, llm=None)
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert result == []
        assert "no LLM adapter configured" in caplog.text
        assert store.list(include_archived=True, limit=500) == []

    def test_fallback_digest_is_deterministic(self, clock: FakeClock) -> None:
        """Test 4: identical state -> identical digest content."""
        contents: list[str] = []
        for _ in range(2):
            fresh_store = MemoryStore(InMemoryBackend(), clock=clock)
            put(fresh_store, "fact alpha", key="alpha", importance=0.9)
            put(fresh_store, "fact beta", key="beta", importance=0.4)
            put(fresh_store, "picked sqlite", key="db-choice", type_=MemoryType.DECISION)
            summarizer = make(fresh_store, clock, llm=None)
            contents.append(summarizer.summarize()[0].content)
        assert contents[0] == contents[1]

    def test_fallback_quotes_decisions_when_flag_true(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        put(store, "ordinary fact", key="fact-1")
        put(store, "DEC-CONTENT-SENTINEL", key="db-choice", type_=MemoryType.DECISION)
        summarizer = make(
            store, clock, llm=None, config=LifecycleConfig(preserve_decisions_verbatim=True)
        )
        digest = summarizer.summarize()[0]
        assert "Decisions preserved verbatim: 1." in digest.content
        assert "DEC-CONTENT-SENTINEL" in digest.content

    def test_fallback_counts_decisions_when_flag_false(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 5: flag False counts decisions instead of quoting them."""
        put(store, "ordinary fact", key="fact-1")
        put(store, "DEC-CONTENT-SENTINEL", key="db-choice", type_=MemoryType.DECISION)
        summarizer = make(
            store, clock, llm=None, config=LifecycleConfig(preserve_decisions_verbatim=False)
        )
        digest = summarizer.summarize()[0]
        assert "Decisions preserved verbatim: 1." in digest.content
        assert "DEC-CONTENT-SENTINEL" not in digest.content

    def test_fallback_orders_top_items_by_importance(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        put(store, "low importance", key="low", importance=0.1)
        put(store, "high importance", key="high", importance=0.9)
        put(store, "mid importance", key="mid", importance=0.5)
        summarizer = make(store, clock, llm=None)
        digest = summarizer.summarize()[0]
        assert digest.content.index("high") < digest.content.index("mid")
        assert digest.content.index("mid") < digest.content.index("low")


# ---------------------------------------------------------------------------
# Call budget (circuit breaker)
# ---------------------------------------------------------------------------


class TestCallBudget:
    def test_budget_two_five_groups(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 6: exactly 2 calls, 2 summaries, 3 groups left live, warned."""
        for i in range(5):
            put(store, f"fact {i}", tags=[f"t{i}"])
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm, config=LifecycleConfig(llm_call_budget=2))
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(result) == 2
        assert len(llm.prompts) == 2
        assert len(summaries(store)) == 2
        live_facts = [e for e in store.list(limit=500) if e.type is MemoryType.FACT]
        assert len(live_facts) == 3  # the remaining 3 groups, unarchived
        exhausted = [r for r in caplog.records if "budget exhausted" in r.message]
        assert len(exhausted) == 1
        assert "3 group(s)" in exhausted[0].getMessage()

    def test_budget_respected_across_chunks(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test 9: one group splitting into 4 chunks under budget 2."""
        entries = []
        for i in range(4):
            entries.append(put(store, f"CONTENT{i:02d} " + "x" * 590, tags=["grp"]))
            clock.advance(seconds=1)
        llm = FakeLLM()
        config = LifecycleConfig(llm_call_budget=2, max_input_tokens=1000, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(result) == 2
        assert len(llm.prompts) == 2
        # Deterministic order: the two oldest entries were summarized.
        archived_ids = {sid for s in result for sid in s.source_entry_ids}
        assert archived_ids == {entries[0].id, entries[1].id}
        for deferred in entries[2:]:
            refreshed = store.get_by_id(deferred.id, touch=False)
            assert refreshed is not None and not refreshed.archived
        exhausted = [r for r in caplog.records if "budget exhausted" in r.message]
        assert len(exhausted) == 1
        assert "2 chunk(s)" in exhausted[0].getMessage()


# ---------------------------------------------------------------------------
# Chunking & truncation
# ---------------------------------------------------------------------------


class TestChunkingTruncation:
    def test_oversize_group_is_split_into_fitting_chunks(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 10: each chunk prompt fits the cap; chunks never mix groups."""
        group_a = []
        for i in range(4):
            group_a.append(put(store, f"AAA{i:02d} " + "a" * 140, tags=["ga"]))
            clock.advance(seconds=1)
        group_b = [put(store, "BBB00 " + "b" * 140, tags=["gb"])]
        llm = FakeLLM()
        config = LifecycleConfig(max_input_tokens=1000, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        result = summarizer.summarize()
        assert len(llm.prompts) >= 3  # group a split (>1 chunk) + group b
        for prompt in llm.prompts:
            assert len(prompt) <= 1000  # CharEstimator: len == tokens
        # No prompt mixes the two groups.
        for prompt in llm.prompts:
            has_a = any(f"AAA{i:02d}" in prompt for i in range(4))
            assert not (has_a and "BBB00" in prompt)
        # Every entry landed in exactly one prompt and one summary back-link.
        all_sources = [sid for s in result for sid in s.source_entry_ids]
        assert sorted(all_sources) == sorted(e.id for e in group_a + group_b)

    def test_single_oversized_entry_truncated_original_intact(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 11: prompt truncated with the marker; stored original is
        byte-identical after the pass (archived, complete)."""
        original = "HEAD-" + "x" * 2000 + "-TAILSENTINEL"
        entry = put(store, original)
        llm = FakeLLM()
        config = LifecycleConfig(max_input_tokens=1000, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        result = summarizer.summarize()
        assert len(result) == 1
        assert len(llm.prompts) == 1
        prompt = llm.prompts[0]
        assert len(prompt) <= 1000
        assert TRUNCATION_MARKER in prompt
        assert "TAILSENTINEL" not in prompt
        assert "HEAD-" in prompt
        archived = store.get_by_id(entry.id, touch=False)
        assert archived is not None
        assert archived.archived and archived.archive_reason is ArchiveReason.SUMMARIZED
        assert archived.content == original  # never modified

    def test_trim_order_is_lowest_effective_importance_first(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 12: two entries, distinct last_accessed_at -> the staler one
        is trimmed when the assembled chunk slightly overflows."""
        content_stale = "STALE-" + "a" * 420
        content_fresh = "FRESH-" + "b" * 420
        put(store, content_stale, key="k1")
        clock.advance(hours=2)
        put(store, content_fresh, key="k2")
        block_stale = f"- [FACT] k1: {content_stale}"
        block_fresh = f"- [FACT] k2: {content_fresh}"
        overhead = len(SUMMARIZE_PROMPT.format(entries=""))
        # Greedy chunking admits both (sum == cap) but the assembled prompt
        # overflows by exactly the joining newline -> the trim path runs.
        cap = overhead + len(block_stale) + len(block_fresh)
        llm = FakeLLM()
        config = LifecycleConfig(max_input_tokens=cap, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        result = summarizer.summarize()
        assert len(result) == 1
        prompt = llm.prompts[0]
        assert len(prompt) <= cap
        assert TRUNCATION_MARKER in prompt
        assert content_fresh in prompt  # fresher entry untouched
        assert content_stale not in prompt  # staler entry trimmed...
        assert "STALE-" in prompt  # ...but its head is kept

    def test_cap_below_overhead_dispatches_over_cap_with_warning(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Review MINOR #4: a cap smaller than the template overhead cannot
        be met even by full truncation — the over-cap prompt is dispatched
        anyway, with a warning (pinned terminal path)."""
        entry = put(store, "some content " + "w" * 500)
        llm = FakeLLM()
        cap = 256  # < len(SUMMARIZE_PROMPT template) with CharEstimator
        config = LifecycleConfig(max_input_tokens=cap, token_safety_margin=0.0)
        summarizer = make(store, clock, llm=llm, config=config, estimator=CharEstimator())
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(SUMMARIZE_PROMPT.format(entries="")) > cap  # premise
        assert len(llm.prompts) == 1  # dispatched anyway
        assert len(llm.prompts[0]) > cap
        assert TRUNCATION_MARKER in llm.prompts[0]  # truncation was attempted
        assert len(result) == 1  # the chunk still summarized
        archived = store.get_by_id(entry.id, touch=False)
        assert archived is not None and archived.archived
        overflow = [r for r in caplog.records if "still exceeds the input cap" in r.message]
        assert len(overflow) == 1


# ---------------------------------------------------------------------------
# Back-links, archival, exemptions (ADR-009 / ADR-006)
# ---------------------------------------------------------------------------


class TestBackLinksArchival:
    def test_happy_path_summary_fields_and_archival(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 13."""
        e1 = put(store, "fact 1", tags=["a", "b"], session_id="s1", importance=0.4)
        e2 = put(store, "fact 2", tags=["a"], session_id="s1", importance=0.9)
        e3 = put(store, "fact 3", tags=["a", "c"], session_id="s1", importance=0.2)
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        result = summarizer.summarize()
        assert len(result) == 1
        summary = result[0]
        assert summary.type is MemoryType.SUMMARY
        assert sorted(summary.source_entry_ids) == sorted([e1.id, e2.id, e3.id])
        assert summary.source.agent_id == AGENT
        assert summary.base_importance == pytest.approx(0.9)
        assert summary.tags == ["a", "b", "c"]
        assert summary.session_id == "s1"
        assert summary.key is None
        for source in (e1, e2, e3):
            archived = store.get_by_id(source.id, touch=False)  # still retrievable
            assert archived is not None
            assert archived.archived
            assert archived.archive_reason is ArchiveReason.SUMMARIZED

    def test_decision_and_pinned_are_never_candidates(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 14: live, unarchived, absent from every prompt."""
        decision = put(store, "DECISION-SENTINEL", type_=MemoryType.DECISION)
        pinned = put(store, "PINNED-SENTINEL", pinned=True)
        put(store, "normal fact one")
        put(store, "normal fact two")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        summarizer.summarize()
        for exempt in (decision, pinned):
            refreshed = store.get_by_id(exempt.id, touch=False)
            assert refreshed is not None and not refreshed.archived
        for prompt in llm.prompts:
            assert "DECISION-SENTINEL" not in prompt
            assert "PINNED-SENTINEL" not in prompt

    def test_summarize_group_mixed_summarizes_only_non_exempt(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 15."""
        decision = put(store, "we chose sqlite", type_=MemoryType.DECISION)
        pinned = put(store, "pinned fact", pinned=True)
        fact = put(store, "plain fact")
        summarizer = make(store, clock, llm=FakeLLM())
        summary = summarizer.summarize_group([decision, pinned, fact])
        assert summary.type is MemoryType.SUMMARY
        assert summary.source_entry_ids == [fact.id]
        assert not store.get_by_id(decision.id, touch=False).archived  # type: ignore[union-attr]
        assert not store.get_by_id(pinned.id, touch=False).archived  # type: ignore[union-attr]
        assert store.get_by_id(fact.id, touch=False).archived  # type: ignore[union-attr]

    def test_summary_created_before_sources_archived(self, clock: FakeClock) -> None:
        """Test 16: create(SUMMARY) precedes every archive of its sources."""
        spy = SpyStore(InMemoryBackend(), clock=clock)
        put(spy, "fact 1", tags=["t1"])
        put(spy, "fact 2", tags=["t2"])
        summarizer = make(spy, clock, llm=FakeLLM())
        spy.calls.clear()
        result = summarizer.summarize()
        assert len(result) == 2
        by_summary = {s.id: set(s.source_entry_ids) for s in result}
        current: set[str] | None = None
        for action, entry_id in spy.calls:
            if action == "create":
                current = by_summary.get(entry_id)
                assert current is not None
            else:
                assert current is not None, "archive before any summary create"
                assert entry_id in current, "archive outside the current chunk"

    def test_concurrently_archived_source_is_skipped_not_fatal(
        self, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Review MINOR #3: a source archived between summary creation and
        the archive loop is logged and skipped; its reason is never
        overwritten and the pass completes normally."""
        racing = RacingStore(InMemoryBackend(), clock=clock)
        raced = put(racing, "raced fact")
        clock.advance(seconds=1)
        clean = put(racing, "clean fact")
        racing.race_target = raced.id
        summarizer = make(racing, clock, llm=FakeLLM())
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(result) == 1  # the pass did not abort
        assert sorted(result[0].source_entry_ids) == sorted([raced.id, clean.id])
        raced_row = racing.get_by_id(raced.id, touch=False)
        assert raced_row is not None and raced_row.archived
        assert raced_row.archive_reason is ArchiveReason.EVICTED  # never relabeled
        clean_row = racing.get_by_id(clean.id, touch=False)
        assert clean_row is not None and clean_row.archived
        assert clean_row.archive_reason is ArchiveReason.SUMMARIZED
        skipped = [r for r in caplog.records if "archived concurrently" in r.message]
        assert len(skipped) == 1
        assert raced.id in skipped[0].getMessage()

    def test_no_recursive_resummarization_within_a_pass(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 17: pass-created summaries are excluded; a pre-existing
        SUMMARY is a valid candidate."""
        old_summary = store.create(
            content="an old digest",
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id=AGENT),
            _allow_summary=True,
            _source_entry_ids=["deadbeefdeadbeefdeadbeefdeadbeef"],
        )
        put(store, "a plain fact")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        result = summarizer.summarize()
        assert len(llm.prompts) == 1  # one group, one call; no recursion
        assert len(result) == 1
        rolled = store.get_by_id(old_summary.id, touch=False)
        assert rolled is not None and rolled.archived  # pre-existing SUMMARY eligible
        assert rolled.archive_reason is ArchiveReason.SUMMARIZED
        fresh = store.get_by_id(result[0].id, touch=False)
        assert fresh is not None and not fresh.archived  # pass output stays live


# ---------------------------------------------------------------------------
# Candidate semantics
# ---------------------------------------------------------------------------


class TestCandidateSemantics:
    def test_candidate_reads_do_not_touch(self, store: MemoryStore, clock: FakeClock) -> None:
        """Test 18: last_accessed_at/access_count unchanged by a pass (D3)."""
        e1 = put(store, "fact one")
        e2 = put(store, "fact two")
        clock.advance(hours=5)
        make(store, clock, llm=None).summarize()  # fallback pass
        make(store, clock, llm=FakeLLM()).summarize()  # real pass
        for entry in (e1, e2):
            refreshed = store.get_by_id(entry.id, touch=False)
            assert refreshed is not None
            assert refreshed.last_accessed_at == T0
            assert refreshed.access_count == 0

    def test_older_than_filters_on_last_accessed_at(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 19: recently-accessed entries are spared; boundary exact."""
        e_old = put(store, "old and idle")
        e_recent = put(store, "old but in active use")
        e_boundary = put(store, "accessed exactly at the cutoff")
        clock.advance(hours=3)
        store.get_by_id(e_boundary.id)  # touch at T0+3h == cutoff instant
        clock.advance(hours=1)
        store.get_by_id(e_recent.id)  # touch at T0+4h (1h before the pass)
        clock.advance(hours=1)  # now = T0+5h; cutoff = T0+3h
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        result = summarizer.summarize(older_than=timedelta(hours=2))
        assert len(result) == 1
        assert result[0].source_entry_ids == [e_old.id]
        for spared in (e_recent, e_boundary):  # strict '<': boundary spared
            refreshed = store.get_by_id(spared.id, touch=False)
            assert refreshed is not None and not refreshed.archived

    def test_second_pass_resummarizes_nothing(self, store: MemoryStore, clock: FakeClock) -> None:
        """Test 20: archived sources are never candidates again."""
        put(store, "fact one")
        put(store, "fact two")
        clock.advance(hours=3)
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        first = summarizer.summarize(older_than=timedelta(hours=1))
        assert len(first) == 1
        second = summarizer.summarize(older_than=timedelta(hours=1))
        assert second == []
        assert len(llm.prompts) == 1  # no LLM calls on the second pass

    def test_session_id_scopes_the_pass(self, store: MemoryStore, clock: FakeClock) -> None:
        """Test 21: lifecycle's end-of-session entry point."""
        s1a = put(store, "s1 fact a", session_id="s1")
        s1b = put(store, "s1 fact b", session_id="s1")
        other = put(store, "s2 fact", session_id="s2")
        unscoped = put(store, "no session fact")
        summarizer = make(store, clock, llm=FakeLLM())
        result = summarizer.summarize(session_id="s1")
        assert len(result) == 1
        assert sorted(result[0].source_entry_ids) == sorted([s1a.id, s1b.id])
        assert result[0].session_id == "s1"
        for spared in (other, unscoped):
            refreshed = store.get_by_id(spared.id, touch=False)
            assert refreshed is not None and not refreshed.archived

    def test_tags_filter_scopes_the_pass(self, store: MemoryStore, clock: FakeClock) -> None:
        tagged = put(store, "tagged fact", tags=["x"])
        spared = put(store, "other fact", tags=["y"])
        summarizer = make(store, clock, llm=FakeLLM())
        result = summarizer.summarize(tags=["x"])
        assert len(result) == 1
        assert result[0].source_entry_ids == [tagged.id]
        refreshed = store.get_by_id(spared.id, touch=False)
        assert refreshed is not None and not refreshed.archived

    def test_max_group_size_slices_groups(self, store: MemoryStore, clock: FakeClock) -> None:
        for i in range(5):
            put(store, f"fact {i}")
            clock.advance(seconds=1)
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        result = summarizer.summarize(max_group_size=2)
        assert [len(s.source_entry_ids) for s in result] == [2, 2, 1]
        assert len(llm.prompts) == 3

    def test_scan_pages_beyond_the_first_page(
        self, store: MemoryStore, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QA coverage (summarizer.py:335): candidates spanning more than one
        ``_PAGE`` of the store scan are ALL collected. Shrinks ``_PAGE`` so a
        handful of entries forces the offset-advance loop; a regression that
        stopped after page one would silently drop later candidates."""
        monkeypatch.setattr("tulving.context.summarizer._PAGE", 2)
        made = []
        for i in range(5):  # 5 entries, page size 2 -> pages [2, 2, 1]
            made.append(put(store, f"fact {i}"))
            clock.advance(seconds=1)
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        result = summarizer.summarize()  # one partition, one group, one chunk
        assert len(result) == 1
        assert sorted(result[0].source_entry_ids) == sorted(e.id for e in made)

    def test_group_ordering_unscoped_before_sessioned_is_deterministic(
        self, store: MemoryStore, clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """QA determinism pin: grouping is session-rank THEN primary tag THEN
        time (blueprint-summarizer). Under a 1-call budget the FIRST group by
        that total order is the one summarized. The unscoped entry (session
        rank 0) must win over a sessioned entry even though its tag sorts
        later ('zzz' > 'aaa') — proving the session-rank bit dominates the tag
        and the ordering is not tag-alphabetical or insertion-order."""
        e_unscoped = put(store, "unscoped fact", tags=["zzz"])
        e_session = put(store, "session fact", tags=["aaa"], session_id="s1")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm, config=LifecycleConfig(llm_call_budget=1))
        with caplog.at_level(logging.WARNING, logger="tulving.summarizer"):
            result = summarizer.summarize()
        assert len(result) == 1
        assert result[0].source_entry_ids == [e_unscoped.id]
        still_live = store.get_by_id(e_session.id, touch=False)
        assert still_live is not None and not still_live.archived


# ---------------------------------------------------------------------------
# Security (LLM egress redaction — CLAUDE.md req #1 extended by ADR-010)
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_sensitive_key_and_token_shapes_redacted_from_prompts(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 22 (prompt half)."""
        put(store, "SUPERSECRETVALUE", key="api_key")
        put(store, "benign text with sk-abcdefghijklmnopqrstuvwx inside")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        for prompt in llm.prompts:
            assert "SUPERSECRETVALUE" not in prompt
            assert "sk-abcdefghijklmnopqrstuvwx" not in prompt
        assert any("[REDACTED]" in prompt for prompt in llm.prompts)

    def test_fallback_digest_content_is_redacted(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 22 (fallback half): the stored digest gets the same treatment."""
        put(store, "SUPERSECRETVALUE", key="api_key")
        put(store, "benign text with sk-abcdefghijklmnopqrstuvwx inside")
        summarizer = make(store, clock, llm=None)
        digest = summarizer.summarize()[0]
        assert "SUPERSECRETVALUE" not in digest.content
        assert "sk-abcdefghijklmnopqrstuvwx" not in digest.content
        assert "[REDACTED]" in digest.content

    def test_custom_key_patterns_reach_every_prompt(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """SEC SEV-002 (prompt half): Memory(sensitive_keys=...) augmentation
        must mask custom-keyed content on LLM egress."""
        put(store, "CUSTOMSECRETVALUE", key="internal_ref")
        put(store, "perfectly benign fact")
        llm = FakeLLM()
        patterns = compile_key_patterns(["internal_ref"])
        summarizer = make(store, clock, llm=llm, key_patterns=patterns)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        for prompt in llm.prompts:
            assert "CUSTOMSECRETVALUE" not in prompt
        assert any("[REDACTED]" in prompt for prompt in llm.prompts)

    def test_custom_key_patterns_reach_fallback_digest(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """SEC SEV-002 (fallback half): the stored digest masks custom-keyed
        content too."""
        put(store, "CUSTOMSECRETVALUE", key="internal_ref")
        put(store, "perfectly benign fact")
        patterns = compile_key_patterns(["internal_ref"])
        summarizer = make(store, clock, llm=None, key_patterns=patterns)
        digest = summarizer.summarize()[0]
        assert "CUSTOMSECRETVALUE" not in digest.content
        assert "[REDACTED]" in digest.content

    def test_custom_key_patterns_scrub_inline_labelled_secret_on_egress(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """QA gap: the custom-key masking every existing test exercises rides
        the ``entry.key``/``is_sensitive_key`` guard. This pins the OTHER
        egress guard — ``redact_text(key_patterns=...)`` masking a custom-key
        *label:value* embedded inline in the content BODY (entry.key is None,
        so the key guard cannot fire). Kills a regression that drops
        ``key_patterns`` from the ``redact_text`` egress calls."""
        put(store, "field notes: internal_ref: INLINESECRET42 was observed")
        llm = FakeLLM()
        patterns = compile_key_patterns(["internal_ref"])
        summarizer = make(store, clock, llm=llm, key_patterns=patterns)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        for prompt in llm.prompts:
            assert "INLINESECRET42" not in prompt
        assert any("[REDACTED]" in prompt for prompt in llm.prompts)

    def test_default_sensitive_key_prose_passes_through(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """v0.2 softening (D-v02-7): a default-sensitive-named key holding
        plain prose is no longer whole-masked on LLM egress."""
        put(store, "auth token TTL is 15 min", key="fact:auth-ttl")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        assert any("auth token TTL is 15 min" in prompt for prompt in llm.prompts)
        assert not any("[REDACTED]" in prompt for prompt in llm.prompts)

    def test_default_sensitive_key_secret_shaped_content_masks(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Default-sensitive key + secret-SHAPED content -> still whole-masked.

        The secret is embedded inside an unrelated sentence and the
        SURROUNDING sentence is asserted absent too (test review MAJOR):
        surgical redact_text alone would strip only the secret substring and
        leave the sentence text intact, so this can only pass via true
        whole-body masking."""
        secret = "sk-live-" + "x" * 24
        content = f"the rotation doc says {secret} expires monthly"
        put(store, content, key="api_key:prod")
        llm = FakeLLM()
        summarizer = make(store, clock, llm=llm)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        for prompt in llm.prompts:
            assert secret not in prompt
            assert "rotation doc says" not in prompt
            assert "expires monthly" not in prompt
        assert any("[REDACTED]" in prompt for prompt in llm.prompts)

    def test_user_declared_key_overrides_default_overlap(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """D-v02-7 Q3 (mandatory): a user-declared pattern masks unconditionally
        even when the key ALSO matches a built-in default, and even for prose."""
        put(store, "rotate quarterly, no value here", key="auth-prod-token")
        llm = FakeLLM()
        explicit = compile_explicit_patterns(["auth-prod"])
        summarizer = make(store, clock, llm=llm, explicit_key_patterns=explicit)
        summarizer.summarize()
        assert llm.prompts, "expected at least one LLM call"
        for prompt in llm.prompts:
            assert "rotate quarterly" not in prompt
        assert any("[REDACTED]" in prompt for prompt in llm.prompts)


# ---------------------------------------------------------------------------
# Purge interplay (mirrors the audit regression owned by test_store.py)
# ---------------------------------------------------------------------------


class TestPurgeInterplay:
    def test_purge_defaults_spare_summarized_sources(
        self, store: MemoryStore, clock: FakeClock
    ) -> None:
        """Test 23: default purge leaves SUMMARIZED sources intact;
        explicit listing removes them."""
        e1 = put(store, "fact one")
        e2 = put(store, "fact two")
        summarizer = make(store, clock, llm=FakeLLM())
        result = summarizer.summarize()
        assert len(result) == 1
        assert store.purge_archived() == 0  # defaults exclude SUMMARIZED
        for source in (e1, e2):
            assert store.get_by_id(source.id, touch=False) is not None
        assert store.purge_archived(reasons=[ArchiveReason.SUMMARIZED]) == 2
        for source in (e1, e2):
            assert store.get_by_id(source.id, touch=False) is None

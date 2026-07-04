"""Tests for tulving.context.decay — written BEFORE implementation.

All time math uses injected ``now`` values — zero ``sleep()`` calls. The
``FakeEvictionStore`` below implements the ``EvictionStore`` protocol
structurally (in-memory list, records ``archive_entry`` calls).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from tulving.context.config import DEFAULT_HALF_LIFE_HOURS, LifecycleConfig
from tulving.context.decay import (
    DecayReport,
    effective_importance,
    evict,
    is_decay_exempt,
    populate_importance,
)
from tulving.entry import MemoryEntry, SourceInfo
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import ConfigError, StorageError

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

HALF_LIVES: dict[MemoryType, float] = dict(DEFAULT_HALF_LIFE_HOURS)


def make_entry(
    *,
    entry_id: str = "e-1",
    type_: MemoryType = MemoryType.FACT,
    base_importance: float = 0.5,
    pinned: bool = False,
    created_at: datetime = T0,
    last_accessed_at: datetime | None = None,
) -> MemoryEntry:
    """A minimal valid entry with an injected timeline."""
    return MemoryEntry(
        id=entry_id,
        content="content",
        type=type_,
        source=SourceInfo(agent_id="agent-a"),
        base_importance=base_importance,
        pinned=pinned,
        created_at=created_at,
        updated_at=created_at,
        last_accessed_at=last_accessed_at,
    )


class FakeEvictionStore:
    """In-memory EvictionStore: yields active entries, records archives."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries: dict[str, MemoryEntry] = {entry.id: entry for entry in entries}
        self.archive_calls: list[tuple[str, ArchiveReason, datetime]] = []

    def iter_active_entries(self) -> Iterator[MemoryEntry]:
        yield from [entry for entry in self.entries.values() if not entry.archived]

    def archive_entry(self, entry_id: str, reason: ArchiveReason, *, now: datetime) -> None:
        self.archive_calls.append((entry_id, reason, now))
        entry = self.entries[entry_id]
        entry.archived = True
        entry.archive_reason = reason
        entry.updated_at = now


class ExplodingEvictionStore(FakeEvictionStore):
    """archive_entry raises StorageError — error-propagation tests."""

    def archive_entry(self, entry_id: str, reason: ArchiveReason, *, now: datetime) -> None:
        raise StorageError("archive failed")


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_naive_now_raises_value_error(self) -> None:
        entry = make_entry()
        with pytest.raises(ValueError):
            effective_importance(entry, datetime(2026, 1, 1), HALF_LIVES)

    def test_missing_type_key_raises_config_error(self) -> None:
        entry = make_entry(type_=MemoryType.FACT)
        with pytest.raises(ConfigError):
            effective_importance(entry, T0, {})

    def test_zero_half_life_raises_config_error(self) -> None:
        entry = make_entry(type_=MemoryType.FACT)
        with pytest.raises(ConfigError):
            effective_importance(entry, T0, {MemoryType.FACT: 0.0})

    def test_negative_half_life_raises_config_error(self) -> None:
        entry = make_entry(type_=MemoryType.FACT)
        with pytest.raises(ConfigError):
            effective_importance(entry, T0, {MemoryType.FACT: -1.0})

    def test_nan_half_life_raises_config_error(self) -> None:
        entry = make_entry(type_=MemoryType.FACT)
        with pytest.raises(ConfigError):
            effective_importance(entry, T0, {MemoryType.FACT: float("nan")})

    def test_bool_half_life_raises_config_error(self) -> None:
        entry = make_entry(type_=MemoryType.FACT)
        with pytest.raises(ConfigError):
            effective_importance(entry, T0, {MemoryType.FACT: True})

    def test_populate_importance_propagates_config_error(self) -> None:
        entries = [make_entry(entry_id="a"), make_entry(entry_id="b")]
        with pytest.raises(ConfigError):
            populate_importance(entries, T0, {})

    def test_evict_propagates_storage_error(self) -> None:
        # An ancient low-base FACT that WILL be selected for eviction.
        entry = make_entry(base_importance=0.2, last_accessed_at=T0 - timedelta(days=3650))
        store = ExplodingEvictionStore([entry])
        with pytest.raises(StorageError):
            evict(store, LifecycleConfig(), now=T0)

    def test_evict_naive_now_raises_value_error_even_on_empty_store(self) -> None:
        """The aware-check is upfront: an empty (or all-exempt) scan must not
        mask a naive injected clock by silently succeeding."""
        with pytest.raises(ValueError):
            evict(FakeEvictionStore([]), LifecycleConfig(), now=datetime(2026, 1, 1))
        with pytest.raises(ValueError):
            evict(
                FakeEvictionStore([make_entry(pinned=True)]),
                LifecycleConfig(),
                now=datetime(2026, 1, 1),
            )

    def test_evict_propagates_config_error_from_corrupted_config(self) -> None:
        config = LifecycleConfig()
        config.half_life_hours[MemoryType.FACT] = 0.0  # hand-corrupted post-validation
        store = FakeEvictionStore([make_entry()])
        with pytest.raises(ConfigError):
            evict(store, config, now=T0)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_clock_skew_clamped_to_age_zero(self) -> None:
        """A future last_accessed_at decays as age 0 — never amplifies."""
        entry = make_entry(base_importance=0.7, last_accessed_at=T0 + timedelta(hours=1))
        assert effective_importance(entry, T0, HALF_LIVES) == 0.7

    def test_base_zero_stays_zero_at_any_age(self) -> None:
        entry = make_entry(base_importance=0.0, last_accessed_at=T0 - timedelta(days=400))
        assert effective_importance(entry, T0, HALF_LIVES) == 0.0

    def test_extreme_age_underflows_toward_zero_without_raising(self) -> None:
        entry = make_entry(base_importance=1.0, last_accessed_at=T0 - timedelta(days=500 * 365))
        value = effective_importance(entry, T0, HALF_LIVES)
        assert value >= 0.0
        assert value == pytest.approx(0.0, abs=1e-12)

    def test_infinite_half_life_means_no_decay(self) -> None:
        """math.isinf -> factor 1.0 for a non-exempt entry (belt and braces)."""
        entry = make_entry(base_importance=0.8, last_accessed_at=T0 - timedelta(days=3650))
        assert effective_importance(entry, T0, {MemoryType.FACT: float("inf")}) == 0.8

    def test_entry_at_exact_threshold_survives_evict(self) -> None:
        """Eviction is strictly < threshold: an entry AT the threshold survives.

        base 0.8 at age 3 x half-life gives exactly 0.8 * 0.125 == 0.1 in
        IEEE-754 (mantissa unchanged, exponent shifted) — equal to the default
        eviction_threshold, so it must NOT be evicted; one hour older IS.
        """
        half_life = HALF_LIVES[MemoryType.FACT]
        at_threshold = make_entry(
            entry_id="at",
            base_importance=0.8,
            last_accessed_at=T0 - timedelta(hours=3 * half_life),
        )
        just_below = make_entry(
            entry_id="below",
            base_importance=0.8,
            last_accessed_at=T0 - timedelta(hours=3 * half_life + 1),
        )
        config = LifecycleConfig()
        assert effective_importance(at_threshold, T0, config.half_life_hours) == 0.1
        store = FakeEvictionStore([at_threshold, just_below])
        report = evict(store, config, now=T0)
        assert report.entries_evicted == 1
        assert not at_threshold.archived
        assert just_below.archived
        assert [call[0] for call in store.archive_calls] == ["below"]

    def test_empty_store_yields_zero_report(self) -> None:
        report = evict(FakeEvictionStore([]), LifecycleConfig(), now=T0)
        assert report == DecayReport(0, 0, 0)

    def test_populate_importance_empty_iterable_returns_empty_list(self) -> None:
        """QA gap: the documented list return holds for zero entries too."""
        result = populate_importance(iter([]), T0, HALF_LIVES)
        assert result == []
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Basic behavior — the formula
# ---------------------------------------------------------------------------


class TestFormula:
    def test_exact_half_life_points_fact(self) -> None:
        base = 0.8
        entry = make_entry(base_importance=base, last_accessed_at=T0)
        assert effective_importance(entry, T0, HALF_LIVES) == base
        assert effective_importance(entry, T0 + timedelta(hours=336), HALF_LIVES) == pytest.approx(
            0.4
        )
        assert effective_importance(entry, T0 + timedelta(hours=672), HALF_LIVES) == pytest.approx(
            0.2
        )

    @pytest.mark.parametrize(
        ("type_", "half_life"),
        [
            (MemoryType.FACT, 336.0),
            (MemoryType.OBSERVATION, 168.0),
            (MemoryType.PLAN, 72.0),
            (MemoryType.SUMMARY, 720.0),
        ],
    )
    def test_canonical_half_life_per_type(self, type_: MemoryType, half_life: float) -> None:
        """One half-life of age halves the base — per the canonical §2.1 table."""
        entry = make_entry(type_=type_, base_importance=0.6, last_accessed_at=T0)
        now = T0 + timedelta(hours=half_life)
        assert effective_importance(entry, now, HALF_LIVES) == pytest.approx(0.3)

    def test_per_type_divergence_plan_decays_faster_than_summary(self) -> None:
        """Identical base and age: PLAN (72 h) decays far more than SUMMARY (720 h)."""
        now = T0 + timedelta(hours=144)
        plan = make_entry(type_=MemoryType.PLAN, base_importance=0.8, last_accessed_at=T0)
        summary = make_entry(type_=MemoryType.SUMMARY, base_importance=0.8, last_accessed_at=T0)
        plan_value = effective_importance(plan, now, HALF_LIVES)
        summary_value = effective_importance(summary, now, HALF_LIVES)
        assert plan_value == pytest.approx(0.2)  # two half-lives
        assert summary_value > 0.6  # a fifth of one half-life
        assert plan_value < summary_value

    def test_never_accessed_decays_from_creation(self) -> None:
        """last_accessed_at normalizes to created_at — not epoch, not exempt."""
        entry = make_entry(base_importance=0.8, created_at=T0, last_accessed_at=None)
        assert entry.last_accessed_at == T0
        value = effective_importance(entry, T0 + timedelta(hours=336), HALF_LIVES)
        assert value == pytest.approx(0.4)

    def test_access_resets_the_decay_clock(self) -> None:
        entry = make_entry(base_importance=0.8, last_accessed_at=T0)
        t1 = T0 + timedelta(hours=500)
        entry.touch(now=t1)
        assert effective_importance(entry, t1, HALF_LIVES) == 0.8

    def test_decision_never_decays_even_with_finite_dict_value(self) -> None:
        """Code exemption wins; the mapping is never consulted for DECISION."""
        entry = make_entry(
            type_=MemoryType.DECISION,
            base_importance=0.9,
            last_accessed_at=T0 - timedelta(days=3650),
        )
        hostile: Mapping[MemoryType, float] = {MemoryType.DECISION: 1.0}
        assert effective_importance(entry, T0, hostile) == 0.9

    def test_pinned_extends_the_exemption_and_unpinned_twin_decays(self) -> None:
        ancient = T0 - timedelta(days=3650)
        pinned = make_entry(
            entry_id="pinned", base_importance=0.9, pinned=True, last_accessed_at=ancient
        )
        plain = make_entry(entry_id="plain", base_importance=0.9, last_accessed_at=ancient)
        assert effective_importance(pinned, T0, HALF_LIVES) == 0.9
        assert effective_importance(plain, T0, HALF_LIVES) < 0.001

    def test_is_decay_exempt_truth_table(self) -> None:
        assert is_decay_exempt(make_entry(pinned=True)) is True
        assert is_decay_exempt(make_entry(type_=MemoryType.DECISION)) is True
        assert is_decay_exempt(make_entry(type_=MemoryType.DECISION, pinned=True)) is True
        assert is_decay_exempt(make_entry(type_=MemoryType.FACT)) is False

    def test_purity_no_mutation_after_repeated_calls(self) -> None:
        """Decay-side half of the idempotency regression: nothing is written."""
        entry = make_entry(base_importance=0.8, last_accessed_at=T0)
        before = entry.to_dict()
        for _ in range(50):
            effective_importance(entry, T0 + timedelta(hours=100), HALF_LIVES)
        assert entry.base_importance == 0.8
        assert entry.last_accessed_at == T0
        assert entry.access_count == 0
        assert entry.importance is None
        assert entry.to_dict() == before

    def test_determinism_identical_inputs_identical_float(self) -> None:
        entry = make_entry(base_importance=0.8, last_accessed_at=T0)
        now = T0 + timedelta(hours=123, minutes=45)
        values = {effective_importance(entry, now, HALF_LIVES) for _ in range(100)}
        assert len(values) == 1

    def test_populate_importance_fills_the_derived_slot(self) -> None:
        now = T0 + timedelta(hours=336)
        entries = [
            make_entry(entry_id="a", base_importance=0.8, last_accessed_at=T0),
            make_entry(entry_id="b", type_=MemoryType.DECISION, base_importance=0.9),
        ]
        stored_before = [entry.to_dict() for entry in entries]
        for snapshot in stored_before:
            snapshot.pop("importance", None)
        result = populate_importance(entries, now, HALF_LIVES)
        assert isinstance(result, list)
        assert result[0].importance == effective_importance(entries[0], now, HALF_LIVES)
        assert result[1].importance == 0.9
        # Stored fields untouched; only the derived slot changed.
        for entry, before in zip(result, stored_before, strict=True):
            after = entry.to_dict()
            after.pop("importance", None)
            assert after == before
        # Repeated calls at the same now produce identical values.
        again = populate_importance(entries, now, HALF_LIVES)
        assert [e.importance for e in again] == [e.importance for e in result]


# ---------------------------------------------------------------------------
# Basic behavior — eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_below_threshold_entry_is_archived_with_evicted_reason(self) -> None:
        doomed = make_entry(
            entry_id="doomed", base_importance=0.2, last_accessed_at=T0 - timedelta(days=3650)
        )
        fresh = make_entry(entry_id="fresh", base_importance=0.9, last_accessed_at=T0)
        store = FakeEvictionStore([doomed, fresh])
        report = evict(store, LifecycleConfig(), now=T0)
        assert report.entries_evicted == 1
        assert store.archive_calls == [("doomed", ArchiveReason.EVICTED, T0)]
        assert doomed.archived is True
        assert doomed.archive_reason is ArchiveReason.EVICTED
        assert fresh.archived is False

    def test_audit_regression_pinned_and_decision_survive_at_floor(self) -> None:
        """MANDATORY AUDIT REGRESSION: exemption at effective importance ~0.

        A DECISION and a pinned FACT with ancient last_accessed_at survive
        evict() while a comparable non-exempt twin IS evicted — so a mutant
        that drops either exemption, or the whole exemption branch, fails.
        """
        ancient = T0 - timedelta(days=3650)
        decision = make_entry(
            entry_id="decision",
            type_=MemoryType.DECISION,
            base_importance=0.5,
            last_accessed_at=ancient,
        )
        pinned_fact = make_entry(
            entry_id="pinned-fact", base_importance=0.5, pinned=True, last_accessed_at=ancient
        )
        plain_twin = make_entry(
            entry_id="plain-twin", base_importance=0.5, last_accessed_at=ancient
        )
        store = FakeEvictionStore([decision, pinned_fact, plain_twin])
        report = evict(store, LifecycleConfig(), now=T0)
        assert report.entries_exempted == 2
        assert report.entries_evicted == 1
        assert decision.archived is False
        assert pinned_fact.archived is False
        assert plain_twin.archived is True
        assert [call[0] for call in store.archive_calls] == ["plain-twin"]

    def test_audit_regression_exempt_base_below_threshold_survives(self) -> None:
        """QA mutation-hardening: exemption must run BEFORE the threshold.

        ``effective_importance`` itself early-returns ``base_importance`` for
        exempt entries, so an exempt entry with base ABOVE the threshold
        survives even if evict() checked the threshold first — that mutant
        escapes the other regression tests. Only an exempt entry whose BASE
        importance is already below the threshold distinguishes the ordering:
        skipped-before-threshold survives; threshold-first evicts it.
        """
        ancient = T0 - timedelta(days=3650)
        low_pinned = make_entry(
            entry_id="low-pinned", base_importance=0.05, pinned=True, last_accessed_at=ancient
        )
        low_decision = make_entry(
            entry_id="low-decision",
            type_=MemoryType.DECISION,
            base_importance=0.05,
            last_accessed_at=ancient,
        )
        low_plain = make_entry(entry_id="low-plain", base_importance=0.05, last_accessed_at=T0)
        store = FakeEvictionStore([low_pinned, low_decision, low_plain])
        config = LifecycleConfig()
        assert low_pinned.base_importance < config.eviction_threshold
        report = evict(store, config, now=T0)
        assert low_pinned.archived is False
        assert low_decision.archived is False
        assert low_plain.archived is True  # fresh but base < threshold: evicted
        assert report == DecayReport(entries_scanned=3, entries_evicted=1, entries_exempted=2)
        assert [call[0] for call in store.archive_calls] == ["low-plain"]

    def test_evict_never_rewrites_stored_importance_of_aged_survivor(self) -> None:
        """QA mutation-hardening: a mid-decay survivor stays bit-identical.

        The other purity tests use age-0 survivors, for which a mutant that
        persists the decayed value back (``base_importance = effective``) is
        an invisible identity write. An AGED survivor (one half-life: 0.8 ->
        effective 0.4, above threshold) exposes it: stored state must be
        untouched and repeated passes at the same ``now`` must not compound.
        """
        aged = make_entry(
            entry_id="aged",
            base_importance=0.8,
            last_accessed_at=T0 - timedelta(hours=HALF_LIVES[MemoryType.FACT]),
        )
        snapshot = aged.to_dict()
        store = FakeEvictionStore([aged])
        config = LifecycleConfig()
        for _ in range(3):
            report = evict(store, config, now=T0)
            assert report == DecayReport(entries_scanned=1, entries_evicted=0, entries_exempted=0)
            assert aged.base_importance == 0.8
            assert aged.to_dict() == snapshot
            assert effective_importance(aged, T0, config.half_life_hours) == pytest.approx(0.4)
        assert store.archive_calls == []

    def test_audit_regression_exempt_only_population_evicts_nothing(self) -> None:
        """Blueprint test 21 verbatim: only exempt entries -> zero evictions."""
        ancient = T0 - timedelta(days=3650)
        decision = make_entry(
            entry_id="d", type_=MemoryType.DECISION, base_importance=0.5, last_accessed_at=ancient
        )
        pinned_fact = make_entry(
            entry_id="p", base_importance=0.5, pinned=True, last_accessed_at=ancient
        )
        store = FakeEvictionStore([decision, pinned_fact])
        report = evict(store, LifecycleConfig(), now=T0)
        assert report == DecayReport(entries_scanned=2, entries_evicted=0, entries_exempted=2)
        assert store.archive_calls == []

    def test_audit_regression_idempotent_across_repeated_runs(self) -> None:
        """MANDATORY AUDIT REGRESSION: repeated startups never compound decay."""
        ancient = T0 - timedelta(days=3650)
        population = [
            make_entry(entry_id="doomed-1", base_importance=0.3, last_accessed_at=ancient),
            make_entry(entry_id="doomed-2", base_importance=0.2, last_accessed_at=ancient),
            make_entry(entry_id="survivor", base_importance=0.9, last_accessed_at=T0),
            make_entry(
                entry_id="pinned", base_importance=0.4, pinned=True, last_accessed_at=ancient
            ),
            make_entry(
                entry_id="decision",
                type_=MemoryType.DECISION,
                base_importance=0.4,
                last_accessed_at=ancient,
            ),
        ]
        store = FakeEvictionStore(population)
        config = LifecycleConfig()

        first = evict(store, config, now=T0)
        assert first.entries_scanned == 5
        assert first.entries_evicted == 2
        survivors_after_first = {
            entry_id: entry.to_dict()
            for entry_id, entry in store.entries.items()
            if not entry.archived
        }

        second = evict(store, config, now=T0)
        third = evict(store, config, now=T0)
        for later in (second, third):
            assert later.entries_scanned == 3  # initial - K
            assert later.entries_evicted == 0
            assert later.entries_exempted == 2
        # Surviving entries are bit-identical across runs: base_importance
        # was never rewritten, so nothing can compound.
        for entry_id, snapshot in survivors_after_first.items():
            assert store.entries[entry_id].to_dict() == snapshot

    def test_fresh_above_threshold_exempt_entries_still_counted_exempted(self) -> None:
        """entries_exempted counts EVERY exempt entry, independent of the
        threshold: fresh pinned and DECISION entries whose hypothetical
        decayed value is far ABOVE the threshold are still skipped-and-
        counted, never threshold-evaluated (blueprint §DecayReport semantics).
        """
        fresh_pinned = make_entry(
            entry_id="fresh-pinned", base_importance=0.9, pinned=True, last_accessed_at=T0
        )
        fresh_decision = make_entry(
            entry_id="fresh-decision",
            type_=MemoryType.DECISION,
            base_importance=0.9,
            last_accessed_at=T0,
        )
        plain_fresh = make_entry(entry_id="plain-fresh", base_importance=0.9, last_accessed_at=T0)
        store = FakeEvictionStore([fresh_pinned, fresh_decision, plain_fresh])
        report = evict(store, LifecycleConfig(), now=T0)
        assert report == DecayReport(entries_scanned=3, entries_evicted=0, entries_exempted=2)
        assert store.archive_calls == []

    def test_scan_is_not_an_access(self) -> None:
        """evict() never touches last_accessed_at or access_count."""
        ancient = T0 - timedelta(days=3650)
        doomed = make_entry(entry_id="doomed", base_importance=0.2, last_accessed_at=ancient)
        survivor = make_entry(entry_id="survivor", base_importance=0.9, last_accessed_at=T0)
        store = FakeEvictionStore([doomed, survivor])
        evict(store, LifecycleConfig(), now=T0)
        assert doomed.last_accessed_at == ancient
        assert survivor.last_accessed_at == T0
        assert doomed.access_count == 0
        assert survivor.access_count == 0

    def test_report_arithmetic(self) -> None:
        ancient = T0 - timedelta(days=3650)
        entries = [
            make_entry(entry_id="e1", base_importance=0.2, last_accessed_at=ancient),
            make_entry(entry_id="e2", base_importance=0.9, last_accessed_at=T0),
            make_entry(entry_id="e3", base_importance=0.9, last_accessed_at=T0),
            make_entry(entry_id="e4", pinned=True, last_accessed_at=ancient),
        ]
        store = FakeEvictionStore(entries)
        report = evict(store, LifecycleConfig(), now=T0)
        survivors = report.entries_scanned - report.entries_evicted - report.entries_exempted
        assert report.entries_scanned == 4
        assert report.entries_evicted == 1
        assert report.entries_exempted == 1
        assert survivors == 2

    def test_default_clock_seam_uses_utcnow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """evict() without now= uses entry.utcnow() — the single clock seam."""
        patched_instant = T0 + timedelta(hours=1)
        monkeypatch.setattr("tulving.context.decay.utcnow", lambda: patched_instant)
        doomed = make_entry(base_importance=0.2, last_accessed_at=T0 - timedelta(days=3650))
        store = FakeEvictionStore([doomed])
        evict(store, LifecycleConfig())
        assert store.archive_calls == [(doomed.id, ArchiveReason.EVICTED, patched_instant)]


# ---------------------------------------------------------------------------
# Serialization / report shape
# ---------------------------------------------------------------------------


class TestDecayReportShape:
    def test_frozen(self) -> None:
        report = DecayReport(1, 0, 0)
        with pytest.raises(FrozenInstanceError):
            report.entries_scanned = 2  # type: ignore[misc]

    def test_no_entries_summarized_attribute(self) -> None:
        """Pins the revision-plan §4 removal: decay does not summarize."""
        report = DecayReport(0, 0, 0)
        assert not hasattr(report, "entries_summarized")

    def test_field_inventory(self) -> None:
        report = DecayReport(entries_scanned=3, entries_evicted=1, entries_exempted=2)
        assert report.entries_scanned == 3
        assert report.entries_evicted == 1
        assert report.entries_exempted == 2

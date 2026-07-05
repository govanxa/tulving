"""Tests for tulving.context.config — written BEFORE implementation.

Pins the D12 canonical policy table (specs/context-decay-eviction.md §2.1),
the merge-onto-defaults semantics, and every ConfigError branch. Created at
build step 9 (memory.py needs `LifecycleConfig`; the decay blueprint says the
module is "available at whichever build step needs it first").
"""

from datetime import timedelta
from typing import Any

import pytest

from tulving.context.config import DEFAULT_HALF_LIFE_HOURS, LifecycleConfig
from tulving.enums import MemoryType
from tulving.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Failure paths first
# ---------------------------------------------------------------------------


class TestFailurePaths:
    @pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
    def test_bad_half_life_values(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours={MemoryType.FACT: bad})

    def test_non_numeric_half_life(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours={MemoryType.FACT: "168"})  # type: ignore[dict-item]

    def test_bool_half_life_rejected(self) -> None:
        # bool is an int subclass — explicitly rejected.
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours={MemoryType.FACT: True})

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours={"fact": 1.0})  # type: ignore[dict-item]

    def test_finite_decision_rejected(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours={MemoryType.DECISION: 24.0})

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_bad_eviction_threshold(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(eviction_threshold=bad)

    def test_bad_inactivity_threshold(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(inactivity_threshold=timedelta(0))

    def test_bad_staleness_days(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(staleness_threshold_days=0)

    @pytest.mark.parametrize("bad", [1.0, -0.1])
    def test_bad_token_safety_margin(self, bad: float) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(token_safety_margin=bad)

    def test_bad_llm_call_budget(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(llm_call_budget=0)

    def test_bad_max_input_tokens(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(max_input_tokens=255)

    def test_config_error_names_type_and_value(self) -> None:
        with pytest.raises(ConfigError) as excinfo:
            LifecycleConfig(half_life_hours={MemoryType.PLAN: -3.0})
        message = str(excinfo.value)
        assert "plan" in message.lower()
        assert "-3" in message


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_legal_boundaries(self) -> None:
        LifecycleConfig(eviction_threshold=0.0)
        LifecycleConfig(eviction_threshold=1.0)
        LifecycleConfig(token_safety_margin=0.0)
        LifecycleConfig(half_life_hours={MemoryType.FACT: float("inf")})
        LifecycleConfig(llm_call_budget=1)
        LifecycleConfig(max_input_tokens=256)


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_defaults_contract_d12(self) -> None:
        config = LifecycleConfig()
        assert config.half_life_hours == {
            MemoryType.FACT: 336.0,
            MemoryType.DECISION: float("inf"),
            MemoryType.OBSERVATION: 168.0,
            MemoryType.PLAN: 72.0,
            MemoryType.SUMMARY: 720.0,
        }
        assert config.eviction_threshold == 0.1
        assert config.inactivity_threshold == timedelta(minutes=30)
        assert config.summarize_on_session_end is True
        assert config.staleness_threshold_days == 30
        assert config.token_safety_margin == 0.15
        # L7 additions (blueprint-memory §Interface assumptions).
        assert config.preserve_decisions_verbatim is True
        assert config.llm_call_budget == 10
        assert config.max_input_tokens == 4000

    def test_merge_semantics(self) -> None:
        overrides: dict[MemoryType, float] = {MemoryType.PLAN: 24.0}
        config = LifecycleConfig(half_life_hours=overrides)
        assert config.half_life_hours[MemoryType.PLAN] == 24.0
        assert config.half_life_hours[MemoryType.FACT] == 336.0
        # Caller's dict is never aliased.
        overrides[MemoryType.PLAN] = 1.0
        assert config.half_life_hours[MemoryType.PLAN] == 24.0
        # And the module defaults are unchanged.
        assert DEFAULT_HALF_LIFE_HOURS[MemoryType.PLAN] == 72.0

    def test_no_shared_state_between_instances(self) -> None:
        first = LifecycleConfig()
        second = LifecycleConfig()
        first.half_life_hours[MemoryType.FACT] = 1.0
        assert second.half_life_hours[MemoryType.FACT] == 336.0

    def test_defaults_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            DEFAULT_HALF_LIFE_HOURS[MemoryType.FACT] = 1.0  # type: ignore[index]

    def test_every_type_present_after_merge(self) -> None:
        config = LifecycleConfig(half_life_hours={MemoryType.PLAN: 24.0})
        for member in MemoryType:
            assert member in config.half_life_hours

    def test_error_is_config_error_not_value_error(self) -> None:
        bad: dict[Any, Any] = {MemoryType.FACT: 0.0}
        with pytest.raises(ConfigError):
            LifecycleConfig(half_life_hours=bad)

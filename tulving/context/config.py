"""Lifecycle policy configuration (D12): the single code home for decay,
eviction, session, staleness, and token-margin knobs.

Default values mirror specs/context-decay-eviction.md §2.1 — the canonical
policy table. Numbers live HERE and nowhere else in the package. Created at
build step 9 (``memory.py`` needs ``LifecycleConfig`` first; the decay
blueprint sanctions creation "at whichever build step needs it first").

As-built divergences from blueprint-lifecycle (see the "As-built note from
build step 9" appended to that blueprint — reconcile before implementing
step 13): ``startup_deadline_seconds`` lives on the ``Memory`` constructor
(default 10.0), not here; ``activity_debounce_seconds`` is omitted until
step 13 implements activity debouncing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from types import MappingProxyType
from typing import Final

from tulving.enums import MemoryType
from tulving.exceptions import ConfigError

#: Canonical per-type half-lives (D2/D12). Read-only; copy before mutating.
DEFAULT_HALF_LIFE_HOURS: Final[Mapping[MemoryType, float]] = MappingProxyType(
    {
        MemoryType.FACT: 336.0,  # 14 days
        MemoryType.DECISION: float("inf"),  # never decays (ADR-006; enforced in code too)
        MemoryType.OBSERVATION: 168.0,  # 7 days
        MemoryType.PLAN: 72.0,  # 3 days
        MemoryType.SUMMARY: 720.0,  # 30 days
    }
)


@dataclass
class LifecycleConfig:
    """Lifecycle policy — shape per architecture.md §3 plus the L7 additions
    (blueprint-memory §Interface assumptions: ``preserve_decisions_verbatim``,
    ``llm_call_budget``, ``max_input_tokens``)."""

    half_life_hours: dict[MemoryType, float] = field(default_factory=dict)
    # ^ user-supplied entries are OVERRIDES: __post_init__ merges them onto
    #   DEFAULT_HALF_LIFE_HOURS, so a partial dict ({PLAN: 24.0}) is legal and
    #   every MemoryType is guaranteed present afterwards.
    eviction_threshold: float = 0.1
    inactivity_threshold: timedelta = timedelta(minutes=30)
    summarize_on_session_end: bool = True  # no-op + logged warning when llm=None
    staleness_threshold_days: int = 30
    token_safety_margin: float = 0.15
    preserve_decisions_verbatim: bool = True
    llm_call_budget: int = 10  # per summarize pass (adapters-llm §2.1)
    max_input_tokens: int = 4000  # per LLM call, pre-margin

    def __post_init__(self) -> None:
        """Merge half-life overrides onto defaults, then validate.

        Raises:
            ConfigError: On any violation. Never raises ValueError — config
                problems are user-facing, not programming errors.
        """
        overrides = self.half_life_hours
        for key in overrides:
            if not isinstance(key, MemoryType):
                raise ConfigError(f"half_life_hours keys must be MemoryType members, got {key!r}")
        merged: dict[MemoryType, float] = dict(DEFAULT_HALF_LIFE_HOURS)
        merged.update(overrides)
        self.half_life_hours = merged
        for member, value in merged.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(
                    f"half-life for {member.value!r} must be a positive number, got {value!r}"
                )
            if math.isnan(value) or value <= 0:
                raise ConfigError(
                    f"half-life for {member.value!r} must be a positive number, got {value!r}"
                )
        if not math.isinf(merged[MemoryType.DECISION]):
            raise ConfigError(
                "DECISION half-life must be float('inf'): decisions never decay "
                "(ADR-006); a finite value would be silently ignored"
            )
        if not 0.0 <= self.eviction_threshold <= 1.0:
            raise ConfigError(
                f"eviction_threshold must be within [0.0, 1.0], got {self.eviction_threshold!r}"
            )
        if self.inactivity_threshold <= timedelta(0):
            raise ConfigError(
                f"inactivity_threshold must be positive, got {self.inactivity_threshold!r}"
            )
        if self.staleness_threshold_days < 1:
            raise ConfigError(
                f"staleness_threshold_days must be >= 1, got {self.staleness_threshold_days!r}"
            )
        if not 0.0 <= self.token_safety_margin < 1.0:
            raise ConfigError(
                f"token_safety_margin must be within [0.0, 1.0), got {self.token_safety_margin!r}"
            )
        if self.llm_call_budget < 1:
            raise ConfigError(f"llm_call_budget must be >= 1, got {self.llm_call_budget!r}")
        if self.max_input_tokens < 256:
            raise ConfigError(f"max_input_tokens must be >= 256, got {self.max_input_tokens!r}")

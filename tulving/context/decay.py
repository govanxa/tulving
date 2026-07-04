"""Lazy decay + reason-aware eviction (revision-plan D2).

The decay formula lives HERE and only here. It is a pure function of time,
computed on read, never persisted: reading an entry N times or restarting N
times leaves stored ``base_importance`` bit-identical. The only write this
module performs is ``evict()``'s archive call — archive state, never
importance values.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from tulving.context.config import LifecycleConfig
from tulving.entry import MemoryEntry, utcnow
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecayReport:
    """Outcome of one eviction pass.

    There is deliberately NO ``entries_summarized`` field — decay does not
    summarize (revision-plan §4; the LifecycleManager orchestrates
    summarization separately).
    """

    entries_scanned: int
    """Active entries examined (exempt ones included)."""

    entries_evicted: int
    """Archived with ``ArchiveReason.EVICTED`` this pass."""

    entries_exempted: int
    """Pinned / DECISION entries skipped before any threshold evaluation."""


class EvictionStore(Protocol):
    """Minimal store surface ``evict()`` needs.

    ``store.py`` / ``memory.py`` satisfy this structurally (no inheritance,
    no import of this module required).
    """

    def iter_active_entries(self) -> Iterator[MemoryEntry]:
        """Yield every non-archived entry for this memory path.

        Must NOT count as an access: no ``touch()``, no ``last_accessed_at``
        updates.
        """
        ...

    def archive_entry(self, entry_id: str, reason: ArchiveReason, *, now: datetime) -> None:
        """Archive one entry: ``archived=True``, ``archive_reason=reason``.

        Persists via the storage backend (DB write; the semantic index
        excludes archived vectors at query time per ADR-015). ``now`` is the
        evaluation instant of the eviction pass; implementations stamp
        ``updated_at`` from the same clock SOURCE, which under a wall clock
        may trail ``now`` by the scan latency — the two are identical only
        under an injected fixed clock.

        Raises:
            MemoryStoreError: When the row cannot be archived.
            StorageError: On backend failure.
        """
        ...


def is_decay_exempt(entry: MemoryEntry) -> bool:
    """True when the entry never decays and never evicts (D2/D6).

    ``entry.pinned`` or ``entry.type is MemoryType.DECISION``. Exported: the
    curator uses the same predicate for ranking annotations.

    Args:
        entry: The entry to test.

    Returns:
        Whether the entry is exempt from decay and eviction.
    """
    return entry.pinned or entry.type is MemoryType.DECISION


def _hours_since_access(entry: MemoryEntry, now: datetime) -> float:
    """Age in hours since last access, clamped to >= 0 (clock skew)."""
    anchor = entry.last_accessed_at
    if anchor is None:  # pragma: no cover - normalized in MemoryEntry.__post_init__
        anchor = entry.created_at
    return max((now - anchor).total_seconds() / 3600.0, 0.0)


def _half_life_for(type_: MemoryType, mapping: Mapping[MemoryType, float]) -> float:
    """Look up a half-life with ConfigError defense (mutated-dict paths)."""
    half_life = mapping.get(type_)
    if half_life is None:
        raise ConfigError(f"no half-life configured for type {type_.value!r}")
    if isinstance(half_life, bool) or math.isnan(half_life) or half_life <= 0:
        raise ConfigError(
            f"half-life for {type_.value!r} must be a positive number, got {half_life!r}"
        )
    return half_life


def effective_importance(
    entry: MemoryEntry,
    now: datetime,
    half_life_hours: Mapping[MemoryType, float],
) -> float:
    """THE decay formula (D2) — pure, no side effects, nothing written.

    ``effective = base_importance * 0.5 ** (hours_since_last_access
    / half_life[entry.type])``. Exempt entries (pinned / DECISION) return
    ``base_importance`` unchanged without consulting the mapping; ages are
    clamped to >= 0 so clock skew never amplifies importance; an infinite
    half-life yields factor 1.0. Does NOT mutate the entry — callers assign
    the result to ``entry.importance`` themselves or use
    :func:`populate_importance`.

    Args:
        entry: The entry to score. May be archived (the formula still
            applies; filtering archived entries is the caller's concern).
        now: Injected clock — REQUIRED and timezone-aware. Pure functions
            have no hidden clock; tests never sleep.
        half_life_hours: Per-type half-lives, normally
            ``LifecycleConfig.half_life_hours`` (post-validation).

    Returns:
        The effective (decayed) importance — deterministic for identical
        inputs.

    Raises:
        ValueError: On a naive ``now`` (programming error, entry.py style).
        ConfigError: When ``entry.type`` is missing from the mapping, or a
            non-positive/NaN half-life is encountered (defense in depth —
            normally impossible after ``LifecycleConfig`` validation, but
            the mapping is a plain dict and could have been mutated).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (naive datetime rejected)")
    if is_decay_exempt(entry):
        return entry.base_importance
    half_life = _half_life_for(entry.type, half_life_hours)
    if math.isinf(half_life):
        return entry.base_importance
    factor: float = 0.5 ** (_hours_since_access(entry, now) / half_life)
    return entry.base_importance * factor


def populate_importance(
    entries: Iterable[MemoryEntry],
    now: datetime,
    half_life_hours: Mapping[MemoryType, float],
) -> list[MemoryEntry]:
    """Convenience for read paths: fill each entry's derived importance slot.

    Sets ``entry.importance`` (the derived, ``compare=False`` slot) via
    :func:`effective_importance` and returns the entries as a list. Mutates
    ONLY the derived slot; stored fields are untouched.

    Args:
        entries: Entries to annotate.
        now: Injected clock — required and timezone-aware.
        half_life_hours: Per-type half-lives.

    Returns:
        The same entries, as a list, with ``importance`` populated.

    Raises:
        ValueError: On a naive ``now``.
        ConfigError: Same conditions as :func:`effective_importance`.
    """
    result = list(entries)
    for entry in result:
        entry.importance = effective_importance(entry, now, half_life_hours)
    return result


def evict(
    store: EvictionStore,
    config: LifecycleConfig,
    *,
    now: datetime | None = None,
) -> DecayReport:
    """Reason-aware eviction pass (D2) — the ONLY writer in this module.

    Archives every non-exempt active entry whose effective importance is
    strictly below ``config.eviction_threshold`` (an entry AT the threshold
    survives) with ``ArchiveReason.EVICTED``. Pinned and DECISION entries are
    skipped BEFORE any threshold evaluation, so they survive even at
    effective importance ~0. Idempotent by construction: nothing about
    importance is written, evicted entries leave the active set, and repeated
    runs at the same ``now`` evict nothing further. The scan is NOT an
    access: it never calls ``touch()`` and never resets the decay clock.

    Args:
        store: Minimal store surface (see :class:`EvictionStore`).
        config: Validated lifecycle policy (threshold + half-lives).
        now: Injected clock; defaults to ``entry.utcnow()`` — the single
            clock seam, at the orchestration boundary only. Tests always
            inject.

    Returns:
        A :class:`DecayReport` with scan/evict/exempt counts.

    Raises:
        ValueError: On a naive injected ``now`` — checked upfront so an
            empty or all-exempt scan cannot mask the programming error.
        ConfigError: Propagated — bad config must be loud.
        MemoryStoreError: Propagated from ``archive_entry`` (the caller —
            ``Memory.startup()`` — owns time-boxing and non-fatality, D8).
        StorageError: Propagated from ``archive_entry``.
    """
    instant = now if now is not None else utcnow()
    if instant.tzinfo is None:
        raise ValueError("now must be timezone-aware (naive datetime rejected)")
    scanned = evicted = exempted = 0
    for entry in store.iter_active_entries():
        scanned += 1
        if is_decay_exempt(entry):
            exempted += 1  # skipped BEFORE threshold evaluation
            continue
        if effective_importance(entry, instant, config.half_life_hours) < (
            config.eviction_threshold
        ):
            store.archive_entry(entry.id, ArchiveReason.EVICTED, now=instant)
            evicted += 1
            logger.debug("evicted entry %s (effective importance below threshold)", entry.id)
    logger.info("eviction pass: scanned=%d evicted=%d exempted=%d", scanned, evicted, exempted)
    return DecayReport(entries_scanned=scanned, entries_evicted=evicted, entries_exempted=exempted)

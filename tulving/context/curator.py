"""Token-budget context curation — the headline primitive (build step 10).

The ``ContextCurator`` selects memories from up to three retrieval sources
(KV exact, semantic, recency), scores them by a weighted blend of
relevance/importance/recency, fits them greedily into a token budget,
formats them into a prompt-ready block, and redacts that block before it
leaves the module. It also owns cold-start orientation (``mode="orient"``).

Imports: stdlib + ``tulving.enums`` (``MemoryType``, ``MatchType``),
``tulving.entry`` (``MemoryEntry``, ``utcnow``), ``tulving.exceptions``
(``ConfigError``), ``tulving.security`` (``redact_text``,
``is_sensitive_key``). Optional at runtime, guarded, lazy: ``tiktoken``.
NEVER hnswlib, never an LLM adapter, never sqlite (blueprint-curator).

Decisions honored:

- **D2** — the curator never computes or stores decayed importance itself.
  It calls the injected :class:`ImportanceEvaluator` with an explicit
  ``now`` and assigns the result to the entry's derived ``importance`` slot
  only for *included* entries.
- **D3** — curator *inclusion* is an access: included entries get one batch
  ``record_access()`` (touch + ``potentially_stale`` auto-clear). Merely
  evaluated entries are never touched.
- **D6** — bad budgets, unknown modes, and invalid weights raise
  ``ConfigError``. No builtin-shadowing names.
- **D10 / security #1** — every emitted block passes through
  ``security.redact_text``; entries whose key is sensitive have their
  content masked entirely.
- **D12** — the only lifecycle number the curator owns is its recency
  half-life (a ranking knob). Half-lives live behind the evaluator; the
  safety margin arrives as a wired scalar.
- **D8** — the constructor validates scalars and stores references only; the
  tiktoken probe is lazy and memoized.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from tulving.entry import MemoryEntry, utcnow
from tulving.enums import MatchType, MemoryType
from tulving.exceptions import ConfigError
from tulving.security import is_sensitive_key, redact_text

logger = logging.getLogger("tulving.curator")

REDACTED_CONTENT: Final[str] = "[REDACTED]"  # pinned to security.REDACTED by a test
TRUNCATION_MARKER: Final[str] = "[… truncated by curator: {elided} tokens elided]"

_STALE_TAG: Final[str] = "potentially_stale"
_SESSION_MARKER_TAG: Final[str] = "session_marker"
_ORIENT_FETCH_LIMIT: Final[int] = 50
_ORIENT_SHARES: Final[tuple[float, ...]] = (0.20, 0.30, 0.25, 0.15, 0.10)
_ORIENT_SECTIONS: Final[tuple[str, ...]] = (
    "--- Session History ---",
    "--- Key Decisions ---",
    "--- Recent Knowledge ---",
    "--- Pinned ---",
    "--- Goal-Relevant ---",
)

# Memoized module-level fallback estimator (probed at most once). Reset in
# tests via monkeypatch to exercise the tiktoken resolution branches.
_default_estimator: TokenEstimator | None = None


# ---------------------------------------------------------------------------
# Token estimation (pluggable; tiktoken optional, never blocking)
# ---------------------------------------------------------------------------


class TokenEstimator(Protocol):
    """Estimate a token count for text (structurally shared with summarizer)."""

    def estimate(self, text: str) -> int:
        """Estimated token count: ``>= 0``, deterministic, roughly monotonic."""
        ...


class HeuristicEstimator:
    """``len(text) // 4`` fallback: 0 for empty text, at least 1 otherwise."""

    def estimate(self, text: str) -> int:
        """Coarse ``len // 4`` estimate; ``0`` for empty, floored at ``1``."""
        if not text:
            return 0
        return max(len(text) // 4, 1)


class TiktokenEstimator:
    """``cl100k_base`` wrapper. Constructed only by :func:`resolve_estimator`
    after a guarded import + encoding load succeeds."""

    def __init__(self, encoding: object) -> None:
        self._encoding = encoding

    def estimate(self, text: str) -> int:
        """Token count via the wrapped tiktoken encoding."""
        return len(self._encoding.encode(text))  # type: ignore[attr-defined]


def resolve_estimator(preferred: TokenEstimator | None = None) -> TokenEstimator:
    """Resolve the token estimator (blueprint/spec §5).

    Resolution order:

    1. ``preferred`` (caller-supplied), returned verbatim — never memoized.
    2. ``tiktoken`` ``cl100k_base`` via a guarded lazy import. Both the
       import and ``get_encoding()`` are wrapped: any failure (missing
       package, failed BPE fetch, corrupt cache) silently selects the
       fallback with a debug log. The probe runs at most once (module memo).
    3. :class:`HeuristicEstimator`.

    Args:
        preferred: A caller-supplied estimator to use unconditionally.

    Returns:
        The resolved :class:`TokenEstimator`.
    """
    if preferred is not None:
        return preferred
    global _default_estimator
    if _default_estimator is None:
        _default_estimator = _probe_default_estimator()
    return _default_estimator


def _probe_default_estimator() -> TokenEstimator:
    """Guarded, one-shot tiktoken probe; falls back to the heuristic."""
    try:
        import tiktoken  # type: ignore[import-not-found]

        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # token counting must never block (spec §5)
        logger.debug("tiktoken unavailable; using heuristic estimator", exc_info=True)
        return HeuristicEstimator()
    logger.debug("using tiktoken cl100k_base estimator")
    return TiktokenEstimator(encoding)


# ---------------------------------------------------------------------------
# Interfaces the curator consumes (faked in tests)
# ---------------------------------------------------------------------------


class ImportanceEvaluator(Protocol):
    """Pure effective-importance evaluator (satisfied by ``DecayManager``)."""

    def effective_importance(self, entry: MemoryEntry, now: datetime | None = None) -> float:
        """Effective (lazily decayed) importance of ``entry`` at ``now``."""
        ...


class RetrievalPort(Protocol):
    """Minimal, access-neutral retrieval surface the curator needs.

    All read methods return ACTIVE (non-archived) entries and must not update
    ``last_accessed_at`` / ``access_count`` (evaluation is not access, D3).
    """

    def lookup_key(self, key: str) -> MemoryEntry | None:
        """Exact active-key match, no touch."""
        ...

    def semantic_candidates(self, query: str, *, top_k: int) -> list[tuple[MemoryEntry, float]]:
        """Top-k semantic hits as ``(entry, score)``; ``[]`` when degraded."""
        ...

    def recent_entries(self, *, limit: int) -> list[MemoryEntry]:
        """Most recently created active entries, newest first."""
        ...

    def list_by(
        self,
        *,
        types: Sequence[MemoryType] | None = None,
        tags: Sequence[str] | None = None,
        pinned_only: bool = False,
        limit: int,
    ) -> list[MemoryEntry]:
        """Filtered active entries, newest-first (orient categories)."""
        ...

    def record_access(self, entry_ids: Sequence[str], *, now: datetime) -> None:
        """Batch touch + ``potentially_stale`` clear for included entries."""
        ...


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


@dataclass
class CuratedContext:
    """The curator's result: prompt-ready, redacted text plus accounting."""

    content: str
    """Formatted, REDACTED, prompt-ready text."""

    entries: list[MemoryEntry]
    """Included entries, in content order.

    SECURITY (SEC-001): these are RAW, UNREDACTED ``MemoryEntry`` objects —
    only ``content`` is the redacted egress string. Never emit ``entry.content``
    (or ``to_dict``) to an egress surface (MCP/export); emit ``content`` only,
    or re-run ``security.redact_text`` on anything derived from ``entries``.
    """

    token_count: int
    """Estimate of ``content`` (0 for a gracefully-empty result)."""

    budget_remaining: int
    """``token_budget - token_count``, never negative."""

    sources_consulted: int
    """Unique candidate entries evaluated (post-dedup, pre-filter)."""


# ---------------------------------------------------------------------------
# Internal candidate model (not exported)
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    entry: MemoryEntry
    relevance: float  # per-source, [0, 1]
    match: MatchType  # KEY | SEMANTIC | TEMPORAL — provenance
    score: float = 0.0  # filled by scoring (query mode)
    effective: float = 0.0  # importance.effective_importance(entry, now)


# ---------------------------------------------------------------------------
# The curator
# ---------------------------------------------------------------------------


class ContextCurator:
    """Token-budget context curation over the :class:`RetrievalPort`.

    Pure compute plus a single batch write (``record_access``); no locks, no
    hnswlib, no LLM, no sqlite. Determinism is a feature: identical store
    state + clock + arguments produce byte-identical ``content``.
    """

    def __init__(
        self,
        retrieval: RetrievalPort,
        importance: ImportanceEvaluator,
        *,
        token_safety_margin: float = 0.15,
        recency_half_life_hours: float = 168.0,
        key_patterns: Sequence[re.Pattern[str]] | None = None,
        estimator: TokenEstimator | None = None,
        semantic_top_k: int = 20,
        recent_limit: int = 10,
        now_fn: Callable[[], datetime] = utcnow,
    ) -> None:
        """Store references and validate scalars (cheap, D8: no I/O, no LLM).

        Args:
            retrieval: Access-neutral retrieval port.
            importance: Effective-importance evaluator (D2).
            token_safety_margin: Fraction of the budget held back as fitting
                slack; must be in ``[0.0, 1.0)``. Wired from ``LifecycleConfig``
                by ``memory.py`` (canonical default 0.15, architecture §3).
            recency_half_life_hours: Curator-local recency ranking knob
                (spec §3.3: 168 h); must be ``> 0``.
            key_patterns: Compiled sensitive-key patterns from
                ``Memory(sensitive_keys=...)``; ``None`` uses security defaults.
            estimator: Token estimator; ``None`` resolves lazily via
                :func:`resolve_estimator` (tiktoken → heuristic).
            semantic_top_k: Semantic candidate fetch size; must be ``>= 1``.
            recent_limit: Recency candidate fetch size; must be ``>= 1``.
            now_fn: Single clock seam (tests inject a fixed clock).

        Raises:
            ConfigError: ``token_safety_margin`` outside ``[0.0, 1.0)``;
                ``recency_half_life_hours <= 0``; ``semantic_top_k`` or
                ``recent_limit`` ``< 1``.
        """
        if not 0.0 <= token_safety_margin < 1.0:
            raise ConfigError(
                f"token_safety_margin must be within [0.0, 1.0), got {token_safety_margin!r}"
            )
        if recency_half_life_hours <= 0:
            raise ConfigError(
                f"recency_half_life_hours must be > 0, got {recency_half_life_hours!r}"
            )
        if semantic_top_k < 1:
            raise ConfigError(f"semantic_top_k must be >= 1, got {semantic_top_k!r}")
        if recent_limit < 1:
            raise ConfigError(f"recent_limit must be >= 1, got {recent_limit!r}")

        self._retrieval = retrieval
        self._importance = importance
        self._token_safety_margin = token_safety_margin
        self._recency_half_life_hours = recency_half_life_hours
        self._key_patterns = key_patterns
        self._preferred_estimator = estimator
        self._resolved_estimator: TokenEstimator | None = None
        self._semantic_top_k = semantic_top_k
        self._recent_limit = recent_limit
        self._now_fn = now_fn

    # ----------------------------------------------------------- public API

    def curate(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        mode: str = "query",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        include_types: list[MemoryType] | None = None,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.4,
    ) -> CuratedContext:
        """Select, rank, fit, format, and redact memories to a token budget.

        Args:
            query: The retrieval query; empty degrades to recency-only.
            token_budget: Hard token budget; must be ``> 0``.
            mode: ``"query"`` or ``"orient"``.
            include_tags: If given, keep only candidates carrying at least one.
            exclude_tags: Drop candidates carrying any of these (exclude wins).
            include_types: If given, keep only these memory types.
            recency_weight: Recency score weight; ``>= 0``.
            importance_weight: Importance score weight; ``>= 0``.
            relevance_weight: Relevance score weight; ``>= 0``.

        Returns:
            A :class:`CuratedContext` (content already redacted).

        Raises:
            ConfigError: ``token_budget <= 0``, unknown ``mode``, any negative
                weight, or all-zero weights.
        """
        if token_budget <= 0:
            raise ConfigError(f"token_budget must be > 0, got {token_budget}")
        if mode not in ("query", "orient"):
            raise ConfigError(f"unknown mode {mode!r}; expected 'query' or 'orient'")
        self._validate_weights(recency_weight, importance_weight, relevance_weight)
        now = self._now_fn()
        if mode == "orient":
            return self._curate_orient(
                query, token_budget, include_tags, exclude_tags, include_types, now
            )
        return self._curate_query(
            query,
            token_budget,
            include_tags,
            exclude_tags,
            include_types,
            recency_weight,
            importance_weight,
            relevance_weight,
            now,
        )

    def estimate_tokens(self, text: str) -> int:
        """Estimate ``text``'s token count via the resolved estimator (raw).

        The safety margin applies to budgets, never to estimates.
        """
        return self._estimate(text)

    # ------------------------------------------------------ query pipeline

    def _curate_query(
        self,
        query: str,
        token_budget: int,
        include_tags: list[str] | None,
        exclude_tags: list[str] | None,
        include_types: list[MemoryType] | None,
        recency_weight: float,
        importance_weight: float,
        relevance_weight: float,
        now: datetime,
    ) -> CuratedContext:
        deduped = self._dedup(self._gather_query(query))
        sources_consulted = len(deduped)
        survivors = [
            cand
            for cand in deduped.values()
            if self._passes_filters(cand.entry, include_tags, exclude_tags, include_types)
        ]
        total_w = recency_weight + importance_weight + relevance_weight
        for cand in survivors:
            cand.effective = self._importance.effective_importance(cand.entry, now)
            recency = self._recency(now, cand.entry)
            cand.score = (
                cand.relevance * relevance_weight
                + cand.effective * importance_weight
                + recency * recency_weight
            ) / total_w
        survivors.sort(key=lambda c: (-c.score, -c.entry.created_at.timestamp(), c.entry.id))

        content_budget = self._content_budget(token_budget, self._query_frame_cost())
        if content_budget < 1:
            return CuratedContext("", [], 0, token_budget, sources_consulted)

        included = self._fit_query(survivors, content_budget, now)
        if not included and survivors:
            # Candidates existed but nothing fit (even truncated) — graceful
            # empty, no side effects (spec §4 edge case 2 / oversize impossible).
            return CuratedContext("", [], 0, token_budget, sources_consulted)

        blocks = [block for _, block in included]
        content, token_count = self._assemble_query(blocks, len(included))
        return self._finish(included, content, token_count, token_budget, sources_consulted, now)

    def _gather_query(self, query: str) -> list[_Candidate]:
        """Gather candidates in the fixed order KV → semantic → recency."""
        gathered: list[_Candidate] = []
        stripped = query.strip()
        if stripped:
            kv_hit = self._retrieval.lookup_key(stripped)
            if kv_hit is not None:
                gathered.append(_Candidate(kv_hit, 1.0, MatchType.KEY))
            for entry, score in self._retrieval.semantic_candidates(
                stripped, top_k=self._semantic_top_k
            ):
                gathered.append(_Candidate(entry, score, MatchType.SEMANTIC))
        for entry in self._retrieval.recent_entries(limit=self._recent_limit):
            gathered.append(_Candidate(entry, 0.0, MatchType.TEMPORAL))
        return gathered

    @staticmethod
    def _dedup(gathered: list[_Candidate]) -> dict[str, _Candidate]:
        """Deduplicate by entry id, keeping the highest relevance (first-seen
        wins on ties; gather order is KV → semantic → recency)."""
        best: dict[str, _Candidate] = {}
        for cand in gathered:
            existing = best.get(cand.entry.id)
            if existing is None or cand.relevance > existing.relevance:
                best[cand.entry.id] = cand
        return best

    def _fit_query(
        self, survivors: list[_Candidate], content_budget: int, now: datetime
    ) -> list[tuple[_Candidate, str]]:
        """Greedy skip-if-doesn't-fit walk; oversize truncation only when the
        greedy walk includes nothing at all."""
        included: list[tuple[_Candidate, str]] = []
        remaining = content_budget
        for cand in survivors:
            block = self._render_block(cand, now)
            cost = self._estimate(block)
            if cost <= remaining:
                included.append((cand, block))
                remaining -= cost
        if included:
            return included
        if survivors:
            top = survivors[0]
            red_header, red_body = self._render_parts(top, now)
            block, ok = self._truncate_to_fit(red_header, red_body, content_budget)
            if ok:
                return [(top, block)]
        return []

    # ----------------------------------------------------- orient pipeline

    def _curate_orient(
        self,
        query: str,
        token_budget: int,
        include_tags: list[str] | None,
        exclude_tags: list[str] | None,
        include_types: list[MemoryType] | None,
        now: datetime,
    ) -> CuratedContext:
        # Gather BEFORE the budget check so a tiny-budget empty result still
        # reports the real evaluated count (parity with query mode, MINOR-2).
        categories, sources_consulted = self._gather_orient(
            query, include_tags, exclude_tags, include_types, now
        )
        content_budget = self._content_budget(token_budget, self._orient_frame_cost())
        if content_budget < 1:
            return CuratedContext("", [], 0, token_budget, sources_consulted)

        sections, included = self._fit_orient(categories, content_budget, now)
        if not included:
            flat = [(name, cand) for name, cands, _ in categories for cand in cands]
            if flat:
                name, top = flat[0]
                red_header, red_body = self._render_parts(top, now)
                block, ok = self._truncate_to_fit(
                    red_header, red_body, content_budget - self._estimate(name)
                )
                if ok:
                    sections = [(name, [block])]
                    included = [top]
                else:
                    return CuratedContext("", [], 0, token_budget, sources_consulted)
            # flat empty ⇒ nothing to include ⇒ header-only frame.

        content, token_count = self._assemble_orient(sections)
        pairs = [(cand, "") for cand in included]
        return self._finish(pairs, content, token_count, token_budget, sources_consulted, now)

    def _gather_orient(
        self,
        query: str,
        include_tags: list[str] | None,
        exclude_tags: list[str] | None,
        include_types: list[MemoryType] | None,
        now: datetime,
    ) -> tuple[list[tuple[str, list[_Candidate], float]], int]:
        """Gather the orient categories with cross-category dedup.

        An entry is charged to the first category that selected it. Returns the
        per-category surviving candidates plus the unique evaluated count
        (post cross-category dedup, pre tag/type filter).

        When ``query`` is empty the goal-relevant category (5) is absent — the
        COMMON cold-start path. The remaining categories' budget shares are
        renormalized to sum to 1.0 so 100% of ``content_budget`` stays usable
        (MINOR-1); the query-present path keeps the canonical five shares
        (which already sum to 1.0, so renormalization is a no-op there).
        """
        limit = _ORIENT_FETCH_LIMIT
        raw: list[list[tuple[MemoryEntry, float, MatchType]]] = [
            [
                (entry, 0.0, MatchType.TEMPORAL)
                for entry in self._retrieval.list_by(tags=[_SESSION_MARKER_TAG], limit=limit)
            ],
            [
                (entry, 0.0, MatchType.TEMPORAL)
                for entry in self._retrieval.list_by(types=[MemoryType.DECISION], limit=limit)
            ],
            [
                (entry, 0.0, MatchType.TEMPORAL)
                for entry in self._retrieval.list_by(types=[MemoryType.SUMMARY], limit=limit)
            ],
            [
                (entry, 0.0, MatchType.TEMPORAL)
                for entry in self._retrieval.list_by(pinned_only=True, limit=limit)
            ],
        ]
        if query.strip():
            raw.append(
                [
                    (entry, score, MatchType.SEMANTIC)
                    for entry, score in self._retrieval.semantic_candidates(
                        query.strip(), top_k=self._semantic_top_k
                    )
                ]
            )

        active_shares = _ORIENT_SHARES[: len(raw)]
        total_share = sum(active_shares)
        normalized = [share / total_share for share in active_shares]

        seen: set[str] = set()
        categories: list[tuple[str, list[_Candidate], float]] = []
        for name, share, items in zip(_ORIENT_SECTIONS[: len(raw)], normalized, raw, strict=True):
            cands: list[_Candidate] = []
            for entry, relevance, match in items:
                if entry.id in seen:
                    continue
                seen.add(entry.id)
                if not self._passes_filters(entry, include_tags, exclude_tags, include_types):
                    continue
                cand = _Candidate(entry, relevance, match)
                cand.effective = self._importance.effective_importance(entry, now)
                cands.append(cand)
            categories.append((name, cands, share))
        return categories, len(seen)

    def _fit_orient(
        self,
        categories: list[tuple[str, list[_Candidate], float]],
        content_budget: int,
        now: datetime,
    ) -> tuple[list[tuple[str, list[str]]], list[_Candidate]]:
        """Fit each category into its share of the budget; unused share rolls
        forward. The section header is charged when its first entry ships."""
        sections: list[tuple[str, list[str]]] = []
        included: list[_Candidate] = []
        rollover = 0
        for name, cands, share in categories:
            budget_for = int(share * content_budget) + rollover
            used = 0
            header_emitted = False
            section_blocks: list[str] = []
            for cand in cands:
                block = self._render_block(cand, now)
                block_cost = self._estimate(block)
                extra = 0 if header_emitted else self._estimate(name)
                if used + extra + block_cost <= budget_for:
                    if not header_emitted:
                        used += extra
                        header_emitted = True
                    section_blocks.append(block)
                    included.append(cand)
                    used += block_cost
            rollover = budget_for - used
            if section_blocks:
                sections.append((name, section_blocks))
        return sections, included

    # -------------------------------------------------- shared finish/format

    def _finish(
        self,
        included: list[tuple[_Candidate, str]],
        content: str,
        token_count: int,
        token_budget: int,
        sources_consulted: int,
        now: datetime,
    ) -> CuratedContext:
        """Populate derived importance, record access (only when non-empty),
        and build the result."""
        if included:
            for cand, _ in included:
                cand.entry.importance = cand.effective
            self._retrieval.record_access([cand.entry.id for cand, _ in included], now=now)
        return CuratedContext(
            content=content,
            entries=[cand.entry for cand, _ in included],
            token_count=token_count,
            budget_remaining=max(0, token_budget - token_count),
            sources_consulted=sources_consulted,
        )

    def _render_block(self, cand: _Candidate, now: datetime) -> str:
        """Rendered, redacted block for one entry."""
        red_header, red_body = self._render_parts(cand, now)
        return f"{red_header}\n{red_body}"

    def _render_parts(self, cand: _Candidate, now: datetime) -> tuple[str, str]:
        """Redacted (header_line, body) for an entry.

        Sensitive-keyed entries render ``REDACTED_CONTENT`` for the body; every
        part is then scanned by ``redact_text`` (content shapes + labelled
        values) before it is ever estimated or emitted.
        """
        entry = cand.entry
        if _STALE_TAG in entry.tags:
            days = self._days_since(now, entry.created_at)
            header = f"[{entry.type.name}, stored {days} days ago, potentially stale]"
        else:
            key = entry.key or ""
            age = self._coarse_age(now, entry.last_accessed_at or entry.created_at)
            header = f"[{entry.type.name}] {key} (importance: {cand.effective:.1f}, {age})"
        if is_sensitive_key(entry.key or "", self._key_patterns):
            body = REDACTED_CONTENT
        else:
            body = entry.content
        return (
            redact_text(header, key_patterns=self._key_patterns),
            redact_text(body, key_patterns=self._key_patterns),
        )

    def _truncate_to_fit(self, header_line: str, body: str, budget: int) -> tuple[str, bool]:
        """Truncate a single oversize block to ``budget`` tokens.

        Character-level binary search against the estimator, keeping the entry
        header line and appending :data:`TRUNCATION_MARKER`. Truncation runs on
        the ALREADY-REDACTED body so a secret can never straddle back into view.

        Returns:
            ``(block_text, True)`` on success, ``("", False)`` when even the
            header line plus the marker cannot fit the budget.
        """
        full = self._estimate(body)

        def block_for(prefix: str, elided: int) -> str:
            marker = TRUNCATION_MARKER.format(elided=elided)
            if prefix:
                return f"{header_line}\n{prefix}\n{marker}"
            return f"{header_line}\n{marker}"

        if self._estimate(block_for("", full)) > budget:
            return "", False
        lo, hi, best = 0, len(body), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self._estimate(block_for(body[:mid], full)) <= budget:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        kept = body[:best].rstrip()
        elided = full - self._estimate(kept)
        return block_for(kept, elided), True

    # --------------------------------------------------------- assembly

    def _assemble_query(self, blocks: list[str], n: int) -> tuple[str, int]:
        """Assemble the query-mode frame around ``blocks`` and count tokens.

        The header embeds counts unknown until fitting ends, so we estimate
        against a placeholder token count, then render the real number; the
        two agree exactly for whitespace-invariant estimators and drift is
        absorbed by the safety margin otherwise (spec §5)."""

        def build(tokens: int) -> str:
            header = f"=== Agent Memory ({n} entries, {tokens} tokens) ==="
            footer = "=== End Memory ==="
            if blocks:
                return header + "\n\n" + "\n\n".join(blocks) + "\n\n" + footer
            return header + "\n\n" + footer

        provisional = self._estimate(self._redact(build(0)))
        content = self._redact(build(provisional))
        return content, self._estimate(content)

    def _assemble_orient(self, sections: list[tuple[str, list[str]]]) -> tuple[str, int]:
        """Assemble the sectioned orient frame; empty sections are omitted."""

        def build(tokens: int) -> str:
            parts = ["=== Project Memory (orient) ==="]
            for name, blocks in sections:
                parts.append(name)
                parts.extend(blocks)
            parts.append(f"=== End Memory ({tokens} tokens) ===")
            return "\n\n".join(parts)

        provisional = self._estimate(self._redact(build(0)))
        content = self._redact(build(provisional))
        return content, self._estimate(content)

    def _query_frame_cost(self) -> int:
        header = "=== Agent Memory (0 entries, 0 tokens) ==="
        footer = "=== End Memory ==="
        return self._estimate(header) + self._estimate(footer)

    def _orient_frame_cost(self) -> int:
        header = "=== Project Memory (orient) ==="
        footer = "=== End Memory (0 tokens) ==="
        return self._estimate(header) + self._estimate(footer)

    def _content_budget(self, token_budget: int, frame_cost: int) -> int:
        effective_budget = math.floor(token_budget * (1.0 - self._token_safety_margin))
        return effective_budget - frame_cost

    # ---------------------------------------------------------- helpers

    @staticmethod
    def _validate_weights(recency: float, importance: float, relevance: float) -> None:
        for name, value in (
            ("recency_weight", recency),
            ("importance_weight", importance),
            ("relevance_weight", relevance),
        ):
            if value < 0:
                raise ConfigError(f"{name} must be >= 0, got {value!r}")
        if recency == 0 and importance == 0 and relevance == 0:
            raise ConfigError("at least one weight must be > 0 (all-zero weights are undefined)")

    @staticmethod
    def _passes_filters(
        entry: MemoryEntry,
        include_tags: list[str] | None,
        exclude_tags: list[str] | None,
        include_types: list[MemoryType] | None,
    ) -> bool:
        tags = set(entry.tags)
        if exclude_tags and tags & set(exclude_tags):
            return False  # exclude wins over include
        if include_tags and not (tags & set(include_tags)):
            return False
        return not (include_types and entry.type not in include_types)

    def _recency(self, now: datetime, entry: MemoryEntry) -> float:
        anchor = entry.last_accessed_at or entry.created_at
        hours = max((now - anchor).total_seconds() / 3600.0, 0.0)
        recency: float = 0.5 ** (hours / self._recency_half_life_hours)
        return recency

    @staticmethod
    def _coarse_age(now: datetime, anchor: datetime) -> str:
        seconds = max((now - anchor).total_seconds(), 0.0)
        days = int(seconds // 86400)
        if days >= 1:
            return f"{days}d ago"
        hours = int(seconds // 3600)
        if hours >= 1:
            return f"{hours}h ago"
        minutes = int(seconds // 60)
        return f"{minutes}m ago"

    @staticmethod
    def _days_since(now: datetime, anchor: datetime) -> int:
        return max((now - anchor).days, 0)

    def _redact(self, text: str) -> str:
        return redact_text(text, key_patterns=self._key_patterns)

    def _estimate(self, text: str) -> int:
        if self._resolved_estimator is None:
            self._resolved_estimator = resolve_estimator(self._preferred_estimator)
        return self._resolved_estimator.estimate(text)

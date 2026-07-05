"""Call-budgeted LLM summarization. Never deletes; archives with back-links (ADR-009).

The only module that spends LLM money. Compresses groups of old memories
into SUMMARY entries, archives the originals recoverably
(``ArchiveReason.SUMMARIZED`` + ``source_entry_ids`` back-links), and backs
``Memory.summarize()``, end-of-session summarization (lifecycle, step 13),
and the MCP server's "llm=None degrades loudly" guarantee.

Key contracts (blueprint-summarizer):

- **Budget** (Kairos circuit breaker): one :class:`~tulving.adapters.llm.CallBudget`
  per pass; the counter increments BEFORE each call, so a raising adapter
  still consumes its slot. Exhaustion stops cleanly and loudly.
- **Write ordering:** summary first, archive second — a crash in between
  leaves harmless duplication, never data loss.
- **D3:** candidate reads are not accesses; nothing here ever touches
  ``last_accessed_at``/``access_count``. **D2:** importance is only read
  (truncation ordering), never written.
- **Security (req #1 / ADR-010):** prompt text leaves the process, so every
  entry body is redacted (sensitive keys -> ``[REDACTED]``, token shapes
  scrubbed) before ``complete()`` — and the fallback digest gets the same
  treatment.

As-built deviations from blueprint-summarizer (both forced by landed code):

- The pass budget reuses :class:`tulving.adapters.llm.CallBudget` (identical
  semantics) instead of re-implementing the sketched private ``_CallBudget``.
- The no-LLM fallback digest is stored as ``MemoryType.OBSERVATION`` (tagged
  ``summarize_skipped``, ``source_entry_ids=[]``), not SUMMARY: the as-built
  store enforces "SUMMARY rows always carry non-empty back-links"
  (blueprint-store; pinned by ``test_store.py``), which is exactly the
  invariant that makes reason-aware purge protection meaningful. The digest
  rolled up nothing, so it must not look like a real summary.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Protocol

from tulving.adapters.llm import CallBudget, LLMAdapter
from tulving.context.config import LifecycleConfig
from tulving.entry import MemoryEntry, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import ConfigError, MemoryStoreError
from tulving.security import REDACTED, is_sensitive_key, redact_text
from tulving.store import MemoryStore

logger = logging.getLogger("tulving.summarizer")

TRUNCATION_MARKER = "[… truncated for summarization]"

SUMMARIZE_PROMPT = """You are summarizing an agent's working memory. Preserve all key facts,
decisions, and their reasoning. Discard verbose details and exploration notes.
Produce a concise digest that another agent could use to understand
what was learned.

Entries to summarize:
{entries}
"""

_PAGE = 500
_TOP_ITEMS = 10
_SNIPPET_CHARS = 120
_FALLBACK_TAG = "summarize_skipped"
_FALLBACK_IMPORTANCE = 0.3


class TokenEstimator(Protocol):
    """Consumed surface of the curator's token estimator (build step 10).

    Declared structurally here so the summarizer never imports the curator;
    the curator's concrete estimator (tiktoken -> ``len // 4`` fallback)
    satisfies it, as does any test fake.
    """

    def estimate(self, text: str) -> int:
        """Estimated token count of ``text``."""
        ...


class ImportanceEvaluator(Protocol):
    """The 'DecayManager-shaped' dependency (blueprint-summarizer, D2).

    ``memory.py``'s ``_DecayEvaluator`` satisfies this structurally — the
    integration pass injects it verbatim. The summarizer only *reads*
    effective importance (truncation ordering); it never writes importance.
    """

    def effective_importance(self, entry: MemoryEntry, now: datetime) -> float:
        """Decayed importance of ``entry`` at ``now`` (pure, D2)."""
        ...


class MemorySummarizer:
    """Compresses old memories into SUMMARY entries under a hard LLM call budget.

    Constructed by ``Memory`` with its bound ``agent_id`` (D7). Never called
    from ``__init__``/``startup()`` (D8). ``llm`` may be None — every path
    then degrades loudly (warning + visible digest, or ``ConfigError`` for
    the explicit ``summarize_group`` request).
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMAdapter | None,
        config: LifecycleConfig,
        token_estimator: TokenEstimator,
        decay: ImportanceEvaluator,
        agent_id: str,
        clock: Callable[[], datetime] = utcnow,
        *,
        key_patterns: Sequence[re.Pattern[str]] | None = None,
    ) -> None:
        """Cheap constructor (D8): store references, validate nothing expensive.

        Args:
            store: The CRUD engine; summaries go through its internal
                SUMMARY-permitting create path, sources through ``archive``.
            llm: Optional adapter. None is legal — ``summarize()`` degrades
                loudly, ``summarize_group()`` refuses with ``ConfigError``.
            config: Lifecycle knobs read here: ``llm_call_budget``,
                ``max_input_tokens``, ``token_safety_margin``,
                ``preserve_decisions_verbatim``.
            token_estimator: The curator's estimator (never blocks on
                tiktoken availability).
            decay: Effective-importance reader used ONLY to order truncation.
            agent_id: Identity stamped into every summary's ``SourceInfo`` (D7).
            clock: Injectable now-source (tests never sleep).
            key_patterns: Compiled sensitive-key patterns from
                ``Memory(sensitive_keys=...)``; None uses the security
                defaults. Needed so user-augmented redaction reaches LLM
                egress (security req #1).

        Raises:
            ConfigError: On an empty ``agent_id`` (D7: identity is
                load-bearing).
        """
        if not agent_id:
            raise ConfigError("agent_id must be non-empty (D7: identity is load-bearing)")
        self._store = store
        self._llm = llm
        self._config = config
        self._estimator = token_estimator
        self._decay = decay
        self._agent_id = agent_id
        self._clock: Callable[[], datetime] = clock
        self._key_patterns: tuple[re.Pattern[str], ...] | None = (
            tuple(key_patterns) if key_patterns is not None else None
        )

    # ------------------------------------------------------------- public API

    def summarize(
        self,
        *,
        older_than: timedelta | None = None,
        tags: list[str] | None = None,
        session_id: str | None = None,
        max_group_size: int = 10,
    ) -> list[MemoryEntry]:
        """Summarize old memories; return the newly created digest entries.

        With an LLM: select candidates (live, non-exempt, filtered), group
        deterministically (session -> primary tag -> creation time), chunk to
        the per-call input cap, then per chunk: spend budget -> call LLM ->
        store SUMMARY -> archive sources. Budget exhaustion logs a warning
        naming the remainder and returns what was completed; an adapter
        exception leaves the failed chunk's entries live and skips the rest
        of that group (chunks of it already summarized stay summarized)
        before moving to the next group.

        Without an LLM (``llm is None``) — NEVER silent: logs a warning, and
        when candidates exist stores ONE deterministic fallback digest
        (tagged ``summarize_skipped``) and returns ``[digest]``. Nothing is
        ever archived without a real summary to back-link; with no
        candidates nothing is stored and ``[]`` is returned.

        Args:
            older_than: Spare entries accessed within this window — filters
                on ``last_accessed_at`` (an entry in active use is never
                rolled up), strict cutoff, keyed to the injected clock.
            tags: Only entries carrying at least one of these tags.
            session_id: Only entries of this session (lifecycle's
                end-of-session entry point).
            max_group_size: Upper bound on entries per group.

        Returns:
            Newly created SUMMARY entries (or the single fallback digest).

        Raises:
            MemoryStoreError: On ``max_group_size < 1``; also propagated from
                store writes (creating a summary or the fallback digest).
            StorageError: Propagated backend failure.
        """
        if max_group_size < 1:
            raise MemoryStoreError(f"max_group_size must be >= 1, got {max_group_size!r}")
        llm = self._llm
        now = self._clock()
        if llm is None:
            logger.warning("summarize skipped: no LLM adapter configured")
        rows = self._scan(older_than=older_than, tags=tags, session_id=session_id, now=now)
        candidates = [entry for entry in rows if self._summarizable(entry)]
        if not candidates:
            return []
        if llm is None:
            decisions = [entry for entry in rows if entry.type is MemoryType.DECISION]
            return [self._fallback_digest(candidates, decisions, now)]

        cap = self._cap(llm)
        chunked_groups = [
            self._chunk(group, cap) for group in self._group(candidates, max_group_size)
        ]
        budget = CallBudget(self._config.llm_call_budget)
        created: list[MemoryEntry] = []
        for group_index, chunks in enumerate(chunked_groups):
            for chunk_index, chunk in enumerate(chunks):
                if not budget.try_acquire():
                    groups_left = len(chunked_groups) - group_index
                    chunks_left = (
                        len(chunks)
                        - chunk_index
                        + sum(len(later) for later in chunked_groups[group_index + 1 :])
                    )
                    logger.warning(
                        "summarize stopped: LLM call budget exhausted (%d/%d calls); "
                        "%d group(s) not fully summarized (%d chunk(s) left for a later pass)",
                        budget.spent,
                        budget.limit,
                        groups_left,
                        chunks_left,
                    )
                    return created
                summary = self._summarize_chunk(chunk, llm, cap, now)
                if summary is None:
                    break  # failed call: leave the rest of this group live
                created.append(summary)
        return created

    def summarize_group(self, entries: list[MemoryEntry]) -> MemoryEntry:
        """Summarize one explicit group into one SUMMARY entry.

        DECISION, pinned, and archived members are filtered out first
        (ADR-006 — no flag can change this). Chunking applies when the group
        exceeds the per-call cap; budget exhaustion mid-group stops cleanly
        (already-summarized chunks stay summarized, the remainder stays live,
        logged).

        Args:
            entries: The group to compress.

        Returns:
            The first (or only) SUMMARY created.

        Raises:
            ConfigError: If ``llm`` is None — an explicit compression request
                cannot be honored (loud by exception; ``summarize()`` is the
                degrading path).
            MemoryStoreError: If nothing summarizable remains after removing
                exempt members, or if every attempted LLM call failed.
        """
        llm = self._llm
        if llm is None:
            raise ConfigError(
                "summarize_group requires an LLM adapter; "
                "llm=None degrades only through summarize()"
            )
        eligible = [entry for entry in entries if self._summarizable(entry)]
        if not eligible:
            raise MemoryStoreError(
                "summarize_group: no summarizable entries "
                "(DECISION, pinned, and archived entries are exempt)"
            )
        now = self._clock()
        cap = self._cap(llm)
        chunks = self._chunk(eligible, cap)
        budget = CallBudget(self._config.llm_call_budget)
        created: list[MemoryEntry] = []
        failed = 0
        for chunk_index, chunk in enumerate(chunks):
            if not budget.try_acquire():
                logger.warning(
                    "summarize_group stopped: LLM call budget exhausted (%d/%d calls); "
                    "%d chunk(s) left live",
                    budget.spent,
                    budget.limit,
                    len(chunks) - chunk_index,
                )
                break
            summary = self._summarize_chunk(chunk, llm, cap, now)
            if summary is None:
                failed += 1
            else:
                created.append(summary)
        if not created:
            raise MemoryStoreError(
                f"summarize_group created no summary ({failed} LLM call(s) failed)"
            )
        return created[0]

    # ----------------------------------------------------- candidate selection

    @staticmethod
    def _summarizable(entry: MemoryEntry) -> bool:
        """Live and non-exempt: never DECISION (ADR-006), never pinned."""
        return not entry.archived and not entry.pinned and entry.type is not MemoryType.DECISION

    def _scan(
        self,
        *,
        older_than: timedelta | None,
        tags: list[str] | None,
        session_id: str | None,
        now: datetime,
    ) -> list[MemoryEntry]:
        """Filtered live rows, paged, WITHOUT touch (D3: not an access)."""
        accessed_before = now - older_than if older_than is not None else None
        rows: list[MemoryEntry] = []
        offset = 0
        while True:
            page = self._store.list(
                tags=tags,
                session_id=session_id,
                accessed_before=accessed_before,
                limit=_PAGE,
                offset=offset,
            )
            rows.extend(page)
            if len(page) < _PAGE:
                return rows
            offset += _PAGE

    def _group(self, candidates: list[MemoryEntry], max_group_size: int) -> list[list[MemoryEntry]]:
        """Deterministic grouping: session -> primary tag -> time (no embeddings)."""
        partitions: dict[tuple[int, str, str], list[MemoryEntry]] = {}
        for entry in candidates:
            session_rank = (0, "") if entry.session_id is None else (1, entry.session_id)
            primary_tag = min(entry.tags) if entry.tags else ""
            key = (session_rank[0], session_rank[1], primary_tag)
            partitions.setdefault(key, []).append(entry)
        groups: list[list[MemoryEntry]] = []
        for key in sorted(partitions):
            members = sorted(partitions[key], key=lambda e: (e.created_at, e.id))
            groups.extend(
                members[i : i + max_group_size] for i in range(0, len(members), max_group_size)
            )
        return groups

    # ------------------------------------------------- chunking & prompt build

    def _cap(self, llm: LLMAdapter) -> int:
        """Per-call input cap in tokens, safety margin applied."""
        limit = min(self._config.max_input_tokens, llm.max_input_tokens)
        return max(int(limit * (1.0 - self._config.token_safety_margin)), 1)

    def _chunk(self, group: list[MemoryEntry], cap: int) -> list[list[MemoryEntry]]:
        """Split one group into sub-groups whose rendered prompts fit ``cap``.

        Order-preserving and greedy over per-entry block estimates; an entry
        whose block alone exceeds the cap becomes its own chunk and is
        truncated at prompt-build time. Chunks never mix groups.
        """
        overhead = self._estimator.estimate(SUMMARIZE_PROMPT.format(entries=""))
        chunks: list[list[MemoryEntry]] = []
        current: list[MemoryEntry] = []
        current_tokens = 0
        for entry in group:
            tokens = self._estimator.estimate(self._block(entry))
            if current and overhead + current_tokens + tokens > cap:
                chunks.append(current)
                current, current_tokens = [], 0
            current.append(entry)
            current_tokens += tokens
        if current:
            chunks.append(current)
        return chunks

    def _label(self, entry: MemoryEntry) -> str:
        return entry.key if entry.key is not None else entry.id

    def _content_for(self, entry: MemoryEntry) -> str:
        """Redacted content: sensitive keys mask entirely, token shapes scrub."""
        if is_sensitive_key(entry.key or "", self._key_patterns):
            return REDACTED
        return redact_text(entry.content, key_patterns=self._key_patterns)

    def _format_block(self, entry: MemoryEntry, content: str) -> str:
        return f"- [{entry.type.name}] {self._label(entry)}: {content}"

    def _block(self, entry: MemoryEntry) -> str:
        """One redacted prompt line for ``entry``."""
        return self._format_block(entry, self._content_for(entry))

    def _assemble(self, chunk: list[MemoryEntry], blocks: dict[str, str]) -> str:
        return SUMMARIZE_PROMPT.format(entries="\n".join(blocks[e.id] for e in chunk))

    def _fit_prompt(self, chunk: list[MemoryEntry], cap: int, now: datetime) -> str:
        """Render the chunk's prompt, trimming content to fit ``cap``.

        Trims lowest-effective-importance entries first (D2 read-only, with
        the injected clock), each trim ending with ``TRUNCATION_MARKER``.
        The stored originals are never modified — truncation affects only
        the LLM input.
        """
        blocks = {entry.id: self._block(entry) for entry in chunk}
        prompt = self._assemble(chunk, blocks)
        if self._estimator.estimate(prompt) <= cap:
            return prompt
        trim_order = sorted(chunk, key=lambda e: (self._decay.effective_importance(e, now), e.id))
        for entry in trim_order:
            blocks[entry.id] = self._shrink(entry, chunk, blocks, cap)
            prompt = self._assemble(chunk, blocks)
            if self._estimator.estimate(prompt) <= cap:
                return prompt
        logger.warning(
            "summarize: prompt still exceeds the input cap after truncation (%d entries)",
            len(chunk),
        )
        return prompt

    def _shrink(
        self,
        entry: MemoryEntry,
        chunk: list[MemoryEntry],
        blocks: dict[str, str],
        cap: int,
    ) -> str:
        """Largest truncated block for ``entry`` that lets the prompt fit.

        Binary search over the (already redacted) content prefix; falls back
        to a marker-only block when even the empty prefix cannot fit (the
        next entry in trim order then absorbs the remainder).
        """
        content = self._content_for(entry)
        best = self._format_block(entry, TRUNCATION_MARKER)
        low, high = 0, len(content)
        while low <= high:
            mid = (low + high) // 2
            candidate = self._format_block(entry, content[:mid] + TRUNCATION_MARKER)
            trial = dict(blocks)
            trial[entry.id] = candidate
            if self._estimator.estimate(self._assemble(chunk, trial)) <= cap:
                best = candidate
                low = mid + 1
            else:
                high = mid - 1
        return best

    # --------------------------------------------------------- chunk execution

    def _summarize_chunk(
        self,
        chunk: list[MemoryEntry],
        llm: LLMAdapter,
        cap: int,
        now: datetime,
    ) -> MemoryEntry | None:
        """One budgeted chunk: call LLM, store SUMMARY, archive sources.

        Summary-first, archive-second (blueprint write ordering): a crash in
        between leaves originals AND the summary visible — harmless
        duplication, never hidden data. Returns None on a failed or empty
        LLM response (the budget slot is already spent by the caller; the
        chunk's entries stay live). Each source archive is individually
        guarded: a source concurrently archived under another reason (e.g.
        evicted) is logged and skipped — its recorded reason is never
        overwritten and the pass never aborts mid-chunk because of it.
        """
        prompt = redact_text(self._fit_prompt(chunk, cap, now), key_patterns=self._key_patterns)
        try:
            response = llm.complete(prompt)
        except Exception as exc:  # any exception = failed call (adapters-llm contract)
            logger.warning(
                "summarize: LLM call failed (%s); %d entries left live for a later pass",
                type(exc).__name__,
                len(chunk),
            )
            return None
        if not response.strip():
            logger.warning(
                "summarize: LLM returned an empty response; %d entries left live", len(chunk)
            )
            return None
        session_ids = {entry.session_id for entry in chunk}
        session_id = next(iter(session_ids)) if len(session_ids) == 1 else None
        base_importance = min(max(max(e.base_importance for e in chunk), 0.0), 1.0)
        summary = self._store.create(
            content=response,
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id=self._agent_id),
            tags=sorted({tag for entry in chunk for tag in entry.tags}),
            base_importance=base_importance,
            session_id=session_id,
            _allow_summary=True,
            _source_entry_ids=[entry.id for entry in chunk],
        )
        for entry in chunk:
            try:
                self._store.archive(entry.id, ArchiveReason.SUMMARIZED)
            except MemoryStoreError:
                # Raced by a concurrent archive (e.g. eviction) between the
                # candidate scan and this write. Log ids only — never content.
                logger.warning(
                    "summarize: source %s was archived concurrently; leaving its "
                    "reason untouched (summary %s still back-links it)",
                    entry.id,
                    summary.id,
                )
        return summary

    # --------------------------------------------------------- no-LLM fallback

    def _fallback_digest(
        self,
        candidates: list[MemoryEntry],
        decisions: list[MemoryEntry],
        now: datetime,
    ) -> MemoryEntry:
        """Deterministic visible trace stored when ``llm=None`` skips a pass.

        Stored as OBSERVATION with ``source_entry_ids=[]`` — it rolled up
        nothing, so nothing is archived and it must never masquerade as a
        real summary (see module docstring deviation note). Architect ruling:
        this is deliberate — SUMMARY is reserved for rows with real source
        back-links (store-enforced) or lifecycle's system session markers;
        the disposable skip digest lives with OBSERVATION's half-life.
        Content is redacted like any outgoing text.
        """
        del now  # deterministic by design: the digest carries no timestamps
        ordering = self._digest_order
        lines = [
            "Summarize pass skipped (no LLM adapter configured).",
            f"Candidates: {len(candidates)} entries.",
            f"Decisions preserved verbatim: {len(decisions)}.",
        ]
        if self._config.preserve_decisions_verbatim and decisions:
            lines.extend(self._block(entry) for entry in sorted(decisions, key=ordering))
        lines.append("Top items:")
        lines.extend(
            self._format_block(entry, self._content_for(entry)[:_SNIPPET_CHARS])
            for entry in sorted(candidates, key=ordering)[:_TOP_ITEMS]
        )
        content = redact_text("\n".join(lines), key_patterns=self._key_patterns)
        return self._store.create(
            content=content,
            type=MemoryType.OBSERVATION,
            source=SourceInfo(agent_id=self._agent_id),
            tags=[_FALLBACK_TAG],
            base_importance=_FALLBACK_IMPORTANCE,
        )

    @staticmethod
    def _digest_order(entry: MemoryEntry) -> tuple[float, datetime, str]:
        """Deterministic digest ordering: importance desc, then age, then id."""
        return (-entry.base_importance, entry.created_at, entry.id)

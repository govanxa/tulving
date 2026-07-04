"""Per-agent sessions, abandonment detection, staleness scanning (D7/D8).

The :class:`LifecycleManager` is the real implementation behind
``memory.py``'s ``LifecycleManagerPort`` seam (blueprint-memory L1-L7): it
owns per-agent session rows, per-session ``last_activity_at``, abandonment
detection (L4 — deterministic, NEVER an LLM call), staleness tagging (L5),
and the deterministic session markers that ``curate(mode="orient")`` reads.

As-built reconciliation against blueprint-lifecycle (its appended step-9
note wins over the blueprint body):

- ``session_start`` returns the session **id string**; ``session_end`` takes
  a session id and returns ``None`` (port shape). The rich
  :class:`SessionSummary` path is :meth:`LifecycleManager.end_session`.
- ``run_startup_tasks``/``StartupReport`` are superseded: ``Memory.startup()``
  owns time-boxing, non-fatality, and the single-runner guard, and folds
  :meth:`check_on_startup` / :meth:`run_staleness_scan` counts into its own
  frozen report. Eviction is likewise orchestrated by ``Memory``.
- The summarizer and staleness scanner are **injected ports** (Protocols
  below) — this module never imports the concrete summarizer or decay
  modules, which are built in parallel.
- ``memory_count`` is DERIVED from the store (the as-built sessions table
  has no such column); it is populated on hydration, never persisted.
- The session context manager lives in ``memory.py`` (``Memory.session()``),
  not here.
- ``record_activity`` must be wired by the Memory integration pass
  (step 13b, pipeline task #5): ``LifecycleManagerPort`` grows
  ``record_activity(session_id, *, write=False)`` there, called from
  ``Memory.search()``/``curate()``. Until that pass lands the
  ``activity_debounce_seconds`` knob is inert in production (it is fully
  exercised by tests) — an owner-sanctioned deferral, do not re-flag.

Markers are persisted directly through the backend with a validated
:class:`MemoryEntry` (type SUMMARY, ``source_entry_ids=[]`` — group SUMMARY
entries carry the back-links) because the store's ``_allow_summary`` path
requires non-empty back-links by design; the D1 supersede discipline is
mirrored for the marker key.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Protocol

from tulving.context.config import LifecycleConfig
from tulving.entry import MemoryEntry, Relationship, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MemoryType, SessionStatus
from tulving.exceptions import ConfigError, MemoryStoreError

if TYPE_CHECKING:
    from tulving.adapters.storage import StorageBackend
    from tulving.store import MemoryStore

logger = logging.getLogger("tulving.lifecycle")

SESSION_KEY_PREFIX: Final[str] = "session:"  # marker keys: f"session:{session.id}"
SESSION_MARKER_IMPORTANCE: Final[float] = 0.7
STALE_TAG: Final[str] = "potentially_stale"

_PAGE: Final[int] = 500
_MARKER_TAGS: Final[tuple[str, ...]] = ("session", "session_marker")
#: Every user-storable type — memory_count never counts system SUMMARY rows.
_COUNTED_TYPES: Final[tuple[MemoryType, ...]] = tuple(
    t for t in MemoryType if t is not MemoryType.SUMMARY
)


# ---------------------------------------------------------------------------
# Injected ports (concrete implementations are built in parallel — steps 11/12)
# ---------------------------------------------------------------------------


class SessionSummarizerPort(Protocol):
    """Consumed slice of ``MemorySummarizer`` (blueprint-summarizer).

    The real ``summarize`` accepts more keyword-only knobs (``older_than``,
    ``tags``, ``max_group_size``) — all defaulted, so it satisfies this
    Protocol structurally. ``llm=None`` degradation (logged warning, no
    archival) is the summarizer's own duty; this module only guarantees the
    call happens outside every startup path (D8).
    """

    def summarize(self, *, session_id: str | None = None) -> list[MemoryEntry]:
        """Roll up the session's memories; return created SUMMARY entries."""
        ...


class StalenessScannerPort(Protocol):
    """Consumed slice of the decay module's staleness detector (step 11)."""

    def scan(self, threshold_days: int) -> list[str]:
        """Tag entries unaccessed for > ``threshold_days``; return their ids."""
        ...


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Session:
    """Persisted session record. Data-only; transitions go through the manager.

    Round-trips via ``to_dict()``/``from_dict()`` following entry.py
    conventions: tz-aware ISO datetimes (naive rejected with ``ValueError``),
    enums by value, unknown keys ignored on ingest. ``memory_count`` is
    derived from the store by the manager (the sessions table carries no
    such column) and defaults to 0 when hydrating a raw backend row.
    """

    id: str  # UUID, minted by the manager
    agent_id: str  # required, non-empty (D7)
    goal: str | None
    started_at: datetime
    last_activity_at: datetime  # never None; starts == started_at
    ended_at: datetime | None = None
    memory_count: int = 0
    status: SessionStatus = SessionStatus.ACTIVE

    def __post_init__(self) -> None:
        """Enforce invariants; raise ``ValueError`` on violation (entry.py pattern)."""
        if not self.id:
            raise ValueError("session id must be non-empty")
        if not self.agent_id:
            raise ValueError("session agent_id must be non-empty (D7)")
        _require_aware(self.started_at, "started_at")
        _require_aware(self.last_activity_at, "last_activity_at")
        if self.memory_count < 0:
            raise ValueError("memory_count must be >= 0")
        if self.status is SessionStatus.ACTIVE:
            if self.ended_at is not None:
                raise ValueError("ACTIVE sessions must not carry ended_at")
        else:
            if self.ended_at is None:
                raise ValueError(f"{self.status.value} sessions must carry ended_at")
            _require_aware(self.ended_at, "ended_at")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict (ISO datetimes, enum values)."""
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "goal": self.goal,
            "started_at": self.started_at.isoformat(),
            "last_activity_at": self.last_activity_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at is not None else None,
            "memory_count": self.memory_count,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Inverse of ``to_dict()``; also hydrates raw backend session rows.

        Unknown keys are ignored (forward-compat); a missing/None
        ``last_activity_at`` normalizes to ``started_at``; a missing
        ``memory_count`` defaults to 0 (backend rows never carry it).

        Raises:
            ValueError: On naive datetime strings or unknown enum values.
            KeyError: On missing required keys.
        """
        started_at = _parse_aware(data["started_at"], "started_at")
        raw_activity = data.get("last_activity_at")
        raw_ended = data.get("ended_at")
        return cls(
            id=data["id"],
            agent_id=data["agent_id"],
            goal=data.get("goal"),
            started_at=started_at,
            last_activity_at=(
                _parse_aware(raw_activity, "last_activity_at")
                if raw_activity is not None
                else started_at
            ),
            ended_at=_parse_aware(raw_ended, "ended_at") if raw_ended is not None else None,
            memory_count=int(data.get("memory_count", 0)),
            status=SessionStatus(data.get("status", SessionStatus.ACTIVE.value)),
        )


@dataclass
class AbandonedSession:
    """One abandonment verdict: the post-transition session and its evidence."""

    session: Session  # post-transition (status=ABANDONED)
    gap: timedelta  # now - last_activity_at at detection time
    marker_entry_id: str


@dataclass
class SessionSummary:
    """Outcome of :meth:`LifecycleManager.end_session`."""

    session: Session  # post-transition (status=ENDED)
    summaries_created: int
    entries_archived: int  # originals archived SUMMARIZED (via back-links)
    decisions_preserved: int  # live DECISION entries in the session
    session_marker: MemoryEntry


# ---------------------------------------------------------------------------
# LifecycleManager
# ---------------------------------------------------------------------------


class LifecycleManager:
    """Per-agent session orchestration over the store/backend (D7).

    One per ``Memory`` instance; ``agent_id`` threaded from the owning
    handle. Thread-safe: session mutations hold an internal
    ``threading.Lock`` (ADR-015: one process, many threads — ``Memory``
    additionally serializes port calls with its own session lock).
    Construction is cheap (D8): references only, no queries, no scans,
    no LLM.

    Satisfies ``tulving.memory.LifecycleManagerPort`` structurally so the
    swap against ``_DefaultLifecycleManager`` is mechanical.
    """

    def __init__(
        self,
        store: MemoryStore,
        backend: StorageBackend,
        config: LifecycleConfig,
        agent_id: str,
        *,
        summarizer: SessionSummarizerPort | None = None,
        staleness: StalenessScannerPort | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        """Bind collaborators; performs no I/O.

        Args:
            store: The CRUD engine (markers, counts, staleness tagging).
            backend: Session-row persistence slice (D3 sessions table).
            config: The single lifecycle policy table (D12).
            agent_id: This handle's bound identity (D7); non-empty.
            summarizer: Injected end-of-session summarization port; None
                degrades loudly (logged warning) when summarization is due.
            staleness: Injected staleness scanner; None uses the built-in
                store-backed scan.
            clock: Injectable now-source (tests never sleep).

        Raises:
            ConfigError: On an empty ``agent_id``.
        """
        if not agent_id:
            raise ConfigError("agent_id must be non-empty (D7: identity is load-bearing)")
        self._store = store
        self._backend = backend
        self._config = config
        self._agent_id = agent_id
        self._summarizer = summarizer
        self._staleness = staleness
        self._clock: Callable[[], datetime] = clock
        self._mutex = threading.Lock()

    # ------------------------------------------------------------- sessions

    def ensure_active_session(self) -> str:
        """This agent's ACTIVE session id, auto-starting an anonymous one (L2).

        Called by ``Memory.store()``: an existing session's
        ``last_activity_at`` is always bumped (write activity is never
        debounced), monotonically — a backwards clock never rewinds it.

        Deliberate asymmetry vs :meth:`session_start`: a STALE ACTIVE
        session is resurrected (bumped and reused), never abandoned —
        continuing to store into it re-legitimizes the session, whereas an
        explicit ``session_start`` declares new work and retires the stale
        one first.

        Returns:
            The ACTIVE session id for this agent.
        """
        with self._mutex:
            row = self._active_row()
            if row is not None:
                session = Session.from_dict(row)
                now = self._clock()
                if now > session.last_activity_at:  # max(now, stored): never rewind
                    self._backend.update_session(session.id, {"last_activity_at": now.isoformat()})
                return session.id
            return self._create_session(goal=None)

    def session_start(self, goal: str | None = None) -> str:
        """Start a session for THIS agent (L3, D7).

        A live ACTIVE session for this agent refuses the start; a *stale*
        one (gap > ``config.inactivity_threshold``) is abandoned first —
        the same deterministic path startup detection uses — and the new
        session starts cleanly. Another agent's ACTIVE session never blocks.

        Args:
            goal: Optional statement of what this session is about.

        Returns:
            The new session id (persisted ACTIVE,
            ``last_activity_at == started_at == now``).

        Raises:
            MemoryStoreError: When this agent already holds a live ACTIVE
                session.
        """
        with self._mutex:
            now = self._clock()
            for row in self._backend.list_sessions(
                agent_id=self._agent_id, status=SessionStatus.ACTIVE.value
            ):
                session = Session.from_dict(row)
                if self._is_stale(session, now):
                    self._abandon(session, now)
                else:
                    raise MemoryStoreError(f"Session already active for agent '{self._agent_id}'")
            return self._create_session(goal=goal)

    def session_end(self, session_id: str) -> None:
        """End an ACTIVE session — port shape; see :meth:`end_session`.

        Args:
            session_id: The session to end.

        Raises:
            MemoryStoreError: When the session is missing, not ACTIVE, or
                belongs to another agent.
        """
        self.end_session(session_id)

    def end_session(self, session_id: str) -> SessionSummary:
        """End an ACTIVE session of this agent (spec §4.1 sequence).

        1. Transition guard: the session must exist, be ACTIVE, and belong
           to this agent (ENDED/ABANDONED are terminal — no resurrection).
        2. If ``config.summarize_on_session_end``: call the injected
           summarizer port (its ``llm=None`` handling is loud by contract);
           with no port wired, log a warning — never silent.
        3. Store the deterministic session marker (always, even when
           summarization is disabled).
        4. Persist ``status=ENDED``, ``ended_at=now``.

        Args:
            session_id: The session to end.

        Returns:
            The :class:`SessionSummary` with post-transition state.

        Raises:
            MemoryStoreError: On the transition-guard violations above.
        """
        with self._mutex:
            row = self._backend.get_session(session_id)
            if (
                row is None
                or row["status"] != SessionStatus.ACTIVE.value
                or row["agent_id"] != self._agent_id
            ):
                raise MemoryStoreError(
                    f"No active session {session_id!r} for agent '{self._agent_id}'"
                )
            session = Session.from_dict(row)

            summaries: list[MemoryEntry] = []
            if self._config.summarize_on_session_end:
                if self._summarizer is None:
                    logger.warning(
                        "session %s ended without summarization: no summarizer wired "
                        "(configure an LLM adapter to enable end-of-session rollups)",
                        session_id,
                    )
                else:
                    summaries = list(self._summarizer.summarize(session_id=session_id))

            now = self._clock()
            memory_count = self._memory_count(session_id)
            decisions = self._session_decisions(session_id)
            # ONE transaction over marker + status flip (DB atomicity): a
            # crash between them can never leave an ENDED session without
            # its marker, or a marker for a still-ACTIVE session.
            with self._backend.transaction():
                marker = self._store_marker(
                    session,
                    now=now,
                    abandoned=False,
                    summaries=summaries,
                    decisions=decisions,
                    memory_count=memory_count,
                    gap=None,
                )
                self._backend.update_session(
                    session_id,
                    {
                        "status": SessionStatus.ENDED.value,
                        "ended_at": now.isoformat(),
                        "last_activity_at": now.isoformat(),
                    },
                )
            ended = replace(
                session,
                status=SessionStatus.ENDED,
                ended_at=now,
                last_activity_at=now,
                memory_count=memory_count,
            )
            return SessionSummary(
                session=ended,
                summaries_created=len(summaries),
                entries_archived=sum(len(s.source_entry_ids) for s in summaries),
                decisions_preserved=len(decisions),
                session_marker=marker,
            )

    def record_activity(self, session_id: str, *, write: bool = True) -> None:
        """Record activity on this agent's ACTIVE session.

        Another agent's session is invisible here (same ownership guard as
        :meth:`end_session`, in this method's no-op idiom): agent A can
        never bump agent B's ``last_activity_at`` (SEC-SEV-002). Updates
        are monotonic — a backwards clock never rewinds activity.

        Args:
            session_id: The session the operation ran under.
            write: True (``store()``): always update ``last_activity_at``.
                False (``search()``/``curate()``): update only when the gap
                exceeds ``config.activity_debounce_seconds`` — read paths
                must not write-amplify, yet a read-only live session never
                looks abandoned.
        """
        with self._mutex:
            row = self._backend.get_session(session_id)
            if (
                row is None
                or row["status"] != SessionStatus.ACTIVE.value
                or row["agent_id"] != self._agent_id
            ):
                logger.debug(
                    "record_activity ignored: session %r is not this agent's ACTIVE session",
                    session_id,
                )
                return
            session = Session.from_dict(row)
            now = self._clock()
            if now <= session.last_activity_at:
                return  # max(now, stored): never rewind (backwards clock)
            if not write:
                gap_seconds = (now - session.last_activity_at).total_seconds()
                if gap_seconds <= self._config.activity_debounce_seconds:
                    return
            self._backend.update_session(session_id, {"last_activity_at": now.isoformat()})

    def close_active_session(self) -> None:
        """Gracefully end this agent's ACTIVE session, if any (L6).

        Used by ``Memory.close()`` — best-effort there; next-startup
        abandonment detection remains the primary recovery path.
        """
        row = self._active_row()
        if row is not None:
            self.end_session(str(row["id"]))

    def get_session(self, session_id: str) -> Session | None:
        """Hydrated session with derived ``memory_count``; None on a miss.

        Another agent's session reads as a miss (same ownership guard as
        :meth:`end_session`, in this method's None idiom): agent A can
        never read agent B's session record (SEC-SEV-002).
        """
        row = self._backend.get_session(session_id)
        if row is None or row["agent_id"] != self._agent_id:
            return None
        session = Session.from_dict(row)
        session.memory_count = self._memory_count(session_id)
        return session

    # ----------------------------------------------------- abandonment (L4)

    def detect_abandoned_sessions(self) -> list[AbandonedSession]:
        """Abandon every ACTIVE session idle past the threshold — any agent.

        A crashed agent may never return, so the scan covers ALL agents'
        ACTIVE sessions; the guarantee that one agent's startup never
        abandons another agent's LIVE session comes from comparing each
        session's OWN ``last_activity_at`` against
        ``config.inactivity_threshold`` (strictly greater), never from
        agent filtering. Deterministic template markers only — NEVER an
        LLM call (D8); abandonment archives NOTHING (ADR-009).

        Returns:
            One :class:`AbandonedSession` per session flipped this pass.
        """
        with self._mutex:
            now = self._clock()
            abandoned: list[AbandonedSession] = []
            for row in self._backend.list_sessions(status=SessionStatus.ACTIVE.value):
                session = Session.from_dict(row)
                if self._is_stale(session, now):
                    abandoned.append(self._abandon(session, now))
            return abandoned

    def check_on_startup(self) -> int:
        """Port shape for ``Memory.startup()``: abandonment count (L4)."""
        return len(self.detect_abandoned_sessions())

    # ------------------------------------------------------- staleness (L5)

    def run_staleness_scan(self) -> int:
        """Tag entries unaccessed for > ``config.staleness_threshold_days``.

        Delegates to the injected scanner port when one is wired (the decay
        module's detector after integration); otherwise runs the built-in
        store-backed scan. Scanning is NOT an access: ``last_accessed_at``
        is never touched (D3) — the tag auto-clears via the store's access
        paths, or in bulk via :meth:`verify_all`.

        Returns:
            The number of entries newly tagged ``potentially_stale``.
        """
        threshold_days = self._config.staleness_threshold_days
        if self._staleness is not None:
            return len(self._staleness.scan(threshold_days))
        now = self._clock()
        cutoff = now - timedelta(days=threshold_days)
        tagged = 0
        offset = 0
        # Page-and-mutate: tagging changes neither the accessed_before
        # membership (last_accessed_at untouched) nor the created_at sort
        # order, so offset paging stays stable — no full materialization.
        while True:
            page = self._store.list(accessed_before=cutoff, limit=_PAGE, offset=offset)
            for entry in page:
                if STALE_TAG in entry.tags:
                    continue
                self._store.update(entry.id, tags=[*entry.tags, STALE_TAG])
                tagged += 1
            if len(page) < _PAGE:
                break
            offset += _PAGE
        if tagged:
            logger.info("staleness scan tagged %d entries as %s", tagged, STALE_TAG)
        return tagged

    def verify_all(self, tags: list[str] | None = None) -> int:
        """Bulk-clear ``potentially_stale`` from matching live entries.

        Explicit human/agent verification — deliberately NOT an access
        (``last_accessed_at`` unchanged; per-entry ``verify()`` in the decay
        module is the access-counting variant).

        Args:
            tags: Only clear entries carrying ANY of these tags; None
                clears every stale-tagged entry.

        Returns:
            The number of entries cleared.
        """
        # Collect-first (clearing the tag SHRINKS the tags=[STALE_TAG] result
        # set, so offset paging while mutating would skip rows) — but hold
        # ids only, never the full hydrated entries.
        matching_ids: list[str] = []
        offset = 0
        while True:
            page = self._store.list(tags=[STALE_TAG], limit=_PAGE, offset=offset)
            for entry in page:
                if tags is not None and not set(tags) & set(entry.tags):
                    continue
                matching_ids.append(entry.id)
            if len(page) < _PAGE:
                break
            offset += _PAGE
        cleared = 0
        for entry_id in matching_ids:
            refreshed = self._store.get_by_id(entry_id, touch=False)
            if refreshed is None or STALE_TAG not in refreshed.tags:
                continue
            self._store.update(entry_id, tags=[t for t in refreshed.tags if t != STALE_TAG])
            cleared += 1
        return cleared

    # -------------------------------------------------------------- internals

    def _create_session(self, goal: str | None) -> str:
        """Persist a new ACTIVE session for this agent; returns its id."""
        now_iso = self._clock().isoformat()
        session_id = uuid.uuid4().hex
        self._backend.create_session(
            {
                "id": session_id,
                "agent_id": self._agent_id,
                "goal": goal,
                "started_at": now_iso,
                "last_activity_at": now_iso,
                "status": SessionStatus.ACTIVE.value,
            }
        )
        return session_id

    def _active_row(self) -> dict[str, Any] | None:
        """This agent's ACTIVE session row, or None."""
        rows = self._backend.list_sessions(
            agent_id=self._agent_id, status=SessionStatus.ACTIVE.value
        )
        return rows[0] if rows else None

    def _is_stale(self, session: Session, now: datetime) -> bool:
        """True when the session is abandonable at ``now``.

        Stale means ``last_activity_at`` deviates from ``now`` by MORE than
        ``config.inactivity_threshold`` in EITHER direction: idle past the
        threshold (the normal case), or implausibly far in the FUTURE — a
        corrupt row or clock jump that, treated as "live", would lock its
        agent out of ``session_start`` indefinitely (SEV-003). Small
        forward skew (within the threshold) still counts as live.
        """
        delta = now - session.last_activity_at
        threshold = self._config.inactivity_threshold
        return delta > threshold or -delta > threshold

    def _abandon(self, session: Session, now: datetime) -> AbandonedSession:
        """Flip one stale ACTIVE session to ABANDONED (deterministic, no LLM).

        Stores the template marker, persists the terminal state, archives
        nothing. ``last_activity_at`` is preserved as evidence of the gap.
        """
        gap = now - session.last_activity_at
        memory_count = self._memory_count(session.id)
        # ONE transaction over marker + status flip (DB atomicity), mirroring
        # end_session.
        with self._backend.transaction():
            marker = self._store_marker(
                session,
                now=now,
                abandoned=True,
                summaries=[],
                decisions=[],
                memory_count=memory_count,
                gap=gap,
            )
            self._backend.update_session(
                session.id,
                {"status": SessionStatus.ABANDONED.value, "ended_at": now.isoformat()},
            )
        flipped = replace(
            session,
            status=SessionStatus.ABANDONED,
            ended_at=now,
            memory_count=memory_count,
        )
        logger.info(
            "session %s (agent %r) abandoned after %s of inactivity",
            session.id,
            session.agent_id,
            gap,
        )
        return AbandonedSession(session=flipped, gap=gap, marker_entry_id=marker.id)

    def _memory_count(self, session_id: str) -> int:
        """Memories stored during the session (system SUMMARY rows excluded)."""
        return self._store.count(
            session_id=session_id,
            types=list(_COUNTED_TYPES),
            include_archived=True,
        )

    def _session_decisions(self, session_id: str) -> list[MemoryEntry]:
        """Live DECISION entries of the session (preserved verbatim, ADR-016)."""
        decisions: list[MemoryEntry] = []
        offset = 0
        while True:
            page = self._store.list(
                session_id=session_id,
                types=[MemoryType.DECISION],
                limit=_PAGE,
                offset=offset,
            )
            decisions.extend(page)
            if len(page) < _PAGE:
                return decisions
            offset += _PAGE

    def _store_marker(
        self,
        session: Session,
        *,
        now: datetime,
        abandoned: bool,
        summaries: list[MemoryEntry],
        decisions: list[MemoryEntry],
        memory_count: int,
        gap: timedelta | None,
    ) -> MemoryEntry:
        """Persist the deterministic session marker; returns the entry.

        MUST run inside an ambient ``backend.transaction()`` — callers hoist
        one transaction over marker + session-status flip so the pair is
        atomic. Written straight through the backend (the store's summary
        path demands back-links; markers deliberately carry none — the
        group SUMMARY entries hold them). The D1 supersede discipline is
        mirrored for the marker key. User writes can never squat this
        namespace: the store rejects ``session:``-prefixed keys at its
        public boundary (SEC-SEV-001).
        """
        content = (
            self._abandoned_marker_content(session, memory_count=memory_count, gap=gap)
            if abandoned
            else self._ended_marker_content(
                session,
                now=now,
                summaries=summaries,
                decisions=decisions,
                memory_count=memory_count,
            )
        )
        tags = [*_MARKER_TAGS, "abandoned"] if abandoned else list(_MARKER_TAGS)
        entry = MemoryEntry(
            id=uuid.uuid4().hex,
            content=content,
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id=session.agent_id),
            key=f"{SESSION_KEY_PREFIX}{session.id}",
            tags=tags,
            session_id=session.id,
            base_importance=SESSION_MARKER_IMPORTANCE,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        old = self._backend.get_by_key(entry.key) if entry.key is not None else None
        if old is not None:
            self._backend.update(
                old["id"],
                {
                    "archived": True,
                    "archive_reason": ArchiveReason.SUPERSEDED.value,
                    "updated_at": now.isoformat(),
                },
            )
            entry.relationships.append(
                Relationship(target_id=str(old["id"]), relationship_type="supersedes")
            )
        self._backend.create(entry.to_dict(), embedding=None)
        return entry

    def _ended_marker_content(
        self,
        session: Session,
        *,
        now: datetime,
        summaries: list[MemoryEntry],
        decisions: list[MemoryEntry],
        memory_count: int,
    ) -> str:
        """Deterministic template for a cleanly ended session (no LLM)."""
        goal = session.goal or "(no goal)"
        lines = [
            f"Session '{goal}': {session.started_at.isoformat()} - {now.isoformat()}. "
            f"{memory_count} memories stored, {len(decisions)} decisions made."
        ]
        if summaries:
            refs = ", ".join(s.key if s.key is not None else s.id for s in summaries)
            lines.append(f"Summaries: {refs}")
        if decisions:
            if self._config.preserve_decisions_verbatim:
                lines.append("Key decisions:")
                lines.extend(f"- {decision.content}" for decision in decisions)
            else:
                lines.append(f"Decisions preserved: {len(decisions)}")
        return "\n".join(lines)

    @staticmethod
    def _abandoned_marker_content(
        session: Session, *, memory_count: int, gap: timedelta | None
    ) -> str:
        """Deterministic template for an abandoned session (D8: never an LLM)."""
        goal = session.goal or "(no goal)"
        return (
            f"Session '{goal}': started {session.started_at.isoformat()}, "
            f"abandoned - recovered at startup after {gap} of inactivity. "
            f"{memory_count} memories stored."
        )


def _require_aware(value: datetime, field_name: str) -> None:
    """Reject naive datetimes (entry.py convention)."""
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (naive datetime rejected)")


def _parse_aware(value: str, field_name: str) -> datetime:
    """Parse an ISO-8601 string that must carry a UTC offset."""
    parsed = datetime.fromisoformat(value)
    _require_aware(parsed, field_name)
    return parsed

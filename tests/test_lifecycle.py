"""Tests for tulving.context.lifecycle — written BEFORE implementation.

Per-agent sessions (D7), abandonment detection (L4 — never an LLM call),
staleness scanning (L5), deterministic session markers, and the injected
summarizer port. InMemory backend, injected FakeClock, no sleeps.

Owns the mandatory audit regression: one agent's startup check never
abandons another agent's live session.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest
from conftest import CLOCK_START, FakeClock

from tulving.adapters.storage import InMemoryBackend, SQLiteBackend
from tulving.context.config import LifecycleConfig
from tulving.context.lifecycle import (
    SESSION_KEY_PREFIX,
    SESSION_MARKER_IMPORTANCE,
    STALE_TAG,
    AbandonedSession,
    LifecycleManager,
    Session,
    SessionSummary,
)
from tulving.entry import MemoryEntry, SourceInfo
from tulving.enums import ArchiveReason, MemoryType, SessionStatus
from tulving.exceptions import ConfigError, MemoryStoreError, StorageError
from tulving.store import MemoryStore

AGENT = "agent-a"
OTHER_AGENT = "agent-b"


# ---------------------------------------------------------------------------
# Fixtures & fakes
# ---------------------------------------------------------------------------


@dataclass
class Env:
    """One backend + store + clock shared by every manager in a test."""

    backend: InMemoryBackend
    store: MemoryStore
    clock: FakeClock


@pytest.fixture
def env(fake_clock: FakeClock) -> Env:
    backend = InMemoryBackend()
    return Env(backend=backend, store=MemoryStore(backend, clock=fake_clock), clock=fake_clock)


def make_manager(
    env: Env,
    *,
    agent_id: str = AGENT,
    config: LifecycleConfig | None = None,
    summarizer: Any = None,
    staleness: Any = None,
) -> LifecycleManager:
    return LifecycleManager(
        env.store,
        env.backend,
        config if config is not None else LifecycleConfig(),
        agent_id,
        summarizer=summarizer,
        staleness=staleness,
        clock=env.clock,
    )


def put(
    env: Env,
    *,
    content: str = "a fact",
    type_: MemoryType = MemoryType.FACT,
    key: str | None = None,
    tags: list[str] | None = None,
    session_id: str | None = None,
    agent: str = AGENT,
    pinned: bool = False,
) -> MemoryEntry:
    return env.store.create(
        content=content,
        type=type_,
        source=SourceInfo(agent_id=agent),
        key=key,
        tags=tags,
        session_id=session_id,
        pinned=pinned,
    )


@dataclass
class RecordingSummarizer:
    """Summarizer port fake: records calls, returns a canned result."""

    result: list[MemoryEntry] = field(default_factory=list)
    calls: list[str | None] = field(default_factory=list)

    def summarize(self, *, session_id: str | None = None) -> list[MemoryEntry]:
        self.calls.append(session_id)
        return list(self.result)


class ArchivingSummarizer:
    """End-to-end fake honoring the real summarizer's write contract:
    one SUMMARY with back-links, sources archived SUMMARIZED. No LLM."""

    def __init__(self, env: Env) -> None:
        self._env = env
        self.calls: list[str | None] = []

    def summarize(self, *, session_id: str | None = None) -> list[MemoryEntry]:
        self.calls.append(session_id)
        sources = [
            entry
            for entry in self._env.store.list(session_id=session_id, limit=500)
            if entry.type is not MemoryType.DECISION and not entry.pinned
        ]
        if not sources:
            return []
        summary = self._env.store.create(
            content="digest of " + ", ".join(sorted(e.id for e in sources)),
            type=MemoryType.SUMMARY,
            source=SourceInfo(agent_id=AGENT),
            session_id=session_id,
            _allow_summary=True,
            _source_entry_ids=[e.id for e in sources],
        )
        for source in sources:
            self._env.store.archive(source.id, ArchiveReason.SUMMARIZED)
        return [summary]


@dataclass
class RecordingScanner:
    """Staleness port fake: records the threshold, returns canned ids."""

    result: list[str] = field(default_factory=list)
    thresholds: list[int] = field(default_factory=list)

    def scan(self, threshold_days: int) -> list[str]:
        self.thresholds.append(threshold_days)
        return list(self.result)


def marker_key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


def session_row(env: Env, session_id: str) -> dict[str, Any]:
    row = env.backend.get_session(session_id)
    assert row is not None
    return row


def ts(value: Any) -> datetime:
    """Parse a backend ISO timestamp (the backend canonicalizes formatting)."""
    assert isinstance(value, str)
    return datetime.fromisoformat(value)


def legacy_backend_row(env: Env, *, key: str, content: str = "legacy squatter") -> MemoryEntry:
    """Persist an entry DIRECTLY via the backend, bypassing the store's
    reserved-prefix guard — models a hostile/legacy row that predates it."""
    now = env.clock.current
    entry = MemoryEntry(
        id=uuid.uuid4().hex,
        content=content,
        type=MemoryType.FACT,
        source=SourceInfo(agent_id=AGENT),
        key=key,
        created_at=now,
        updated_at=now,
        last_accessed_at=now,
    )
    env.backend.create(entry.to_dict(), embedding=None)
    return entry


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_constructor_rejects_empty_agent_id(self, env: Env) -> None:
        with pytest.raises(ConfigError):
            LifecycleManager(env.store, env.backend, LifecycleConfig(), "", clock=env.clock)

    def test_session_start_refuses_live_active_session_naming_agent(self, env: Env) -> None:
        manager = make_manager(env)
        manager.session_start(goal="first")
        with pytest.raises(MemoryStoreError, match=AGENT):
            manager.session_start(goal="second")

    def test_session_end_unknown_id_raises_no_active_session(self, env: Env) -> None:
        manager = make_manager(env)
        with pytest.raises(MemoryStoreError, match="No active session"):
            manager.session_end("missing-id")

    def test_double_end_raises_no_active_session(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        manager.session_end(sid)
        with pytest.raises(MemoryStoreError, match="No active session"):
            manager.session_end(sid)

    def test_end_of_abandoned_session_raises_terminal(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=31)
        assert manager.check_on_startup() == 1
        with pytest.raises(MemoryStoreError, match="No active session"):
            manager.session_end(sid)

    def test_ending_another_agents_session_is_refused(self, env: Env) -> None:
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        sid = manager_b.session_start()
        with pytest.raises(MemoryStoreError, match="No active session"):
            manager_a.session_end(sid)
        assert session_row(env, sid)["status"] == SessionStatus.ACTIVE.value

    def test_record_activity_on_ended_session_is_a_noop(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        manager.session_end(sid)
        before = session_row(env, sid)
        env.clock.advance(minutes=5)
        manager.record_activity(sid)  # no raise
        assert session_row(env, sid) == before

    def test_record_activity_on_unknown_session_is_a_noop(self, env: Env) -> None:
        manager = make_manager(env)
        manager.record_activity("no-such-session")  # no raise

    def test_config_rejects_negative_activity_debounce(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(activity_debounce_seconds=-1.0)

    def test_config_rejects_infinite_activity_debounce(self) -> None:
        with pytest.raises(ConfigError):
            LifecycleConfig(activity_debounce_seconds=float("inf"))
        with pytest.raises(ConfigError):
            LifecycleConfig(activity_debounce_seconds=float("nan"))

    def test_config_rejects_debounce_at_or_past_inactivity_threshold(self) -> None:
        # default inactivity_threshold is 30 min = 1800 s
        with pytest.raises(ConfigError):
            LifecycleConfig(activity_debounce_seconds=1800.0)
        with pytest.raises(ConfigError):
            LifecycleConfig(activity_debounce_seconds=7200.0)
        # the cross-check bites from the threshold side too (default 60 s)
        with pytest.raises(ConfigError):
            LifecycleConfig(inactivity_threshold=timedelta(seconds=59))
        # just under the threshold is legal
        config = LifecycleConfig(activity_debounce_seconds=1799.0)
        assert config.activity_debounce_seconds == 1799.0

    def test_session_invariants(self) -> None:
        now = CLOCK_START
        with pytest.raises(ValueError):
            Session(id="", agent_id=AGENT, goal=None, started_at=now, last_activity_at=now)
        with pytest.raises(ValueError):
            Session(id="s1", agent_id="", goal=None, started_at=now, last_activity_at=now)
        naive = datetime(2026, 7, 3, 12, 0, 0)
        with pytest.raises(ValueError):
            Session(id="s1", agent_id=AGENT, goal=None, started_at=naive, last_activity_at=now)
        with pytest.raises(ValueError):
            Session(id="s1", agent_id=AGENT, goal=None, started_at=now, last_activity_at=naive)
        # ended_at set iff status != ACTIVE
        with pytest.raises(ValueError):
            Session(
                id="s1",
                agent_id=AGENT,
                goal=None,
                started_at=now,
                last_activity_at=now,
                ended_at=now,
                status=SessionStatus.ACTIVE,
            )
        with pytest.raises(ValueError):
            Session(
                id="s1",
                agent_id=AGENT,
                goal=None,
                started_at=now,
                last_activity_at=now,
                ended_at=None,
                status=SessionStatus.ENDED,
            )
        with pytest.raises(ValueError):
            Session(
                id="s1",
                agent_id=AGENT,
                goal=None,
                started_at=now,
                last_activity_at=now,
                memory_count=-1,
            )

    def test_session_from_dict_rejects_naive_datetimes(self) -> None:
        data = {
            "id": "s1",
            "agent_id": AGENT,
            "goal": None,
            "started_at": "2026-07-03T12:00:00",  # naive
            "last_activity_at": "2026-07-03T12:00:00+00:00",
            "ended_at": None,
            "status": "active",
        }
        with pytest.raises(ValueError):
            Session.from_dict(data)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_gap_exactly_at_threshold_is_not_abandoned(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=30)  # gap == threshold, strictly > required
        assert manager.check_on_startup() == 0
        assert session_row(env, sid)["status"] == SessionStatus.ACTIVE.value
        assert env.store.get_by_key(marker_key(sid), touch=False) is None

    def test_gap_just_over_threshold_is_abandoned_with_exact_gap(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=30, seconds=1)
        abandoned = manager.detect_abandoned_sessions()
        assert len(abandoned) == 1
        assert abandoned[0].gap == timedelta(minutes=30, seconds=1)
        assert abandoned[0].session.id == sid
        assert abandoned[0].session.status is SessionStatus.ABANDONED

    def test_read_activity_inside_debounce_window_writes_nothing(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        started = session_row(env, sid)["last_activity_at"]
        env.clock.advance(seconds=60)  # == debounce, strictly > required
        manager.record_activity(sid, write=False)
        assert session_row(env, sid)["last_activity_at"] == started

    def test_read_activity_past_debounce_window_updates(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(seconds=61)
        manager.record_activity(sid, write=False)
        assert ts(session_row(env, sid)["last_activity_at"]) == env.clock.current

    def test_write_activity_always_updates_even_inside_debounce(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(seconds=1)
        manager.record_activity(sid, write=True)
        assert ts(session_row(env, sid)["last_activity_at"]) == env.clock.current


# ---------------------------------------------------------------------------
# Sessions — start / activity / end
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_session_start_persists_active_with_synchronized_timestamps(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start(goal="build the thing")
        assert isinstance(sid, str) and sid
        row = session_row(env, sid)
        assert row["status"] == SessionStatus.ACTIVE.value
        assert row["agent_id"] == AGENT
        assert row["goal"] == "build the thing"
        assert ts(row["started_at"]) == env.clock.current
        assert row["last_activity_at"] == row["started_at"]
        assert row["ended_at"] is None

    def test_other_agents_active_session_never_blocks(self, env: Env) -> None:
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        sid_b = manager_b.session_start(goal="b's work")
        sid_a = manager_a.session_start(goal="a's work")  # must not raise (D7)
        assert sid_a != sid_b
        assert session_row(env, sid_b)["status"] == SessionStatus.ACTIVE.value

    def test_stale_own_session_is_abandoned_then_fresh_start(self, env: Env) -> None:
        manager = make_manager(env)
        old_sid = manager.session_start(goal="stale work")
        env.clock.advance(minutes=31)
        new_sid = manager.session_start(goal="fresh work")
        assert new_sid != old_sid
        assert session_row(env, old_sid)["status"] == SessionStatus.ABANDONED.value
        assert session_row(env, new_sid)["status"] == SessionStatus.ACTIVE.value
        marker = env.store.get_by_key(marker_key(old_sid), touch=False)
        assert marker is not None
        assert "abandoned" in marker.tags

    def test_ensure_active_session_creates_anonymous_session(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.ensure_active_session()
        row = session_row(env, sid)
        assert row["goal"] is None
        assert row["status"] == SessionStatus.ACTIVE.value

    def test_ensure_active_session_returns_existing_and_bumps_activity(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start(goal="ongoing")
        env.clock.advance(minutes=5)
        assert manager.ensure_active_session() == sid
        assert ts(session_row(env, sid)["last_activity_at"]) == env.clock.current

    def test_get_session_hydrates_with_derived_memory_count(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        put(env, session_id=sid)
        put(env, session_id=sid, type_=MemoryType.DECISION, content="decided")
        session = manager.get_session(sid)
        assert session is not None
        assert session.memory_count == 2
        assert session.status is SessionStatus.ACTIVE
        assert manager.get_session("missing") is None

    def test_close_active_session_ends_it(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        manager.close_active_session()
        assert session_row(env, sid)["status"] == SessionStatus.ENDED.value
        assert env.store.get_by_key(marker_key(sid), touch=False) is not None

    def test_close_active_session_without_one_is_a_noop(self, env: Env) -> None:
        manager = make_manager(env)
        manager.close_active_session()  # no raise

    def test_ensure_active_session_mints_fresh_after_terminal_sessions(self, env: Env) -> None:
        manager = make_manager(env)
        ended_sid = manager.session_start(goal="done work")
        manager.session_end(ended_sid)
        abandoned_sid = manager.session_start(goal="crashed work")
        env.clock.advance(minutes=31)
        assert manager.check_on_startup() == 1  # abandoned_sid flips

        fresh_sid = manager.ensure_active_session()
        assert fresh_sid not in {ended_sid, abandoned_sid}
        assert session_row(env, fresh_sid)["status"] == SessionStatus.ACTIVE.value
        # terminal rows untouched
        assert session_row(env, ended_sid)["status"] == SessionStatus.ENDED.value
        assert session_row(env, abandoned_sid)["status"] == SessionStatus.ABANDONED.value

    def test_ensure_active_session_resurrects_stale_session_not_abandons(self, env: Env) -> None:
        """Deliberate asymmetry vs session_start: continuing to store into a
        stale session re-legitimizes it — no abandonment, no marker."""
        manager = make_manager(env)
        sid = manager.session_start(goal="long pause")
        env.clock.advance(minutes=45)  # well past the 30-min threshold

        assert manager.ensure_active_session() == sid
        row = session_row(env, sid)
        assert row["status"] == SessionStatus.ACTIVE.value
        assert ts(row["last_activity_at"]) == env.clock.current  # bumped
        assert env.store.get_by_key(marker_key(sid), touch=False) is None  # no marker


# ---------------------------------------------------------------------------
# End of session — markers, summarizer port, purge interplay
# ---------------------------------------------------------------------------


class TestEndOfSession:
    def test_end_session_happy_path_marker_and_counts(self, env: Env) -> None:
        summarizer = RecordingSummarizer()
        manager = make_manager(env, summarizer=summarizer)
        sid = manager.session_start(goal="ship it")
        put(env, session_id=sid, content="fact one")
        put(env, session_id=sid, type_=MemoryType.DECISION, content="use sqlite")
        env.clock.advance(minutes=10)

        summary = manager.end_session(sid)

        assert isinstance(summary, SessionSummary)
        assert summary.session.status is SessionStatus.ENDED
        assert summary.session.ended_at == env.clock.current
        assert summary.session.memory_count == 2
        assert summary.summaries_created == 0
        assert summary.entries_archived == 0
        assert summary.decisions_preserved == 1
        assert summarizer.calls == [sid]

        row = session_row(env, sid)
        assert row["status"] == SessionStatus.ENDED.value
        assert ts(row["ended_at"]) == env.clock.current

        marker = summary.session_marker
        assert marker.key == marker_key(sid)
        assert marker.type is MemoryType.SUMMARY
        assert marker.tags == ["session", "session_marker"]
        assert marker.base_importance == SESSION_MARKER_IMPORTANCE
        assert marker.source_entry_ids == []
        assert marker.session_id == sid
        # persisted and hydratable through the store
        stored = env.store.get_by_key(marker_key(sid), touch=False)
        assert stored is not None and stored.id == marker.id

    def test_session_end_port_shim_returns_none(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        assert manager.session_end(sid) is None

    def test_summarize_disabled_skips_summarizer_but_stores_marker(self, env: Env) -> None:
        summarizer = RecordingSummarizer()
        config = LifecycleConfig(summarize_on_session_end=False)
        manager = make_manager(env, config=config, summarizer=summarizer)
        sid = manager.session_start()
        manager.session_end(sid)
        assert summarizer.calls == []
        assert env.store.get_by_key(marker_key(sid), touch=False) is not None
        assert session_row(env, sid)["status"] == SessionStatus.ENDED.value

    def test_no_summarizer_wired_degrades_loudly_archives_nothing(
        self, env: Env, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = make_manager(env)  # summarizer=None, summarize_on_session_end=True
        sid = manager.session_start()
        put(env, session_id=sid)
        with caplog.at_level(logging.WARNING, logger="tulving.lifecycle"):
            manager.session_end(sid)
        assert any("summar" in record.message.lower() for record in caplog.records)
        archived = env.store.list(
            include_archived=True, archive_reasons=[ArchiveReason.SUMMARIZED], limit=10
        )
        assert archived == []
        assert env.store.get_by_key(marker_key(sid), touch=False) is not None
        assert session_row(env, sid)["status"] == SessionStatus.ENDED.value

    def test_marker_quotes_decisions_verbatim_by_default(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start(goal="decide things")
        put(env, session_id=sid, type_=MemoryType.DECISION, content="we chose hnswlib")
        summary = manager.end_session(sid)
        assert "we chose hnswlib" in summary.session_marker.content

    def test_marker_counts_decisions_when_verbatim_disabled(self, env: Env) -> None:
        config = LifecycleConfig(preserve_decisions_verbatim=False)
        manager = make_manager(env, config=config)
        sid = manager.session_start()
        put(env, session_id=sid, type_=MemoryType.DECISION, content="we chose hnswlib")
        summary = manager.end_session(sid)
        assert "we chose hnswlib" not in summary.session_marker.content
        assert "1" in summary.session_marker.content

    def test_marker_references_created_summaries(self, env: Env) -> None:
        summarizer = ArchivingSummarizer(env)
        manager = make_manager(env, summarizer=summarizer)
        sid = manager.session_start()
        put(env, session_id=sid, content="rolled up")
        summary = manager.end_session(sid)
        assert summary.summaries_created == 1
        created = summary.session_marker.content
        assert summarizer.calls == [sid]
        assert any(entry.id in created for entry in env.store.list(types=[MemoryType.SUMMARY]))

    def test_marker_supersedes_colliding_legacy_key(self, env: Env) -> None:
        # The store now refuses session:* keys (SEC-SEV-001), so a collision
        # can only come from a hostile/legacy backend row — the marker write
        # still supersedes it defensively.
        manager = make_manager(env)
        sid = manager.session_start()
        squatter = legacy_backend_row(env, key=marker_key(sid))
        summary = manager.end_session(sid)
        old = env.store.get_by_id(squatter.id, touch=False)
        assert old is not None and old.archived
        assert old.archive_reason is ArchiveReason.SUPERSEDED
        assert any(
            rel.relationship_type == "supersedes" and rel.target_id == squatter.id
            for rel in summary.session_marker.relationships
        )

    def test_summarized_sources_survive_default_purge(self, env: Env) -> None:
        """Audit regression mirror: purge refuses SUMMARIZED unless explicit."""
        summarizer = ArchivingSummarizer(env)
        manager = make_manager(env, summarizer=summarizer)
        sid = manager.session_start()
        source = put(env, session_id=sid, content="to be rolled up")
        manager.end_session(sid)

        archived = env.store.get_by_id(source.id, touch=False)
        assert archived is not None and archived.archive_reason is ArchiveReason.SUMMARIZED

        env.store.purge_archived()  # defaults exclude SUMMARIZED
        assert env.store.get_by_id(source.id, touch=False) is not None

        env.store.purge_archived(reasons=[ArchiveReason.SUMMARIZED])
        assert env.store.get_by_id(source.id, touch=False) is None


# ---------------------------------------------------------------------------
# Abandonment detection (L4)
# ---------------------------------------------------------------------------


class TestAbandonmentDetection:
    def test_startup_never_abandons_another_agents_live_session(self, env: Env) -> None:
        """MANDATORY AUDIT REGRESSION #5 — written to bite under mutation.

        Phase 1: agent A's session is 1 minute old; agent B's startup check
        must leave it untouched (status, timestamps, no marker). Phase 2:
        after 31 idle minutes the SAME check abandons it — proving the
        earlier skip was the threshold logic, not agent filtering or a
        dead detector.
        """
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        sid_a = manager_a.session_start(goal="a's live work")
        env.clock.advance(minutes=1)

        # B starts up: A's session is live and MUST be untouched.
        before = session_row(env, sid_a)
        assert manager_b.check_on_startup() == 0
        after = session_row(env, sid_a)
        assert after["status"] == SessionStatus.ACTIVE.value
        assert after["last_activity_at"] == before["last_activity_at"]
        assert after["ended_at"] is None
        assert after == before
        assert env.store.get_by_key(marker_key(sid_a), touch=False) is None

        # 31 more idle minutes: the same detector NOW abandons it.
        env.clock.advance(minutes=31)
        assert manager_b.check_on_startup() == 1
        final = session_row(env, sid_a)
        assert final["status"] == SessionStatus.ABANDONED.value
        assert ts(final["ended_at"]) == env.clock.current
        marker = env.store.get_by_key(marker_key(sid_a), touch=False)
        assert marker is not None
        assert "abandoned" in marker.tags

    def test_detection_is_per_session_not_store_wide(self, env: Env) -> None:
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        stale_sid = manager_a.session_start(goal="stale")
        env.clock.advance(minutes=25)
        live_sid = manager_b.session_start(goal="live")  # fresh activity
        env.clock.advance(minutes=10)  # stale gap 35min; live gap 10min

        abandoned = manager_a.detect_abandoned_sessions()
        assert [a.session.id for a in abandoned] == [stale_sid]
        assert abandoned[0].gap == timedelta(minutes=35)
        assert session_row(env, live_sid)["status"] == SessionStatus.ACTIVE.value

    def test_abandonment_stores_marker_archives_nothing_calls_no_summarizer(self, env: Env) -> None:
        summarizer = RecordingSummarizer()
        manager = make_manager(env, summarizer=summarizer)
        sid = manager.session_start()
        entry = put(env, session_id=sid, content="must stay live")
        env.clock.advance(minutes=31)

        abandoned = manager.detect_abandoned_sessions()

        assert len(abandoned) == 1
        assert summarizer.calls == []  # NEVER an LLM path during startup (D8)
        marker = env.store.get_by_key(marker_key(sid), touch=False)
        assert marker is not None
        assert marker.tags == ["session", "session_marker", "abandoned"]
        assert marker.base_importance == SESSION_MARKER_IMPORTANCE
        assert marker.source_entry_ids == []
        assert abandoned[0].marker_entry_id == marker.id
        # abandonment archives NOTHING (persistent-by-default, ADR-009)
        still_live = env.store.get_by_id(entry.id, touch=False)
        assert still_live is not None and not still_live.archived

    def test_repeated_startup_checks_are_idempotent(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=31)
        assert manager.check_on_startup() == 1
        first = session_row(env, sid)
        marker_count = len(env.store.list(tags=["session_marker"], limit=100))

        # Same injected now: nothing further happens.
        assert manager.check_on_startup() == 0
        assert session_row(env, sid) == first
        assert len(env.store.list(tags=["session_marker"], limit=100)) == marker_count

        # Later startups never re-abandon a terminal session either.
        env.clock.advance(hours=5)
        assert manager.check_on_startup() == 0
        assert len(env.store.list(tags=["session_marker"], limit=100)) == marker_count


# ---------------------------------------------------------------------------
# Staleness (L5) & verify_all
# ---------------------------------------------------------------------------


class TestStaleness:
    def test_scan_tags_only_entries_unaccessed_past_threshold(self, env: Env) -> None:
        manager = make_manager(env)
        old = put(env, content="old news")
        env.clock.advance(days=31)
        fresh = put(env, content="fresh")

        assert manager.run_staleness_scan() == 1
        old_entry = env.store.get_by_id(old.id, touch=False)
        fresh_entry = env.store.get_by_id(fresh.id, touch=False)
        assert old_entry is not None and STALE_TAG in old_entry.tags
        assert fresh_entry is not None and STALE_TAG not in fresh_entry.tags

    def test_scan_is_not_an_access_and_never_double_tags(self, env: Env) -> None:
        manager = make_manager(env)
        entry = put(env)
        accessed_at = entry.last_accessed_at
        env.clock.advance(days=31)

        assert manager.run_staleness_scan() == 1
        again = env.store.get_by_id(entry.id, touch=False)
        assert again is not None
        assert again.last_accessed_at == accessed_at  # scan is NOT an access
        assert again.tags.count(STALE_TAG) == 1

        assert manager.run_staleness_scan() == 0  # already tagged: idempotent
        final = env.store.get_by_id(entry.id, touch=False)
        assert final is not None and final.tags.count(STALE_TAG) == 1

    def test_scan_delegates_to_injected_scanner_port(self, env: Env) -> None:
        scanner = RecordingScanner(result=["id-1", "id-2"])
        config = LifecycleConfig(staleness_threshold_days=7)
        manager = make_manager(env, config=config, staleness=scanner)
        put(env)
        env.clock.advance(days=31)
        assert manager.run_staleness_scan() == 2
        assert scanner.thresholds == [7]
        # the native scan did not also run
        entry = env.store.list(limit=10)[0]
        assert STALE_TAG not in entry.tags

    def test_verify_all_clears_tag_without_counting_as_access(self, env: Env) -> None:
        manager = make_manager(env)
        entry = put(env, tags=["topic"])
        env.clock.advance(days=31)
        manager.run_staleness_scan()
        tagged = env.store.get_by_id(entry.id, touch=False)
        assert tagged is not None and STALE_TAG in tagged.tags

        assert manager.verify_all() == 1
        cleared = env.store.get_by_id(entry.id, touch=False)
        assert cleared is not None
        assert STALE_TAG not in cleared.tags
        assert cleared.tags == ["topic"]
        assert cleared.last_accessed_at == entry.last_accessed_at  # NOT an access

    def test_verify_all_with_tag_filter_only_touches_matches(self, env: Env) -> None:
        manager = make_manager(env)
        alpha = put(env, content="alpha", tags=["alpha"])
        beta = put(env, content="beta", tags=["beta"])
        env.clock.advance(days=31)
        assert manager.run_staleness_scan() == 2

        assert manager.verify_all(tags=["alpha"]) == 1
        alpha_entry = env.store.get_by_id(alpha.id, touch=False)
        beta_entry = env.store.get_by_id(beta.id, touch=False)
        assert alpha_entry is not None and STALE_TAG not in alpha_entry.tags
        assert beta_entry is not None and STALE_TAG in beta_entry.tags

    def test_verify_all_with_nothing_stale_returns_zero(self, env: Env) -> None:
        manager = make_manager(env)
        put(env)
        assert manager.verify_all() == 0


# ---------------------------------------------------------------------------
# Security — reserved marker namespace (SEC-SEV-001) & agent isolation (SEV-002)
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_squat_after_end_is_rejected_and_marker_survives(self, env: Env) -> None:
        """SEC-SEV-001 regression: after a session ends, a user store() on
        the marker key must be REFUSED — the genuine marker stays the live
        key-holder instead of being D1-superseded."""
        manager = make_manager(env)
        sid = manager.session_start(goal="protect me")
        summary = manager.end_session(sid)

        with pytest.raises(MemoryStoreError, match="reserved"):
            put(env, key=marker_key(sid), content="squat attempt")

        survivor = env.store.get_by_key(marker_key(sid), touch=False)
        assert survivor is not None
        assert survivor.id == summary.session_marker.id
        assert not survivor.archived
        assert survivor.content != "squat attempt"

    def test_marker_read_path_returns_lifecycle_tagged_summary(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        manager.session_end(sid)

        marker = env.store.get_by_key(f"session:{sid}", touch=False)
        assert marker is not None
        assert marker.type is MemoryType.SUMMARY
        assert "session_marker" in marker.tags
        assert marker.session_id == sid

    def test_reserved_prefix_constants_stay_in_sync(self) -> None:
        from tulving.store import _RESERVED_SESSION_KEY_PREFIX

        assert _RESERVED_SESSION_KEY_PREFIX == SESSION_KEY_PREFIX

    def test_record_activity_ignores_foreign_session(self, env: Env) -> None:
        """SEC-SEV-002: agent A can never bump agent B's last_activity_at."""
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        sid_b = manager_b.session_start()
        before = session_row(env, sid_b)["last_activity_at"]
        env.clock.advance(minutes=5)

        manager_a.record_activity(sid_b, write=True)
        assert session_row(env, sid_b)["last_activity_at"] == before

        manager_b.record_activity(sid_b, write=True)  # the owner still can
        assert ts(session_row(env, sid_b)["last_activity_at"]) == env.clock.current

    def test_get_session_hides_foreign_session(self, env: Env) -> None:
        """SEC-SEV-002: agent A can never read agent B's session record."""
        manager_a = make_manager(env)
        manager_b = make_manager(env, agent_id=OTHER_AGENT)
        sid_b = manager_b.session_start(goal="b's secret goal")

        assert manager_a.get_session(sid_b) is None
        own = manager_b.get_session(sid_b)
        assert own is not None and own.goal == "b's secret goal"


# ---------------------------------------------------------------------------
# Clock robustness (reviewer L1 / SEC SEV-003)
# ---------------------------------------------------------------------------


class TestClockRobustness:
    def test_backwards_clock_never_rewinds_record_activity(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=10)
        manager.record_activity(sid, write=True)
        high_water = session_row(env, sid)["last_activity_at"]

        env.clock.advance(minutes=-5)  # clock jumps backwards
        manager.record_activity(sid, write=True)
        assert session_row(env, sid)["last_activity_at"] == high_water
        manager.record_activity(sid, write=False)
        assert session_row(env, sid)["last_activity_at"] == high_water

    def test_backwards_clock_never_rewinds_ensure_active_session(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        env.clock.advance(minutes=10)
        manager.record_activity(sid, write=True)
        high_water = session_row(env, sid)["last_activity_at"]

        env.clock.advance(minutes=-7)
        assert manager.ensure_active_session() == sid
        assert session_row(env, sid)["last_activity_at"] == high_water

    def test_far_future_activity_does_not_lock_out_session_start(self, env: Env) -> None:
        """SEV-003: a corrupt/far-future last_activity_at must not make
        'Session already active' permanent — it is treated as stale."""
        manager = make_manager(env)
        old_sid = manager.session_start()
        future = env.clock.current + timedelta(days=10)
        env.backend.update_session(old_sid, {"last_activity_at": future.isoformat()})

        new_sid = manager.session_start(goal="recovered")
        assert new_sid != old_sid
        assert session_row(env, old_sid)["status"] == SessionStatus.ABANDONED.value
        assert session_row(env, new_sid)["status"] == SessionStatus.ACTIVE.value

    def test_far_future_session_is_abandonable_by_detection(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start()
        future = env.clock.current + timedelta(days=10)
        env.backend.update_session(sid, {"last_activity_at": future.isoformat()})

        assert manager.check_on_startup() == 1
        assert session_row(env, sid)["status"] == SessionStatus.ABANDONED.value

    def test_small_future_skew_still_counts_as_live(self, env: Env) -> None:
        """Forward skew WITHIN the threshold is live: session_start refuses
        and detection leaves it alone (no false abandonment)."""
        manager = make_manager(env)
        sid = manager.session_start()
        near_future = env.clock.current + timedelta(minutes=5)  # threshold 30m
        env.backend.update_session(sid, {"last_activity_at": near_future.isoformat()})

        assert manager.check_on_startup() == 0
        with pytest.raises(MemoryStoreError, match="already active"):
            manager.session_start()
        assert session_row(env, sid)["status"] == SessionStatus.ACTIVE.value


# ---------------------------------------------------------------------------
# Port compliance (mechanical swap for memory.py's _DefaultLifecycleManager)
# ---------------------------------------------------------------------------


class TestPortCompliance:
    def test_satisfies_lifecycle_manager_port(self, env: Env) -> None:
        from tulving.memory import LifecycleManagerPort

        manager = make_manager(env)
        port: LifecycleManagerPort = manager  # structural — mypy-visible too
        assert port is manager
        sig = inspect.signature(manager.session_start)
        assert sig.parameters["goal"].default is None
        for name in (
            "ensure_active_session",
            "session_start",
            "session_end",
            "check_on_startup",
            "run_staleness_scan",
            "close_active_session",
        ):
            assert callable(getattr(manager, name))

    def test_port_return_types(self, env: Env) -> None:
        manager = make_manager(env)
        sid = manager.session_start(goal=None)
        assert isinstance(sid, str)
        assert manager.session_end(sid) is None
        assert isinstance(manager.ensure_active_session(), str)
        assert isinstance(manager.check_on_startup(), int)
        assert isinstance(manager.run_staleness_scan(), int)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_session_round_trips_via_to_dict_from_dict(self) -> None:
        started = CLOCK_START
        ended = CLOCK_START + timedelta(hours=1)
        session = Session(
            id="s1",
            agent_id=AGENT,
            goal="round trip",
            started_at=started,
            last_activity_at=ended,
            ended_at=ended,
            memory_count=3,
            status=SessionStatus.ENDED,
        )
        assert Session.from_dict(session.to_dict()) == session

    def test_from_dict_ignores_unknown_keys_and_defaults_memory_count(self) -> None:
        data = {
            "id": "s1",
            "agent_id": AGENT,
            "goal": None,
            "started_at": CLOCK_START.isoformat(),
            "last_activity_at": CLOCK_START.isoformat(),
            "ended_at": None,
            "status": "active",
            "v2_column": "ignored",
        }
        session = Session.from_dict(data)
        assert session.memory_count == 0
        assert session.status is SessionStatus.ACTIVE

    def test_from_dict_normalizes_missing_last_activity_to_started(self) -> None:
        data = {
            "id": "s1",
            "agent_id": AGENT,
            "started_at": CLOCK_START.isoformat(),
            "last_activity_at": None,
            "status": "active",
        }
        session = Session.from_dict(data)
        assert session.last_activity_at == session.started_at

    def test_abandoned_session_carries_post_transition_state(self, env: Env) -> None:
        manager = make_manager(env)
        manager.session_start(goal="will crash")
        env.clock.advance(minutes=45)
        result = manager.detect_abandoned_sessions()
        assert len(result) == 1
        abandoned: AbandonedSession = result[0]
        assert abandoned.session.status is SessionStatus.ABANDONED
        assert abandoned.session.ended_at == env.clock.current
        assert abandoned.gap == timedelta(minutes=45)
        assert isinstance(abandoned.marker_entry_id, str) and abandoned.marker_entry_id


# ---------------------------------------------------------------------------
# SQLite-backed marker+flip atomicity (DB engineer gap — parity for the
# InMemory single-transaction guarantee, on the real backend)
# ---------------------------------------------------------------------------


class _FlipFailsSQLiteBackend(SQLiteBackend):
    """Real SQLite backend that crashes at the session STATUS flip only.

    Models a crash landing BETWEEN the marker write and the status flip that
    ``end_session``/``_abandon`` hoist under one ``backend.transaction()``:
    the marker INSERT has already run when the flip raises. Atomicity demands
    the whole scope roll back — session stays ACTIVE, no orphan marker row.
    """

    def update_session(self, session_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        if "status" in fields:
            raise StorageError("injected crash at session status flip")
        return super().update_session(session_id, fields)


class TestSQLiteMarkerFlipAtomicity:
    def _wire(self, backend: SQLiteBackend, clock: FakeClock) -> LifecycleManager:
        store = MemoryStore(backend, clock=clock)
        return LifecycleManager(store, backend, LifecycleConfig(), AGENT, clock=clock)

    def test_end_session_marker_and_flip_atomic_on_sqlite(
        self, tmp_path: Any, fake_clock: FakeClock
    ) -> None:
        backend = _FlipFailsSQLiteBackend(tmp_path / "life.db")
        try:
            manager = self._wire(backend, fake_clock)
            store = MemoryStore(backend, clock=fake_clock)
            sid = manager.session_start(goal="atomic end")

            with pytest.raises(StorageError, match="injected crash"):
                manager.end_session(sid)

            # Session untouched (still ACTIVE, no ended_at) — the flip rolled back.
            row = backend.get_session(sid)
            assert row is not None
            assert row["status"] == SessionStatus.ACTIVE.value
            assert row["ended_at"] is None
            # AND the marker INSERT rolled back with it: no orphan marker row.
            assert store.get_by_key(marker_key(sid), touch=False) is None
        finally:
            backend.close()

    def test_abandon_marker_and_flip_atomic_on_sqlite(
        self, tmp_path: Any, fake_clock: FakeClock
    ) -> None:
        backend = _FlipFailsSQLiteBackend(tmp_path / "life.db")
        try:
            manager = self._wire(backend, fake_clock)
            store = MemoryStore(backend, clock=fake_clock)
            sid = manager.session_start(goal="atomic abandon")
            fake_clock.advance(minutes=31)

            with pytest.raises(StorageError, match="injected crash"):
                manager.detect_abandoned_sessions()

            row = backend.get_session(sid)
            assert row is not None
            assert row["status"] == SessionStatus.ACTIVE.value  # NOT flipped to ABANDONED
            assert row["ended_at"] is None
            assert store.get_by_key(marker_key(sid), touch=False) is None
        finally:
            backend.close()


# ---------------------------------------------------------------------------
# Pagination boundaries (pins the page-full continue branches: staleness scan,
# verify_all, _session_decisions) — _PAGE shrunk so tiny sets span pages
# ---------------------------------------------------------------------------


class TestPaginationBranches:
    def test_staleness_scan_spans_multiple_pages(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tulving.context.lifecycle._PAGE", 2)
        manager = make_manager(env)
        for i in range(5):
            put(env, content=f"old-{i}")
        env.clock.advance(days=31)
        assert manager.run_staleness_scan() == 5  # 3 pages (2 + 2 + 1)
        for entry in env.store.list(limit=10):
            assert STALE_TAG in entry.tags

    def test_verify_all_spans_multiple_pages(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tulving.context.lifecycle._PAGE", 2)
        manager = make_manager(env)
        for i in range(5):
            put(env, content=f"old-{i}", tags=["topic"])
        env.clock.advance(days=31)
        assert manager.run_staleness_scan() == 5
        assert manager.verify_all() == 5
        for entry in env.store.list(limit=10):
            assert STALE_TAG not in entry.tags

    def test_end_session_collects_decisions_across_pages(
        self, env: Env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tulving.context.lifecycle._PAGE", 2)
        manager = make_manager(env)
        sid = manager.session_start()
        for i in range(5):
            put(env, session_id=sid, type_=MemoryType.DECISION, content=f"decision-{i}")
        summary = manager.end_session(sid)
        assert summary.decisions_preserved == 5

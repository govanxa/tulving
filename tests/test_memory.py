"""Tests for tulving.memory — written BEFORE implementation.

Integration-level: ``InMemoryBackend`` + a fake semantic index (the real
``tulving/semantic_index.py`` is being built in parallel; Memory codes
against the blueprint-semantic-index interface through an injectable seam)
+ stub lifecycle managers; ``SQLiteBackend`` (default construction) for
lock/restart/persistence tests. No sleeps — injected ``FakeClock`` +
``startup_deadline_seconds`` force every timing path.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from conftest import FakeClock

from tulving.adapters.embeddings import HashEmbedder
from tulving.adapters.storage import InMemoryBackend
from tulving.enums import ArchiveReason, MatchType, MemoryType, SessionStatus
from tulving.exceptions import (
    ConfigError,
    MemoryStoreError,
    SecurityError,
    StorageError,
    VectorIndexError,
)
from tulving.memory import Memory, SearchResult, StartupReport

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------


class _ReconcileReport:
    """Duck-typed stand-in for semantic_index.ReconcileReport."""

    def __init__(self, *, full_rebuild: bool = False) -> None:
        self.added_from_blobs = 0
        self.dropped_orphans = 0
        self.full_rebuild = full_rebuild
        self.compacted = False
        self.tombstone_fraction = 0.0


class FakeSemanticIndex:
    """In-memory stand-in implementing blueprint-semantic-index's consumed
    surface: (entry_id, score) pairs, tombstone remove, open/reconcile."""

    def __init__(self) -> None:
        self.vectors: dict[str, str] = {}
        self.tombstoned: set[str] = set()
        self.opened = False
        self.open_error: str | None = None
        self.reconcile_report: Any = _ReconcileReport()
        self.calls: list[str] = []
        self.search_calls = 0
        self.on_rebuild: Callable[[FakeSemanticIndex], None] | None = None

    def open(self) -> None:
        self.calls.append("open")
        if self.open_error is not None:
            raise VectorIndexError(self.open_error)
        self.opened = True

    def reconcile(self) -> Any:
        self.calls.append("reconcile")
        return self.reconcile_report

    def add(self, entry_id: str, text: str) -> list[float]:
        if entry_id in self.vectors and entry_id not in self.tombstoned:
            raise VectorIndexError(f"entry {entry_id!r} is already live in the index")
        self.vectors[entry_id] = text
        self.tombstoned.discard(entry_id)
        return [0.0]

    def remove(self, entry_id: str) -> bool:
        if entry_id not in self.vectors or entry_id in self.tombstoned:
            return False
        self.tombstoned.add(entry_id)
        return True

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]:
        self.search_calls += 1
        query_words = set(query.lower().split())
        scored: list[tuple[str, float]] = []
        for entry_id, text in self.vectors.items():
            if entry_id in self.tombstoned:
                continue
            if filter_fn is not None and not filter_fn(entry_id):
                continue
            if text == query:
                score = 1.0
            else:
                words = set(text.lower().split())
                union = query_words | words
                score = len(query_words & words) / len(union) if union else 0.0
            if score > 0.0:
                scored.append((entry_id, score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]

    def rebuild(self, *, re_embed: bool = False) -> None:
        self.calls.append("rebuild")
        self.open_error = None
        self.tombstoned.clear()
        if self.on_rebuild is not None:
            self.on_rebuild(self)

    def flush(self) -> None:
        self.calls.append("flush")

    def close(self) -> None:
        self.calls.append("close")
        self.opened = False


class StubLifecycle:
    """Stub for the blueprint's L1-L7 lifecycle seam."""

    def __init__(self, *, abandoned: int = 0, stale: int = 0) -> None:
        self.calls: list[str] = []
        self.abandoned = abandoned
        self.stale = stale
        self.close_raises = False

    def ensure_active_session(self) -> str:
        self.calls.append("ensure_active_session")
        return "stub-session"

    def session_start(self, goal: str | None = None) -> str:
        self.calls.append("session_start")
        return "stub-session"

    def session_end(self, session_id: str) -> None:
        self.calls.append("session_end")

    def check_on_startup(self) -> int:
        self.calls.append("check_on_startup")
        return self.abandoned

    def run_staleness_scan(self) -> int:
        self.calls.append("run_staleness_scan")
        return self.stale

    def close_active_session(self) -> None:
        self.calls.append("close_active_session")
        if self.close_raises:
            raise RuntimeError("lifecycle close exploded")


class FakeCurator:
    """Records delegation; stands in for ContextCurator (build step 10)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def curate(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        mode: str = "query",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> Any:
        self.calls.append(
            {
                "query": query,
                "token_budget": token_budget,
                "mode": mode,
                "include_tags": include_tags,
                "exclude_tags": exclude_tags,
            }
        )
        return f"curated:{query}"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_semantic() -> FakeSemanticIndex:
    return FakeSemanticIndex()


def make_memory(
    tmp_path: Path,
    clock: FakeClock,
    *,
    semantic: FakeSemanticIndex | None = None,
    subdir: str = "mem",
    **kwargs: Any,
) -> Memory:
    kwargs.setdefault("storage_backend", InMemoryBackend())
    kwargs.setdefault("agent_id", "agent-x")
    return Memory(
        path=str(tmp_path / subdir),
        clock=clock,
        semantic_index=semantic,
        **kwargs,
    )


@pytest.fixture
def memory(tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex) -> Any:
    handle = make_memory(tmp_path, fake_clock, semantic=fake_semantic)
    yield handle
    handle.close()


def store_fact(memory: Memory, content: str, **kwargs: Any) -> Any:
    kwargs.setdefault("type", MemoryType.FACT)
    return memory.store(content, **kwargs)


# ---------------------------------------------------------------------------
# Failure paths — constructor validation (D7/D8)
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_empty_agent_id_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            Memory(path=str(tmp_path / "m"), agent_id="", storage_backend=InMemoryBackend())

    def test_invalid_embedding_adapter_rejected(self, tmp_path: Path) -> None:
        class NotAnAdapter:
            pass

        with pytest.raises(ConfigError):
            Memory(
                path=str(tmp_path / "m"),
                embedding_adapter=NotAnAdapter(),  # type: ignore[arg-type]
                storage_backend=InMemoryBackend(),
            )

    def test_llm_without_complete_rejected(self, tmp_path: Path) -> None:
        class NotAnLLM:
            complete = "not callable"

        with pytest.raises(ConfigError):
            Memory(
                path=str(tmp_path / "m"),
                llm_adapter=NotAnLLM(),  # type: ignore[arg-type]
                storage_backend=InMemoryBackend(),
            )

    def test_llm_none_is_legal(self, tmp_path: Path) -> None:
        handle = Memory(
            path=str(tmp_path / "m"), llm_adapter=None, storage_backend=InMemoryBackend()
        )
        handle.close()

    def test_init_never_touches_adapter_identity(self, tmp_path: Path) -> None:
        """D8: constructing with an adapter whose dimension/model access raises
        must succeed — identity comparison is startup()'s job."""

        class ExplodingIdentity(HashEmbedder):
            @property
            def dimension(self) -> int:
                raise RuntimeError("model load attempted in __init__")

            @property
            def model_id(self) -> str:
                raise RuntimeError("model load attempted in __init__")

        handle = Memory(
            path=str(tmp_path / "m"),
            embedding_adapter=ExplodingIdentity(32),
            storage_backend=InMemoryBackend(),
            semantic_index=FakeSemanticIndex(),
        )
        handle.close()


# ---------------------------------------------------------------------------
# Failure paths — read-only refusal
# ---------------------------------------------------------------------------


class TestReadOnly:
    @pytest.fixture
    def ro(self, tmp_path: Path, fake_clock: FakeClock) -> Any:
        # A read-only handle coexisting with the live writer — the sanctioned
        # ADR-015 pattern (read_only skips the writer lock).
        backend = InMemoryBackend()
        writer = make_memory(tmp_path, fake_clock, storage_backend=backend)
        store_fact(writer, "shared fact", key="k")
        handle = Memory(
            path=str(tmp_path / "mem"),
            agent_id="agent-x",
            storage_backend=backend,
            clock=fake_clock,
            read_only=True,
        )
        yield handle
        handle.close()
        writer.close()

    def test_every_write_method_refused(self, ro: Memory) -> None:
        writes: list[Callable[[], Any]] = [
            lambda: ro.store("x", type=MemoryType.FACT),
            lambda: ro.update("some-id", content="x"),
            lambda: ro.pin("some-id"),
            lambda: ro.unpin("some-id"),
            lambda: ro.forget("k"),
            lambda: ro.forget_by_id("some-id"),
            lambda: ro.forget_by_tags(["t"]),
            lambda: ro.forget_by_age(timedelta(days=1)),
            lambda: ro.purge_archived(),
            lambda: ro.summarize(),
            lambda: ro.session("goal"),
            lambda: ro.rebuild_index(),
        ]
        for write in writes:
            with pytest.raises(MemoryStoreError, match="read_only"):
                write()

    def test_reads_work(self, ro: Memory) -> None:
        assert ro.get("k") is not None
        assert ro.list_keys() == ["k"]
        assert ro.search("missing-query") == []

    def test_read_only_get_never_touches(self, ro: Memory) -> None:
        entry = ro.get("k")
        assert entry is not None
        raw = ro._store.get_by_id(entry.id, touch=False)
        assert raw is not None
        assert raw.access_count == 0

    def test_read_only_search_never_touches(self, ro: Memory) -> None:
        results = ro.search("k")  # exact-key hit
        assert results
        assert results[0].match_type is MatchType.KEY
        raw = ro._store.get_by_id(results[0].entry.id, touch=False)
        assert raw is not None
        assert raw.access_count == 0


# ---------------------------------------------------------------------------
# MANDATORY AUDIT REGRESSION — second writer on the same path is refused
# ---------------------------------------------------------------------------


class TestAdvisoryLock:
    def test_second_writer_refused_in_process(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        first = Memory(path=str(path), agent_id="a")
        try:
            with pytest.raises(StorageError) as excinfo:
                Memory(path=str(path), agent_id="b")
            message = str(excinfo.value)
            assert "read_only" in message
            assert str(path) in message
        finally:
            first.close()

    def test_read_only_second_handle_allowed(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        first = Memory(path=str(path), agent_id="a")
        try:
            second = Memory(path=str(path), agent_id="b", read_only=True)
            second.close()
        finally:
            first.close()

    def test_close_releases_the_lock(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        first = Memory(path=str(path), agent_id="a")
        first.close()
        assert first._lock is not None
        first._lock.release()  # double release is a silent no-op
        second = Memory(path=str(path), agent_id="b")
        second.close()

    def test_lock_file_diagnostics_are_content_free(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        handle = Memory(path=str(path), agent_id="a")
        try:
            handle.store("SUPERSECRETCONTENT", type=MemoryType.FACT, key="secret_key")
            text = (path / "tulving.lock").read_text(encoding="utf-8")
            payload = json.loads(text.splitlines()[0])
            assert payload["pid"] == os.getpid()
            assert payload["hostname"]
            assert payload["acquired_at"]
            assert "SUPERSECRETCONTENT" not in text
            assert "secret_key" not in text
        finally:
            handle.close()

    def test_stale_lock_file_does_not_block(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        path.mkdir(parents=True)
        (path / "tulving.lock").write_bytes(b"\x00garbage-not-json\xff")
        handle = Memory(path=str(path), agent_id="a")
        handle.close()

    def test_cross_process_refusal_and_crash_recovery(self, tmp_path: Path) -> None:
        """True subprocess smoke: a second process is refused; killing the
        holder (simulated crash) releases the kernel lock."""
        mem_dir = tmp_path / "xproc"
        sentinel = tmp_path / "child-ready"
        script = (
            "import sys, time\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "from tulving.memory import Memory\n"
            f"m = Memory(path={str(mem_dir)!r}, agent_id='child')\n"
            f"Path({str(sentinel)!r}).write_text('ready', encoding='utf-8')\n"
            "time.sleep(60)\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        child = subprocess.Popen(
            [sys.executable, "-c", script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 20.0
            while not sentinel.exists():
                if child.poll() is not None or time.monotonic() > deadline:
                    _, stderr = child.communicate(timeout=5)
                    pytest.fail(f"child never acquired the lock: {stderr.decode()!r}")
                time.sleep(0.05)
            with pytest.raises(StorageError):
                Memory(path=str(mem_dir), agent_id="parent")
        finally:
            child.kill()
            child.wait(timeout=10)
        # The kernel released the crashed holder's lock: acquisition succeeds.
        recovered = Memory(path=str(mem_dir), agent_id="parent")
        recovered.close()


# ---------------------------------------------------------------------------
# startup() — lazy, single-runner, time-boxed, non-fatal, idempotent
# ---------------------------------------------------------------------------


class TestStartup:
    def test_lazy_trigger_runs_once(
        self, tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex
    ) -> None:
        stub = StubLifecycle()
        handle = make_memory(tmp_path, fake_clock, semantic=fake_semantic, lifecycle_manager=stub)
        try:
            assert handle.get("missing") is None
            assert stub.calls.count("check_on_startup") == 1
            assert fake_semantic.calls.count("open") == 1
            handle.get("missing")
            assert stub.calls.count("check_on_startup") == 1
            report = handle.startup()
            assert report.ran is False
            assert stub.calls.count("check_on_startup") == 1
        finally:
            handle.close()

    def test_first_startup_reports_ran(self, memory: Memory) -> None:
        report = memory.startup()
        assert isinstance(report, StartupReport)
        assert report.ran is True
        assert memory.startup().ran is False

    def test_concurrent_startup_single_runner(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        stub = StubLifecycle()
        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=stub)
        try:
            reports: list[StartupReport] = []
            barrier = threading.Barrier(8)

            def run() -> None:
                barrier.wait()
                reports.append(handle.startup())

            threads = [threading.Thread(target=run) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(reports) == 8
            assert stub.calls.count("check_on_startup") == 1
        finally:
            handle.close()

    def test_lifecycle_hooks_wired_and_folded(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        stub = StubLifecycle(abandoned=3, stale=7)
        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=stub)
        try:
            report = handle.startup()
            assert report.sessions_abandoned == 3
            assert report.entries_marked_stale == 7
            assert stub.calls.count("check_on_startup") == 1
            assert stub.calls.count("run_staleness_scan") == 1
        finally:
            handle.close()

    def test_startup_non_fatal_and_content_free_errors(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        stub = StubLifecycle()
        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=stub)
        try:

            def exploding_eviction(now: datetime) -> Any:
                raise StorageError("eviction pass exploded")

            handle._run_eviction = exploding_eviction  # type: ignore[method-assign]
            report = handle.startup()
            assert report.ran is True
            assert any("evict" in err for err in report.errors)
            assert report.decay is None
            # Other tasks still ran.
            assert "check_on_startup" in stub.calls
            assert "run_staleness_scan" in stub.calls
            assert all("SECRETCONTENT" not in err for err in report.errors)
        finally:
            handle.close()

    def test_zero_deadline_defers_everything(
        self, tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex
    ) -> None:
        stub = StubLifecycle()
        handle = make_memory(
            tmp_path,
            fake_clock,
            semantic=fake_semantic,
            lifecycle_manager=stub,
            startup_deadline_seconds=0.0,
        )
        try:
            report = handle.startup()
            assert set(report.deferred) == {
                "index_reconcile",
                "abandoned_sessions",
                "evict",
                "staleness_scan",
            }
            assert report.errors == ()
            assert fake_semantic.calls == []
            assert stub.calls == []
            # Cached — a later manual startup() does not re-run (deferral
            # waits for the next process start, D8).
            assert handle.startup().ran is False
            assert stub.calls == []
        finally:
            handle.close()

    def test_reconcile_report_folded(
        self, tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex
    ) -> None:
        fake_semantic.reconcile_report = _ReconcileReport(full_rebuild=True)
        handle = make_memory(tmp_path, fake_clock, semantic=fake_semantic)
        try:
            report = handle.startup()
            assert report.reconcile is not None
            assert report.reconcile.full_rebuild is True  # type: ignore[attr-defined]
            entry = store_fact(handle, "reconciled entry")
            hits = handle.search("reconciled entry")
            assert [r.entry.id for r in hits] == [entry.id]
        finally:
            handle.close()

    def test_adapter_mismatch_is_loud_degraded_recoverable(
        self,
        tmp_path: Path,
        fake_clock: FakeClock,
        fake_semantic: FakeSemanticIndex,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fake_semantic.open_error = (
            "memory was embedded with 'hash-old' (32d) but the active adapter is "
            "'hash-new' (32d). Run rebuild(re_embed=True) to re-embed."
        )
        handle = make_memory(tmp_path, fake_clock, semantic=fake_semantic)
        try:
            report = handle.startup()
            assert report.ran is True
            assert any("hash-old" in e and "hash-new" in e for e in report.errors)
            assert report.reconcile is None
            entry = store_fact(handle, "kv only entry", key="kv only entry")
            with caplog.at_level(logging.WARNING, logger="tulving.memory"):
                hits = handle.search("kv only entry")
            assert [r.match_type for r in hits] == [MatchType.KEY]
            assert any("semantic" in rec.message.lower() for rec in caplog.records)
            assert fake_semantic.search_calls == 0
            # Recovery: explicit rebuild restores semantic search.
            store = handle._store
            fake_semantic.on_rebuild = lambda idx: idx.vectors.update(
                {e.id: e.content for e in store.list(limit=100)}
            )
            handle.rebuild_index(re_embed=True)
            hits = handle.search("kv only entry")
            assert entry.id in {r.entry.id for r in hits}
            assert fake_semantic.search_calls == 1
        finally:
            handle.close()

    def test_read_only_startup_writes_nothing(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "ro-startup"
        writer = Memory(path=str(path), agent_id="a", clock=clock)
        writer.store("evictable", type=MemoryType.FACT, importance=0.2)
        writer.close()
        clock.advance(hours=336 * 10)
        reader = Memory(path=str(path), agent_id="a", clock=clock, read_only=True)
        report = reader.startup()
        assert {"abandoned_sessions", "evict", "staleness_scan"} <= set(report.deferred)
        reader.close()
        verify = Memory(path=str(path), agent_id="a", clock=clock)
        archived = verify._store.list(include_archived=True, limit=100)
        assert all(not e.archived for e in archived)
        verify.close()

    def test_decay_idempotency_across_repeated_startups(self, tmp_path: Path) -> None:
        """MANDATORY AUDIT REGRESSION: 3 simulated restarts; first evicts
        K > 0, later passes evict 0; surviving rows are bit-identical —
        decay wrote nothing, repeated startup is a no-op for importance."""
        clock = FakeClock()
        path = tmp_path / "idempotent"
        seeder = Memory(path=str(path), agent_id="a", clock=clock)
        seeder.store("stale low fact", type=MemoryType.FACT, importance=0.3, key="low")
        seeder.store("old decision", type=MemoryType.DECISION, importance=0.3)
        pinned = seeder.store("pinned low fact", type=MemoryType.FACT, importance=0.3)
        seeder.pin(pinned.id)
        seeder.store("high fact", type=MemoryType.FACT, importance=0.9, key="high")
        seeder.close()
        # 0.3 * 0.5**3 = 0.0375 < 0.1 (evicted); 0.9 * 0.5**3 = 0.1125 (survives).
        clock.advance(hours=336 * 3)

        snapshots: list[list[dict[str, Any]]] = []
        evicted: list[int] = []
        for _ in range(3):
            handle = Memory(path=str(path), agent_id="a", clock=clock)
            report = handle.startup()
            assert report.decay is not None
            evicted.append(report.decay.entries_evicted)
            rows = handle._store.list(include_archived=True, limit=100)
            snapshots.append(sorted((e.to_dict() for e in rows), key=lambda r: str(r["id"])))
            handle.close()

        assert evicted[0] == 1  # only the unpinned low FACT
        assert evicted[1] == 0
        assert evicted[2] == 0
        assert snapshots[0] == snapshots[1] == snapshots[2]

    def test_pinned_and_decision_survive_eviction_at_floor(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "exempt"
        seeder = Memory(path=str(path), agent_id="a", clock=clock)
        decision = seeder.store("keep decision", type=MemoryType.DECISION, importance=0.3)
        pinned = seeder.store("keep pinned", type=MemoryType.FACT, importance=0.3)
        seeder.pin(pinned.id)
        seeder.close()
        clock.advance(hours=336 * 50)  # effective importance ~ 0
        handle = Memory(path=str(path), agent_id="a", clock=clock)
        try:
            report = handle.startup()
            assert report.decay is not None
            assert report.decay.entries_evicted == 0
            assert report.decay.entries_exempted == 2
            for entry_id in (decision.id, pinned.id):
                entry = handle.get_by_id(entry_id)
                assert entry is not None and not entry.archived
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# Store & retrieve (integration happy paths)
# ---------------------------------------------------------------------------


class TestStoreAndRetrieve:
    def test_round_trip(self, memory: Memory, fake_clock: FakeClock) -> None:
        entry = store_fact(memory, "the sky is blue", key="sky", importance=0.7)
        assert entry.id
        assert entry.source.agent_id == "agent-x"
        assert entry.importance == pytest.approx(0.7)  # age 0 => base
        got = memory.get("sky")
        assert got is not None
        assert got.id == entry.id
        assert got.access_count == 1
        assert got.importance is not None
        assert memory.get("missing") is None

    def test_caller_source_agent_id_overridden(self, memory: Memory) -> None:
        from tulving.entry import SourceInfo

        entry = store_fact(
            memory,
            "with source",
            source=SourceInfo(agent_id="impostor", step_id="s1", run_id="r1"),
        )
        assert entry.source.agent_id == "agent-x"
        assert entry.source.step_id == "s1"
        assert entry.source.run_id == "r1"

    def test_supersede_end_to_end(self, memory: Memory, fake_semantic: FakeSemanticIndex) -> None:
        old = store_fact(memory, "old unique content", key="k")
        new = store_fact(memory, "new unique content", key="k")
        current = memory.get("k")
        assert current is not None and current.id == new.id
        old_entry = memory.get_by_id(old.id)
        assert old_entry is not None
        assert old_entry.archived is True
        assert old_entry.archive_reason is ArchiveReason.SUPERSEDED
        # The archived vector was tombstoned: old content no longer surfaces.
        hits = memory.search("old unique content")
        assert old.id not in {r.entry.id for r in hits}

    def test_search_merges_and_dedups(self, memory: Memory) -> None:
        entry = store_fact(memory, "alpha beta", key="alpha beta")
        results = memory.search("alpha beta")
        matching = [r for r in results if r.entry.id == entry.id]
        assert len(matching) == 1
        assert matching[0].match_type is MatchType.KEY
        assert matching[0].score == 1.0

    def test_semantic_scores_clamped_and_typed(self, memory: Memory) -> None:
        store_fact(memory, "gamma delta epsilon")
        results = memory.search("gamma delta")
        assert results
        for result in results:
            assert isinstance(result, SearchResult)
            assert 0.0 <= result.score <= 1.0
            assert result.match_type is MatchType.SEMANTIC

    def test_search_filters_tags_and_types(self, memory: Memory) -> None:
        tagged = store_fact(memory, "omega shared words", tags=["wanted"])
        store_fact(memory, "omega shared other words", tags=["unwanted"])
        hits = memory.search("omega shared", tags=["wanted"])
        assert {r.entry.id for r in hits} == {tagged.id}
        decision = memory.store("omega decision words", type=MemoryType.DECISION)
        hits = memory.search("omega", types=[MemoryType.DECISION])
        assert {r.entry.id for r in hits} == {decision.id}

    def test_kv_candidate_also_filtered(self, memory: Memory) -> None:
        store_fact(memory, "keyed content", key="thekey", tags=["a"])
        assert memory.search("thekey", tags=["other"]) == []

    def test_min_importance_uses_effective_importance(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        old = store_fact(memory, "ancient news story", importance=0.9)
        fake_clock.advance(hours=336 * 4)  # 0.9 -> 0.05625
        fresh = store_fact(memory, "fresh news story", importance=0.2)
        hits = memory.search("news story", min_importance=0.1)
        ids = {r.entry.id for r in hits}
        assert fresh.id in ids
        assert old.id not in ids

    def test_search_touches_in_one_batch(
        self, memory: Memory, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        hit_one = store_fact(memory, "quux common words here")
        hit_two = store_fact(memory, "quux common words there")
        unmatched = store_fact(memory, "completely unrelated")
        calls: list[list[str]] = []
        original = memory._store.touch_entries

        def spy(entry_ids: list[str], now: datetime | None = None) -> int:
            calls.append(list(entry_ids))
            return original(entry_ids, now)

        monkeypatch.setattr(memory._store, "touch_entries", spy)
        results = memory.search("quux common words")
        assert {r.entry.id for r in results} == {hit_one.id, hit_two.id}
        assert len(calls) == 1
        assert set(calls[0]) == {hit_one.id, hit_two.id}
        for result in results:
            assert result.entry.access_count == 1
            assert result.entry.last_accessed_at == fake_clock.current
        raw = memory._store.get_by_id(unmatched.id, touch=False)
        assert raw is not None and raw.access_count == 0

    def test_empty_query_returns_empty(
        self, memory: Memory, fake_semantic: FakeSemanticIndex
    ) -> None:
        store_fact(memory, "anything")
        before = fake_semantic.search_calls
        assert memory.search("   ") == []
        assert fake_semantic.search_calls == before

    def test_top_k_below_one_rejected(self, memory: Memory) -> None:
        with pytest.raises(MemoryStoreError):
            memory.search("q", top_k=0)

    def test_batch_stored_entries_immediately_searchable(self, memory: Memory) -> None:
        stored = [store_fact(memory, f"looped entry number {i}") for i in range(5)]
        for entry in stored:
            hits = memory.search(entry.content, top_k=10)
            assert entry.id in {r.entry.id for r in hits}

    def test_no_embedder_degrades_loudly(
        self, tmp_path: Path, fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        handle = make_memory(tmp_path, fake_clock)  # no semantic index at all
        try:
            entry = store_fact(handle, "kv text", key="kv text")
            with caplog.at_level(logging.WARNING, logger="tulving.memory"):
                hits = handle.search("kv text")
                handle.search("kv text")
            assert [r.match_type for r in hits] == [MatchType.KEY]
            semantic_warnings = [rec for rec in caplog.records if "semantic" in rec.message.lower()]
            assert len(semantic_warnings) == 1  # once per instance, not per call
            assert entry.id == hits[0].entry.id
        finally:
            handle.close()

    def test_index_failure_never_fails_store(
        self, memory: Memory, fake_semantic: FakeSemanticIndex, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        memory.startup()

        def exploding_add(entry_id: str, text: str) -> list[float]:
            raise VectorIndexError("index full")

        monkeypatch.setattr(fake_semantic, "add", exploding_add)
        entry = store_fact(memory, "durable despite index", key="durable")
        assert memory.get("durable") is not None
        assert entry.id

    def test_list_keys(self, memory: Memory) -> None:
        store_fact(memory, "one", key="cfg:alpha")
        store_fact(memory, "two", key="cfg:beta")
        store_fact(memory, "three", key="other")
        assert memory.list_keys() == ["cfg:alpha", "cfg:beta", "other"]
        assert memory.list_keys(prefix="cfg:") == ["cfg:alpha", "cfg:beta"]

    def test_get_by_id_returns_archived(self, memory: Memory) -> None:
        entry = store_fact(memory, "will be archived", key="k")
        memory.forget("k")
        archived = memory.get_by_id(entry.id)
        assert archived is not None
        assert archived.archived is True
        assert archived.importance is not None  # formula applies to archived too


# ---------------------------------------------------------------------------
# Edit — the owner-approved rebase semantics (D2)
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_importance_is_explicit_rebase(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        entry = store_fact(memory, "to rebase", importance=0.8)
        fake_clock.advance(hours=336 * 2)  # effective 0.8 -> 0.2
        updated = memory.update(entry.id, importance=0.6)
        assert updated.base_importance == 0.6
        assert updated.last_accessed_at == fake_clock.current  # fresh anchor
        assert updated.importance == pytest.approx(0.6)  # effective == base now
        assert updated.updated_at == fake_clock.current

    def test_plain_update_never_touches_importance(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        entry = store_fact(memory, "original words", importance=0.8)
        anchor = entry.last_accessed_at
        fake_clock.advance(hours=5)
        updated = memory.update(entry.id, content="changed words")
        assert updated.base_importance == 0.8
        assert updated.last_accessed_at == anchor

    def test_update_out_of_range_importance(self, memory: Memory) -> None:
        entry = store_fact(memory, "x")
        with pytest.raises(MemoryStoreError):
            memory.update(entry.id, importance=1.5)

    def test_update_content_re_embeds(self, memory: Memory) -> None:
        entry = store_fact(memory, "unique original phrasing")
        memory.update(entry.id, content="unique replacement phrasing")
        hits = memory.search("unique replacement phrasing")
        assert entry.id in {r.entry.id for r in hits}
        hits = memory.search("original")
        assert entry.id not in {r.entry.id for r in hits}

    def test_update_all_none_is_noop(self, memory: Memory) -> None:
        entry = store_fact(memory, "unchanged")
        current = memory.update(entry.id)
        assert current.content == "unchanged"

    def test_update_missing_id(self, memory: Memory) -> None:
        with pytest.raises(MemoryStoreError):
            memory.update("missing", content="x")

    def test_pin_unpin_round_trip(self, memory: Memory) -> None:
        entry = store_fact(memory, "pin me")
        memory.pin(entry.id)
        got = memory.get_by_id(entry.id)
        assert got is not None and got.pinned is True
        memory.unpin(entry.id)
        got = memory.get_by_id(entry.id)
        assert got is not None and got.pinned is False
        with pytest.raises(MemoryStoreError):
            memory.pin("missing")


# ---------------------------------------------------------------------------
# Curate seam — RetrievalPort implementation + delegation
# ---------------------------------------------------------------------------


class TestCurateSeam:
    def test_curate_without_curator_raises(self, memory: Memory) -> None:
        with pytest.raises(NotImplementedError, match=r"[Cc]urator"):
            memory.curate("query")

    def test_curate_delegates_to_injected_curator(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        curator = FakeCurator()
        handle = make_memory(tmp_path, fake_clock, curator=curator)
        try:
            result = handle.curate(
                "the query", token_budget=1234, mode="orient", include_tags=["a"]
            )
            assert result == "curated:the query"
            assert curator.calls == [
                {
                    "query": "the query",
                    "token_budget": 1234,
                    "mode": "orient",
                    "include_tags": ["a"],
                    "exclude_tags": None,
                }
            ]
        finally:
            handle.close()

    def test_port_lookup_key_never_touches(self, memory: Memory) -> None:
        entry = store_fact(memory, "keyed", key="k")
        port = memory._retrieval
        found = port.lookup_key("k")
        assert found is not None and found.id == entry.id
        raw = memory._store.get_by_id(entry.id, touch=False)
        assert raw is not None and raw.access_count == 0
        assert port.lookup_key("missing") is None

    def test_port_semantic_candidates(self, memory: Memory) -> None:
        entry = store_fact(memory, "semantic candidate text")
        pairs = memory._retrieval.semantic_candidates("semantic candidate text", top_k=5)
        assert [(p[0].id, p[1]) for p in pairs] == [(entry.id, 1.0)]
        raw = memory._store.get_by_id(entry.id, touch=False)
        assert raw is not None and raw.access_count == 0

    def test_port_semantic_candidates_empty_without_embedder(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        handle = make_memory(tmp_path, fake_clock)
        try:
            store_fact(handle, "anything")
            assert handle._retrieval.semantic_candidates("anything", top_k=5) == []
        finally:
            handle.close()

    def test_port_recent_and_list_by(self, memory: Memory, fake_clock: FakeClock) -> None:
        first = store_fact(memory, "first")
        fake_clock.advance(hours=1)
        second = store_fact(memory, "second")
        memory.pin(first.id)
        port = memory._retrieval
        recents = port.recent_entries(limit=10)
        assert [e.id for e in recents] == [second.id, first.id]
        pinned = port.list_by(pinned_only=True, limit=10)
        assert [e.id for e in pinned] == [first.id]
        decisions = port.list_by(types=[MemoryType.DECISION], limit=10)
        assert decisions == []

    def test_port_record_access_touches_and_clears_stale(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        included = store_fact(memory, "included", tags=["potentially_stale", "keep"])
        excluded = store_fact(memory, "excluded", tags=["potentially_stale"])
        fake_clock.advance(hours=2)
        memory._retrieval.record_access([included.id], now=fake_clock.current)
        after = memory._store.get_by_id(included.id, touch=False)
        assert after is not None
        assert after.access_count == 1
        assert after.last_accessed_at == fake_clock.current
        assert after.tags == ["keep"]
        untouched = memory._store.get_by_id(excluded.id, touch=False)
        assert untouched is not None
        assert untouched.access_count == 0
        assert "potentially_stale" in untouched.tags

    def test_port_record_access_noop_in_read_only(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        backend = InMemoryBackend()
        writer = make_memory(tmp_path, fake_clock, storage_backend=backend)
        entry = store_fact(writer, "shared", key="k")
        reader = Memory(
            path=str(tmp_path / "mem"),
            agent_id="agent-x",
            storage_backend=backend,
            clock=fake_clock,
            read_only=True,
        )
        try:
            reader._retrieval.record_access([entry.id], now=fake_clock.current)
            raw = reader._store.get_by_id(entry.id, touch=False)
            assert raw is not None and raw.access_count == 0
        finally:
            reader.close()
            writer.close()


# ---------------------------------------------------------------------------
# Forget family & purge
# ---------------------------------------------------------------------------


class TestForgetFamily:
    def test_forget_key(self, memory: Memory) -> None:
        entry = store_fact(memory, "forget me please now", key="k")
        assert memory.forget("k") is True
        assert memory.get("k") is None
        hits = memory.search("forget me please now")
        assert entry.id not in {r.entry.id for r in hits}
        assert memory.forget("missing") is False

    def test_forget_hard(self, memory: Memory, fake_semantic: FakeSemanticIndex) -> None:
        entry = store_fact(memory, "hard delete", key="k")
        assert memory.forget("k", hard=True) is True
        assert memory.get_by_id(entry.id) is None
        assert entry.id in fake_semantic.tombstoned

    def test_forget_by_id(self, memory: Memory) -> None:
        entry = store_fact(memory, "unkeyed entry")
        assert memory.forget_by_id(entry.id) is True
        archived = memory.get_by_id(entry.id)
        assert archived is not None and archived.archived is True
        assert memory.forget_by_id(entry.id) is False  # already archived
        assert memory.forget_by_id("missing") is False

    def test_forget_by_tags(self, memory: Memory) -> None:
        one = store_fact(memory, "one", tags=["a"])
        two = store_fact(memory, "two", tags=["a", "b"])
        store_fact(memory, "three", tags=["c"])
        assert memory.forget_by_tags(["a"]) == 2
        for entry_id in (one.id, two.id):
            archived = memory.get_by_id(entry_id)
            assert archived is not None
            assert archived.archive_reason is ArchiveReason.FORGOTTEN
        with pytest.raises(MemoryStoreError):
            memory.forget_by_tags([])

    def test_forget_by_age(self, memory: Memory, fake_clock: FakeClock) -> None:
        old_fact = store_fact(memory, "old fact")
        old_decision = memory.store("old decision", type=MemoryType.DECISION)
        old_pinned = store_fact(memory, "old pinned")
        memory.pin(old_pinned.id)
        fake_clock.advance(days=30)
        recent = store_fact(memory, "recent fact")
        count = memory.forget_by_age(timedelta(days=7))
        assert count == 1
        archived = memory.get_by_id(old_fact.id)
        assert archived is not None and archived.archived is True
        for survivor in (old_decision.id, old_pinned.id, recent.id):
            entry = memory.get_by_id(survivor)
            assert entry is not None and not entry.archived

    def test_forget_by_age_can_include_decisions(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        decision = memory.store("old decision", type=MemoryType.DECISION)
        fake_clock.advance(days=30)
        count = memory.forget_by_age(timedelta(days=7), preserve_decisions=False)
        assert count == 1
        archived = memory.get_by_id(decision.id)
        assert archived is not None and archived.archived is True

    def test_purge_passthrough_protects_summarized(self, memory: Memory) -> None:
        forgotten = store_fact(memory, "forgotten", key="k")
        memory.forget("k")
        summarized = store_fact(memory, "summarized source")
        memory._store.archive(summarized.id, ArchiveReason.SUMMARIZED)
        assert memory.purge_archived() == 1
        assert memory.get_by_id(forgotten.id) is None
        survivor = memory.get_by_id(summarized.id)
        assert survivor is not None
        assert survivor.archive_reason is ArchiveReason.SUMMARIZED


# ---------------------------------------------------------------------------
# Sessions & close
# ---------------------------------------------------------------------------


class TestSessions:
    def test_first_store_auto_starts_session(self, memory: Memory) -> None:
        entry = store_fact(memory, "auto session")
        sessions = memory._backend.list_sessions(agent_id="agent-x", status="active")
        assert len(sessions) == 1
        assert entry.session_id == sessions[0]["id"]

    def test_concurrent_first_stores_single_session(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        handle = make_memory(tmp_path, fake_clock)
        try:
            handle.startup()
            barrier = threading.Barrier(8)
            errors: list[Exception] = []

            def run(i: int) -> None:
                barrier.wait()
                try:
                    handle.store(f"entry {i}", type=MemoryType.FACT)
                except Exception as exc:  # pragma: no cover - failure diagnostics
                    errors.append(exc)

            threads = [threading.Thread(target=run, args=(i,)) for i in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert errors == []
            sessions = handle._backend.list_sessions(agent_id="agent-x", status="active")
            assert len(sessions) == 1
        finally:
            handle.close()

    def test_session_context_manager(self, memory: Memory) -> None:
        with memory.session("do the thing") as session_id:
            entry = store_fact(memory, "inside session")
            assert entry.session_id == session_id
        session = memory._backend.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.ENDED.value

    def test_session_end_called_on_exception(self, memory: Memory) -> None:
        with pytest.raises(RuntimeError, match="body"):
            with memory.session("goal") as session_id:
                raise RuntimeError("body exploded")
        session = memory._backend.get_session(session_id)
        assert session is not None
        assert session["status"] == SessionStatus.ENDED.value

    def test_second_concurrent_session_refused(self, memory: Memory) -> None:
        with memory.session("first"):
            with pytest.raises(MemoryStoreError, match="agent-x"):
                with memory.session("second"):
                    pass  # pragma: no cover

    def test_other_agents_session_never_conflicts(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        backend = InMemoryBackend()
        agent_a = make_memory(tmp_path, fake_clock, storage_backend=backend, subdir="a")
        agent_b = make_memory(
            tmp_path, fake_clock, storage_backend=backend, subdir="b", agent_id="agent-y"
        )
        try:
            with agent_a.session("a-goal"), agent_b.session("b-goal"):
                pass
        finally:
            agent_a.close()
            agent_b.close()

    def test_summarize_seam(self, memory: Memory) -> None:
        with pytest.raises(NotImplementedError, match=r"[Ss]ummarizer"):
            memory.summarize()


class TestClose:
    def test_close_is_idempotent(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        handle = make_memory(tmp_path, fake_clock)
        handle.close()
        handle.close()

    def test_public_call_after_close_raises(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        handle = make_memory(tmp_path, fake_clock)
        handle.close()
        with pytest.raises(StorageError):
            handle.get("k")

    def test_close_never_raises_from_lifecycle(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        stub = StubLifecycle()
        stub.close_raises = True
        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=stub)
        handle.startup()
        handle.close()  # must not raise
        assert "close_active_session" in stub.calls

    def test_close_flushes_semantic_index(
        self, tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex
    ) -> None:
        handle = make_memory(tmp_path, fake_clock, semantic=fake_semantic)
        handle.startup()
        handle.close()
        assert "close" in fake_semantic.calls

    def test_context_manager_protocol(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        with make_memory(tmp_path, fake_clock) as handle:
            store_fact(handle, "inside with", key="k")
        with pytest.raises(StorageError):
            handle.get("k")

    def test_rebuild_index_without_embedder_raises(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        handle = make_memory(tmp_path, fake_clock)
        try:
            with pytest.raises(ConfigError):
                handle.rebuild_index()
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# Construction warnings & containment (security #2, ADR-015 #4)
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_cloud_sync_warning_with_injected_backend(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        synced = tmp_path / "OneDrive" / "mem"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handle = Memory(
                path=str(synced),
                agent_id="a",
                storage_backend=InMemoryBackend(),
                clock=fake_clock,
            )
            handle.close()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "ADR-015" in str(user_warnings[0].message)

    def test_cloud_sync_single_warning_with_default_backend(self, tmp_path: Path) -> None:
        synced = tmp_path / "OneDrive" / "mem"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handle = Memory(path=str(synced), agent_id="a")
            handle.close()
        user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1  # the backend's — never two

    def test_clean_path_warns_nothing(self, tmp_path: Path, fake_clock: FakeClock) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            handle = make_memory(tmp_path, fake_clock, subdir="clean")
            handle.close()
        assert [w for w in caught if issubclass(w.category, UserWarning)] == []

    def test_hostile_symlink_containment(self, tmp_path: Path) -> None:
        mem_dir = tmp_path / "mem"
        mem_dir.mkdir()
        outside = tmp_path / "outside.db"
        outside.write_bytes(b"")
        try:
            os.symlink(outside, mem_dir / "tulving.db")
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/user")
        with pytest.raises(SecurityError):
            Memory(path=str(mem_dir), agent_id="a")


# ---------------------------------------------------------------------------
# SQLite persistence round trip
# ---------------------------------------------------------------------------


class TestSQLitePersistence:
    def test_restart_round_trip(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "persist"
        first = Memory(path=str(path), agent_id="a", clock=clock)
        first.store("persisted fact", type=MemoryType.FACT, key="k", importance=0.9)
        first.close()
        second = Memory(path=str(path), agent_id="a", clock=clock)
        entry = second.get("k")
        assert entry is not None
        assert entry.content == "persisted fact"
        second.close()


# ---------------------------------------------------------------------------
# Real semantic-index integration (hnswlib + HashEmbedder, default wiring)
# ---------------------------------------------------------------------------


class TestRealSemanticIntegration:
    """End-to-end with the real tulving.semantic_index (no injection)."""

    def test_store_search_supersede_with_real_index(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "real"
        handle = Memory(
            path=str(path),
            agent_id="a",
            embedding_adapter=HashEmbedder(32),
            clock=clock,
        )
        try:
            report = handle.startup()
            assert report.reconcile is not None
            entry = handle.store("the exact stored text", type=MemoryType.FACT)
            hits = handle.search("the exact stored text")
            assert hits
            assert hits[0].entry.id == entry.id
            assert hits[0].match_type is MatchType.SEMANTIC
            assert hits[0].score == pytest.approx(1.0, abs=1e-5)
            # Supersede: the archived vector never surfaces again.
            old = handle.store("keyed old text", type=MemoryType.FACT, key="k")
            handle.store("keyed new text", type=MemoryType.FACT, key="k")
            hits = handle.search("keyed old text", top_k=10)
            assert old.id not in {r.entry.id for r in hits}
        finally:
            handle.close()

    def test_index_cache_survives_restart(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "real-restart"
        first = Memory(
            path=str(path), agent_id="a", embedding_adapter=HashEmbedder(32), clock=clock
        )
        entry = first.store("survives the restart", type=MemoryType.FACT)
        first.close()
        assert (path / "tulving.hnsw").exists()
        second = Memory(
            path=str(path), agent_id="a", embedding_adapter=HashEmbedder(32), clock=clock
        )
        try:
            report = second.startup()
            assert report.reconcile is not None
            assert report.reconcile.full_rebuild is False  # flushed cache reused
            hits = second.search("survives the restart")
            assert [r.entry.id for r in hits] == [entry.id]
        finally:
            second.close()

    def test_adapter_mismatch_refused_then_rebuilt_for_real(self, tmp_path: Path) -> None:
        clock = FakeClock()
        path = tmp_path / "real-mismatch"

        class RenamedEmbedder(HashEmbedder):
            @property
            def model_id(self) -> str:
                return f"tulving/hash-embedder-v2-{self.dimension}"

        first = Memory(
            path=str(path), agent_id="a", embedding_adapter=HashEmbedder(32), clock=clock
        )
        entry = first.store("mismatch survivor", type=MemoryType.FACT)
        first.close()
        second = Memory(
            path=str(path), agent_id="a", embedding_adapter=RenamedEmbedder(32), clock=clock
        )
        try:
            report = second.startup()
            assert any("index_reconcile" in err for err in report.errors)
            assert second.search("mismatch survivor") == []  # semantic disabled
            second.rebuild_index(re_embed=True)
            hits = second.search("mismatch survivor")
            assert [r.entry.id for r in hits] == [entry.id]
        finally:
            second.close()


# ---------------------------------------------------------------------------
# Branch coverage: decay seam, seam errors, degraded paths
# ---------------------------------------------------------------------------


class TestDecaySeamFunctions:
    """Direct unit tests for the pure-default decay seam (until step 11)."""

    def test_naive_now_rejected(self, memory: Memory) -> None:
        from tulving.memory import _effective_importance

        entry = store_fact(memory, "x")
        with pytest.raises(ValueError):
            _effective_importance(entry, datetime(2026, 1, 1), {MemoryType.FACT: 336.0})

    def test_missing_type_key_is_config_error(self, memory: Memory, fake_clock: FakeClock) -> None:
        from tulving.memory import _effective_importance

        entry = store_fact(memory, "x")
        with pytest.raises(ConfigError):
            _effective_importance(entry, fake_clock.current, {})

    def test_corrupted_half_life_is_config_error(
        self, memory: Memory, fake_clock: FakeClock
    ) -> None:
        from tulving.memory import _effective_importance

        entry = store_fact(memory, "x")
        with pytest.raises(ConfigError):
            _effective_importance(entry, fake_clock.current, {MemoryType.FACT: 0.0})

    def test_infinite_half_life_never_decays(self, memory: Memory, fake_clock: FakeClock) -> None:
        from tulving.memory import _effective_importance

        entry = store_fact(memory, "x", importance=0.8)
        fake_clock.advance(hours=100_000)
        value = _effective_importance(entry, fake_clock.current, {MemoryType.FACT: float("inf")})
        assert value == 0.8


class TestConstructorEdges:
    def test_negative_deadline_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            Memory(
                path=str(tmp_path / "m"),
                storage_backend=InMemoryBackend(),
                startup_deadline_seconds=-1.0,
            )

    def test_llm_without_any_complete_attribute(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError):
            Memory(
                path=str(tmp_path / "m"),
                llm_adapter=object(),  # type: ignore[arg-type]
                storage_backend=InMemoryBackend(),
            )

    def test_properties(self, memory: Memory) -> None:
        assert memory.agent_id == "agent-x"
        assert memory.read_only is False

    def test_lock_released_when_backend_construction_fails(self, tmp_path: Path) -> None:
        """A failure AFTER lock acquisition must not leave the path locked."""
        path = tmp_path / "failing"
        path.mkdir()
        (path / "tulving.db").mkdir()  # a directory where the DB file must go
        with pytest.raises(StorageError):
            Memory(path=str(path), agent_id="a")
        (path / "tulving.db").rmdir()
        recovered = Memory(path=str(path), agent_id="a")
        recovered.close()

    def test_refusal_message_survives_garbage_diagnostics(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        holder = Memory(path=str(path), agent_id="a")
        try:
            # Corrupt the diagnostics region (offset 0 is not the lock byte).
            (path / "tulving.lock").write_bytes(b"\xff\xfenot-json")
            with pytest.raises(StorageError, match="read_only"):
                Memory(path=str(path), agent_id="b")
        finally:
            holder.close()


class TestStartupAfterClose:
    def test_startup_and_rebuild_after_close_raise(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        handle = make_memory(tmp_path, fake_clock)
        handle.close()
        with pytest.raises(StorageError):
            handle.startup()
        with pytest.raises(StorageError):
            handle.rebuild_index()


class TestSessionSeamEdges:
    def test_double_end_is_refused(self, memory: Memory) -> None:
        with memory.session("goal") as session_id:
            pass
        with pytest.raises(MemoryStoreError, match=r"[Nn]o active session"):
            memory._lifecycle.session_end(session_id)

    def test_session_end_error_propagates_on_clean_exit(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        class EndExplodes(StubLifecycle):
            def session_end(self, session_id: str) -> None:
                raise RuntimeError("end exploded")

        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=EndExplodes())
        try:
            with pytest.raises(RuntimeError, match="end exploded"):
                with handle.session("goal"):
                    pass
        finally:
            handle.close()

    def test_body_exception_never_masked_by_session_end_error(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        class EndExplodes(StubLifecycle):
            def session_end(self, session_id: str) -> None:
                raise RuntimeError("end exploded")

        handle = make_memory(tmp_path, fake_clock, lifecycle_manager=EndExplodes())
        try:
            with pytest.raises(ValueError, match="the body"):
                with handle.session("goal"):
                    raise ValueError("the body")
        finally:
            handle.close()


class TestDegradedPaths:
    def test_semantic_search_failure_degrades_to_kv(
        self,
        memory: Memory,
        fake_semantic: FakeSemanticIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = store_fact(memory, "keyed", key="keyed")

        def exploding_search(query: str, *, top_k: int = 5, filter_fn: Any = None) -> Any:
            raise VectorIndexError("query failed")

        monkeypatch.setattr(fake_semantic, "search", exploding_search)
        hits = memory.search("keyed")
        assert [r.entry.id for r in hits] == [entry.id]
        assert hits[0].match_type is MatchType.KEY

    def test_stale_index_ids_dropped_on_hydration(
        self, memory: Memory, fake_semantic: FakeSemanticIndex
    ) -> None:
        entry = store_fact(memory, "will be archived quietly")
        # Archive BEHIND the index (vector stays live in the graph): the
        # hydration post-filter must still exclude it; unknown ids drop too.
        memory._store.archive(entry.id, ArchiveReason.FORGOTTEN)
        fake_semantic.vectors["ghost-id"] = "will be archived quietly"
        assert memory.search("will be archived quietly") == []

    def test_update_reembed_failure_swallowed(
        self,
        memory: Memory,
        fake_semantic: FakeSemanticIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entry = store_fact(memory, "before")

        def exploding_add(entry_id: str, text: str) -> Any:
            raise VectorIndexError("index full")

        monkeypatch.setattr(fake_semantic, "add", exploding_add)
        updated = memory.update(entry.id, content="after")
        assert updated.content == "after"

    def test_tombstone_failure_swallowed(
        self,
        memory: Memory,
        fake_semantic: FakeSemanticIndex,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store_fact(memory, "to forget", key="k")

        def exploding_remove(entry_id: str) -> bool:
            raise VectorIndexError("mapping corrupt")

        monkeypatch.setattr(fake_semantic, "remove", exploding_remove)
        assert memory.forget("k") is True

    def test_close_swallows_component_failures(
        self, tmp_path: Path, fake_clock: FakeClock, fake_semantic: FakeSemanticIndex
    ) -> None:
        class ClosingExplodesBackend(InMemoryBackend):
            def close(self) -> None:
                raise StorageError("backend close exploded")

        def exploding_close() -> None:
            raise VectorIndexError("index close exploded")

        fake_semantic.close = exploding_close  # type: ignore[method-assign]
        handle = make_memory(
            tmp_path,
            fake_clock,
            semantic=fake_semantic,
            storage_backend=ClosingExplodesBackend(),
        )
        handle.startup()
        handle.close()  # must not raise

    def test_update_all_none_missing_id(self, memory: Memory) -> None:
        with pytest.raises(MemoryStoreError):
            memory.update("missing")

    def test_search_kv_hit_filtered_by_type(self, memory: Memory) -> None:
        store_fact(memory, "typed content", key="typedkey")
        assert memory.search("typedkey", types=[MemoryType.DECISION]) == []


class TestBulkAndPaging:
    def test_forget_by_tags_hard(self, memory: Memory, fake_semantic: FakeSemanticIndex) -> None:
        entry = store_fact(memory, "hard bulk", tags=["bulk"])
        assert memory.forget_by_tags(["bulk"], hard=True) == 1
        assert memory.get_by_id(entry.id) is None
        assert entry.id in fake_semantic.tombstoned

    def test_bulk_forget_skips_already_archived(self, memory: Memory) -> None:
        entry = store_fact(memory, "raced entry", tags=["race"])
        memory._store.archive(entry.id, ArchiveReason.SUMMARIZED)
        assert memory._bulk_forget([entry]) == 0
        after = memory.get_by_id(entry.id)
        assert after is not None
        assert after.archive_reason is ArchiveReason.SUMMARIZED  # never relabeled

    def test_paged_listing_walks_pages(
        self, memory: Memory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tulving.memory._PAGE", 2)
        stored = [store_fact(memory, f"page walk {i}", tags=["paged"]) for i in range(5)]
        listed = memory._all_active_entries(tags=["paged"])
        assert {e.id for e in listed} == {e.id for e in stored}
        memory.forget_by_tags(["paged"])
        assert len(memory._all_archived_ids()) == 5


class TestSummarizerSeam:
    def test_summarize_delegates_to_injected_summarizer(
        self, tmp_path: Path, fake_clock: FakeClock
    ) -> None:
        class FakeSummarizer:
            def __init__(self) -> None:
                self.calls: list[Any] = []

            def summarize(
                self,
                *,
                older_than: timedelta | None = None,
                tags: list[str] | None = None,
            ) -> list[Any]:
                self.calls.append((older_than, tags))
                return []

        summarizer = FakeSummarizer()
        handle = make_memory(tmp_path, fake_clock, summarizer=summarizer)
        try:
            assert handle.summarize(older_than=timedelta(days=1), tags=["t"]) == []
            assert summarizer.calls == [(timedelta(days=1), ["t"])]
        finally:
            handle.close()


# ---------------------------------------------------------------------------
# Fix round: C2/DB-HIGH-2 (read-only never writes), SEC-SEV-001 (hostile lock
# diagnostics), DB-M2 (no unlink on release)
# ---------------------------------------------------------------------------


class TestReadOnlyNeverBootstraps:
    """C2/DB-HIGH-2: a read_only handle never creates or migrates anything."""

    def test_read_only_on_nonexistent_dir_refused(self, tmp_path: Path) -> None:
        missing = tmp_path / "never-created"
        with pytest.raises(MemoryStoreError, match="read_only"):
            Memory(path=str(missing), agent_id="a", read_only=True)
        assert not missing.exists()  # nothing was bootstrapped

    def test_read_only_on_empty_dir_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(MemoryStoreError, match="writer"):
            Memory(path=str(empty), agent_id="a", read_only=True)
        assert list(empty.iterdir()) == []  # no db, no lock, no index created

    def test_read_only_with_embedder_and_deleted_cache_writes_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        clock = FakeClock()
        path = tmp_path / "ro-index"
        writer = Memory(
            path=str(path), agent_id="a", embedding_adapter=HashEmbedder(32), clock=clock
        )
        writer.store("seeded entry one", type=MemoryType.FACT, key="one")
        writer.store("seeded entry two", type=MemoryType.FACT)
        writer.close()
        # Simulate a lost cache: the read-only handle must NOT regenerate it.
        (path / "tulving.hnsw").unlink()

        def snapshot() -> list[dict[str, Any]]:
            probe = Memory(path=str(path), agent_id="a", clock=clock)
            rows = sorted(
                (e.to_dict() for e in probe._store.list(include_archived=True, limit=100)),
                key=lambda r: str(r["id"]),
            )
            sessions = probe._backend.list_sessions()
            probe.close()
            return [*rows, *sessions]

        before = snapshot()
        reader = Memory(
            path=str(path),
            agent_id="a",
            embedding_adapter=HashEmbedder(32),
            clock=clock,
            read_only=True,
        )
        try:
            reader.startup()
            with caplog.at_level(logging.WARNING, logger="tulving.memory"):
                hits = reader.search("seeded entry one")
            # Loud degradation: KV-only at best, plus a warning.
            assert all(r.match_type is MatchType.KEY for r in hits)
            assert any("semantic" in rec.message.lower() for rec in caplog.records)
        finally:
            reader.close()
        assert not (path / "tulving.hnsw").exists()  # never regenerated
        assert not (path / "tulving.hnsw.tmp").exists()
        assert snapshot() == before  # zero logical DB writes


class TestHostileLockDiagnostics:
    """SEC-SEV-001: lock-file contents are untrusted; the refusal message
    must never reflect them raw."""

    def test_refusal_message_sanitized_and_bounded(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        holder = Memory(path=str(path), agent_id="a")
        try:
            hostile = json.dumps(
                {
                    "pid": 99999999999999999,  # implausible — must be discarded
                    "hostname": "\x1b[31mEVIL\x1b[0m",
                    "acquired_at": "\x1b]0;pwned\x07sk-" + "A" * 2048,
                }
            )
            # The diagnostics region (offset 0) is writable by other handles;
            # only the lock byte at 4096 is kernel-locked.
            (path / "tulving.lock").write_text(hostile, encoding="utf-8")
            with pytest.raises(StorageError) as excinfo:
                Memory(path=str(path), agent_id="b")
            message = str(excinfo.value)
            assert "\x1b" not in message
            assert "\x07" not in message
            assert all(ch.isprintable() for ch in message)
            assert "sk-" not in message
            assert "99999999999999999" not in message
            assert len(message) < 400
            assert "read_only" in message  # remedy survives
        finally:
            holder.close()

    def test_valid_but_oversize_acquired_at_discarded(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        holder = Memory(path=str(path), agent_id="a")
        try:
            hostile = json.dumps({"pid": os.getpid(), "acquired_at": "2026-01-01" * 50})
            (path / "tulving.lock").write_text(hostile, encoding="utf-8")
            with pytest.raises(StorageError) as excinfo:
                Memory(path=str(path), agent_id="b")
            message = str(excinfo.value)
            assert f"pid {os.getpid()}" in message  # valid field kept
            assert "since" not in message  # oversize field discarded
        finally:
            holder.close()


class TestLockFileNotUnlinked:
    """DB-M2: release() never unlinks (unlink-on-release is a POSIX TOCTOU)."""

    def test_release_leaves_the_file_and_reacquire_works(self, tmp_path: Path) -> None:
        path = tmp_path / "locked"
        first = Memory(path=str(path), agent_id="a")
        first.close()
        assert (path / "tulving.lock").exists()  # existence is meaningless by design
        second = Memory(path=str(path), agent_id="b")
        second.close()
        assert (path / "tulving.lock").exists()

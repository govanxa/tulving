"""Memory: the Tulving public API.

Composes storage, indexes, curation, decay, summarization, and sessions
behind one per-agent handle (D7). Owns the advisory single-writer lock
(ADR-015), the explicit ``startup()`` pass (D8), and the write-ordering
discipline across the DB/index boundary (DB-first, index-second, reconcile
at startup).

There is no encryption at rest in v0.1 (ADR-010): do not store secrets you
cannot afford on disk.

Component wiring (this is the step-13b integration pass — all four Context
modules are now live behind Memory):

- ``curator``: ``curate()`` delegates to :class:`ContextCurator`, built
  lazily over the ``RetrievalPort`` (``_RetrievalAdapter``); the shared
  token estimator is resolved once per handle (``_get_estimator``).
- ``summarizer``: ``summarize()`` delegates to :class:`MemorySummarizer`
  (consuming ``self._llm``; ``llm=None`` degrades loudly there). The same
  summarizer backs lifecycle end-of-session rollups via a lazy port adapter.
- ``lifecycle``: the real :class:`LifecycleManager` owns per-agent session
  rows, abandonment detection (L4), staleness scanning (L5), and session
  markers. ``record_activity(write=False)`` keeps a read-heavy live session
  from looking abandoned (search/curate).
- decay: delegated to ``tulving.context.decay`` — the D2 formula lives there
  and only there (D12); ``_EvictionAdapter`` bridges the ``EvictionStore``
  protocol onto the store + index tombstoning.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import socket
import sys
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Protocol

from tulving.adapters.embeddings import EmbeddingAdapter
from tulving.adapters.llm import LLMAdapter
from tulving.adapters.storage import SQLiteBackend, StorageBackend, cloud_sync_risk
from tulving.context.config import LifecycleConfig
from tulving.context.curator import (
    ContextCurator,
    CuratedContext,
    TokenEstimator,
    resolve_estimator,
)
from tulving.context.decay import DecayReport
from tulving.context.decay import effective_importance as _effective_importance
from tulving.context.decay import evict as _decay_evict
from tulving.context.lifecycle import LifecycleManager
from tulving.context.summarizer import MemorySummarizer
from tulving.entry import MemoryEntry, Relationship, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MatchType, MemoryType
from tulving.exceptions import (
    ConfigError,
    MemoryStoreError,
    StorageError,
    TulvingError,
    VectorIndexError,
)
from tulving.export import ImportReport, MemoryExporter, MemoryImporter
from tulving.kv_index import KVIndex
from tulving.security import (
    compile_explicit_patterns,
    compile_key_patterns,
    contain_path,
    redact_text,
)
from tulving.semantic_index import ReconcileReport, SemanticIndex
from tulving.store import MemoryStore

logger = logging.getLogger("tulving.memory")

LOCK_FILENAME: Final[str] = "tulving.lock"
DB_FILENAME: Final[str] = "tulving.db"
INDEX_FILENAME: Final[str] = "tulving.hnsw"

_STALE_TAG: Final[str] = "potentially_stale"
_PAGE: Final[int] = 500
# The kernel lock byte sits past the diagnostics region so other processes
# can always READ the pid/host JSON at offset 0 (Windows region locks would
# otherwise block the read).
_LOCK_BYTE_OFFSET: Final[int] = 4096

_EMBEDDING_ADAPTER_SURFACE: Final[tuple[str, ...]] = (
    "embed",
    "embed_batch",
    "model_id",
    "dimension",
    "distance_metric",
    "normalizes",
)


def _file_size(path: Path) -> int:
    """Byte size of ``path``, or 0 when it is absent (maintenance ``inspect``/``vacuum``)."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


# ---------------------------------------------------------------------------
# Decay wiring (formula + eviction pass live in context/decay.py — D12)
# ---------------------------------------------------------------------------


class _EvictionAdapter:
    """``context.decay.EvictionStore`` over Memory's store + index.

    ``archive_entry`` mirrors the pre-step-11 eviction write exactly: the
    store archives the row (DB-first, ADR-015; timestamps come from the
    same clock source as the pass's ``now``) and the vector is tombstoned
    best-effort.
    """

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def iter_active_entries(self) -> Iterator[MemoryEntry]:
        """Paged active-entry scan; access-neutral (never touches)."""
        yield from self._memory._all_active_entries()

    def archive_entry(self, entry_id: str, reason: ArchiveReason, *, now: datetime) -> None:
        """Archive one row, then tombstone its vector (logged, never raised).

        ``now`` (the pass's evaluation instant) is accepted per the protocol
        but not threaded through: ``store.archive`` stamps ``updated_at``
        from its own read of the same clock SOURCE, which under the
        production wall clock may trail ``now`` by the scan latency (the two
        coincide under an injected fixed clock, as in tests).
        """
        del now  # same clock source; store.archive takes its own reading
        self._memory._store.archive(entry_id, reason)
        self._memory._semantic_remove(entry_id)


class _DecayEvaluator:
    """Bridges the curator/summarizer 'DecayManager-shaped' expectation onto
    the pure decay function (satisfies the curator's ``ImportanceEvaluator``
    protocol structurally)."""

    def __init__(self, half_life_hours: Mapping[MemoryType, float]) -> None:
        self._half_lives: dict[MemoryType, float] = dict(half_life_hours)

    def effective_importance(self, entry: MemoryEntry, now: datetime | None = None) -> float:
        """Effective importance of ``entry`` at ``now`` (default: utcnow)."""
        return _effective_importance(entry, now if now is not None else utcnow(), self._half_lives)


# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """One search hit: the entry, its score, and how it matched."""

    entry: MemoryEntry
    score: float  # [0.0, 1.0]; KV exact = 1.0, semantic = 1 - cosine distance
    match_type: MatchType  # KEY | SEMANTIC (TEMPORAL is curator-internal in v0.1)


@dataclass(frozen=True)
class StoreStats:
    """Read-only snapshot for ``tulving maintenance inspect`` (D2/D3-safe).

    Counts and on-disk sizes only — never computes or mutates importance,
    never bumps ``last_accessed_at``.
    """

    live_count: int
    archived_count: int
    archived_by_reason: dict[str, int]  # keyed by ArchiveReason.value, all reasons present
    total_count: int
    db_bytes: int
    wal_bytes: int
    index_bytes: int
    total_bytes: int
    schema_version: int
    embedding_model_id: str | None
    embedding_dimension: int | None
    distance_metric: str | None


@dataclass(frozen=True)
class VacuumResult:
    """Outcome of ``Memory.vacuum()``."""

    bytes_before: int
    bytes_after: int
    bytes_reclaimed: int  # max(before - after, 0)


@dataclass(frozen=True)
class StartupReport:
    """Outcome of one ``startup()`` pass.

    All counts zero / fields None for tasks that were skipped (read_only,
    no embedder, deadline) or failed.
    """

    ran: bool  # False = cached report from an earlier call
    duration_seconds: float
    reconcile: ReconcileReport | None  # None: no embedder, skipped, or failed
    decay: DecayReport | None
    sessions_abandoned: int
    entries_marked_stale: int
    deferred: tuple[str, ...]  # task names skipped by the deadline/read_only
    errors: tuple[str, ...]  # non-fatal task failures (redaction-safe text)


# ---------------------------------------------------------------------------
# Consumed-component protocols (structural seams for steps 8/10/12/13)
# ---------------------------------------------------------------------------


class SemanticIndexPort(Protocol):
    """Consumed surface of ``tulving.semantic_index.SemanticIndex``
    (blueprint-semantic-index). The real class satisfies this structurally."""

    def open(self) -> None: ...
    def reconcile(self) -> Any: ...
    def add(self, entry_id: str, text: str) -> list[float]: ...
    def remove(self, entry_id: str) -> bool: ...
    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filter_fn: Callable[[str], bool] | None = None,
    ) -> list[tuple[str, float]]: ...
    def rebuild(self, *, re_embed: bool = False) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class CuratorPort(Protocol):
    """Consumed surface of ``ContextCurator`` (build step 10)."""

    def curate(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        mode: str = "query",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> CuratedContext: ...


class SummarizerPort(Protocol):
    """Consumed surface of ``MemorySummarizer`` (the ``Memory(summarizer=...)``
    injection seam; the default is built lazily in ``_get_summarizer``)."""

    def summarize(
        self,
        *,
        older_than: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]: ...


class LifecycleManagerPort(Protocol):
    """blueprint-memory's L1-L7 lifecycle seam.

    Satisfied structurally by ``tulving.context.lifecycle.LifecycleManager``.
    ``record_activity`` was added by the step-13b integration pass so the
    ``activity_debounce_seconds`` knob is exercised in production: read paths
    (``search``/``curate``) bump ``last_activity_at`` with ``write=False`` so a
    read-heavy live session never looks abandoned, without write-amplifying.
    """

    def ensure_active_session(self) -> str: ...
    def session_start(self, goal: str | None = None) -> str: ...
    def session_end(self, session_id: str) -> None: ...
    def check_on_startup(self) -> int: ...
    def run_staleness_scan(self) -> int: ...
    def close_active_session(self) -> None: ...
    def record_activity(self, session_id: str, *, write: bool = True) -> None: ...


# ---------------------------------------------------------------------------
# Advisory single-writer lock (ADR-015)
# ---------------------------------------------------------------------------


class _AdvisoryLock:
    """Single-writer advisory lock for one memory path (ADR-015).

    Mechanism: open/create ``<dir>/tulving.lock`` (O_CREAT | O_RDWR — NOT
    O_EXCL; the file's *existence* is meaningless, only the kernel lock
    matters), then take a non-blocking exclusive lock and HOLD the fd for
    the process lifetime:

    - Windows: ``msvcrt.locking(fd, LK_NBLCK, 1)`` on one byte past the
      diagnostics region (so other processes can still read it);
    - POSIX: ``fcntl.flock(fd, LOCK_EX | LOCK_NB)``.

    The kernel releases the lock the instant the holding process dies —
    crash recovery needs no PID probing and no stale-file deletion. The
    diagnostics JSON line ``{pid, hostname, acquired_at}`` is INFORMATIONAL
    ONLY (error messages), never part of the locking decision, and
    content-free (security req #1). Because the file can live in a shared or
    cloud-synced directory, its contents are treated as UNTRUSTED when read
    back: fields are validated, sanitized, and length-bounded before they
    appear in any error message.

    ``release()`` never unlinks the file: the file's existence carries no
    meaning by design, and unlink-on-release is a classic POSIX TOCTOU (two
    writers can both "hold" a lock through an orphaned inode when one
    deletes the path another has already opened).
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: int | None = None

    def acquire(self) -> None:
        """Take the exclusive lock or raise ``StorageError`` with a remedy."""
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR)
        try:
            self._lock_fd(fd)
        except OSError as exc:
            os.close(fd)
            raise StorageError(self._refusal_message()) from exc
        self._fd = fd
        self._write_diagnostics(fd)

    def release(self) -> None:
        """Unlock and close. Idempotent, never raises.

        The lock file is deliberately NOT unlinked (see class docstring):
        deleting it would open a delete/reopen race in which two writers
        each hold an exclusive lock on a different inode of "the" path.
        """
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            self._unlock_fd(fd)
        except OSError:  # pragma: no cover - kernel unlock virtually never fails
            logger.debug("advisory lock unlock failed", exc_info=True)
        finally:
            with suppress(OSError):
                os.close(fd)

    @staticmethod
    def _lock_fd(fd: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - POSIX branch; exercised on POSIX platforms
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, _LOCK_BYTE_OFFSET, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:  # pragma: no cover - POSIX branch; exercised on POSIX platforms
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_UN)

    def _write_diagnostics(self, fd: int) -> None:
        """pid/host/timestamp only — informational, never user data."""
        try:
            payload = json.dumps(
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    "acquired_at": utcnow().isoformat(),
                }
            )
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, payload.encode("utf-8") + b"\n")
        except OSError:  # pragma: no cover - diagnostics are best-effort
            logger.debug("could not write lock diagnostics", exc_info=True)

    def _refusal_message(self) -> str:
        """Actionable refusal text; holder diagnostics are UNTRUSTED input.

        The lock file may live in a shared/synced directory, so anything
        read back is bounded (first 4 KB), validated per field (pid must be
        a plausible int, acquired_at must parse as ISO-8601), stripped of
        non-printable characters, and length-capped. Fields failing
        validation are silently discarded — the message never reflects raw
        file content (SEC-SEV-001).
        """
        holder = ""
        try:
            # Bounded RAW read (2 KB): unbuffered so the request range can
            # never overlap the kernel-locked byte at _LOCK_BYTE_OFFSET —
            # Windows fails any read whose range crosses a locked region,
            # and buffered readers request full 8 KB blocks.
            fd = os.open(self._path, os.O_RDONLY)
            try:
                raw = os.read(fd, _LOCK_BYTE_OFFSET // 2)
            finally:
                os.close(fd)
            first_line = raw.decode("utf-8", errors="replace").splitlines()[0]
            info = json.loads(first_line)
            parts: list[str] = []
            pid = info.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool) and 0 < pid < 2**32:
                parts.append(f"pid {pid}")
            acquired = info.get("acquired_at")
            if isinstance(acquired, str) and len(acquired) <= 64:
                cleaned = "".join(ch for ch in acquired if ch.isprintable())[:40]
                try:
                    datetime.fromisoformat(cleaned)
                except ValueError:
                    pass
                else:
                    parts.append(f"since {cleaned}")
            if parts:
                holder = f" ({', '.join(parts)})"
        except Exception:
            logger.debug("could not read lock holder diagnostics", exc_info=True)
        return (
            f"another Tulving process{holder} is writing to {self._path.parent} — "
            "close the other session, or open with Memory(..., read_only=True)"
        )


# ---------------------------------------------------------------------------
# Lifecycle end-of-session summarization adapter
# ---------------------------------------------------------------------------


class _LifecycleSummarizerAdapter:
    """``SessionSummarizerPort`` backed by Memory's lazy ``MemorySummarizer``.

    Construction is deferred: the summarizer (and its one-shot token-estimator
    probe) is built on the FIRST end-of-session rollup via
    ``Memory._get_summarizer()``, never at handle construction (D8). Wired only
    when an LLM is configured — with ``llm=None`` the ``LifecycleManager``
    receives ``summarizer=None`` and degrades loudly on its own (logged
    warning), so no skip-digest is spawned on every session end.
    """

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def summarize(self, *, session_id: str | None = None) -> list[MemoryEntry]:
        """Roll up ``session_id``'s memories; return created SUMMARY entries."""
        return self._memory._get_summarizer().summarize(session_id=session_id)


class _SessionContextManager:
    """``with memory.session(goal):`` wrapper — ``__exit__`` always ends the
    session, even when the body raised (blueprint L3)."""

    def __init__(self, memory: Memory, goal: str | None) -> None:
        self._memory = memory
        self._goal = goal
        self._session_id: str | None = None

    def __enter__(self) -> str:
        with self._memory._session_lock:
            self._session_id = self._memory._lifecycle.session_start(self._goal)
            # Read paths (search/curate) bump THIS session's activity.
            self._memory._active_session_id = self._session_id
        return self._session_id

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        session_id = self._session_id
        if session_id is not None:
            try:
                with self._memory._session_lock:
                    self._memory._lifecycle.session_end(session_id)
                    if self._memory._active_session_id == session_id:
                        self._memory._active_session_id = None
            except Exception:
                if exc_type is None:
                    raise
                # Never mask the body's exception with an end-of-session error.
                logger.warning("session_end failed while unwinding an exception", exc_info=True)


# ---------------------------------------------------------------------------
# The curator's RetrievalPort — production implementation
# ---------------------------------------------------------------------------


class _RetrievalAdapter:
    """The curator's ``RetrievalPort`` (blueprint-curator): access-neutral
    reads over the store/index, plus the single transactional
    ``record_access`` write (touch + ``potentially_stale`` auto-clear)."""

    def __init__(self, memory: Memory) -> None:
        self._memory = memory

    def lookup_key(self, key: str) -> MemoryEntry | None:
        """Exact active-key match, NO touch (bypasses KVIndex.get's touch)."""
        return self._memory._store.get_by_key(key, touch=False)

    def semantic_candidates(self, query: str, *, top_k: int) -> list[tuple[MemoryEntry, float]]:
        """Top-k semantic hits, hydrated, access-neutral; [] when degraded."""
        return self._memory._semantic_pairs(query, top_k=top_k, filter_fn=None)

    def recent_entries(self, *, limit: int) -> list[MemoryEntry]:
        """Most recently created active entries, newest first. NO touch."""
        return self._memory._store.list(limit=limit)

    def list_by(
        self,
        *,
        types: Sequence[MemoryType] | None = None,
        tags: Sequence[str] | None = None,
        pinned_only: bool = False,
        limit: int,
    ) -> list[MemoryEntry]:
        """Filtered active entries (orient mode's category walks). NO touch."""
        return self._memory._store.list(
            types=list(types) if types is not None else None,
            tags=list(tags) if tags is not None else None,
            pinned=True if pinned_only else None,
            limit=limit,
        )

    def record_access(self, entry_ids: Sequence[str], *, now: datetime) -> None:
        """THE one curator-triggered write: batched touch + stale-tag clear,
        one transaction. No-op (logged) on a read-only handle."""
        memory = self._memory
        if memory._read_only:
            logger.debug("record_access skipped: read_only handle")
            return
        with memory._backend.transaction():
            memory._store.touch_entries(list(entry_ids), now)
            for entry_id in entry_ids:
                entry = memory._store.get_by_id(entry_id, touch=False)
                if entry is not None and _STALE_TAG in entry.tags:
                    memory._store.update(
                        entry_id, tags=[tag for tag in entry.tags if tag != _STALE_TAG]
                    )


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class Memory:
    """Persistent, typed, searchable working memory for one agent (D7).

    The only class most users ever touch. ``__init__`` is cheap (D8):
    resolve/contain paths, warn on synced paths, take the advisory writer
    lock, open the backend, validate adapters structurally, and build the
    component graph. All lifecycle work lives in :meth:`startup` —
    explicit, time-boxed, non-fatal, single-runner, idempotent, and lazily
    triggered on first use.
    """

    def __init__(
        self,
        path: str | Path = "./tulving_memory",
        agent_id: str = "default",
        embedding_adapter: EmbeddingAdapter | None = None,
        llm_adapter: LLMAdapter | None = None,
        storage_backend: StorageBackend | None = None,
        sensitive_keys: list[str] | None = None,
        lifecycle: LifecycleConfig | None = None,
        *,
        read_only: bool = False,
        startup_deadline_seconds: float = 10.0,
        clock: Callable[[], datetime] = utcnow,
        semantic_index: SemanticIndexPort | None = None,
        curator: CuratorPort | None = None,
        summarizer: SummarizerPort | None = None,
        lifecycle_manager: LifecycleManagerPort | None = None,
    ) -> None:
        """Open (or create) a memory path and bind this handle's identity.

        Args:
            path: Memory directory; created if absent (writer handles only —
                a read-only handle refuses a nonexistent store).
            agent_id: Identity bound at construction (D7); non-empty.
            embedding_adapter: Optional embedder enabling semantic search.
            llm_adapter: Optional LLM. Validated and retained on the handle;
                consumed by the summarizer (``summarize()`` and end-of-session
                rollups). ``None`` degrades loudly (warning + visible digest),
                never silently.
            storage_backend: Injected backend; None opens SQLite in ``path``.
            sensitive_keys: Extra sensitive-key patterns for redaction. Merged
                with the built-in defaults for the surgical ``redact_text``
                scrub AND compiled separately as explicit-only patterns
                (v0.2 D-v02-7 Q3): a match on THESE always whole-masks the
                entry's content unconditionally, even when the key also
                matches a built-in default.
            lifecycle: Policy knobs; None uses ``LifecycleConfig()`` defaults.
            read_only: Skip the writer lock; refuse writes, never touch,
                never create/migrate anything (a nonexistent store is
                refused with ``MemoryStoreError``).
            startup_deadline_seconds: Time-box for the ``startup()`` pass.
            clock: Injectable now-source (tests never sleep).
            semantic_index: Injection seam for the semantic index (tests /
                advanced wiring); None builds the default when possible.
            curator: Injection seam for the curator (tests / advanced wiring);
                None builds a default :class:`ContextCurator` lazily.
            summarizer: Injection seam for ``summarize()`` (tests / advanced
                wiring); None builds a default :class:`MemorySummarizer`
                lazily. Note: end-of-session rollups always use the default
                summarizer (an injected port need not accept ``session_id``).
            lifecycle_manager: Injection seam for the lifecycle manager (tests
                / advanced wiring); None builds the real
                :class:`LifecycleManager`.

        Raises:
            ConfigError: Empty ``agent_id``, structurally invalid adapters,
                or bad configuration values.
            MemoryStoreError: ``read_only=True`` on a nonexistent store —
                a read-only handle never bootstraps one.
            StorageError: Another process holds the writer lock, or the
                backend cannot open.
            SecurityError: A derived internal path escapes the memory dir
                (hostile symlink/junction).
        """
        if not agent_id:
            raise ConfigError("agent_id must be non-empty (D7: identity is load-bearing)")
        if startup_deadline_seconds < 0:
            raise ConfigError("startup_deadline_seconds must be >= 0")
        self._validate_adapters(embedding_adapter, llm_adapter)

        self._agent_id = agent_id
        self._read_only = read_only
        self._clock: Callable[[], datetime] = clock
        self._startup_deadline = startup_deadline_seconds

        memory_dir = Path(path).expanduser().resolve()
        if read_only:
            # C2/DB-HIGH-2: a read-only handle must never create or migrate
            # anything — not the directory, not the database, not the index.
            if not memory_dir.is_dir():
                raise MemoryStoreError(
                    f"no memory store exists at {memory_dir} — read_only handles "
                    "never create one; open a writer first, or check the path"
                )
            if storage_backend is None and not (memory_dir / DB_FILENAME).is_file():
                raise MemoryStoreError(
                    f"no database found at {memory_dir} — read_only handles never "
                    "bootstrap one; open a writer first, or check the path"
                )
        else:
            memory_dir.mkdir(parents=True, exist_ok=True)
        db_path = contain_path(memory_dir / DB_FILENAME, memory_dir)
        index_path = contain_path(memory_dir / INDEX_FILENAME, memory_dir)
        contain_path(memory_dir / (INDEX_FILENAME + ".tmp"), memory_dir)
        lock_path = contain_path(memory_dir / LOCK_FILENAME, memory_dir)
        self._memory_dir = memory_dir

        if storage_backend is not None:
            # The default SQLiteBackend performs the identical check itself —
            # one warning either way, never two, never a failure (ADR-015 #4).
            reason = cloud_sync_risk(memory_dir)
            if reason is not None:
                warnings.warn(
                    f"{reason}; advisory locking and SQLite WAL are unreliable on "
                    "synced/network storage (ADR-015)",
                    UserWarning,
                    stacklevel=2,
                )

        self._lock: _AdvisoryLock | None = None
        if not read_only:
            lock = _AdvisoryLock(lock_path)
            lock.acquire()
            self._lock = lock
        try:
            self._backend: StorageBackend = (
                storage_backend if storage_backend is not None else SQLiteBackend(db_path)
            )
            self._config = lifecycle if lifecycle is not None else LifecycleConfig()
            self._key_patterns = compile_key_patterns(sensitive_keys)
            # ONLY the user-declared patterns (v0.2 D-v02-7 Q3): threaded
            # SEPARATELY from the merged self._key_patterns so a user
            # declaration masks unconditionally, even on overlap with a
            # built-in default. Never merge these two sets.
            self._explicit_key_patterns = compile_explicit_patterns(sensitive_keys)
            # Consumed by the MemorySummarizer (summarize() + end-of-session
            # rollups). None degrades loudly there — never called elsewhere.
            self._llm = llm_adapter
            # Retained for the JSON import path (re-embedding); None imports
            # without vectors (build step 15). The semantic index also holds it.
            self._embedding_adapter = embedding_adapter

            self._store = MemoryStore(self._backend, clock=clock)
            self._kv = KVIndex(self._store)
            self._semantic: SemanticIndexPort | None = (
                semantic_index
                if semantic_index is not None
                else self._build_semantic_index(
                    embedding_adapter, self._backend, index_path, read_only=read_only
                )
            )
            self._semantic_open = False
            self._semantic_warned = False
            self._evaluator = _DecayEvaluator(self._config.half_life_hours)
            self._retrieval = _RetrievalAdapter(self)
            self._curator = curator
            # Injected summarizer seam (tests / advanced wiring); the default
            # MemorySummarizer is built lazily+memoized in _get_summarizer (D8:
            # its token-estimator probe must not run at construction).
            self._summarizer = summarizer
            self._summarizer_impl: MemorySummarizer | None = None
            # One token estimator serves the whole handle (curator + summarizer);
            # resolved lazily so the tiktoken probe never runs at construction.
            self._estimator: TokenEstimator | None = None
            self._lifecycle: LifecycleManagerPort = (
                lifecycle_manager
                if lifecycle_manager is not None
                else LifecycleManager(
                    self._store,
                    self._backend,
                    self._config,
                    agent_id,
                    # llm=None -> no session-end summarizer wired: the manager
                    # degrades loudly on its own (never a silent skip, never a
                    # skip-digest on every session end).
                    summarizer=(
                        _LifecycleSummarizerAdapter(self) if self._llm is not None else None
                    ),
                    clock=clock,
                )
            )
        except BaseException:
            if self._lock is not None:
                self._lock.release()
            raise

        self._startup_lock = threading.Lock()
        self._startup_report: StartupReport | None = None
        self._session_lock = threading.Lock()
        # Dedicated lock for the lazy component builders (_get_estimator/
        # _get_curator/_get_summarizer). It is always the INNERMOST lock: the
        # builders acquire no other lock while holding it and resolve the shared
        # estimator before taking it, so it is safe to acquire while _mutex/
        # _session_lock are held (end-of-session rollup) without a cycle.
        self._build_lock = threading.Lock()
        # Best-known ACTIVE session for read-path activity bumps (search/curate);
        # set by store()/session() writes, never by a read-only handle.
        self._active_session_id: str | None = None
        self._closed = False

    # ------------------------------------------------------------ properties

    @property
    def agent_id(self) -> str:
        """The identity bound at construction (D7)."""
        return self._agent_id

    @property
    def read_only(self) -> bool:
        """Whether this handle refuses writes (and never touches)."""
        return self._read_only

    @property
    def semantic_available(self) -> bool:
        """Whether semantic (vector) retrieval is actually live on this handle.

        ``True`` only when an embedding adapter is configured AND the semantic
        index is open and enabled — the single flag (``_semantic_open``) that
        every semantic-dependent code path in this class already consults
        before touching the index (search, store's index delta, update's
        re-embed, import's reconcile). It is ``False`` — not an error — for
        every "semantic unavailable" condition this class recognizes: no
        embedder configured, the read-only-diverged-cache path
        (``_build_semantic_index``/``_run_startup``), and the
        adapter-mismatch/unrecoverable-cache path that disables the index
        during ``startup()``. In every ``False`` case, retrieval has already
        silently degraded to KV-exact + recency — callers that need to know
        *which* regime actually served a query (e.g. ``tulving eval``'s
        history log) should read this property rather than infer it from
        whether an embedding adapter was merely configured.

        Before ``startup()`` has run, this is ``False`` (the pre-startup
        default) even when an adapter is configured — call ``startup()``
        (directly, or via any public method's lazy trigger) first if the
        answer must reflect the index's real state.
        """
        return self._semantic is not None and self._semantic_open

    # ------------------------------------------------------- static helpers

    @staticmethod
    def _validate_adapters(
        embedding_adapter: EmbeddingAdapter | None, llm_adapter: LLMAdapter | None
    ) -> None:
        """STRUCTURAL adapter validation only (D8): attribute presence via
        ``getattr_static`` — never invokes properties (a LocalEmbedder's
        ``dimension`` would load its model)."""
        if embedding_adapter is not None:
            for name in _EMBEDDING_ADAPTER_SURFACE:
                try:
                    inspect.getattr_static(embedding_adapter, name)
                except AttributeError:
                    raise ConfigError(
                        f"embedding_adapter does not satisfy the EmbeddingAdapter "
                        f"protocol: missing {name!r}"
                    ) from None
        if llm_adapter is not None:
            try:
                complete = inspect.getattr_static(llm_adapter, "complete")
            except AttributeError:
                complete = None
            if complete is None or not callable(complete):
                raise ConfigError(
                    "llm_adapter must expose a callable .complete(prompt); "
                    "llm_adapter=None is legal (summarization degrades loudly)"
                )

    @staticmethod
    def _build_semantic_index(
        adapter: EmbeddingAdapter | None,
        backend: StorageBackend,
        index_path: Path,
        *,
        read_only: bool,
    ) -> SemanticIndexPort | None:
        """Default semantic-index construction (cheap per its blueprint, D8).

        C2/DB-HIGH-2 contract with the semantic-index owner: a read-only
        handle passes ``read_only=True`` — in that mode ``open()`` loads the
        cache only when it is clean, NEVER writes meta/labels/.hnsw, and
        enters a disabled state on divergence (Memory treats disabled as
        semantic-unavailable and degrades loudly to KV-only). Until the
        ``read_only`` parameter lands in ``SemanticIndex`` (capability probe
        below), a read-only handle leaves semantic DISABLED entirely rather
        than risk index writes.
        """
        if adapter is None:
            return None
        accepts_read_only = "read_only" in inspect.signature(SemanticIndex.__init__).parameters
        if read_only:
            if not accepts_read_only:
                logger.warning(
                    "read_only handle: SemanticIndex does not support read_only "
                    "opening yet; semantic search is disabled on this handle "
                    "(KV retrieval still works)"
                )
                return None
            return SemanticIndex(adapter, backend, index_path=index_path, read_only=True)
        if accepts_read_only:
            return SemanticIndex(adapter, backend, index_path=index_path, read_only=False)
        return SemanticIndex(adapter, backend, index_path=index_path)

    # -------------------------------------------------------------- lifecycle

    def startup(self) -> StartupReport:
        """Run the startup pass: explicit, time-boxed, non-fatal, idempotent.

        The first completed run caches its report; every later call (and
        every concurrent caller) receives the cached report with
        ``ran=False``. Lazily invoked by every public method.

        Returns:
            The :class:`StartupReport` for this pass (or the cached one).

        Raises:
            StorageError: When the handle is closed.
        """
        if self._closed:
            raise StorageError("memory is closed")
        with self._startup_lock:
            if self._startup_report is not None:
                return replace(self._startup_report, ran=False)
            report = self._run_startup()
            self._startup_report = report
            return report

    def _run_startup(self) -> StartupReport:
        start = time.monotonic()
        deadline = start + self._startup_deadline
        deferred: list[str] = []
        errors: list[str] = []
        reconcile_report: Any | None = None
        decay_report: DecayReport | None = None
        sessions_abandoned = 0
        entries_marked_stale = 0
        now = self._clock()

        def expired() -> bool:
            return time.monotonic() >= deadline

        # Task 1: index reconciliation (ADR-015 duty; read_only opens only —
        # a read_only index enters a DISABLED state on divergence instead of
        # rebuilding/writing (C2/DB-HIGH-2 contract); disabled == unavailable.
        if self._semantic is not None:
            if expired():
                deferred.append("index_reconcile")
            else:
                try:
                    self._semantic.open()
                    # Coordinated contract with semantic_index: a read-only
                    # open on a divergent/missing cache leaves the index in a
                    # disabled state (its open() warning is the loud part).
                    # Probe both spellings until a public property is agreed.
                    disabled = bool(
                        getattr(
                            self._semantic,
                            "disabled",
                            getattr(self._semantic, "_disabled", False),
                        )
                    )
                    self._semantic_open = not disabled
                    if not self._semantic_open:
                        logger.warning(
                            "read_only handle: semantic index diverged from the "
                            "active adapter/cache and stays disabled (no writes "
                            "in read-only mode); search degrades to KV-only"
                        )
                    elif not self._read_only:
                        reconcile_report = self._semantic.reconcile()
                except TulvingError as exc:
                    # Adapter mismatch / unrecoverable cache: maximally loud,
                    # never fatal. Serving the wrong geometry would be silent
                    # corruption — refusal is the feature.
                    logger.error(
                        "semantic index unavailable: %s — semantic search is "
                        "disabled; run rebuild_index(re_embed=True) to recover",
                        exc,
                    )
                    errors.append(self._safe_error("index_reconcile", exc))

        def run_abandoned() -> None:
            nonlocal sessions_abandoned
            sessions_abandoned = self._lifecycle.check_on_startup()

        def run_evict() -> None:
            nonlocal decay_report
            decay_report = self._run_eviction(now)

        def run_staleness() -> None:
            nonlocal entries_marked_stale
            entries_marked_stale = self._lifecycle.run_staleness_scan()

        write_tasks: tuple[tuple[str, Callable[[], None]], ...] = (
            ("abandoned_sessions", run_abandoned),
            ("evict", run_evict),
            ("staleness_scan", run_staleness),
        )
        for name, task in write_tasks:
            if self._read_only or expired():
                deferred.append(name)
                continue
            try:
                task()
            except TulvingError as exc:
                logger.warning("startup task %s failed (non-fatal): %s", name, exc)
                errors.append(self._safe_error(name, exc))

        return StartupReport(
            ran=True,
            duration_seconds=time.monotonic() - start,
            reconcile=reconcile_report,
            decay=decay_report,
            sessions_abandoned=sessions_abandoned,
            entries_marked_stale=entries_marked_stale,
            deferred=tuple(deferred),
            errors=tuple(errors),
        )

    @staticmethod
    def _safe_error(task_name: str, exc: TulvingError) -> str:
        """Redaction-safe error text for StartupReport (security req #1)."""
        return f"{task_name}: {redact_text(str(exc))}"

    def _run_eviction(self, now: datetime) -> DecayReport:
        """Reason-aware eviction pass (D2): archive-state writes only.

        Delegates to ``tulving.context.decay.evict`` via ``_EvictionAdapter``
        (D12: the formula and the pass live in one place). Exemptions
        (pinned/DECISION) are checked BEFORE any threshold evaluation;
        eviction is strictly ``<`` the threshold.
        """
        return _decay_evict(_EvictionAdapter(self), self._config, now=now)

    def close(self) -> None:
        """Idempotent, best-effort shutdown; never raises.

        Ends this agent's active session (best-effort — next-startup
        abandonment detection is the primary mechanism), flushes/closes the
        index, closes the backend, and releases the writer lock. Any public
        call afterwards raises ``StorageError``.
        """
        if self._closed:
            return
        self._closed = True
        if not self._read_only:
            try:
                self._lifecycle.close_active_session()
            except Exception:
                logger.warning("close: ending the active session failed", exc_info=True)
        if self._semantic is not None and self._semantic_open:
            try:
                self._semantic.close()
            except Exception:
                logger.warning("close: semantic index close failed", exc_info=True)
            self._semantic_open = False
        try:
            self._backend.close()
        except Exception:
            logger.warning("close: backend close failed", exc_info=True)
        if self._lock is not None:
            self._lock.release()

    def __enter__(self) -> Memory:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def rebuild_index(self, *, re_embed: bool = True) -> None:
        """Rebuild the semantic index — the explicit adapter-migration path.

        Works precisely when ``startup()`` refused an adapter mismatch: calls
        ``semantic.rebuild(re_embed=...)`` directly, then opens the index if
        it is not open.

        Raises:
            MemoryStoreError: On a read-only handle.
            ConfigError: When no semantic index / embedder is configured.
            VectorIndexError: Propagated from the rebuild itself.
        """
        if self._closed:
            raise StorageError("memory is closed")
        self._require_writable()
        if self._semantic is None:
            raise ConfigError(
                "rebuild_index requires an embedding adapter / semantic index; none is configured"
            )
        self._semantic.rebuild(re_embed=re_embed)
        if not self._semantic_open:
            self._semantic.open()
            self._semantic_open = True

    # ------------------------------------------------------- store & retrieve

    def store(
        self,
        content: str,
        *,
        type: MemoryType,
        key: str | None = None,
        tags: list[str] | None = None,
        importance: float = 0.5,
        source: SourceInfo | None = None,
        relationships: list[Relationship] | None = None,
    ) -> MemoryEntry:
        """Store a memory; a duplicate live key supersedes, never raises (D1).

        Args:
            content: The memory text; non-empty.
            type: The memory type (SUMMARY is system-only and rejected).
            key: Optional address; storing on a live key supersedes it.
            tags: Optional organization tags.
            importance: Base importance in [0, 1]; immutable except via the
                explicit ``update(importance=...)`` rebase.
            source: Optional provenance; its ``agent_id`` is always
                overridden by this handle's bound identity (D7).
            relationships: Optional links to other entries.

        Returns:
            The stored entry (derived ``importance`` populated; age 0 means
            it equals the base).
        """
        self._ensure_started()
        self._require_writable()
        with self._session_lock:
            session_id = self._lifecycle.ensure_active_session()
            # Remember it so read paths (search/curate) can bump its activity.
            self._active_session_id = session_id
        entry = self._store.create(
            content=content,
            type=type,
            source=self._bind_source(source),
            key=key,
            tags=tags,
            base_importance=importance,
            relationships=relationships,
            session_id=session_id,
        )
        # Index delta, DB-first (ADR-015): the row is durable; an index
        # failure is logged and swallowed — startup()/rebuild repair the cache.
        if self._semantic is not None and self._semantic_open:
            try:
                superseded = next(
                    (
                        rel.target_id
                        for rel in entry.relationships
                        if rel.relationship_type == "supersedes"
                    ),
                    None,
                )
                if superseded is not None:
                    self._semantic.remove(superseded)
                self._semantic.add(entry.id, content)
            except VectorIndexError as exc:
                logger.warning(
                    "index update failed after DB commit for entry %s: %s — the row "
                    "is durable and KV/recency-retrievable. startup() reconciliation "
                    "restores the vector only when its embedding BLOB was persisted; "
                    "if the embedding itself failed there is no BLOB, so run "
                    "rebuild_index(re_embed=True) to restore semantic search",
                    entry.id,
                    exc,
                )
        entry.importance = self._evaluator.effective_importance(entry, self._clock())
        return entry

    def _bind_source(self, source: SourceInfo | None) -> SourceInfo:
        """D7: the bound identity always wins; caller context fields flow."""
        if source is None:
            return SourceInfo(agent_id=self._agent_id)
        if source.agent_id != self._agent_id:
            logger.debug(
                "caller-supplied source.agent_id %r overridden by bound identity %r (D7)",
                source.agent_id,
                self._agent_id,
            )
        return SourceInfo(
            agent_id=self._agent_id,
            step_id=source.step_id,
            run_id=source.run_id,
            workflow_name=source.workflow_name,
        )

    def get(self, key: str) -> MemoryEntry | None:
        """Active entry holding ``key``; None on a miss, never an error.

        Counts as an access (D3) unless the handle is read-only."""
        self._ensure_started()
        return self._with_importance(self._store.get_by_key(key, touch=not self._read_only))

    def get_by_id(self, entry_id: str) -> MemoryEntry | None:
        """Entry by id; archived entries ARE returned (back-link traversal).

        Touch applies to live entries only (store contract); importance is
        populated for archived entries too — filtering is the caller's."""
        self._ensure_started()
        return self._with_importance(self._store.get_by_id(entry_id, touch=not self._read_only))

    def _with_importance(self, entry: MemoryEntry | None) -> MemoryEntry | None:
        if entry is not None:
            entry.importance = self._evaluator.effective_importance(entry, self._clock())
        return entry

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        tags: list[str] | None = None,
        types: list[MemoryType] | None = None,
        min_importance: float = 0.0,
    ) -> list[SearchResult]:
        """Merged KV-exact + semantic search with one batched touch (D3).

        ``min_importance`` filters on EFFECTIVE (decayed) importance — D2's
        read-time value. Results may number fewer than ``top_k`` after
        filtering; an empty/whitespace query returns ``[]`` (search is
        query-driven; "what matters generally" is ``curate()``'s job).

        Raises:
            MemoryStoreError: When ``top_k`` < 1.
        """
        self._ensure_started()
        if top_k < 1:
            raise MemoryStoreError("top_k must be >= 1")
        stripped = query.strip()
        if not stripped:
            return []
        self._record_read_activity()
        now = self._clock()

        candidates: dict[str, tuple[MemoryEntry, float, MatchType]] = {}

        def offer(entry: MemoryEntry, score: float, match: MatchType) -> None:
            existing = candidates.get(entry.id)
            if existing is None or score > existing[1]:
                candidates[entry.id] = (entry, score, match)

        kv_hit = self._store.get_by_key(stripped, touch=False)
        if kv_hit is not None:
            offer(kv_hit, 1.0, MatchType.KEY)
        for entry, score in self._semantic_pairs(
            stripped, top_k=top_k, filter_fn=self._search_filter(tags, types)
        ):
            offer(entry, score, MatchType.SEMANTIC)

        results: list[SearchResult] = []
        for entry, score, match in candidates.values():
            if tags is not None and not set(tags) & set(entry.tags):
                continue
            if types is not None and entry.type not in types:
                continue
            effective = self._evaluator.effective_importance(entry, now)
            entry.importance = effective
            if effective < min_importance:
                continue
            results.append(
                SearchResult(entry=entry, score=min(max(score, 0.0), 1.0), match_type=match)
            )
        results.sort(key=lambda r: (-r.score, -r.entry.created_at.timestamp(), r.entry.id))
        results = results[:top_k]
        if results and not self._read_only:
            # ONE batched statement (D3); mirror onto the returned objects so
            # DB and result agree without a second read.
            self._store.touch_entries([r.entry.id for r in results], now)
            for result in results:
                result.entry.touch(now)
        return results

    def _search_filter(
        self, tags: list[str] | None, types: list[MemoryType] | None
    ) -> Callable[[str], bool] | None:
        """Push tags/types filters through the index's over-fetch."""
        if tags is None and types is None:
            return None
        allowed = {entry.id for entry in self._all_active_entries(tags=tags, types=types)}
        return allowed.__contains__

    def _semantic_pairs(
        self,
        query: str,
        *,
        top_k: int,
        filter_fn: Callable[[str], bool] | None,
    ) -> list[tuple[MemoryEntry, float]]:
        """Semantic hits hydrated to entries, access-neutral, archived
        dropped (post-filter belt-and-braces). Degrades loudly to []."""
        if self._semantic is None or not self._semantic_open:
            self._warn_semantic_disabled()
            return []
        try:
            pairs = self._semantic.search(query, top_k=top_k, filter_fn=filter_fn)
        except VectorIndexError as exc:
            logger.warning("semantic search failed: %s", exc)
            return []
        hydrated: list[tuple[MemoryEntry, float]] = []
        for entry_id, score in pairs:
            entry = self._store.get_by_id(entry_id, touch=False)
            if entry is None or entry.archived:
                continue  # vanished mid-flight / archived-vector post-filter
            hydrated.append((entry, score))
        return hydrated

    def _warn_semantic_disabled(self) -> None:
        """One WARNING per instance — loud degradation, never silent."""
        if self._semantic_warned:
            return
        self._semantic_warned = True
        logger.warning(
            "semantic search is unavailable (no embedding adapter, or the index "
            "failed to open); search degrades to exact-key matches. If an adapter "
            "mismatch was reported, run rebuild_index(re_embed=True)."
        )

    def list_keys(self, prefix: str | None = None) -> list[str]:
        """Active keys, sorted ascending. NOT an access — no touch (D3)."""
        self._ensure_started()
        return self._kv.keys(prefix or "")

    # ------------------------------------------------------------------ curate

    def curate(
        self,
        query: str,
        *,
        token_budget: int = 4000,
        mode: str = "query",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> CuratedContext:
        """Token-budget context curation — delegation to the ContextCurator.

        Returns a ``CuratedContext`` whose ``content`` is ALREADY redacted;
        ``entries`` carries raw, unredacted ``MemoryEntry`` objects (never emit
        those to an egress surface — see ``CuratedContext``). Uses an injected
        ``Memory(curator=...)`` when supplied, otherwise a default
        ``ContextCurator`` built lazily over Memory's ``RetrievalPort``
        (``_RetrievalAdapter``), its decay-backed ``ImportanceEvaluator``
        (``_DecayEvaluator``), the wired ``token_safety_margin``
        (``LifecycleConfig``), and the compiled sensitive-key patterns.
        """
        self._ensure_started()
        self._record_read_activity()
        return self._get_curator().curate(
            query,
            token_budget=token_budget,
            mode=mode,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

    def _get_curator(self) -> CuratorPort:
        """The injected curator, or a lazily-built default over the port (D8).

        Lazy so the tiktoken probe never runs at construction; memoized under
        ``_build_lock`` (double-checked) so one curator — sharing the handle's
        single estimator — serves the whole lifetime even when curate() and
        summarize() first-fire concurrently (ADR-015: one process, many
        threads). The estimator is resolved BEFORE the lock, so ``_build_lock``
        is never taken re-entrantly.
        """
        curator = self._curator
        if curator is None:
            estimator = self._get_estimator()
            with self._build_lock:
                if self._curator is None:
                    self._curator = ContextCurator(
                        self._retrieval,
                        self._evaluator,
                        token_safety_margin=self._config.token_safety_margin,
                        key_patterns=self._key_patterns,
                        explicit_key_patterns=self._explicit_key_patterns,
                        estimator=estimator,
                        now_fn=self._clock,
                    )
                curator = self._curator
        return curator

    def _get_estimator(self) -> TokenEstimator:
        """The handle's shared token estimator (curator + summarizer), memoized.

        Resolved lazily via :func:`resolve_estimator` (tiktoken -> heuristic)
        so the one-shot probe never runs at construction (D8); double-checked
        under ``_build_lock`` so exactly one instance serves the whole handle
        even under concurrent first-access.
        """
        estimator = self._estimator
        if estimator is None:
            with self._build_lock:
                if self._estimator is None:
                    self._estimator = resolve_estimator()
                estimator = self._estimator
        return estimator

    def _get_summarizer(self) -> MemorySummarizer:
        """The default :class:`MemorySummarizer`, built lazily and memoized (D8).

        Consumes ``self._llm`` (``llm=None`` degrades loudly there), the shared
        token estimator, the decay-backed importance evaluator, this handle's
        bound identity (D7), the injected clock, and the compiled sensitive-key
        patterns (so user-augmented redaction reaches LLM egress, security
        req #1). Lazy so the estimator probe never runs at construction;
        double-checked under ``_build_lock``. The estimator is resolved BEFORE
        the lock (never re-entrant); may run while the lifecycle ``_mutex`` is
        held (end-of-session rollup) — ``_build_lock`` is always the innermost
        lock, so no ordering hazard.
        """
        summarizer = self._summarizer_impl
        if summarizer is None:
            estimator = self._get_estimator()
            with self._build_lock:
                if self._summarizer_impl is None:
                    self._summarizer_impl = MemorySummarizer(
                        self._store,
                        self._llm,
                        self._config,
                        estimator,
                        self._evaluator,
                        self._agent_id,
                        self._clock,
                        key_patterns=self._key_patterns,
                        explicit_key_patterns=self._explicit_key_patterns,
                    )
                summarizer = self._summarizer_impl
        return summarizer

    # -------------------------------------------------------------------- edit

    def update(
        self,
        entry_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Edit an entry; None means leave unchanged (all-None is a no-op).

        ``importance=`` is an EXPLICIT REBASE (owner-approved D2 resolution):
        it sets a new ``base_importance`` AND resets the decay anchor
        (``last_accessed_at = now``) — "as of now, this memory is worth x".
        Ordinary content/tags updates never touch importance or the anchor.
        A content change re-embeds (DB committed first; index failures are
        logged and swallowed).

        Raises:
            MemoryStoreError: On a missing id or out-of-range importance.
        """
        self._ensure_started()
        self._require_writable()
        now = self._clock()
        entry: MemoryEntry | None = None
        if content is not None or tags is not None:
            entry = self._store.update(entry_id, content=content, tags=tags)
            if content is not None and self._semantic is not None and self._semantic_open:
                try:
                    self._semantic.remove(entry_id)
                    self._semantic.add(entry_id, content)
                except VectorIndexError as exc:
                    logger.warning(
                        "re-embed failed after DB commit for entry %s: %s", entry_id, exc
                    )
        if importance is not None:
            entry = self._store.rebase_importance(entry_id, importance, now=now)
        if entry is None:
            entry = self._store.get_by_id(entry_id, touch=False)
            if entry is None:
                raise MemoryStoreError(f"no entry with id {entry_id!r}")
        entry.importance = self._evaluator.effective_importance(entry, now)
        return entry

    def pin(self, entry_id: str) -> None:
        """Exempt an entry from decay and eviction (D2).

        Raises:
            MemoryStoreError: On a missing id.
        """
        self._set_pinned(entry_id, True)

    def unpin(self, entry_id: str) -> None:
        """Remove the eviction exemption.

        Raises:
            MemoryStoreError: On a missing id.
        """
        self._set_pinned(entry_id, False)

    def _set_pinned(self, entry_id: str, pinned: bool) -> None:
        self._ensure_started()
        self._require_writable()
        self._store.update(entry_id, pinned=pinned)

    # ------------------------------------------------------------------ forget

    def forget(self, key: str, *, hard: bool = False) -> bool:
        """Forget the active entry holding ``key``; False on a miss.

        Soft (default) archives with reason FORGOTTEN; ``hard=True`` removes
        the row and its embedding. The index vector is tombstoned either way.
        """
        self._ensure_started()
        self._require_writable()
        entry = self._store.get_by_key(key, touch=False)
        if entry is None:
            return False
        self._store.forget(key, hard=hard)
        self._semantic_remove(entry.id)
        return True

    def forget_by_id(self, entry_id: str, *, hard: bool = False) -> bool:
        """Forget an entry by id (unkeyed entries included — MCP A3).

        Missing or already-archived ids return False, never raise."""
        self._ensure_started()
        self._require_writable()
        forgotten = self._store.forget_by_id(entry_id, hard=hard)
        if forgotten:
            self._semantic_remove(entry_id)
        return forgotten

    def forget_by_tags(self, tags: list[str], *, hard: bool = False) -> int:
        """Forget every active entry carrying ANY of ``tags``; returns count.

        Raises:
            MemoryStoreError: On an empty tag list (it would be a footgun).
        """
        self._ensure_started()
        self._require_writable()
        if not tags:
            raise MemoryStoreError("forget_by_tags requires a non-empty tag list")
        matches = self._all_active_entries(tags=list(tags))
        return self._bulk_forget(matches, hard=hard)

    def forget_by_age(
        self,
        older_than: timedelta,
        *,
        types: list[MemoryType] | None = None,
        preserve_decisions: bool = True,
        hard: bool = False,
    ) -> int:
        """Forget entries CREATED more than ``older_than`` ago.

        "Age" means ``created_at`` (a hard age cutoff — unlike the
        summarizer's ``older_than``, which spares recently used entries).
        Pinned entries are ALWAYS spared; DECISION entries are spared unless
        ``preserve_decisions=False``.
        """
        self._ensure_started()
        self._require_writable()
        cutoff = self._clock() - older_than
        matches = self._all_active_entries(created_before=cutoff, types=types)
        eligible = [
            entry
            for entry in matches
            if not entry.pinned and not (preserve_decisions and entry.type is MemoryType.DECISION)
        ]
        return self._bulk_forget(eligible, hard=hard)

    def _bulk_forget(self, entries: list[MemoryEntry], *, hard: bool = False) -> int:
        """Archive (or hard-delete) in ONE transaction; tombstone after."""
        now_iso = self._clock().isoformat()
        forgotten: list[str] = []
        with self._backend.transaction():
            for entry in entries:
                row = self._backend.read(entry.id)
                if row is None or row["archived"]:
                    continue  # raced/already handled; never relabel a reason
                if hard:
                    self._backend.delete(entry.id)
                else:
                    self._backend.update(
                        entry.id,
                        {
                            "archived": True,
                            "archive_reason": ArchiveReason.FORGOTTEN.value,
                            "updated_at": now_iso,
                        },
                    )
                forgotten.append(entry.id)
        for entry_id in forgotten:
            self._semantic_remove(entry_id)
        return len(forgotten)

    def purge_archived(
        self,
        *,
        reasons: list[ArchiveReason] | None = None,
        older_than: timedelta | None = None,
    ) -> int:
        """Hard-delete archived rows — pure passthrough to the store.

        Reason-aware: SUMMARIZED sources are protected unless explicitly
        listed (store contract). Purged ids are also tombstoned in the
        index so the in-memory maps stay honest until compaction.
        """
        self._ensure_started()
        self._require_writable()
        before = self._all_archived_ids()
        purged = self._store.purge_archived(reasons=reasons, older_than=older_than)
        if purged:
            for entry_id in before:
                if self._backend.read(entry_id) is None:
                    self._semantic_remove(entry_id)
        return purged

    def _all_archived_ids(self) -> set[str]:
        ids: set[str] = set()
        offset = 0
        while True:
            page = self._backend.list(
                {"archive_reasons": [reason.value for reason in ArchiveReason]},
                _PAGE,
                offset,
            )
            ids.update(str(row["id"]) for row in page)
            if len(page) < _PAGE:
                return ids
            offset += _PAGE

    # -------------------------------------------------------------- maintenance

    def store_stats(self) -> StoreStats:
        """Aggregate counts + on-disk sizes + meta (``tulving maintenance inspect``).

        Read-only (D2/D3-safe): counts and ``os.stat`` sizes only — never
        computes or mutates importance, never runs a decay pass, never bumps
        ``last_accessed_at``. Legal on a ``read_only=True`` handle.

        Returns:
            The :class:`StoreStats` snapshot.
        """
        self._ensure_started()
        live_count = self._store.count()
        archived_by_reason = {
            reason.value: self._store.count(include_archived=True, archive_reasons=[reason])
            for reason in ArchiveReason
        }
        archived_count = sum(archived_by_reason.values())
        meta = self._store.get_meta()
        db_bytes = _file_size(self._memory_dir / DB_FILENAME)
        wal_bytes = _file_size(self._memory_dir / (DB_FILENAME + "-wal"))
        index_bytes = _file_size(self._memory_dir / INDEX_FILENAME)
        return StoreStats(
            live_count=live_count,
            archived_count=archived_count,
            archived_by_reason=archived_by_reason,
            total_count=live_count + archived_count,
            db_bytes=db_bytes,
            wal_bytes=wal_bytes,
            index_bytes=index_bytes,
            total_bytes=db_bytes + wal_bytes + index_bytes,
            schema_version=meta["schema_version"],
            embedding_model_id=meta["embedding_model_id"],
            embedding_dimension=meta["embedding_dimension"],
            distance_metric=meta["distance_metric"],
        )

    def purgeable_count(
        self,
        *,
        reasons: list[ArchiveReason] | None = None,
        older_than: timedelta | None = None,
    ) -> int:
        """How many archived rows ``purge_archived`` would delete (read-only dry-run).

        Backs ``tulving maintenance purge --dry-run`` and its confirmation
        prompt; never deletes.

        Returns:
            The count of matching archived rows.
        """
        self._ensure_started()
        return self._store.count_archived(reasons=reasons, older_than=older_than)

    def vacuum(self) -> VacuumResult:
        """Reclaim SQLite file space (``tulving maintenance vacuum``).

        Requires a writable handle (holds the writer lock, ADR-015).
        Measures the ``.db`` file before/after ``backend.vacuum()`` — a full
        ``VACUUM``, which needs roughly 2x the database size in temporary
        space and holds the write lock for its duration.

        Returns:
            The :class:`VacuumResult`.

        Raises:
            MemoryStoreError: On a read-only handle.
            StorageError: Propagated from the backend on failure.
        """
        self._ensure_started()
        self._require_writable()
        db_path = self._memory_dir / DB_FILENAME
        bytes_before = _file_size(db_path)
        self._backend.vacuum()
        bytes_after = _file_size(db_path)
        return VacuumResult(
            bytes_before=bytes_before,
            bytes_after=bytes_after,
            bytes_reclaimed=max(bytes_before - bytes_after, 0),
        )

    # ---------------------------------------------------------------- sessions

    def session(self, goal: str | None = None) -> _SessionContextManager:
        """A ``with``-scoped session for this agent (per-agent, D7).

        ``__exit__`` always ends the session, even when the body raised.
        A second concurrent session for THIS agent raises
        ``MemoryStoreError`` at ``__enter__``; other agents never conflict.
        """
        self._ensure_started()
        self._require_writable()
        return _SessionContextManager(self, goal)

    def summarize(
        self,
        *,
        older_than: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]:
        """Summarize old memories into SUMMARY digests (call-budgeted).

        Delegates to the injected ``Memory(summarizer=...)`` seam when supplied,
        otherwise to a default :class:`MemorySummarizer` built lazily over this
        handle's store, LLM, config, shared token estimator, decay evaluator,
        bound identity (D7), clock, and sensitive-key patterns.

        With ``llm=None`` the summarizer degrades LOUDLY (a logged warning plus,
        when candidates exist, one deterministic visible digest) — never
        silently, never raising.

        Args:
            older_than: Spare entries accessed within this window
                (``last_accessed_at`` cutoff) — an entry in active use is
                never rolled up.
            tags: Only summarize entries carrying at least one of these tags.

        Returns:
            The newly created SUMMARY digests (or the single fallback digest
            when ``llm=None`` and candidates exist; ``[]`` when none exist).

        Raises:
            MemoryStoreError: On a read-only handle, or propagated from store
                writes.
        """
        self._ensure_started()
        self._require_writable()
        summarizer = self._summarizer if self._summarizer is not None else self._get_summarizer()
        return summarizer.summarize(older_than=older_than, tags=tags)

    # --------------------------------------------------------------- export/import

    def export_json(
        self,
        path: str | Path,
        *,
        include_embeddings: bool = False,
        include_archived: bool = False,
        include_sensitive: bool = False,
        allowed_root: str | Path | None = None,
    ) -> None:
        """Export this store to a JSON file (build step 15; thin convenience).

        Assembles a :class:`MemoryExporter` over this handle's store, compiled
        sensitive-key patterns (so the user's ``sensitive_keys`` reach the
        export surface — security req #1, threaded both merged AND
        explicit-only per D-v02-7 Q3), and containment root (default: the
        memory directory). Redaction is ON unless ``include_sensitive=True``.

        Args:
            path: Destination path (leaf-validated, directory-contained).
            include_embeddings: Emit per-entry vectors from the BLOBs.
            include_archived: Include archived entries (lossless chains).
            include_sensitive: Opt OUT of redaction (plaintext backup).
            allowed_root: Containment root; defaults to the memory directory.

        Raises:
            SecurityError: On a path violation.
            StorageError: On serialization or I/O failure.
        """
        self._ensure_started()
        exporter = MemoryExporter(
            self._store,
            allowed_root=allowed_root if allowed_root is not None else self._memory_dir,
            sensitive_key_patterns=self._key_patterns,
            explicit_key_patterns=self._explicit_key_patterns,
        )
        exporter.to_json(
            path,
            include_embeddings=include_embeddings,
            include_archived=include_archived,
            include_sensitive=include_sensitive,
        )

    def import_json(self, path: str | Path, *, on_key_conflict: str = "skip") -> ImportReport:
        """Import a JSON export into this store (build step 15; thin convenience).

        Assembles a :class:`MemoryImporter` over this handle's store and active
        embedding adapter, runs the import, then best-effort reconciles the
        semantic index so imported vectors (persisted as BLOBs by ``restore``)
        become searchable in this session — a cache failure is logged, never
        fatal (ADR-015; the BLOBs remain the source of truth).

        Args:
            path: Source export file (read-only; deliberately not contained).
            on_key_conflict: ``"skip"`` (default) or ``"supersede"`` (D1).

        Returns:
            The :class:`ImportReport`.

        Raises:
            MemoryStoreError: On a read-only handle or a malformed/incompatible
                file.
            StorageError: On I/O or embedding failure.
            ValueError: On an unknown ``on_key_conflict``.
        """
        self._ensure_started()
        self._require_writable()
        importer = MemoryImporter(self._store, embedder=self._embedding_adapter)
        report = importer.from_json(path, on_key_conflict=on_key_conflict)
        if report.entries_imported and self._semantic is not None and self._semantic_open:
            try:
                self._semantic.reconcile()
            except VectorIndexError as exc:
                logger.warning(
                    "post-import index reconcile failed: %s — imported vectors are "
                    "durable BLOBs; run rebuild_index() to make them searchable",
                    exc,
                )
        return report

    # --------------------------------------------------------------- internals

    def _record_read_activity(self) -> None:
        """Bump the ACTIVE session's ``last_activity_at`` for a read path.

        Called by ``search()``/``curate()`` with ``write=False`` so a
        read-heavy live session never looks abandoned, without write-amplifying
        (the debounce lives in ``LifecycleManager.record_activity``). No-op on a
        read-only handle (never writes, D-read_only) and when no session is
        active — a pure read never starts a session. Best-effort: a stale
        cached id is a safe no-op in ``record_activity`` (ownership/ACTIVE
        guard), so failures here never break the read.
        """
        if self._read_only:
            return
        session_id = self._active_session_id
        if session_id is None:
            return
        self._lifecycle.record_activity(session_id, write=False)

    def _ensure_started(self) -> None:
        """Lazy startup trigger — every public method calls this first."""
        if self._closed:
            raise StorageError("memory is closed")
        if self._startup_report is None:
            self.startup()

    def _require_writable(self) -> None:
        if self._read_only:
            raise MemoryStoreError(
                "memory opened read_only=True; write operations are unavailable on this handle"
            )

    def _semantic_remove(self, entry_id: str) -> None:
        """Tombstone one vector; cache failures are logged, never raised."""
        if self._semantic is None or not self._semantic_open:
            return
        try:
            self._semantic.remove(entry_id)
        except VectorIndexError as exc:
            logger.warning("index tombstone failed for entry %s: %s", entry_id, exc)

    def _all_active_entries(self, **filters: Any) -> list[MemoryEntry]:
        """Paged full listing of active entries under ``store.list`` filters."""
        entries: list[MemoryEntry] = []
        offset = 0
        while True:
            page = self._store.list(limit=_PAGE, offset=offset, **filters)
            entries.extend(page)
            if len(page) < _PAGE:
                return entries
            offset += _PAGE

"""Memory: the Tulving public API.

Composes storage, indexes, curation, decay, summarization, and sessions
behind one per-agent handle (D7). Owns the advisory single-writer lock
(ADR-015), the explicit ``startup()`` pass (D8), and the write-ordering
discipline across the DB/index boundary (DB-first, index-second, reconcile
at startup).

There is no encryption at rest in v0.1 (ADR-010): do not store secrets you
cannot afford on disk.

Integration seams (this is build step 9; steps 10-13 land later):

- ``curator`` / ``summarizer``: ``curate()``/``summarize()`` raise
  ``NotImplementedError`` until steps 10/12 wire the real components; the
  curator's ``RetrievalPort`` (``_RetrievalAdapter``) is fully implemented.
- lifecycle: ``_DefaultLifecycleManager`` implements the blueprint's L1-L7
  seam minimally (real per-agent session rows; abandonment/staleness are
  deferred no-ops until step 13).
- decay: the D2 formula lives here as a pure default
  (``_effective_importance``) until ``context/decay.py`` lands (step 11).
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import os
import socket
import sys
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
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
from tulving.entry import MemoryEntry, Relationship, SourceInfo, utcnow
from tulving.enums import ArchiveReason, MatchType, MemoryType, SessionStatus
from tulving.exceptions import (
    ConfigError,
    MemoryStoreError,
    StorageError,
    TulvingError,
    VectorIndexError,
)
from tulving.kv_index import KVIndex
from tulving.security import compile_key_patterns, contain_path, redact_text
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

# ---------------------------------------------------------------------------
# Decay seam (pure default until context/decay.py lands — build step 11)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecayReport:
    """Outcome of one eviction pass.

    TODO(step 11): replace with ``tulving.context.decay.DecayReport`` when
    the decay module lands — the field inventory matches blueprint-decay
    exactly, so the swap is type-shape compatible.
    """

    entries_scanned: int
    entries_evicted: int
    entries_exempted: int


def _is_decay_exempt(entry: MemoryEntry) -> bool:
    """True when the entry never decays and never evicts (D2/D6)."""
    return entry.pinned or entry.type is MemoryType.DECISION


def _effective_importance(
    entry: MemoryEntry,
    now: datetime,
    half_life_hours: Mapping[MemoryType, float],
) -> float:
    """THE decay formula (D2), computed on read, never written back.

    Pure default seam: the implementation matches blueprint-decay verbatim.
    TODO(step 11): delegate to ``tulving.context.decay.effective_importance``
    once ``context/decay.py`` lands, keeping the formula in one place (D12).

    Raises:
        ValueError: On a naive ``now`` (programming error, entry.py style).
        ConfigError: On a missing type key or a non-positive/NaN half-life
            (defense in depth — normally impossible post LifecycleConfig).
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware (naive datetime rejected)")
    if _is_decay_exempt(entry):
        return entry.base_importance
    half_life = half_life_hours.get(entry.type)
    if half_life is None:
        raise ConfigError(f"no half-life configured for type {entry.type.value!r}")
    if isinstance(half_life, bool) or math.isnan(half_life) or half_life <= 0:
        raise ConfigError(
            f"half-life for {entry.type.value!r} must be a positive number, got {half_life!r}"
        )
    if math.isinf(half_life):
        return entry.base_importance
    anchor = entry.last_accessed_at
    if anchor is None:  # pragma: no cover - normalized in MemoryEntry.__post_init__
        anchor = entry.created_at
    hours = max((now - anchor).total_seconds() / 3600.0, 0.0)
    factor: float = 0.5 ** (hours / half_life)
    return entry.base_importance * factor


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
    ) -> Any: ...


class SummarizerPort(Protocol):
    """Consumed surface of ``MemorySummarizer`` (build step 12)."""

    def summarize(
        self,
        *,
        older_than: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> list[MemoryEntry]: ...


class LifecycleManagerPort(Protocol):
    """blueprint-memory's L1-L7 lifecycle seam (build step 13)."""

    def ensure_active_session(self) -> str: ...
    def session_start(self, goal: str | None = None) -> str: ...
    def session_end(self, session_id: str) -> None: ...
    def check_on_startup(self) -> int: ...
    def run_staleness_scan(self) -> int: ...
    def close_active_session(self) -> None: ...


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
# Default lifecycle seam (until context/lifecycle.py — build step 13)
# ---------------------------------------------------------------------------


class _DefaultLifecycleManager:
    """Minimal, deterministic, LLM-free lifecycle seam (blueprint L1-L7).

    Real per-agent session rows via the backend's session methods; the
    write-heavy startup duties are deferred no-ops:

    TODO(step 13): replace with ``context.lifecycle.LifecycleManager`` —
    abandonment detection (L4), staleness scanning (L5), session markers,
    and end-of-session summarization all land there. Thread safety is
    provided by ``Memory._session_lock`` around every session mutation.
    """

    def __init__(
        self,
        backend: StorageBackend,
        config: LifecycleConfig,
        agent_id: str,
        clock: Callable[[], datetime],
    ) -> None:
        self._backend = backend
        self._config = config
        self._agent_id = agent_id
        self._clock = clock

    def _active_session_id(self) -> str | None:
        sessions = self._backend.list_sessions(
            agent_id=self._agent_id, status=SessionStatus.ACTIVE.value
        )
        return str(sessions[0]["id"]) if sessions else None

    def _create_session(self, goal: str | None) -> str:
        now = self._clock().isoformat()
        session_id = uuid.uuid4().hex
        self._backend.create_session(
            {
                "id": session_id,
                "agent_id": self._agent_id,
                "goal": goal,
                "started_at": now,
                "last_activity_at": now,
                "status": SessionStatus.ACTIVE.value,
            }
        )
        return session_id

    def ensure_active_session(self) -> str:
        """This agent's ACTIVE session id, auto-starting an anonymous one (L2)."""
        active = self._active_session_id()
        if active is not None:
            self._backend.update_session(active, {"last_activity_at": self._clock().isoformat()})
            return active
        return self._create_session(goal=None)

    def session_start(self, goal: str | None = None) -> str:
        """Start a session; refuse when THIS agent already has one (L3, D7)."""
        if self._active_session_id() is not None:
            raise MemoryStoreError(f"Session already active for agent '{self._agent_id}'")
        return self._create_session(goal)

    def session_end(self, session_id: str) -> None:
        """End an ACTIVE session; summarization is deferred loudly (L3)."""
        session = self._backend.get_session(session_id)
        if session is None or session["status"] != SessionStatus.ACTIVE.value:
            raise MemoryStoreError(f"No active session {session_id!r} to end")
        if self._config.summarize_on_session_end:
            logger.warning(
                "session %s ended without summarization: the summarizer lands at "
                "build step 12 (deterministic session markers land with lifecycle, "
                "step 13)",
                session_id,
            )
        now = self._clock().isoformat()
        self._backend.update_session(
            session_id,
            {
                "status": SessionStatus.ENDED.value,
                "ended_at": now,
                "last_activity_at": now,
            },
        )

    def check_on_startup(self) -> int:
        """Abandoned-session detection hook (L4).

        TODO(step 13): per-session ``last_activity_at`` vs
        ``config.inactivity_threshold``, per-agent correct, never an LLM
        call. No-op until ``context/lifecycle.py`` lands.
        """
        return 0

    def run_staleness_scan(self) -> int:
        """Staleness-tagging hook (L5). TODO(step 13): no-op until lifecycle."""
        return 0

    def close_active_session(self) -> None:
        """Best-effort graceful close used by ``Memory.close()`` (L6)."""
        active = self._active_session_id()
        if active is not None:
            self.session_end(active)


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
            llm_adapter: Optional LLM. Validated and retained on the handle,
                but INERT until the summarizer lands (build step 12) — no
                code path calls it yet; None degrades loudly downstream.
            storage_backend: Injected backend; None opens SQLite in ``path``.
            sensitive_keys: Extra sensitive-key patterns for redaction.
            lifecycle: Policy knobs; None uses ``LifecycleConfig()`` defaults.
            read_only: Skip the writer lock; refuse writes, never touch,
                never create/migrate anything (a nonexistent store is
                refused with ``MemoryStoreError``).
            startup_deadline_seconds: Time-box for the ``startup()`` pass.
            clock: Injectable now-source (tests never sleep).
            semantic_index: Injection seam for the semantic index (tests /
                advanced wiring); None builds the default when possible.
            curator: Injection seam until ContextCurator lands (step 10).
            summarizer: Injection seam until MemorySummarizer lands (step 12).
            lifecycle_manager: Injection seam until LifecycleManager lands
                (step 13); None uses the deterministic default seam.

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
            # CR-M1: retained for the summarizer (build step 12); inert until
            # then — no code path in this module calls it.
            self._llm = llm_adapter

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
            self._summarizer = summarizer
            self._lifecycle: LifecycleManagerPort = (
                lifecycle_manager
                if lifecycle_manager is not None
                else _DefaultLifecycleManager(self._backend, self._config, agent_id, clock)
            )
        except BaseException:
            if self._lock is not None:
                self._lock.release()
            raise

        self._startup_lock = threading.Lock()
        self._startup_report: StartupReport | None = None
        self._session_lock = threading.Lock()
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

        Pure-default seam: TODO(step 11) delegate to
        ``context.decay.evict`` via an ``_EvictionAdapter`` once the decay
        module lands. Exemptions (pinned/DECISION) are checked BEFORE any
        threshold evaluation; eviction is strictly ``<`` the threshold.
        """
        scanned = evicted = exempted = 0
        for entry in self._all_active_entries():
            scanned += 1
            if _is_decay_exempt(entry):
                exempted += 1
                continue
            effective = _effective_importance(entry, now, self._config.half_life_hours)
            if effective < self._config.eviction_threshold:
                self._store.archive(entry.id, ArchiveReason.EVICTED)
                self._semantic_remove(entry.id)
                evicted += 1
                logger.debug("evicted entry %s (effective importance below threshold)", entry.id)
        if evicted:
            logger.info(
                "eviction pass: scanned=%d evicted=%d exempted=%d", scanned, evicted, exempted
            )
        return DecayReport(scanned, evicted, exempted)

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
    ) -> Any:
        """Token-budget context curation — delegation to the ContextCurator.

        Returns ``CuratedContext`` (content ALREADY redacted) once build
        step 10 lands. The return type is ``Any`` only until then.

        Raises:
            NotImplementedError: Until a curator is wired (build step 10) —
                Memory's ``RetrievalPort`` implementation is ready.
        """
        self._ensure_started()
        # TODO(step 10): this seam MUST be replaced by real ContextCurator
        # wiring BEFORE mcp/server.py (step 14) is built — the MCP `curate`
        # and `orient` tools depend on it.
        if self._curator is None:
            raise NotImplementedError(
                "ContextCurator lands at build step 10; Memory's RetrievalPort "
                "(_RetrievalAdapter) is ready — construct the curator over it and "
                "inject via Memory(curator=...) or wire the default here."
            )
        return self._curator.curate(
            query,
            token_budget=token_budget,
            mode=mode,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )

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

        Raises:
            NotImplementedError: Until MemorySummarizer lands (build step
                12) — inject via ``Memory(summarizer=...)`` until then.
        """
        self._ensure_started()
        self._require_writable()
        # TODO(step 12): this seam MUST be replaced by real MemorySummarizer
        # wiring (consuming self._llm) BEFORE mcp/server.py (step 14) is
        # built — session-end summarization and llm=None loud degradation
        # depend on it.
        if self._summarizer is None:
            raise NotImplementedError(
                "MemorySummarizer lands at build step 12; inject one via "
                "Memory(summarizer=...) until then."
            )
        return self._summarizer.summarize(older_than=older_than, tags=tags)

    # --------------------------------------------------------------- internals

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

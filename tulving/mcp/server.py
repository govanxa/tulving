"""Tulving MCP server: a thin stdio wrapper over Memory (six tools).

Local-only by design: stdio transport, never a network listener (CLAUDE.md
security req #5). Every tool response goes through ``security.redact_text`` at
one choke point (``_emit``). Requires ``pip install "tulving[mcp]"``.

Thin-wrapper discipline: a handler only converts arguments, calls one
``Memory`` method (always off the event loop via ``_offload``), formats the
result, and redacts it. No business logic lives here.
"""

from __future__ import annotations

import argparse
import functools
import logging
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, Literal, TypeVar

from tulving import Memory, MemoryType
from tulving.adapters.embeddings import LocalEmbedder, OpenAIEmbedder
from tulving.adapters.llm import DEFAULT_MODEL, AnthropicAdapter
from tulving.exceptions import ConfigError, StorageError, TulvingError
from tulving.security import redact_text

if TYPE_CHECKING:  # import-time-free: type checkers see it, runtime never does
    from collections.abc import Callable

    from mcp.server.fastmcp import FastMCP
    from pydantic import Field
else:
    # ``pydantic`` ships with the ``[mcp]`` extra (it is ``mcp``'s dependency),
    # so it is present whenever ``build_server`` runs. Guarded here so a
    # core-only ``import tulving.mcp.server`` still succeeds; the annotated
    # ``Field`` bound below is only resolved (by FastMCP's schema build) once
    # the extra is installed.
    try:
        from pydantic import Field
    except ImportError:  # pragma: no cover - exercised only on a core-only install
        Field = None

logger = logging.getLogger("tulving.mcp")

T = TypeVar("T")

EXIT_OK: Final[int] = 0
EXIT_CONFIG: Final[int] = 1  # missing extra, bad adapter choice, bad config value
EXIT_LOCKED: Final[int] = 2  # single-writer lock refused (ADR-015)

DEFAULT_MEMORY_PATH: Final[str] = "./tulving_memory"
DEFAULT_TOKEN_BUDGET: Final[int] = 4000

# Valid memory-type strings for the search filter (includes SUMMARY - system
# entries are searchable even though callers cannot store() them).
_VALID_TYPES: Final[tuple[str, ...]] = tuple(t.value for t in MemoryType)

NO_LLM_NOTE: Final[str] = (
    "[note: no LLM adapter configured - this briefing contains key decisions and "
    "deterministic session markers, not generated summaries]"
)

_NO_LLM_WARNING: Final[str] = (
    "no LLM adapter configured - session summarization and orient digests degrade "
    "to deterministic markers (set TULVING_LLM_ADAPTER=claude or --llm claude to "
    "enable summaries)"
)

# Boot-time loud half of the offline-mode degradation contract (stderr, once).
_NO_EMBEDDER_WARNING: Final[str] = (
    "no embedding adapter configured (--embedding none): semantic search "
    "(memory_search) is disabled; store/get/curate/forget/list_keys work, and "
    "curate ranks by exact-key match + importance/recency. To enable "
    "find-by-meaning, restart with --embedding local (needs tulving[local]) or "
    "--embedding openai (needs tulving[openai] + OPENAI_API_KEY)."
)

# Response-surface loud half: the text of the ToolError memory_search raises in offline mode.
_SEARCH_DISABLED_NOTE: Final[str] = (
    "semantic search (find-by-meaning) is unavailable in offline mode "
    "(--embedding none): no embedder is configured. Use memory_curate for "
    "relevance-ranked context (exact-key + importance/recency) or memory_get for "
    "an exact key. To enable semantic search, restart the server with --embedding "
    "local or --embedding openai."
)

# Tool description shown when search is disabled (resident-token cost is tiny and worth it).
_DESC_SEARCH_DISABLED: Final[str] = (
    "DISABLED in offline mode (--embedding none): semantic search requires an "
    "embedder. Use memory_curate or memory_get instead."
)

_MCP_HINT: Final[str] = (
    'the MCP server requires the [mcp] extra; install it with: pip install "tulving[mcp]"'
)

# Terse tool descriptions (spec mcp-server.md §3). Resident tool definitions
# cost context tokens every turn - do not elaborate (ADR-016).
_DESC_STORE: Final[str] = (
    "Store a memory (fact, decision, observation, or plan) for later retrieval. "
    "Storing to an existing key supersedes the old entry."
)
_DESC_GET: Final[str] = "Retrieve a memory by its exact key."
_DESC_SEARCH: Final[str] = "Search memories by meaning. Returns the most relevant entries."
_DESC_CURATE: Final[str] = (
    "Get the most relevant memories fitted to a token budget, as prompt-ready text. "
    "mode 'orient' returns a cold-start briefing (key decisions, session history) "
    "instead of query-ranked results."
)
_DESC_FORGET: Final[str] = (
    "Remove memories by key, id, or tags. Archives by default; hard=true deletes."
)
_DESC_LIST: Final[str] = "List memory keys, optionally filtered by prefix (e.g. 'decision:')."


def _require_mcp() -> Any:
    """Import and return the ``FastMCP`` class, gated behind the ``[mcp]`` extra.

    Called only from ``build_server``/``main`` - never at module import time, so
    ``import tulving.mcp.server`` succeeds on a core-only install and the
    console-script surfaces OUR message rather than a traceback.

    Returns:
        The ``FastMCP`` class.

    Raises:
        ConfigError: When the ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise ConfigError(_MCP_HINT) from exc
    return FastMCP


@dataclass(frozen=True)
class ServerSettings:
    """Resolved configuration: CLI args override env vars override defaults."""

    memory_path: Path
    embedding: str = "local"  # {local, openai, none}
    llm: str = "none"  # {none, claude}
    llm_model: str | None = None
    default_token_budget: int = DEFAULT_TOKEN_BUDGET
    read_only: bool = False
    llm_configured: bool = False  # derived: an LLM adapter is wired
    embedding_configured: bool = True  # derived: an embedder is wired (False iff embedding=="none")


def _emit(text: str) -> str:
    """THE redaction choke point (security req #1).

    Every string a handler returns and every ``ToolError`` message goes through
    here. Idempotent: re-redacting already-redacted curator output is a no-op by
    ``security.py``'s contract.
    """
    return redact_text(text)


async def _offload(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a synchronous ``Memory`` call on a worker thread.

    ``Memory`` is synchronous and embedding is CPU-bound; every ``Memory`` call
    routes through here so a handler never blocks the event loop (spec §4.1).
    ``Memory`` is thread-safe in-process (ADR-015), so the anyio default limiter
    is fine (no tuning in v0.1). ``anyio`` is imported lazily (``[mcp]`` extra).
    """
    import anyio

    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


# --------------------------------------------------------------------------- formatting


def _format_entry(entry: Any) -> str:
    """One retrieved entry as terse, human-readable text (ADR-012 #4).

    The memory's own key is shown BARE in the header (never as ``key: value``):
    the redaction choke point masks the value after any sensitive label, and
    ``key`` is one such label - so a labelled form would eat legitimate
    metadata. The curator uses the same bare-key convention.
    """
    importance = entry.importance if entry.importance is not None else entry.base_importance
    tags = ", ".join(entry.tags) if entry.tags else "(none)"
    header = f"[{entry.type.value}] {entry.key}" if entry.key else f"[{entry.type.value}]"
    lines = [
        header,
        f"id: {entry.id}",
        f"tags: {tags}",
        f"importance: {importance:.2f}",
        f"created: {entry.created_at.isoformat()}",
        "---",
        entry.content,
    ]
    return "\n".join(lines)


def _format_results(results: list[Any]) -> str:
    """Search results as numbered lines (score | match | type | bare key)."""
    if not results:
        return "No matching memories."
    lines = []
    for i, result in enumerate(results, start=1):
        entry = result.entry
        key = entry.key if entry.key else "(no key)"
        lines.append(
            f"{i}. [{result.score:.2f} {result.match_type.value} | "
            f"{entry.type.value} | {key}] {entry.content}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- server build


def build_server(memory: Memory, settings: ServerSettings) -> Any:
    """Register the six tools on a ``FastMCP('tulving')`` and return it.

    Pure wiring: takes a constructed ``Memory`` so tests pass a mock and never
    touch storage, adapters, or a transport.

    Args:
        memory: The assembled ``Memory`` handle every tool delegates to.
        settings: Resolved configuration (default budget, LLM-configured flag).

    Returns:
        The configured ``FastMCP`` instance (typed ``Any`` - ``mcp`` is optional).
    """
    fastmcp_cls = _require_mcp()
    from mcp.server.fastmcp.exceptions import ToolError

    server: FastMCP = fastmcp_cls("tulving")

    def _guard(fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a handler body: translate errors into redacted tool errors.

        ``TulvingError`` -> redacted ``ToolError`` (actionable, ``isError``).
        Any other exception -> generic message + server-side ``logger.exception``,
        so tracebacks and raw text (which may hold paths/content) never cross the
        stdio boundary.
        """

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolError:
                raise
            except TulvingError as exc:
                raise ToolError(_emit(str(exc))) from None
            except Exception:
                logger.exception("unexpected error in a tulving-mcp handler")
                raise ToolError("internal error - see the tulving-mcp server log") from None

        return wrapper

    @server.tool(description=_DESC_STORE)
    @_guard
    async def memory_store(
        content: str,
        type: Literal["fact", "decision", "observation", "plan"],
        key: str | None = None,
        tags: list[str] | None = None,
        importance: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5,
    ) -> str:
        entry = await _offload(
            memory.store,
            content=content,
            type=MemoryType(type),
            key=key,
            tags=tags or [],
            importance=importance,
        )
        suffix = f" with key '{entry.key}'" if entry.key else ""
        return _emit(f"Stored memory {entry.id}{suffix}")

    @server.tool(description=_DESC_GET)
    @_guard
    async def memory_get(key: str) -> str:
        entry = await _offload(memory.get, key)
        if entry is None:
            return _emit(f"No memory found for key '{key}'.")
        return _emit(_format_entry(entry))

    search_desc = _DESC_SEARCH if settings.embedding_configured else _DESC_SEARCH_DISABLED

    @server.tool(description=search_desc)
    @_guard
    async def memory_search(
        query: str,
        top_k: int = 5,
        tags: list[str] | None = None,
        types: list[str] | None = None,
    ) -> str:
        if not settings.embedding_configured:
            # Loud, actionable, redacted - short-circuits BEFORE _offload; Memory.search
            # is never called (it would silently narrow to KV-exact - see Design D-A).
            raise ToolError(_emit(_SEARCH_DISABLED_NOTE))
        parsed_types: list[MemoryType] | None = None
        if types:
            for t in types:
                if t not in _VALID_TYPES:
                    raise ToolError(
                        _emit(f"unknown memory type '{t}' - valid: {', '.join(_VALID_TYPES)}")
                    )
            parsed_types = [MemoryType(t) for t in types]
        results = await _offload(
            memory.search,
            query,
            top_k=top_k,
            tags=tags,
            types=parsed_types,
        )
        return _emit(_format_results(results))

    @server.tool(description=_DESC_CURATE)
    @_guard
    async def memory_curate(
        query: str,
        token_budget: int | None = None,
        mode: Literal["query", "orient"] = "query",
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
    ) -> str:
        budget = token_budget if token_budget is not None else settings.default_token_budget
        ctx = await _offload(
            memory.curate,
            query,
            token_budget=budget,
            mode=mode,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
        )
        body = (
            f"{ctx.content}\n---\n[tokens: {ctx.token_count}, budget remaining: "
            f"{ctx.budget_remaining}, sources consulted: {ctx.sources_consulted}]"
        )
        if mode == "orient" and not settings.llm_configured:
            body += "\n" + NO_LLM_NOTE
        return _emit(body)

    @server.tool(description=_DESC_FORGET)
    @_guard
    async def memory_forget(
        key: str | None = None,
        id: str | None = None,
        tags: list[str] | None = None,
        hard: bool = False,
    ) -> str:
        # Empty string / empty list count as NOT supplied (truthy test), so
        # tags=[] or key="" trips the exactly-one gate instead of routing to
        # forget_by_tags([]) / forget("").
        selectors = [s for s in (key, id, tags) if s]
        if len(selectors) != 1:
            raise ToolError(_emit("provide exactly one of: key, id, tags"))
        verb = "Deleted" if hard else "Archived"
        if key:
            hit = await _offload(memory.forget, key, hard=hard)
            if not hit:
                return _emit(f"No memory found for key '{key}'.")
            return _emit(f"{verb} memory with key '{key}'.")
        if id:
            hit = await _offload(memory.forget_by_id, id, hard=hard)
            if not hit:
                return _emit(f"No memory found for id '{id}'.")
            return _emit(f"{verb} memory with id '{id}'.")
        assert tags  # the exactly-one gate guarantees tags is the live selector
        count = await _offload(memory.forget_by_tags, tags, hard=hard)
        return _emit(f"{verb} {count} memories.")

    @server.tool(description=_DESC_LIST)
    @_guard
    async def memory_list_keys(prefix: str | None = None) -> str:
        keys = await _offload(memory.list_keys, prefix=prefix)
        if not keys:
            return _emit("No keys stored.")
        return _emit("\n".join(keys))

    return server


# --------------------------------------------------------------------------- adapters


def _build_embedder(name: str) -> Any:
    """Construct the embedding adapter selected by ``name`` (``None`` for offline mode)."""
    if name == "none":
        return None
    if name == "local":
        return LocalEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    raise ConfigError(f"unknown embedding adapter {name!r}; choose 'local', 'openai', or 'none'")


def _build_llm(settings: ServerSettings) -> Any:
    """Construct the LLM adapter selected by ``settings.llm`` (or ``None``)."""
    if settings.llm == "none":
        return None
    if settings.llm == "claude":
        return AnthropicAdapter(model=settings.llm_model or DEFAULT_MODEL)
    raise ConfigError(f"unknown llm adapter {settings.llm!r}; choose 'none' or 'claude'")


def create_memory(settings: ServerSettings) -> Memory:
    """Construct adapters from ``settings`` and return the ``Memory`` handle.

    The advisory writer lock is taken inside ``Memory.__init__``; a refusal
    raises ``StorageError`` (ADR-015), surfaced by ``main`` as ``EXIT_LOCKED``.

    Raises:
        ConfigError: Unknown adapter choice, a missing extra, or a missing
            credential (from the adapter constructor).
        StorageError: The writer lock is held by another process.
        TulvingError: Other construction failures (e.g. read-only, missing store).
    """
    embedder = _build_embedder(settings.embedding)
    llm = _build_llm(settings)
    return Memory(
        settings.memory_path,
        embedding_adapter=embedder,
        llm_adapter=llm,
        read_only=settings.read_only,
    )


# --------------------------------------------------------------------------- CLI / main


def _build_parser() -> argparse.ArgumentParser:
    """Build the ``tulving-mcp`` argument parser (flags fall back to env vars)."""
    parser = argparse.ArgumentParser(
        prog="tulving-mcp",
        description="Run the Tulving memory server over stdio for MCP clients.",
        epilog=(
            "Security: memories are stored UNENCRYPTED at rest (v0.1) - do not "
            "store secrets you cannot afford on disk. The server is local-only "
            "(stdio); it never opens a network listener."
        ),
    )
    parser.add_argument("--memory-path", default=None, help="Memory storage directory.")
    parser.add_argument(
        "--embedding",
        choices=["local", "openai", "none"],
        default=None,
        help="Embedding backend. 'none' = torch-free offline mode (semantic search disabled).",
    )
    parser.add_argument(
        "--llm", choices=["none", "claude"], default=None, help="LLM for summarization."
    )
    parser.add_argument(
        "--read-only", action="store_true", help="Open without the writer lock; refuse writes."
    )
    return parser


def _resolve_settings(args: argparse.Namespace) -> ServerSettings:
    """Merge CLI args, env vars, and defaults into ``ServerSettings``.

    Raises:
        ConfigError: When ``TULVING_DEFAULT_TOKEN_BUDGET`` is not an integer.
    """
    memory_path = args.memory_path or os.environ.get("TULVING_MEMORY_PATH") or DEFAULT_MEMORY_PATH
    embedding = args.embedding or os.environ.get("TULVING_EMBEDDING_ADAPTER") or "local"
    llm = args.llm or os.environ.get("TULVING_LLM_ADAPTER") or "none"
    llm_model = os.environ.get("TULVING_LLM_MODEL") or None

    budget_raw = os.environ.get("TULVING_DEFAULT_TOKEN_BUDGET")
    if budget_raw:
        try:
            budget = int(budget_raw)
        except ValueError as exc:
            raise ConfigError("TULVING_DEFAULT_TOKEN_BUDGET must be an integer") from exc
    else:
        budget = DEFAULT_TOKEN_BUDGET

    return ServerSettings(
        memory_path=Path(memory_path).expanduser().resolve(),
        embedding=embedding,
        llm=llm,
        llm_model=llm_model,
        default_token_budget=budget,
        read_only=args.read_only,
        llm_configured=llm != "none",
        embedding_configured=embedding != "none",
    )


def _install_signal_handlers() -> None:
    """Best-effort shutdown signals (spec §4). Windows/thread limits are FINE.

    Startup-time abandoned-session detection is the primary recovery path
    (ADR-013), so failure to install a handler is only logged at debug level.
    """

    def _request_shutdown(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    for name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            signal.signal(sig, _request_shutdown)
        except (ValueError, OSError, AttributeError):
            logger.debug("could not install %s handler (non-fatal)", name, exc_info=True)


def _fail(message: str, code: int) -> int:
    """Print a redacted diagnostic to stderr and return an exit code."""
    print(_emit(message), file=sys.stderr)
    return code


def main(argv: list[str] | None = None) -> int:
    """Console entry point (``tulving-mcp = tulving.mcp.server:main``)."""
    args = _build_parser().parse_args(argv)

    try:
        settings = _resolve_settings(args)
    except ConfigError as exc:
        return _fail(str(exc), EXIT_CONFIG)

    try:
        _require_mcp()
    except ConfigError as exc:
        return _fail(str(exc), EXIT_CONFIG)

    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        memory = create_memory(settings)
    except ConfigError as exc:
        return _fail(str(exc), EXIT_CONFIG)
    except StorageError:
        return _fail(
            f"another Tulving process is using {settings.memory_path} - close the "
            "other session, or start with --read-only",
            EXIT_LOCKED,
        )
    except TulvingError as exc:
        return _fail(str(exc), EXIT_CONFIG)

    if not settings.llm_configured:
        logger.warning(_NO_LLM_WARNING)  # the loud half of the degradation contract

    if not settings.embedding_configured:
        logger.warning(_NO_EMBEDDER_WARNING)  # loud half of the offline-mode degradation contract

    try:
        report = memory.startup()
        logger.info("startup complete: %s", report)
    except TulvingError as exc:
        logger.warning("startup() failed (non-fatal): %s", exc)  # D8: degraded, never fatal

    _install_signal_handlers()

    try:
        server = build_server(memory, settings)
        server.run()  # FastMCP default stdio transport (local-only, security req #5)
    except KeyboardInterrupt:
        logger.info("shutdown requested")
    finally:
        try:
            memory.close()
        except Exception:
            logger.warning("close() failed during shutdown (non-fatal)", exc_info=True)

    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

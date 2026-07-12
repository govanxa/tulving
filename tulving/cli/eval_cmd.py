"""Tulving CLI -- `tulving eval`: is Tulving helping, measured against your real store?

Read-only measurement of dump-vs-``curate()`` token cost, optional probe-set
correctness scoring against a local OpenAI-compatible endpoint, a versioned JSON
history log, and a self-contained HTML trend report. A pure consumer of the public
API (``Memory``, ``curate``, ``list_keys``, ``get``) plus one mandated internal reuse
(``tulving.context.curator.resolve_estimator``). Adds no engine features, no core
dependencies.

The store is opened READ-ONLY (``Memory(..., read_only=True)``): this subcommand never
mutates the user's memory (see the read-only migration caveat, blueprint-eval-cli).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from re import Pattern
from typing import Any, Final

from tulving import Memory, __version__
from tulving.adapters.embeddings import EmbeddingAdapter
from tulving.cli import eval_report
from tulving.cli._util import EXIT_ERROR, EXIT_OK, emit
from tulving.cli.eval_scoring import AnswerScorer, load_probes, score_probes
from tulving.context.curator import TokenEstimator, resolve_estimator
from tulving.exceptions import ConfigError, MemoryStoreError, StorageError, TulvingError
from tulving.security import REDACTED as _REDACTED_BODY
from tulving.security import compile_key_patterns, is_sensitive_key, redact_text

NAME: Final[str] = "eval"
HISTORY_SCHEMA_VERSION: Final[int] = 1
DEFAULT_HISTORY: Final[str] = "eval_history.json"
DEFAULT_HTML: Final[str] = "report.html"
DEFAULT_BUDGET: Final[int] = 1500
DEFAULT_LM_URL: Final[str] = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL: Final[str] = "local-model"
DEFAULT_GENERIC_QUERY: Final[str] = "resuming work - what do I need to know?"
DEFAULT_PROBES_FILENAME: Final[str] = "probes.json"
SESSION_KEY_PREFIX: Final[str] = "session:"

STARTER_PROBES: Final[list[dict[str, str]]] = [
    {"q": "What did we decide about the datastore?", "expect": "SQLite"},
    {"q": "Which auth scheme did we choose?", "expect": "JWT"},
    {"q": "How long do auth tokens live?", "expect": "15"},
]

# Set by register() so run() can call parser.error() for the conditionally-required
# --store check (argparse OWNS exit 2; see the dispatcher's exit-code table).
_parser: argparse.ArgumentParser | None = None


@dataclass(frozen=True)
class EvalRun:
    """One history row (serializes 1:1 to the log's ``runs[]`` entries)."""

    timestamp: str
    store_path: str
    agent_id: str
    store_size: int
    budget: int
    dump_tokens: int
    curate_tokens: int
    reduction: float
    estimator: str
    embedding: str
    correctness: dict[str, str] | None
    model: str | None
    tulving_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe plain dict (counts + metadata only)."""
        return {
            "timestamp": self.timestamp,
            "store_path": self.store_path,
            "agent_id": self.agent_id,
            "store_size": self.store_size,
            "budget": self.budget,
            "dump_tokens": self.dump_tokens,
            "curate_tokens": self.curate_tokens,
            "reduction": self.reduction,
            "estimator": self.estimator,
            "embedding": self.embedding,
            "correctness": self.correctness,
            "model": self.model,
            "tulving_version": self.tulving_version,
        }


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Register the ``eval`` subcommand parser (D8: cheap, no I/O)."""
    global _parser
    parser = subparsers.add_parser(
        NAME,
        help="measure whether Tulving is helping, against your real store",
        description=(
            "Read-only measurement of dump-vs-curate token cost, optional probe "
            "correctness scoring, and a self-contained HTML trend report."
        ),
    )
    parser.add_argument("--store", help="path to your real memory store (opened read-only)")
    parser.add_argument(
        "--agent-id", default="default", help="agent id bound at open (default: default)"
    )
    parser.add_argument(
        "--embedding",
        choices=("none", "local", "openai"),
        default="none",
        help="embedder to match your MCP server (default: none)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=DEFAULT_BUDGET,
        help=f"curate token budget (default: {DEFAULT_BUDGET})",
    )
    parser.add_argument(
        "--history",
        default=DEFAULT_HISTORY,
        help=f"history log file (default: {DEFAULT_HISTORY})",
    )
    parser.add_argument(
        "--html",
        dest="html",
        default=DEFAULT_HTML,
        help=f"report output path (default: {DEFAULT_HTML})",
    )
    parser.add_argument("--out", dest="html", default=DEFAULT_HTML, help=argparse.SUPPRESS)
    parser.add_argument("--no-report", action="store_true", help="skip HTML render (log only)")
    parser.add_argument(
        "--probes", help="JSON file of {q, expect} probes; enables correctness scoring"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"model name reported to the endpoint (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--lm-url", default=DEFAULT_LM_URL, help="OpenAI-compatible chat-completions URL"
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="per-request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--init-probes",
        nargs="?",
        const=DEFAULT_PROBES_FILENAME,
        default=None,
        metavar="PATH",
        help=f"write a starter probes file (default {DEFAULT_PROBES_FILENAME}) and exit; "
        "never opens a store",
    )
    parser.set_defaults(_handler=run)
    _parser = parser


def _init_probes(target: Path) -> int:
    """Write a starter probes file; refuse to overwrite an existing one."""
    if target.exists():
        emit(f"{target} already exists -- not overwriting.", stream=sys.stderr)
        return EXIT_ERROR
    target.write_text(json.dumps(STARTER_PROBES, indent=2), encoding="utf-8")
    emit(f"Wrote starter probes to {target}. Edit it with your own known-answer questions.")
    return EXIT_OK


def run(args: argparse.Namespace) -> int:
    """Execute the ``eval`` subcommand.

    Returns:
        A process exit code (``EXIT_OK``/``EXIT_ERROR`` from ``tulving.cli._util``).
    """
    if args.init_probes is not None:
        return _init_probes(Path(args.init_probes))

    if not args.store:
        assert _parser is not None  # register() always runs before run() (ruling 1)
        _parser.error("--store is required (or use --init-probes). See --help.")

    history_path = Path(args.history)
    html_path = Path(args.html)

    try:
        mem = _open_store(args.store, args.agent_id, args.embedding)
    except (MemoryStoreError, StorageError, ConfigError) as exc:
        emit(f"could not open store: {redact_text(str(exc))}", stream=sys.stderr)
        return EXIT_ERROR

    try:
        probes: list[dict[str, str]] = []
        if args.probes:
            probe_path = Path(args.probes)
            if not probe_path.exists():
                emit(
                    f"probes file not found: {probe_path} (create one with --init-probes)",
                    stream=sys.stderr,
                )
                return EXIT_ERROR
            probes = load_probes(probe_path)

        key_patterns = compile_key_patterns()
        dump_text, store_size = _build_dump(mem, key_patterns)
        queries = [probe["q"] for probe in probes]
        dump_tokens, curate_tokens, reduction, estimator_label = _measure(
            mem, dump_text, queries, args.budget
        )

        correctness: dict[str, str] | None = None
        model_used: str | None = None
        if probes:
            scorer = AnswerScorer(args.lm_url, args.model, args.timeout)
            try:
                correctness = score_probes(
                    scorer,
                    probes,
                    dump=dump_text,
                    curate_fn=lambda q: mem.curate(q, token_budget=args.budget).content,
                )
                model_used = args.model
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                MemoryStoreError,
                json.JSONDecodeError,
            ) as exc:
                emit(
                    f"model scoring failed ({redact_text(str(exc))}); is the endpoint "
                    "running? correctness skipped -- token efficiency still recorded.",
                    stream=sys.stderr,
                )

        run_row = EvalRun(
            timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
            store_path=str(args.store),
            agent_id=args.agent_id,
            store_size=store_size,
            budget=args.budget,
            dump_tokens=dump_tokens,
            curate_tokens=curate_tokens,
            reduction=reduction,
            estimator=estimator_label,
            embedding=args.embedding,
            correctness=correctness,
            model=model_used,
            tulving_version=__version__,
        )
        append_run(history_path, run_row)

        if not args.no_report:
            _write_report(history_path, html_path)
    except TulvingError as exc:
        emit(redact_text(str(exc)), stream=sys.stderr)
        return EXIT_ERROR
    finally:
        mem.close()

    if reduction >= 2:
        emit(
            f"Verdict: curate returns the relevant slice in ~{reduction}x fewer tokens than "
            "a full dump, and the gap widens as your store grows."
        )
    else:
        emit(
            f"Verdict: store still small ({store_size} memories) so curate's fixed overhead "
            f"({curate_tokens} tok) exceeds the tiny dump ({dump_tokens} tok). The win appears "
            "as the store grows past ~20 memories -- keep logging runs to watch it cross over."
        )
    return EXIT_OK


def _open_store(path: str, agent_id: str, embedding: str) -> Memory:
    """Open the real store READ-ONLY (never writes, safe alongside a live writer).

    Args:
        path: The store directory.
        agent_id: Identity bound at open (D7).
        embedding: ``"none"`` (default; no torch), ``"local"``, or ``"openai"`` --
            matching the user's MCP server so ``curate()`` measures the same way.

    Returns:
        A started, read-only :class:`Memory` handle.
    """
    embedder: EmbeddingAdapter | None = None
    if embedding == "local":
        from tulving import LocalEmbedder

        embedder = LocalEmbedder()
    elif embedding == "openai":
        from tulving import OpenAIEmbedder

        embedder = OpenAIEmbedder()
    mem = Memory(path, agent_id=agent_id, embedding_adapter=embedder, read_only=True)
    mem.startup()  # idempotent; defers all write tasks on a read-only handle
    return mem


def _build_dump(mem: Memory, key_patterns: Sequence[Pattern[str]]) -> tuple[str, int]:
    """The whole live store as one redacted prompt block, plus the live entry count.

    Skips reserved ``session:`` lifecycle markers (internal session-history rows, not
    memories an agent would dump into a prompt). Mirrors the curator's redaction:
    mask the body of any sensitive-keyed entry, then scan the whole block with
    ``redact_text`` (security req #1).
    """
    lines: list[str] = []
    for key in mem.list_keys():
        if key.startswith(SESSION_KEY_PREFIX):
            continue
        entry = mem.get(key)
        if entry is None:
            continue
        body = _REDACTED_BODY if is_sensitive_key(entry.key or "", key_patterns) else entry.content
        lines.append(f"[{entry.type.value}] {entry.key} {body}")
    dump = redact_text("\n".join(lines), key_patterns=key_patterns)
    return dump, len(lines)


def _estimator_label(estimator: TokenEstimator) -> str:
    """``TiktokenEstimator`` -> ``"tiktoken"``; anything else -> ``"heuristic"``."""
    return "tiktoken" if type(estimator).__name__ == "TiktokenEstimator" else "heuristic"


def _measure(
    mem: Memory, dump_text: str, queries: list[str], budget: int
) -> tuple[int, int, float, str]:
    """Token-efficiency measurement: dump tokens, curate tokens, reduction, estimator label.

    Both sides of the ratio use the SAME estimator (``resolve_estimator()``, mandated
    reuse) so the ratio is honest. ``curate_tokens == 0`` never divides by zero.
    """
    estimator = resolve_estimator()
    dump_tokens = estimator.estimate(dump_text)
    effective_queries = queries or [DEFAULT_GENERIC_QUERY]
    totals = [mem.curate(q, token_budget=budget).token_count for q in effective_queries]
    curate_tokens = round(sum(totals) / len(totals))
    reduction = round(dump_tokens / max(curate_tokens, 1), 1)
    return dump_tokens, curate_tokens, reduction, _estimator_label(estimator)


def load_history(path: Path) -> list[dict[str, Any]]:
    """Load the history log's ``runs[]``, tolerant of a legacy bare-list file.

    Args:
        path: The history log path.

    Returns:
        The ``runs`` list (``[]`` when the file is absent, corrupt, or unrecognized).

    Raises:
        MemoryStoreError: The file's ``schema_version`` is newer than this Tulving
            supports -- refuses to risk corrupting a newer log.
    """
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        emit(f"could not read {path}; starting a fresh log.", stream=sys.stderr)
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        schema_version = raw.get("schema_version")
        if isinstance(schema_version, int) and schema_version > HISTORY_SCHEMA_VERSION:
            raise MemoryStoreError(
                f"{path} has schema_version {schema_version}, newer than this Tulving "
                f"supports ({HISTORY_SCHEMA_VERSION}); refusing to append -- upgrade tulving"
            )
        runs = raw.get("runs")
        if isinstance(runs, list):
            return runs
    emit(f"{path} has an unexpected format; starting a fresh log.", stream=sys.stderr)
    return []


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via temp-file + ``os.replace`` (DB LOW fix).

    Avoids a torn/partial file if the process is interrupted mid-write (a plain
    ``write_text`` truncates in place first); the temp file lives in the SAME
    directory so ``os.replace`` is an atomic rename on both POSIX and Windows.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_name)
        raise


def append_run(path: Path, run: EvalRun) -> None:
    """Append one run to the versioned history log (creating it if absent)."""
    runs = load_history(path)
    runs.append(run.to_dict())
    payload = {"schema_version": HISTORY_SCHEMA_VERSION, "runs": runs}
    _atomic_write_text(path, json.dumps(payload, indent=2))


def _write_report(history_path: Path, out_path: Path) -> int:
    """Render the history log to ``out_path``; returns the count of runs rendered."""
    runs = load_history(history_path)
    if not runs:
        return 0
    _atomic_write_text(out_path, eval_report.render(runs))
    return len(runs)

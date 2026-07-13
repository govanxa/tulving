"""Tulving CLI -- `tulving maintenance`: user-facing housekeeping for a store that
archives-and-never-deletes.

The store archives and never deletes (D1 supersede, D2 eviction, `forget`, session
abandonment). ``curate()`` stays token-bounded no matter how large the store grows, so
growth is a disk-only concern. This module surfaces the reclaim engine as four
subcommands: ``inspect`` (counts/sizes/meta), ``purge`` (wrap
``MemoryStore.purge_archived`` with a CLI-layer safety floor), ``vacuum`` (SQLite
``VACUUM``), ``export`` (wrap ``export_json``, default-on redaction).

Mutating ops (``purge``, ``vacuum``) open a WRITER ``Memory`` handle and take the
advisory lock (ADR-015); read ops (``inspect``, ``export``) open a READ-ONLY handle and
never touch. This module never constructs an embedder or LLM (torch-free, mcp-free). A
writer open also runs ``Memory``'s normal startup housekeeping (decay eviction, session
abandonment checks) before the requested op -- documented on ``purge``/``vacuum``'s
``--help`` text, not silent.

Writer-open failures (``purge``/``vacuum``) are deliberately NOT caught locally: a
nonexistent store is already refused by an earlier read-only precheck, so the only
realistic failure left is the advisory-lock refusal (or a genuine backend failure) --
both propagate uncaught to the dispatcher's ``guard_main``, the single classifier that
tells a lock refusal (``EXIT_LOCKED``) apart from any other ``StorageError``
(``EXIT_ERROR``). A local ``except StorageError: return EXIT_LOCKED`` here would
mismap non-lock failures (migration errors, disk full, a NUL path) to ``EXIT_LOCKED``
too -- exactly the review MAJOR this module used to have.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any, Final

from tulving import ArchiveReason, Memory, StoreStats
from tulving.cli._util import EXIT_ERROR, EXIT_OK, confirm, emit
from tulving.exceptions import MemoryStoreError, TulvingError
from tulving.security import redact_text

NAME: Final[str] = "maintenance"

_REASON_CHOICES: Final[tuple[str, ...]] = (
    "evicted",
    "superseded",
    "forgotten",
    "abandoned",
    "summarized",
)
_DURATION_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)([smhdw])$")
_DURATION_UNITS: Final[dict[str, int]] = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86400,
    "w": 604800,
}

_SAFETY_FLOOR_MESSAGE: Final[str] = (
    "refusing to purge every archived row without a safety floor -- pass "
    "--older-than DURATION (e.g. 30d) or --all to confirm a full purge."
)
_AGENT_ID_HELP: Final[str] = "agent id bound at open (default: default)"


# ---------------------------------------------------------------------------
# dispatcher registration surface (the D-v02-6 contract)
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """Attach the ``maintenance`` command tree to the top-level dispatcher.

    Import-safe and side-effect-free (no Memory construction, no I/O). Each leaf
    subparser calls ``set_defaults(_handler=<callable(args) -> int>)``; the
    dispatcher invokes ``args._handler(args)`` and uses the returned int as the
    process exit code (D-v02-6).
    """
    parser = subparsers.add_parser(
        NAME,
        help="housekeeping for a store that archives-and-never-deletes",
        description=(
            "Inspect disk usage, reclaim archived rows, shrink the SQLite file, or "
            "take a redacted backup. Mutating ops take the advisory writer lock "
            "(ADR-015); read ops never touch the store (D2/D3)."
        ),
    )
    sub = parser.add_subparsers(dest="maintenance_command", required=True)
    _register_inspect(sub)
    _register_purge(sub)
    _register_vacuum(sub)
    _register_export(sub)


def _register_inspect(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = sub.add_parser("inspect", help="counts, sizes, and meta for a store")
    parser.add_argument("--store", required=True, help="path to the memory store")
    parser.add_argument("--agent-id", default="default", help=_AGENT_ID_HELP)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.set_defaults(_handler=_cmd_inspect)


def _register_purge(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = sub.add_parser(
        "purge",
        help="hard-delete archived rows (reason-aware; SUMMARIZED protected by default)",
        description=(
            "Hard-delete archived rows, reason-aware (SUMMARIZED protected unless "
            "--include-summarized or --reason summarized is given). --dry-run needs "
            "no --all/--older-than -- it deletes nothing, so the safety floor does not "
            "apply to it; note its preview count can differ from what a later, "
            "unconfirmed purge actually deletes if another writer archives/purges rows "
            "in between (TOCTOU -- the preview is a snapshot, not a lock). The writer "
            "open also runs Memory's normal startup housekeeping (decay eviction, "
            "session abandonment checks) before purging."
        ),
    )
    parser.add_argument("--store", required=True, help="path to the memory store")
    parser.add_argument("--agent-id", default="default", help=_AGENT_ID_HELP)
    parser.add_argument(
        "--older-than",
        type=_parse_duration,
        default=None,
        metavar="DURATION",
        help="only purge rows archived before this age, e.g. 30d/12h/90m/2w/45s",
    )
    parser.add_argument("--all", action="store_true", help="force a full purge with no age cutoff")
    parser.add_argument(
        "--reason",
        action="append",
        dest="reasons",
        choices=_REASON_CHOICES,
        default=None,
        help="only purge this archive reason (repeatable); default: everything but summarized",
    )
    parser.add_argument(
        "--include-summarized",
        action="store_true",
        help="also purge SUMMARIZED sources (ADR-009: normally protected)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the would-delete count; delete nothing (no --all/--older-than required)",
    )
    parser.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt (non-interactive)"
    )
    parser.set_defaults(_handler=_cmd_purge)


def _register_vacuum(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = sub.add_parser(
        "vacuum",
        help="reclaim SQLite file space (VACUUM)",
        description=(
            "Reclaim SQLite file space with a full VACUUM. Needs roughly 2x the "
            "database size in temporary space and holds the write lock for the whole "
            "duration (ADR-015) -- expect it to be slow on OneDrive/network-synced "
            "paths. The writer open also runs Memory's normal startup housekeeping "
            "(decay eviction, session abandonment checks) before the vacuum itself."
        ),
    )
    parser.add_argument("--store", required=True, help="path to the memory store")
    parser.add_argument("--agent-id", default="default", help=_AGENT_ID_HELP)
    parser.add_argument(
        "--yes", action="store_true", help="skip the confirmation prompt (non-interactive)"
    )
    parser.set_defaults(_handler=_cmd_vacuum)


def _register_export(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = sub.add_parser("export", help="write a redacted JSON backup")
    parser.add_argument("--store", required=True, help="path to the memory store")
    parser.add_argument("--agent-id", default="default", help=_AGENT_ID_HELP)
    parser.add_argument("--out", required=True, help="destination file (leaf-validated)")
    parser.add_argument("--include-archived", action="store_true", help="include archived entries")
    parser.add_argument("--include-embeddings", action="store_true", help="emit per-entry vectors")
    parser.add_argument(
        "--include-sensitive",
        action="store_true",
        help="opt OUT of redaction -- writes plaintext secrets to disk",
    )
    parser.set_defaults(_handler=_cmd_export)


# ---------------------------------------------------------------------------
# handlers
# ---------------------------------------------------------------------------


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Print counts/sizes/meta for a store; read-only, never touches (D2/D3)."""
    store_path = _resolve_store_path(args)
    try:
        mem = _open_memory(store_path, read_only=True, agent_id=args.agent_id)
    except MemoryStoreError as exc:
        emit(f"no memory store found: {redact_text(str(exc))}", stream=sys.stderr)
        return EXIT_ERROR
    try:
        stats = mem.store_stats()
    except TulvingError as exc:
        emit(redact_text(str(exc)), stream=sys.stderr)
        return EXIT_ERROR
    finally:
        mem.close()
    if args.json:
        emit(json.dumps(_stats_payload(stats), indent=2))
    else:
        emit(_format_stats(stats))
    return EXIT_OK


def _cmd_purge(args: argparse.Namespace) -> int:
    """Hard-delete archived rows; CLI-layer safety floor + confirmation."""
    store_path = _resolve_store_path(args)
    reasons = _reasons_from_args(args)
    older_than = args.older_than

    if not args.dry_run and older_than is None and not args.all:
        emit(_SAFETY_FLOOR_MESSAGE, stream=sys.stderr)
        return EXIT_ERROR

    try:
        reader = _open_memory(store_path, read_only=True, agent_id=args.agent_id)
    except MemoryStoreError as exc:
        emit(f"no memory store found: {redact_text(str(exc))}", stream=sys.stderr)
        return EXIT_ERROR
    try:
        would_delete = reader.purgeable_count(reasons=reasons, older_than=older_than)
    finally:
        reader.close()

    if args.dry_run:
        emit(f"would delete {would_delete} archived row(s) (dry run; nothing deleted).")
        return EXIT_OK
    if would_delete == 0:
        emit("nothing to purge.")
        return EXIT_OK
    if not confirm(f"purge {would_delete} archived row(s)?", assume_yes=args.yes):
        emit("purge declined; nothing deleted.", stream=sys.stderr)
        return EXIT_ERROR

    # The read-only preview above already proved the store exists; a
    # StorageError from THIS writer open (lock refusal, or a genuine backend
    # failure) is intentionally left uncaught -- see the module docstring for
    # why guard_main, not this handler, classifies it.
    writer = _open_memory(store_path, read_only=False, agent_id=args.agent_id)
    try:
        deleted = writer.purge_archived(reasons=reasons, older_than=older_than)
    except TulvingError as exc:
        emit(redact_text(str(exc)), stream=sys.stderr)
        return EXIT_ERROR
    finally:
        writer.close()
    emit(
        f"purged {deleted} archived row(s). Run 'tulving maintenance vacuum' to reclaim disk space."
    )
    return EXIT_OK


def _cmd_vacuum(args: argparse.Namespace) -> int:
    """Reclaim SQLite file space; confirmation-gated, writer-only.

    A read-only existence precheck runs first (mirrors purge's preview): without
    it, a typo'd/nonexistent --store would silently mkdir a brand-new empty store
    on the writer open below and "successfully" vacuum nothing at EXIT_OK.
    """
    store_path = _resolve_store_path(args)
    try:
        precheck = _open_memory(store_path, read_only=True, agent_id=args.agent_id)
    except MemoryStoreError as exc:
        emit(f"no memory store found: {redact_text(str(exc))}", stream=sys.stderr)
        return EXIT_ERROR
    precheck.close()

    if not confirm(f"vacuum store at {store_path}?", assume_yes=args.yes):
        emit("vacuum declined; nothing changed.", stream=sys.stderr)
        return EXIT_ERROR

    # The precheck above already proved the store exists; see _cmd_purge's
    # matching comment (and the module docstring) for why a StorageError from
    # THIS writer open is left uncaught for guard_main to classify.
    writer = _open_memory(store_path, read_only=False, agent_id=args.agent_id)
    try:
        result = writer.vacuum()
    except TulvingError as exc:
        emit(redact_text(str(exc)), stream=sys.stderr)
        return EXIT_ERROR
    finally:
        writer.close()
    emit(
        f"vacuum reclaimed {_human_bytes(result.bytes_reclaimed)} "
        f"({_human_bytes(result.bytes_before)} -> {_human_bytes(result.bytes_after)})."
    )
    return EXIT_OK


def _cmd_export(args: argparse.Namespace) -> int:
    """Write a redacted JSON backup; read-only, never blocks on the writer lock."""
    store_path = _resolve_store_path(args)
    try:
        mem = _open_memory(store_path, read_only=True, agent_id=args.agent_id)
    except MemoryStoreError as exc:
        emit(f"no memory store found: {redact_text(str(exc))}", stream=sys.stderr)
        return EXIT_ERROR
    try:
        mem.export_json(
            args.out,
            include_archived=args.include_archived,
            include_embeddings=args.include_embeddings,
            include_sensitive=args.include_sensitive,
            allowed_root=Path.cwd(),
        )
    except TulvingError as exc:
        # SecurityError (bad path/leaf) IS a TulvingError -- one branch,
        # same EXIT_ERROR mapping either way; no separate clause needed.
        emit(redact_text(str(exc)), stream=sys.stderr)
        return EXIT_ERROR
    finally:
        mem.close()
    if args.include_sensitive:
        emit(
            "WARNING: --include-sensitive wrote UNREDACTED plaintext to disk "
            "(no encryption at rest, ADR-010) -- protect this file accordingly.",
            stream=sys.stderr,
        )
    emit(f"exported to {args.out} (redacted={not args.include_sensitive}).")
    return EXIT_OK


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _parse_duration(text: str) -> timedelta:
    """Parse ``<int><unit>`` (unit in s/m/h/d/w) into a ``timedelta``.

    Used as an argparse ``type=`` -- a raised ``ValueError`` becomes an argparse
    usage error (``EXIT_USAGE``), never a handler-level failure. This is also why
    an out-of-range amount must raise ``ValueError`` too, not let ``timedelta``'s
    own ``OverflowError`` escape: ``OverflowError`` is neither caught by argparse's
    ``type=`` machinery nor a ``TulvingError`` guard_main recognizes, so it would
    otherwise surface as a raw, unredacted traceback under an undefined exit code
    (security-flagged regression).

    Raises:
        ValueError: On any string not matching ``<non-negative int><unit>``, or on
            an amount so large the equivalent ``timedelta`` would overflow.
    """
    match = _DURATION_RE.match(text)
    if not match:
        raise ValueError(
            f"invalid duration {text!r}; expected <int><unit> where unit is one of "
            "s/m/h/d/w (e.g. 30d, 12h, 90m, 2w, 45s)"
        )
    amount = int(match.group(1))
    unit = match.group(2)
    try:
        return timedelta(seconds=amount * _DURATION_UNITS[unit])
    except OverflowError as exc:
        raise ValueError(
            f"duration {text!r} is too large (max is about {timedelta.max.days} days)"
        ) from exc


def _reasons_from_args(args: argparse.Namespace) -> list[ArchiveReason] | None:
    """``--reason`` (verbatim) else ``--include-summarized`` (all five) else None."""
    reasons = getattr(args, "reasons", None)
    if reasons:
        return [ArchiveReason(value) for value in reasons]
    if getattr(args, "include_summarized", False):
        return list(ArchiveReason)
    return None


def _open_memory(store: Path, *, read_only: bool, agent_id: str = "default") -> Memory:
    """Open a torch-free, mcp-free ``Memory`` handle (monkeypatch seam in tests).

    Constructs no embedder and no LLM -- maintenance never builds either.
    """
    return Memory(
        store, agent_id=agent_id, embedding_adapter=None, llm_adapter=None, read_only=read_only
    )


def _resolve_store_path(args: argparse.Namespace) -> Path:
    """``args.store`` -> an expanded, resolved absolute path."""
    return Path(args.store).expanduser().resolve()


def _human_bytes(n: int) -> str:
    """ASCII-only ``B``/``KB``/``MB``/``GB``/``TB`` byte formatting."""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _stats_payload(stats: StoreStats) -> dict[str, Any]:
    """``StoreStats`` -> a JSON-safe plain dict (``--json`` output)."""
    return {
        "live_count": stats.live_count,
        "archived_count": stats.archived_count,
        "archived_by_reason": dict(stats.archived_by_reason),
        "total_count": stats.total_count,
        "db_bytes": stats.db_bytes,
        "wal_bytes": stats.wal_bytes,
        "index_bytes": stats.index_bytes,
        "total_bytes": stats.total_bytes,
        "schema_version": stats.schema_version,
        "embedding_model_id": stats.embedding_model_id,
        "embedding_dimension": stats.embedding_dimension,
        "distance_metric": stats.distance_metric,
    }


def _format_stats(stats: StoreStats) -> str:
    """ASCII table rendering of a ``StoreStats`` snapshot (default ``inspect`` output)."""
    lines = [
        f"live: {stats.live_count}",
        f"archived: {stats.archived_count}",
    ]
    for reason in sorted(stats.archived_by_reason):
        lines.append(f"  {reason}: {stats.archived_by_reason[reason]}")
    lines.append(f"total: {stats.total_count}")
    lines.append(f"db size: {_human_bytes(stats.db_bytes)}")
    lines.append(f"wal size: {_human_bytes(stats.wal_bytes)}")
    lines.append(f"index size: {_human_bytes(stats.index_bytes)}")
    lines.append(f"total disk: {_human_bytes(stats.total_bytes)}")
    lines.append(f"schema version: {stats.schema_version}")
    embedding = stats.embedding_model_id if stats.embedding_model_id else "none configured"
    lines.append(f"embedding: {embedding}")
    if stats.embedding_dimension is not None:
        lines.append(f"dimension: {stats.embedding_dimension}")
    if stats.distance_metric is not None:
        lines.append(f"distance metric: {stats.distance_metric}")
    return "\n".join(lines)


__all__ = ["NAME", "register"]

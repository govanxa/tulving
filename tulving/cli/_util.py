"""Tulving CLI — shared helpers (D-v02-6, ruling 4). Stdlib only.

This is a **public contract surface** within the CLI package: signatures are fixed and
consumed by every subcommand module (``eval_cmd``, and the future ``maintenance``).
Imports only stdlib + ``tulving.security.redact_text`` + ``tulving.exceptions`` — no
``Memory``, no embedders, no I/O beyond the console/streams.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from collections.abc import Callable
from typing import Final, TextIO

from tulving.exceptions import StorageError, TulvingError
from tulving.security import redact_text

EXIT_OK: Final[int] = 0
EXIT_ERROR: Final[int] = 1
EXIT_USAGE: Final[int] = 2  # argparse OWNS this code; never returned manually
EXIT_LOCKED: Final[int] = 3

# Substring of the writer-lock refusal message raised by Memory's
# ``_AdvisoryLock._refusal_message`` (memory.py) — used to distinguish a lock
# refusal StorageError from any other StorageError (e.g. migration failure).
_LOCK_REFUSAL_MARKER: Final[str] = "is writing to"


def emit(text: str, *, stream: TextIO | None = None) -> None:
    """ASCII-safe console write (cp1252 Windows consoles).

    Encodes ``text.encode("ascii", "replace").decode("ascii")`` before writing, so a
    store path / model name with non-cp1252 characters can never raise
    ``UnicodeEncodeError``. Appends a newline. Never raises.

    Args:
        text: The text to print.
        stream: Output stream; ``None`` (default) resolves to the CURRENT
            ``sys.stdout`` at call time -- never a reference captured once at
            import time, so output redirection and test capture behave correctly.
    """
    safe = text.encode("ascii", "replace").decode("ascii")
    target = stream if stream is not None else sys.stdout
    with contextlib.suppress(Exception):  # pragma: no branch - console writes never fail
        print(safe, file=target)


def confirm(prompt: str, *, assume_yes: bool = False) -> bool:
    """y/N confirmation for destructive actions (maintenance ``--purge``/``--vacuum``).

    - ``assume_yes=True`` (wired from a subcommand's ``--yes`` flag) -> ``True``, no prompt.
    - Otherwise, when stdin is a TTY, prompts ``f"{prompt} [y/N] "``; only an explicit
      'y'/'yes' (case-insensitive) returns ``True``; default is ``False``.
    - **Non-interactive (stdin not a TTY) and no ``--yes`` -> returns ``False`` (REFUSES).**
      Destructive ops never proceed unattended without ``--yes``.

    Prompt is written via :func:`emit` (ASCII-safe). Never raises.

    Args:
        prompt: The question to ask (without the ``[y/N]`` suffix).
        assume_yes: Skip the prompt and return ``True`` unconditionally.

    Returns:
        Whether the action was confirmed.
    """
    if assume_yes:
        return True
    try:
        interactive = sys.stdin.isatty()
    except Exception:  # stdin without a working isatty() is exotic but not fatal
        interactive = False
    if not interactive:
        return False
    emit(f"{prompt} [y/N] ")
    try:
        response = input().strip().lower()
    except (EOFError, OSError):
        return False
    return response in ("y", "yes")


def _is_lock_refusal(message: str) -> bool:
    """Whether a ``StorageError`` message is the single-writer lock refusal (ADR-015)."""
    return _LOCK_REFUSAL_MARKER in message


def guard_main(handler: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """Invoke a subcommand handler under the redaction-guarded top-level guard.

    Returns the handler's int on success. Translates:
      - ``TulvingError`` -> ``emit(redact_text(str(exc)), stream=sys.stderr)`` +
        ``EXIT_LOCKED`` when it is a lock-refusal ``StorageError``, else ``EXIT_ERROR``;
      - ``KeyboardInterrupt`` -> 130.

    Unexpected non-Tulving exceptions are NOT swallowed (they surface as tracebacks — a
    bug, not a user error). ``redact_text`` guarantees no secret leaks into the message
    (security req #1).

    Args:
        handler: The subcommand's ``run`` (or a stub), called with ``args``.
        args: The parsed argparse namespace.

    Returns:
        The process exit code.
    """
    try:
        return handler(args)
    except TulvingError as exc:
        is_lock_refusal = isinstance(exc, StorageError) and _is_lock_refusal(str(exc))
        code = EXIT_LOCKED if is_lock_refusal else EXIT_ERROR
        emit(redact_text(str(exc)), stream=sys.stderr)
        return code
    except KeyboardInterrupt:
        return 130

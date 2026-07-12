"""Tulving CLI dispatcher — the `tulving` console-script entry point.

Owns the subcommand registry and routes to each subcommand's ``run(args)`` via the
``_handler`` default (D-v02-6, ruling 1). Stdlib only; heavy/optional imports live
inside subcommand ``run()`` bodies so ``tulving --help`` stays cheap (D8).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Final

from tulving import __version__
from tulving.cli._util import (
    EXIT_ERROR,
    EXIT_LOCKED,
    EXIT_OK,
    EXIT_USAGE,
    confirm,
    emit,
    guard_main,
)

logger = logging.getLogger(__name__)

PROG: Final[str] = "tulving"

# The one shared edit surface (blueprint-eval-cli): adding a subcommand = create the
# module + append one string here.
_SUBCOMMAND_MODULES: tuple[str, ...] = (
    "tulving.cli.eval_cmd",  # owned by blueprint-eval-cli
    "tulving.cli.maintenance",  # owned by the parallel maintenance architect
)

__all__ = [
    "EXIT_ERROR",
    "EXIT_LOCKED",
    "EXIT_OK",
    "EXIT_USAGE",
    "PROG",
    "confirm",
    "emit",
    "guard_main",
    "main",
]


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level parser and register every importable subcommand module.

    A subcommand module that fails to import (e.g. a sibling not yet merged) is
    skipped with a debug log — the dispatcher never hard-fails because a sibling
    subcommand is absent.
    """
    parser = argparse.ArgumentParser(
        prog=PROG,
        description="Tulving -- the context-budget engine for AI agents.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for module_name in _SUBCOMMAND_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            logger.debug(
                "subcommand module %s is not available; skipping", module_name, exc_info=True
            )
            continue
        module.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse args and dispatch to the selected subcommand's handler.

    Argparse owns usage errors (``--help``, ``--version``, missing/invalid
    subcommand) via ``SystemExit`` — translated here into an int exit code so this
    function never raises ``SystemExit`` itself.

    Args:
        argv: Argument vector (``None`` uses ``sys.argv[1:]``).

    Returns:
        The process exit code.
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        # ruling 1: every registered subcommand's register() sets this default.
        # (argparse.Namespace's typeshed stub declares a catch-all __getattr__ -> Any,
        # so this dynamic attribute needs no type: ignore.)
        handler: Callable[[argparse.Namespace], int] = args._handler
        return guard_main(handler, args)
    except SystemExit as exc:
        # Covers argparse's own usage errors/--help/--version AND a subcommand
        # calling parser.error(...) from within run() (e.g. eval_cmd's missing
        # --store check) -- neither path may escape main() as a raised SystemExit.
        code = exc.code
        if code is None:
            return EXIT_OK
        if isinstance(code, int):
            return code
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised via the console script
    sys.exit(main())

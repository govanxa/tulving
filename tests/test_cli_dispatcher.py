"""Tests for the `tulving` console-script dispatcher (tulving/cli/__init__.py + _util.py).

Written BEFORE implementation. Covers the dispatcher contract (D-v02-6): exit codes,
subcommand routing via `_handler`, sibling-import isolation, and the shared `_util`
helpers (`emit`, `confirm`, `guard_main`) other subcommands (maintenance) build against.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Sequence

import pytest

import tulving.cli as cli
from tulving import __version__
from tulving.cli import _util
from tulving.exceptions import MemoryStoreError, StorageError

# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_no_subcommand_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = cli.main([])
        assert code == _util.EXIT_USAGE

    def test_unknown_subcommand_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        code = cli.main(["bogus"])
        assert code == _util.EXIT_USAGE

    def test_eval_missing_store_returns_exit_usage_via_parser_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """REVIEW MEDIUM regression: eval_cmd.run() calls `_parser.error(...)` for a
        missing --store (argparse OWNS exit 2), which raises SystemExit *from inside*
        the handler -- i.e. inside guard_main's try, not argparse's own parse_args.
        main() must still translate that to EXIT_USAGE (2), not let it escape or
        misclassify it as EXIT_ERROR/EXIT_LOCKED."""
        code = cli.main(["eval"])
        assert code == _util.EXIT_USAGE
        assert "--store" in capsys.readouterr().err

    def test_main_does_not_swallow_unexpected_exceptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REVIEW MEDIUM regression: the except SystemExit wrap around parse+dispatch
        must not accidentally widen into swallowing a genuine bug. A handler raising
        a non-Tulving, non-SystemExit, non-KeyboardInterrupt exception must still
        propagate out of main() as a real traceback."""
        module_name = "tulving.cli._stub_raising_subcommand"

        class _StubModule:
            NAME = "stubraise"

            @staticmethod
            def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
                def _boom(_args: argparse.Namespace) -> int:
                    raise ValueError("a real bug, not a user error")

                parser = subparsers.add_parser("stubraise")
                parser.set_defaults(_handler=_boom)

        import sys

        monkeypatch.setitem(sys.modules, module_name, _StubModule)
        monkeypatch.setattr(cli, "_SUBCOMMAND_MODULES", (module_name,))
        with pytest.raises(ValueError, match="a real bug, not a user error"):
            cli.main(["stubraise"])

    def test_missing_maintenance_module_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cli,
            "_SUBCOMMAND_MODULES",
            ("tulving.cli.eval_cmd", "tulving.cli._nonexistent_subcommand"),
        )
        parser = cli._build_parser()
        # Still registers eval; never hard-fails on the absent sibling.
        help_text = parser.format_help()
        assert "eval" in help_text

    def test_guard_main_tulvingerror_maps_to_exit_error_and_redacts(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise MemoryStoreError("secret=abcdef12345")

        code = _util.guard_main(handler, argparse.Namespace())
        assert code == _util.EXIT_ERROR
        captured = capsys.readouterr()
        assert "[REDACTED]" in captured.err
        assert "abcdef12345" not in captured.err

    def test_guard_main_lock_refusal_maps_to_exit_locked(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise StorageError(
                "another Tulving process (pid 123) is writing to C:\\store — "
                "close the other session, or open with Memory(..., read_only=True)"
            )

        code = _util.guard_main(handler, argparse.Namespace())
        assert code == _util.EXIT_LOCKED

    def test_guard_main_other_storageerror_maps_to_exit_error(self) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise StorageError("migration failed: corrupt schema")

        code = _util.guard_main(handler, argparse.Namespace())
        assert code == _util.EXIT_ERROR

    def test_guard_main_keyboardinterrupt_returns_130(self) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise KeyboardInterrupt

        code = _util.guard_main(handler, argparse.Namespace())
        assert code == 130

    def test_guard_main_unexpected_exception_propagates(self) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise ValueError("not a tulving error")

        with pytest.raises(ValueError, match="not a tulving error"):
            _util.guard_main(handler, argparse.Namespace())


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_confirm_non_interactive_refuses_without_yes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: False)
        assert _util.confirm("proceed?") is False

    def test_confirm_assume_yes_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: False)
        assert _util.confirm("proceed?", assume_yes=True) is True

    def test_confirm_exotic_stdin_isatty_raises_refuses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """100% coverage requirement (security-adjacent module): stdin without a
        working isatty() (e.g. some wrapped/piped stream) must degrade to
        non-interactive refusal, never raise."""

        def _raising_isatty() -> bool:
            raise RuntimeError("exotic stdin: no isatty support")

        monkeypatch.setattr(_util.sys.stdin, "isatty", _raising_isatty)
        assert _util.confirm("proceed?") is False

    def test_confirm_interactive_yes_returns_true(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "y")
        assert _util.confirm("proceed?") is True
        assert "[y/N]" in capsys.readouterr().out

    def test_confirm_interactive_yes_full_word(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "YES")
        assert _util.confirm("proceed?") is True

    def test_confirm_interactive_other_input_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "sure")
        assert _util.confirm("proceed?") is False

    def test_confirm_interactive_eof_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", _raise_eof)
        assert _util.confirm("proceed?") is False

    def test_emit_is_ascii_safe(self) -> None:
        stream = io.StringIO()
        _util.emit("model → done", stream=stream)
        output = stream.getvalue()
        assert "→" not in output
        assert output.strip() != ""

    def test_exit_constants_values(self) -> None:
        assert (_util.EXIT_OK, _util.EXIT_ERROR, _util.EXIT_USAGE, _util.EXIT_LOCKED) == (
            0,
            1,
            2,
            3,
        )


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_version_flag_prints_version_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli.main(["--version"])
        captured = capsys.readouterr()
        assert __version__ in captured.out
        assert code == _util.EXIT_OK

    def test_eval_registered(self) -> None:
        parser = cli._build_parser()
        help_text = parser.format_help()
        assert "eval" in help_text

    def test_run_returns_handler_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module_name = "tulving.cli._stub_subcommand"

        class _StubModule:
            NAME = "stub"

            @staticmethod
            def register(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
                parser = subparsers.add_parser("stub")
                parser.set_defaults(_handler=lambda _args: 7)

        import sys

        monkeypatch.setitem(sys.modules, module_name, _StubModule)
        monkeypatch.setattr(cli, "_SUBCOMMAND_MODULES", (module_name,))
        code = cli.main(["stub"])
        assert code == 7

    def test_emit_appends_newline(self) -> None:
        stream = io.StringIO()
        _util.emit("hello", stream=stream)
        assert stream.getvalue() == "hello\n"

    def test_emit_never_raises_on_arbitrary_text(self) -> None:
        stream = io.StringIO()
        _util.emit("", stream=stream)  # must not raise
        _util.emit("plain ascii text", stream=stream)


def _all_option_strings(parser: argparse.ArgumentParser) -> Sequence[str]:
    options: list[str] = []
    for action in parser._actions:
        options.extend(action.option_strings)
    return options


class TestSecurity:
    """emit/guard_main sit on the redaction-guarded egress path."""

    def test_guard_main_redacts_before_emit(self, capsys: pytest.CaptureFixture[str]) -> None:
        def handler(_args: argparse.Namespace) -> int:
            raise MemoryStoreError("api_key=sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567")

        _util.guard_main(handler, argparse.Namespace())
        captured = capsys.readouterr()
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567" not in captured.err

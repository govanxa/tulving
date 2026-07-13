"""Tests for tulving.cli.maintenance -- written BEFORE implementation.

Uses real temp SQLite dirs (tmp_path): VACUUM and the advisory lock need a real file.
Seeds data by opening a normal writer Memory, storing/forgetting/archiving, then
closing it before invoking the CLI. Handlers are exercised via register() into a
throwaway parser, then args._handler(args); confirmations use --yes except where the
prompt itself is under test. No embedder, no network, no [mcp].
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

import tulving.cli as cli_dispatcher
from tulving import Memory
from tulving.cli import _util, maintenance
from tulving.cli._util import EXIT_ERROR, EXIT_LOCKED, EXIT_OK
from tulving.enums import ArchiveReason, MemoryType
from tulving.exceptions import ConfigError, StorageError


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    maintenance.register(subparsers)
    return parser.parse_args(["maintenance", *argv])


def _seed_writer_store(tmp_path: Path, entries: list[tuple[str, str, MemoryType]]) -> Path:
    path = tmp_path / "store"
    writer = Memory(path=str(path), agent_id="default")
    for key, content, mtype in entries:
        writer.store(content, type=mtype, key=key)
    writer.close()
    return path


def _seed_archived(tmp_path: Path, reasons: dict[str, ArchiveReason]) -> Path:
    """One live fact per key, archived with the given reason -- reaches into
    the internal store (mirrors tests/test_memory.py's own pattern) since the
    public API has no verb that mints an arbitrary ArchiveReason directly."""
    path = tmp_path / "store"
    writer = Memory(path=str(path), agent_id="default")
    try:
        for key, reason in reasons.items():
            entry = writer.store(f"content for {key}", type=MemoryType.FACT, key=key)
            writer._store.archive(entry.id, reason)
    finally:
        writer.close()
    return path


def _inspect_stats(path: Path) -> dict[str, Any]:
    mem = Memory(path=str(path), agent_id="checker", read_only=True)
    try:
        return maintenance._stats_payload(mem.store_stats())
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# Registration & argument surface
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_wires_every_leaf_and_performs_no_io(self, tmp_path: Path) -> None:
        before = list(tmp_path.iterdir())
        cases = (
            ("inspect", maintenance._cmd_inspect, []),
            ("purge", maintenance._cmd_purge, []),
            ("vacuum", maintenance._cmd_vacuum, []),
            ("export", maintenance._cmd_export, ["--out", "x"]),
        )
        for name, handler, extra in cases:
            args = _parse_args([name, "--store", str(tmp_path), *extra])
            assert args._handler is handler
        assert list(tmp_path.iterdir()) == before

    def test_maintenance_registered_on_the_dispatcher(self) -> None:
        parser = cli_dispatcher._build_parser()
        help_text = parser.format_help()
        assert "maintenance" in help_text


class TestDurationParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("30d", timedelta(days=30)),
            ("12h", timedelta(hours=12)),
            ("90m", timedelta(minutes=90)),
            ("2w", timedelta(weeks=2)),
            ("45s", timedelta(seconds=45)),
        ],
    )
    def test_valid_durations(self, text: str, expected: timedelta) -> None:
        assert maintenance._parse_duration(text) == expected

    @pytest.mark.parametrize("text", ["30", "5x", "", "-1d", "1.5h"])
    def test_invalid_durations_raise_value_error(self, text: str) -> None:
        with pytest.raises(ValueError):
            maintenance._parse_duration(text)

    def test_overflowing_duration_raises_value_error_not_overflow_error(self) -> None:
        """SECURITY regression: timedelta(seconds=huge) raises OverflowError,
        which is neither caught by argparse's type= machinery (only
        ValueError/TypeError/ArgumentTypeError) nor a TulvingError guard_main
        recognizes -- it must be re-raised as ValueError instead."""
        with pytest.raises(ValueError):
            maintenance._parse_duration("99999999999999999999d")

    def test_bad_older_than_exits_usage_via_argparse(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        code = cli_dispatcher.main(
            ["maintenance", "purge", "--store", str(tmp_path), "--older-than", "bogus", "--yes"]
        )
        assert code == _util.EXIT_USAGE
        assert capsys.readouterr().err  # argparse's usage message, no traceback

    def test_huge_older_than_exits_usage_cleanly_no_traceback(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """SECURITY regression end-to-end: a huge --older-than must produce a
        clean EXIT_USAGE, never an unredacted traceback with an undefined
        exit code."""
        code = cli_dispatcher.main(
            [
                "maintenance",
                "purge",
                "--store",
                str(tmp_path),
                "--older-than",
                "99999999999999999999d",
                "--yes",
            ]
        )
        assert code == _util.EXIT_USAGE
        err = capsys.readouterr().err
        assert err  # argparse's usage message
        assert "Traceback" not in err


class TestReasonMapping:
    def test_explicit_reasons_used_verbatim(self) -> None:
        args = _parse_args(
            ["purge", "--store", "x", "--reason", "evicted", "--reason", "forgotten"]
        )
        assert maintenance._reasons_from_args(args) == [
            ArchiveReason.EVICTED,
            ArchiveReason.FORGOTTEN,
        ]

    def test_no_flags_returns_none(self) -> None:
        args = _parse_args(["purge", "--store", "x"])
        assert maintenance._reasons_from_args(args) is None

    def test_include_summarized_returns_all_five(self) -> None:
        args = _parse_args(["purge", "--store", "x", "--include-summarized"])
        assert maintenance._reasons_from_args(args) == list(ArchiveReason)


class TestAgentId:
    """REVIEW MINOR (ruling: DO IT) -- parity with eval_cmd's --agent-id."""

    @pytest.mark.parametrize(
        ("cmd", "extra"),
        [
            ("inspect", []),
            ("purge", []),
            ("vacuum", []),
            ("export", ["--out", "x"]),
        ],
    )
    def test_defaults_to_default_on_every_subcommand(self, cmd: str, extra: list[str]) -> None:
        args = _parse_args([cmd, "--store", "x", *extra])
        assert args.agent_id == "default"

    def test_flows_through_to_open_memory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        captured: list[str] = []
        original = maintenance._open_memory

        def spy(store: Path, *, read_only: bool, agent_id: str = "default") -> Memory:
            captured.append(agent_id)
            return original(store, read_only=read_only, agent_id=agent_id)

        monkeypatch.setattr(maintenance, "_open_memory", spy)
        args = _parse_args(["inspect", "--store", str(path), "--agent-id", "custom-agent"])
        code = args._handler(args)
        assert code == EXIT_OK
        assert captured == ["custom-agent"]


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


class TestInspect:
    def test_empty_store(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _seed_writer_store(tmp_path, [])
        args = _parse_args(["inspect", "--store", str(path)])
        code = args._handler(args)
        assert code == EXIT_OK
        out = capsys.readouterr().out
        assert "live: 0" in out
        assert "archived: 0" in out
        for reason in ArchiveReason:
            assert f"{reason.value}: 0" in out
        assert "db size:" in out
        assert "embedding: none configured" in out

    def test_archived_heavy_store(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _seed_archived(
            tmp_path,
            {
                "e1": ArchiveReason.EVICTED,
                "e2": ArchiveReason.EVICTED,
                "f1": ArchiveReason.FORGOTTEN,
                "s1": ArchiveReason.SUMMARIZED,
            },
        )
        args = _parse_args(["inspect", "--store", str(path)])
        code = args._handler(args)
        assert code == EXIT_OK
        stats = _inspect_stats(path)
        assert stats["archived_count"] == 4
        assert stats["total_count"] == stats["live_count"] + stats["archived_count"]
        assert sum(stats["archived_by_reason"].values()) == stats["archived_count"]
        assert stats["archived_by_reason"]["evicted"] == 2
        assert stats["archived_by_reason"]["forgotten"] == 1
        assert stats["archived_by_reason"]["summarized"] == 1

    def test_json_output(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        path = _seed_writer_store(tmp_path, [("k1", "one", MemoryType.FACT)])
        args = _parse_args(["inspect", "--store", str(path), "--json"])
        code = args._handler(args)
        assert code == EXIT_OK
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["live_count"] >= 1  # the fact + an internal session marker
        assert payload == maintenance._stats_payload(
            Memory(path=str(path), agent_id="x", read_only=True).store_stats()
        )
        assert out.isascii()

    def test_read_only_no_lock_with_a_concurrent_writer(self, tmp_path: Path) -> None:
        path = tmp_path / "store"
        writer = Memory(path=str(path), agent_id="a")
        try:
            writer.store("x", type=MemoryType.FACT, key="k")
            args = _parse_args(["inspect", "--store", str(path)])
            code = args._handler(args)
            assert code == EXIT_OK
        finally:
            writer.close()

    def test_never_touches_last_accessed_at(self, tmp_path: Path) -> None:
        path = _seed_writer_store(tmp_path, [("k", "content", MemoryType.FACT)])
        args = _parse_args(["inspect", "--store", str(path)])
        assert args._handler(args) == EXIT_OK
        reader = Memory(path=str(path), agent_id="checker", read_only=True)
        try:
            raw = reader._store.get_by_key("k", touch=False)
            assert raw is not None
            assert raw.access_count == 0
        finally:
            reader.close()

    def test_nonexistent_store(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        args = _parse_args(["inspect", "--store", str(missing)])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert not missing.exists()


# ---------------------------------------------------------------------------
# purge (includes mandated regressions)
# ---------------------------------------------------------------------------


class TestPurge:
    def test_refuses_summarized_unless_explicit(self, tmp_path: Path) -> None:
        path = _seed_archived(
            tmp_path, {"sumkey": ArchiveReason.SUMMARIZED, "evickey": ArchiveReason.EVICTED}
        )
        args = _parse_args(["purge", "--store", str(path), "--all", "--yes"])
        assert args._handler(args) == EXIT_OK
        stats = _inspect_stats(path)
        assert stats["archived_by_reason"]["summarized"] == 1
        assert stats["archived_by_reason"]["evicted"] == 0

    def test_explicit_reason_summarized_purges_it(self, tmp_path: Path) -> None:
        path = _seed_archived(tmp_path, {"sumkey": ArchiveReason.SUMMARIZED})
        args = _parse_args(
            ["purge", "--store", str(path), "--all", "--reason", "summarized", "--yes"]
        )
        assert args._handler(args) == EXIT_OK
        assert _inspect_stats(path)["archived_by_reason"]["summarized"] == 0

    def test_include_summarized_purges_everything(self, tmp_path: Path) -> None:
        path = _seed_archived(
            tmp_path, {"sumkey": ArchiveReason.SUMMARIZED, "evickey": ArchiveReason.EVICTED}
        )
        args = _parse_args(
            ["purge", "--store", str(path), "--all", "--include-summarized", "--yes"]
        )
        assert args._handler(args) == EXIT_OK
        assert _inspect_stats(path)["archived_count"] == 0

    def test_safety_floor_refuses_without_older_than_or_all(self, tmp_path: Path) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        args = _parse_args(["purge", "--store", str(path), "--yes"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert _inspect_stats(path)["archived_count"] == 1

    def test_older_than_excludes_too_recent_rows(self, tmp_path: Path) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        args = _parse_args(["purge", "--store", str(path), "--older-than", "365d", "--yes"])
        assert args._handler(args) == EXIT_OK
        assert _inspect_stats(path)["archived_count"] == 1

    def test_older_than_includes_elapsed_rows(self, tmp_path: Path) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        args = _parse_args(["purge", "--store", str(path), "--older-than", "0s", "--yes"])
        assert args._handler(args) == EXIT_OK
        assert _inspect_stats(path)["archived_count"] == 0

    def test_dry_run_deletes_nothing_and_opens_no_writer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        opened_read_only: list[bool] = []
        original = maintenance._open_memory

        def spy(store: Path, *, read_only: bool, agent_id: str = "default") -> Memory:
            opened_read_only.append(read_only)
            return original(store, read_only=read_only, agent_id=agent_id)

        monkeypatch.setattr(maintenance, "_open_memory", spy)
        args = _parse_args(["purge", "--store", str(path), "--all", "--dry-run"])
        code = args._handler(args)
        assert code == EXIT_OK
        assert opened_read_only == [True]  # only the preview handle -- no writer
        assert _inspect_stats(path)["archived_count"] == 1
        assert "would delete 1" in capsys.readouterr().out

    def test_bare_dry_run_previews_delete_all_with_no_floor_required(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """REVIEW MINOR + DB note: --dry-run alone (no --all, no --older-than)
        is intentionally exempt from the safety floor -- it deletes nothing,
        so there is nothing for the floor to guard. Pin that explicitly."""
        path = _seed_archived(tmp_path, {"e1": ArchiveReason.EVICTED, "e2": ArchiveReason.EVICTED})
        args = _parse_args(["purge", "--store", str(path), "--dry-run"])
        code = args._handler(args)
        assert code == EXIT_OK
        assert "would delete 2" in capsys.readouterr().out
        assert _inspect_stats(path)["archived_count"] == 2  # nothing deleted

    def test_help_documents_dry_run_floor_exemption_and_toctou(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        maintenance.register(subparsers)
        with pytest.raises(SystemExit):
            parser.parse_args(["maintenance", "purge", "--help"])
        help_text = capsys.readouterr().out
        assert "--dry-run" in help_text
        assert "safety floor" in help_text
        assert "TOCTOU" in help_text

    def test_interactive_decline_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "n")
        args = _parse_args(["purge", "--store", str(path), "--all"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert _inspect_stats(path)["archived_count"] == 1

    def test_interactive_accept_proceeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "y")
        args = _parse_args(["purge", "--store", str(path), "--all"])
        code = args._handler(args)
        assert code == EXIT_OK
        assert _inspect_stats(path)["archived_count"] == 0

    def test_noninteractive_without_yes_never_calls_input(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: False)

        def _boom() -> str:
            raise AssertionError("input() must never be called non-interactively")

        monkeypatch.setattr("builtins.input", _boom)
        args = _parse_args(["purge", "--store", str(path), "--all"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert _inspect_stats(path)["archived_count"] == 1

    def test_second_writer_refused(self, tmp_path: Path) -> None:
        """The writer-open StorageError is deliberately left uncaught by the
        handler (review MAJOR fix) -- guard_main is the single classifier that
        maps the lock-refusal marker to EXIT_LOCKED, exactly as the real
        dispatcher invokes every subcommand handler."""
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        held = Memory(path=str(path), agent_id="holder")
        try:
            args = _parse_args(["purge", "--store", str(path), "--all", "--yes"])
            code = _util.guard_main(args._handler, args)
            assert code == EXIT_LOCKED
        finally:
            held.close()
        assert _inspect_stats(path)["archived_count"] == 1

    def test_writer_open_non_lock_storage_error_maps_to_exit_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REVIEW MAJOR regression: a non-lock StorageError from the writer
        open (migration failure, corrupt meta, disk full, a NUL path) must
        map to EXIT_ERROR=1, NOT EXIT_LOCKED=3 -- only guard_main's
        ``is writing to`` marker check earns the LOCKED code."""
        path = _seed_archived(tmp_path, {"k": ArchiveReason.EVICTED})
        original = maintenance._open_memory

        def flaky(store: Path, *, read_only: bool, agent_id: str = "default") -> Memory:
            if not read_only:
                raise StorageError("migration failed: corrupt schema")
            return original(store, read_only=read_only, agent_id=agent_id)

        monkeypatch.setattr(maintenance, "_open_memory", flaky)
        args = _parse_args(["purge", "--store", str(path), "--all", "--yes"])
        code = _util.guard_main(args._handler, args)
        assert code == EXIT_ERROR

    def test_nothing_to_purge_is_ok(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "live", MemoryType.FACT)])
        args = _parse_args(["purge", "--store", str(path), "--all", "--yes"])
        code = args._handler(args)
        assert code == EXIT_OK
        assert "nothing to purge" in capsys.readouterr().out

    def test_nonexistent_store(self, tmp_path: Path) -> None:
        missing = tmp_path / "missing"
        args = _parse_args(["purge", "--store", str(missing), "--all", "--yes"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert not missing.exists()


# ---------------------------------------------------------------------------
# vacuum
# ---------------------------------------------------------------------------


class TestVacuum:
    def test_reclaims_after_purge(self, tmp_path: Path) -> None:
        path = tmp_path / "store"
        writer = Memory(path=str(path), agent_id="default")
        big = "x" * 5000
        for i in range(200):
            entry = writer.store(big, type=MemoryType.FACT, key=f"k{i}")
            writer._store.archive(entry.id, ArchiveReason.EVICTED)
        writer.close()

        purge_args = _parse_args(["purge", "--store", str(path), "--all", "--yes"])
        assert purge_args._handler(purge_args) == EXIT_OK

        db_path = path / "tulving.db"
        size_before = db_path.stat().st_size
        vacuum_args = _parse_args(["vacuum", "--store", str(path), "--yes"])
        code = vacuum_args._handler(vacuum_args)
        assert code == EXIT_OK
        size_after = db_path.stat().st_size
        assert size_after < size_before

    def test_needs_confirmation_interactive_decline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr("builtins.input", lambda: "n")
        args = _parse_args(["vacuum", "--store", str(path)])
        code = args._handler(args)
        assert code == EXIT_ERROR

    def test_noninteractive_without_yes_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        monkeypatch.setattr(_util.sys.stdin, "isatty", lambda: False)
        args = _parse_args(["vacuum", "--store", str(path)])
        code = args._handler(args)
        assert code == EXIT_ERROR

    def test_second_writer_refused(self, tmp_path: Path) -> None:
        """See the matching purge note: the writer-open StorageError is left
        uncaught for guard_main -- the real dispatcher's single classifier --
        to map to EXIT_LOCKED."""
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        held = Memory(path=str(path), agent_id="holder")
        try:
            args = _parse_args(["vacuum", "--store", str(path), "--yes"])
            code = _util.guard_main(args._handler, args)
            assert code == EXIT_LOCKED
        finally:
            held.close()

    def test_writer_open_non_lock_storage_error_maps_to_exit_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """REVIEW MAJOR regression, vacuum side: a non-lock StorageError from
        the writer open must map to EXIT_ERROR=1, not EXIT_LOCKED=3."""
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        original = maintenance._open_memory

        def flaky(store: Path, *, read_only: bool, agent_id: str = "default") -> Memory:
            if not read_only:
                raise StorageError("disk full: cannot extend database file")
            return original(store, read_only=read_only, agent_id=agent_id)

        monkeypatch.setattr(maintenance, "_open_memory", flaky)
        args = _parse_args(["vacuum", "--store", str(path), "--yes"])
        code = _util.guard_main(args._handler, args)
        assert code == EXIT_ERROR

    def test_on_tight_store_is_benign(self, tmp_path: Path) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        args = _parse_args(["vacuum", "--store", str(path), "--yes"])
        code = args._handler(args)
        assert code == EXIT_OK

    def test_nonexistent_store(self, tmp_path: Path) -> None:
        """MANDATED (VACUUM EXISTENCE GUARD, found independently by all three
        reviewers): a typo'd/nonexistent --store must be refused BEFORE
        confirming or opening a writer -- not silently mkdir a brand-new
        empty store and report "0 B reclaimed" at EXIT_OK."""
        missing = tmp_path / "missing"
        args = _parse_args(["vacuum", "--store", str(missing), "--yes"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert not missing.exists()
        assert list(tmp_path.iterdir()) == []

    def test_help_documents_vacuum_caveats(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        maintenance.register(subparsers)
        with pytest.raises(SystemExit):
            parser.parse_args(["maintenance", "vacuum", "--help"])
        help_text = capsys.readouterr().out
        assert "2x" in help_text
        assert "write lock" in help_text
        assert "OneDrive" in help_text


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExport:
    def test_redacts_by_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "store"
        writer = Memory(path=str(path), agent_id="default")
        writer.store(
            "token is sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567",
            type=MemoryType.FACT,
            key="note",
        )
        writer.store("hunter2secretvalue", type=MemoryType.FACT, key="api_key")
        writer.close()
        monkeypatch.chdir(tmp_path)
        args = _parse_args(["export", "--store", str(path), "--out", "backup"])
        code = args._handler(args)
        assert code == EXIT_OK
        out_path = tmp_path / "backup.json"
        text = out_path.read_text(encoding="utf-8")
        payload = json.loads(text)
        assert payload["redacted"] is True
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ01234567" not in text
        assert "hunter2secretvalue" not in text

    def test_include_sensitive_opts_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "store"
        writer = Memory(path=str(path), agent_id="default")
        writer.store("hunter2secretvalue", type=MemoryType.FACT, key="api_key")
        writer.close()
        monkeypatch.chdir(tmp_path)
        args = _parse_args(
            ["export", "--store", str(path), "--out", "backup", "--include-sensitive"]
        )
        code = args._handler(args)
        assert code == EXIT_OK
        text = (tmp_path / "backup.json").read_text(encoding="utf-8")
        payload = json.loads(text)
        assert payload["redacted"] is False
        assert "hunter2secretvalue" in text
        assert "WARNING" in capsys.readouterr().err

    def test_path_traversal_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        monkeypatch.chdir(tmp_path)
        args = _parse_args(["export", "--store", str(path), "--out", "../evil"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert not (tmp_path.parent / "evil.json").exists()

    def test_bad_leaf_name_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _seed_writer_store(tmp_path, [("k", "x", MemoryType.FACT)])
        monkeypatch.chdir(tmp_path)
        args = _parse_args(["export", "--store", str(path), "--out", "bad name"])
        code = args._handler(args)
        assert code == EXIT_ERROR

    def test_nonexistent_store(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        missing = tmp_path / "missing"
        monkeypatch.chdir(tmp_path)
        args = _parse_args(["export", "--store", str(missing), "--out", "backup"])
        code = args._handler(args)
        assert code == EXIT_ERROR
        assert not missing.exists()

    def test_is_read_only_never_blocks_on_writer_lock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "store"
        held = Memory(path=str(path), agent_id="holder")
        try:
            held.store("x", type=MemoryType.FACT, key="k")
            monkeypatch.chdir(tmp_path)
            args = _parse_args(["export", "--store", str(path), "--out", "backup"])
            code = args._handler(args)
            assert code == EXIT_OK
        finally:
            held.close()

    def test_flags_flow_through_to_export_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _SpyMemory:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []

            def export_json(self, path: Any, **kwargs: Any) -> None:
                self.calls.append({"path": path, **kwargs})

            def close(self) -> None:
                pass

        spy = _SpyMemory()
        monkeypatch.setattr(
            maintenance, "_open_memory", lambda store, *, read_only, agent_id="default": spy
        )
        args = _parse_args(
            [
                "export",
                "--store",
                str(tmp_path),
                "--out",
                "backup",
                "--include-archived",
                "--include-embeddings",
            ]
        )
        code = args._handler(args)
        assert code == EXIT_OK
        assert len(spy.calls) == 1
        call = spy.calls[0]
        assert call["include_archived"] is True
        assert call["include_embeddings"] is True
        assert call["include_sensitive"] is False


# ---------------------------------------------------------------------------
# human-readable byte formatting
# ---------------------------------------------------------------------------


class TestGenericTulvingErrorMapping:
    """Any *other* TulvingError reaching a handler (not the lock refusal, not
    a nonexistent-store MemoryStoreError, not an export SecurityError) still
    maps to EXIT_ERROR with a redacted message -- exercised via the
    ``_open_memory`` monkeypatch seam with a stub handle."""

    def test_inspect_store_stats_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Stub:
            def store_stats(self) -> Any:
                raise ConfigError("boom")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            maintenance, "_open_memory", lambda store, *, read_only, agent_id="default": _Stub()
        )
        args = _parse_args(["inspect", "--store", str(tmp_path)])
        assert args._handler(args) == EXIT_ERROR

    def test_purge_purge_archived_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Stub:
            def purgeable_count(self, **kwargs: Any) -> int:
                return 1

            def purge_archived(self, **kwargs: Any) -> int:
                raise ConfigError("boom")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            maintenance, "_open_memory", lambda store, *, read_only, agent_id="default": _Stub()
        )
        args = _parse_args(["purge", "--store", str(tmp_path), "--all", "--yes"])
        assert args._handler(args) == EXIT_ERROR

    def test_vacuum_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Stub:
            def vacuum(self) -> Any:
                raise ConfigError("boom")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            maintenance, "_open_memory", lambda store, *, read_only, agent_id="default": _Stub()
        )
        args = _parse_args(["vacuum", "--store", str(tmp_path), "--yes"])
        assert args._handler(args) == EXIT_ERROR

    def test_export_non_security_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Stub:
            def export_json(self, path: Any, **kwargs: Any) -> None:
                raise ConfigError("boom")

            def close(self) -> None:
                pass

        monkeypatch.setattr(
            maintenance, "_open_memory", lambda store, *, read_only, agent_id="default": _Stub()
        )
        args = _parse_args(["export", "--store", str(tmp_path), "--out", "backup"])
        assert args._handler(args) == EXIT_ERROR


class TestHumanBytes:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, "0 B"),
            (512, "512 B"),
            (2048, "2.0 KB"),
            (5 * 1024 * 1024, "5.0 MB"),
            (3 * 1024**3, "3.0 GB"),
            (2 * 1024**4, "2.0 TB"),
        ],
    )
    def test_formats_ascii_only(self, n: int, expected: str) -> None:
        result = maintenance._human_bytes(n)
        assert result == expected
        assert result.isascii()


class TestFormatStats:
    """Direct unit coverage of the ASCII table renderer, including the
    embedder-configured branches (dimension/distance metric present) that a
    torch-free maintenance handle never exercises end-to-end."""

    def test_includes_dimension_and_distance_metric_when_present(self) -> None:
        stats = maintenance.StoreStats(
            live_count=1,
            archived_count=0,
            archived_by_reason={r.value: 0 for r in ArchiveReason},
            total_count=1,
            db_bytes=100,
            wal_bytes=0,
            index_bytes=50,
            total_bytes=150,
            schema_version=2,
            embedding_model_id="hash-embedder",
            embedding_dimension=8,
            distance_metric="cosine",
        )
        text = maintenance._format_stats(stats)
        assert "embedding: hash-embedder" in text
        assert "dimension: 8" in text
        assert "distance metric: cosine" in text

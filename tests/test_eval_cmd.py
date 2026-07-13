"""Tests for tulving.cli.eval_cmd — written BEFORE implementation.

A read-only `Memory` requires an on-disk DB, so read-only tests create a real SQLite
store via a short-lived writer handle first, then reopen it read-only (mirrors
tests/test_memory.py). `tulving.context.curator._default_estimator` is monkeypatched to
a `HeuristicEstimator` so token counts are deterministic and tiktoken-independent.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import tulving.context.curator as curator_mod
from tulving import Memory
from tulving.cli import _util, eval_cmd
from tulving.context.curator import HeuristicEstimator
from tulving.enums import MemoryType
from tulving.exceptions import MemoryStoreError, StorageError
from tulving.security import compile_key_patterns


@pytest.fixture(autouse=True)
def _heuristic_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test in this module gets a deterministic, tiktoken-independent estimator."""
    monkeypatch.setattr(curator_mod, "_default_estimator", HeuristicEstimator())


def _parse_eval_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    eval_cmd.register(subparsers)
    return parser.parse_args(["eval", *argv])


def _seed_writer_store(tmp_path: Path, entries: list[tuple[str, str, MemoryType]]) -> Path:
    path = tmp_path / "store"
    writer = Memory(path=str(path), agent_id="default")
    for key, content, mtype in entries:
        writer.store(content, type=mtype, key=key)
    writer.close()
    return path


def _default_run(**overrides: Any) -> eval_cmd.EvalRun:
    base: dict[str, Any] = {
        "timestamp": "2026-07-01T00:00:00+00:00",
        "store_path": "p",
        "agent_id": "default",
        "store_size": 1,
        "budget": 1500,
        "dump_tokens": 10,
        "curate_tokens": 5,
        "reduction": 2.0,
        "estimator": "heuristic",
        "embedding": "none",
        "correctness": None,
        "model": None,
        "tulving_version": "0.2.0",
    }
    base.update(overrides)
    return eval_cmd.EvalRun(**base)


class _StubCurated:
    def __init__(self, token_count: int) -> None:
        self.token_count = token_count
        self.content = "curated content"


class _StubMemory:
    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def curate(self, _query: str, *, token_budget: int) -> _StubCurated:
        return _StubCurated(self._token_count)


class _StubEntry:
    """Duck-typed stand-in for MemoryEntry (only .key/.content/.type are used)."""

    def __init__(self, key: str, content: str, mtype: MemoryType) -> None:
        self.key = key
        self.content = content
        self.type = mtype


class _StubDumpMemory:
    """Duck-typed stand-in for Memory exposing only what `_build_dump` needs."""

    def __init__(self, entries: dict[str, _StubEntry]) -> None:
        self._entries = entries

    def list_keys(self) -> list[str]:
        return list(self._entries)

    def get(self, key: str) -> _StubEntry | None:
        return self._entries.get(key)


def _run_eval_with_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]],
    *,
    extra_args: list[str] | None = None,
) -> tuple[int, Path, Path]:
    """Drive eval_cmd.run() end-to-end with a stubbed AnswerScorer transport."""
    store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
    probes_path = tmp_path / "probes.json"
    probes_path.write_text(json.dumps([{"q": "What is it?", "expect": "hello"}]), encoding="utf-8")
    history = tmp_path / "h.json"
    report = tmp_path / "r.html"

    import tulving.cli.eval_scoring as eval_scoring_mod

    original_init = eval_scoring_mod.AnswerScorer.__init__

    def patched_init(
        self: Any, url: str, model: str, timeout: int, _transport: Any = transport
    ) -> None:
        original_init(self, url, model, timeout, transport)

    monkeypatch.setattr(eval_scoring_mod.AnswerScorer, "__init__", patched_init)
    args = _parse_eval_args(
        [
            "--store",
            str(store_path),
            "--probes",
            str(probes_path),
            "--history",
            str(history),
            "--html",
            str(report),
            *(extra_args or []),
        ]
    )
    code = eval_cmd.run(args)
    return code, history, report


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_history_log_refuses_newer_schema(self, tmp_path: Path) -> None:
        history = tmp_path / "hist.json"
        history.write_text(json.dumps({"schema_version": 99, "runs": []}), encoding="utf-8")
        with pytest.raises(MemoryStoreError):
            eval_cmd.load_history(history)

    def test_history_log_refuses_newer_schema_integration(self, tmp_path: Path) -> None:
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        history = tmp_path / "hist.json"
        history.write_text(json.dumps({"schema_version": 99, "runs": []}), encoding="utf-8")
        report = tmp_path / "r.html"
        args = _parse_eval_args(
            ["--store", str(store_path), "--history", str(history), "--html", str(report)]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_ERROR
        # untouched: still the same too-new payload, no append happened
        assert json.loads(history.read_text())["schema_version"] == 99

    def test_history_log_corrupt_starts_fresh(self, tmp_path: Path) -> None:
        history = tmp_path / "hist.json"
        history.write_text("{not json", encoding="utf-8")
        runs = eval_cmd.load_history(history)
        assert runs == []

    def test_init_probes_writes_starter_and_refuses_overwrite(self, tmp_path: Path) -> None:
        target = tmp_path / "probes.json"
        args = _parse_eval_args(["--init-probes", str(target)])
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK
        assert target.exists()
        data = json.loads(target.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) > 0

        code2 = eval_cmd.run(args)
        assert code2 == _util.EXIT_ERROR
        # not clobbered
        assert json.loads(target.read_text(encoding="utf-8")) == data

    def test_missing_probe_file_degrades(self, tmp_path: Path) -> None:
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        args = _parse_eval_args(
            [
                "--store",
                str(store_path),
                "--probes",
                str(tmp_path / "does_not_exist.json"),
                "--history",
                str(tmp_path / "h.json"),
                "--html",
                str(tmp_path / "r.html"),
            ]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_ERROR

    def test_read_only_stale_schema_migration_is_handled(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def fake_open_store(_path: str, _agent_id: str, _embedding: str) -> Memory:
            raise StorageError("schema migration failed: corrupt row")

        monkeypatch.setattr(eval_cmd, "_open_store", fake_open_store)
        args = _parse_eval_args(
            [
                "--store",
                str(tmp_path / "wherever"),
                "--history",
                str(tmp_path / "h.json"),
                "--html",
                str(tmp_path / "r.html"),
            ]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_ERROR

    def test_load_history_unexpected_dict_format_starts_fresh(self, tmp_path: Path) -> None:
        """QA gap: a syntactically-valid JSON object that is neither a bare list
        nor a recognizable {schema_version, runs} shape must warn and fall back
        to an empty log, not crash or silently misinterpret the file."""
        history = tmp_path / "hist.json"
        history.write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")
        runs = eval_cmd.load_history(history)
        assert runs == []

    def test_atomic_write_text_removes_temp_file_and_reraises_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QA gap: `_atomic_write_text`'s cleanup-on-failure branch was untested.
        A failure during the atomic rename must remove the leftover temp file and
        propagate the original exception -- never leave a stray `.tmp` file or
        swallow the error."""
        target = tmp_path / "out.json"

        def failing_replace(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(eval_cmd.os, "replace", failing_replace)
        with pytest.raises(OSError, match="disk full"):
            eval_cmd._atomic_write_text(target, "content")

        assert not target.exists()
        leftover = list(tmp_path.iterdir())
        assert leftover == []


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_reduction_math(self) -> None:
        mem = _StubMemory(token_count=100)
        dump_text = "x" * 400  # heuristic estimator: len // 4 == 100
        dump_tok, curate_tok, reduction, label = eval_cmd._measure(mem, dump_text, [], 1500)  # type: ignore[arg-type]
        assert dump_tok == 100
        assert curate_tok == 100
        assert reduction == round(100 / 100, 1)
        assert label == "heuristic"

    def test_reduction_math_curate_zero_no_div_by_zero(self) -> None:
        mem = _StubMemory(token_count=0)
        dump_tok, curate_tok, reduction, _label = eval_cmd._measure(mem, "hello world", [], 1500)  # type: ignore[arg-type]
        assert curate_tok == 0
        assert reduction == round(dump_tok / 1, 1)

    def test_empty_store_records_run_without_crashing(self, tmp_path: Path) -> None:
        store_path = tmp_path / "empty_store"
        writer = Memory(path=str(store_path), agent_id="default")
        writer.close()
        history = tmp_path / "h.json"
        args = _parse_eval_args(
            [
                "--store",
                str(store_path),
                "--history",
                str(history),
                "--html",
                str(tmp_path / "r.html"),
            ]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK
        payload = json.loads(history.read_text(encoding="utf-8"))
        row = payload["runs"][0]
        assert row["store_size"] == 0
        assert row["dump_tokens"] == 0
        assert row["reduction"] == 0.0

    def test_history_log_reads_legacy_bare_list(self, tmp_path: Path) -> None:
        history = tmp_path / "hist.json"
        legacy_row = {"timestamp": "2020-01-01T00:00:00+00:00", "store_size": 1}
        history.write_text(json.dumps([legacy_row]), encoding="utf-8")

        runs = eval_cmd.load_history(history)
        assert runs == [legacy_row]

        eval_cmd.append_run(history, _default_run())
        payload = json.loads(history.read_text(encoding="utf-8"))
        assert payload["schema_version"] == eval_cmd.HISTORY_SCHEMA_VERSION
        assert payload["runs"][0] == legacy_row
        assert payload["runs"][1] == _default_run().to_dict()

    def test_write_report_returns_zero_for_empty_history(self, tmp_path: Path) -> None:
        """QA gap: `_write_report`'s empty-history short-circuit (no runs -> no HTML
        file, count 0) was untested."""
        history = tmp_path / "hist.json"  # never created -> load_history returns []
        out = tmp_path / "r.html"
        count = eval_cmd._write_report(history, out)
        assert count == 0
        assert not out.exists()

    def test_no_report_flag_skips_html_render(self, tmp_path: Path) -> None:
        """QA gap: `--no-report` must skip the HTML render entirely (log-only run)."""
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        history = tmp_path / "h.json"
        report = tmp_path / "r.html"
        args = _parse_eval_args(
            [
                "--store",
                str(store_path),
                "--history",
                str(history),
                "--html",
                str(report),
                "--no-report",
            ]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK
        assert history.exists()
        assert not report.exists()


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_session_keys_excluded_from_dump(self) -> None:
        # NOTE: MemoryStore.create() itself refuses to persist a caller-supplied
        # "session:"-prefixed key (SEC-SEV-001, store.py) -- real session markers
        # live in a dedicated sessions table, never as keyed MemoryEntry rows. The
        # skip in `_build_dump` is therefore defensive/forward-looking; exercised
        # here against a duck-typed stub (only `.list_keys()`/`.get()` are used).
        entries = {
            "fact:a": _StubEntry("fact:a", "alpha fact", MemoryType.FACT),
            "session:2026-07-01T00:00:00": _StubEntry(
                "session:2026-07-01T00:00:00", "session marker content", MemoryType.FACT
            ),
            "decision:b": _StubEntry("decision:b", "beta decision", MemoryType.DECISION),
        }
        mem = _StubDumpMemory(entries)
        dump_text, count = eval_cmd._build_dump(mem, compile_key_patterns())  # type: ignore[arg-type]

        assert count == 2
        assert "session:" not in dump_text
        assert "session marker content" not in dump_text

    def test_dump_is_redacted(self, tmp_path: Path) -> None:
        """Default-sensitive key + secret-SHAPED content -> still whole-masked
        (v0.2 softening, D-v02-7: the key alone is no longer enough).

        The secret is embedded inside an unrelated sentence and the
        SURROUNDING sentence is asserted absent too (test review MAJOR /
        QA-strengthened, mirroring test_curator/test_summarizer/test_export's
        equivalent sites): a byte-identical secret-only fixture cannot
        distinguish true whole-body masking from ``redact_text``'s surgical
        substring scrub alone, since the latter already strips the raw
        secret value wherever it appears in the dump. Embedding it mid-
        sentence means only the whole-mask branch in ``_build_dump`` removes
        the surrounding prose."""
        secret_value = "sk-" + "z" * 24
        sentence = f"the rotation doc says {secret_value} expires monthly"
        store_path = _seed_writer_store(
            tmp_path,
            [("fact:a", "token sk-ABCDEFGHIJKLMNOPQRSTUVWX inline", MemoryType.FACT)],
        )
        writer = Memory(path=str(store_path), agent_id="default")
        writer.store(sentence, type=MemoryType.FACT, key="api_key:prod")
        writer.close()

        reader = Memory(path=str(store_path), agent_id="default", read_only=True)
        reader.startup()
        try:
            dump_text, _count = eval_cmd._build_dump(reader, compile_key_patterns())
        finally:
            reader.close()

        assert "[REDACTED]" in dump_text
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in dump_text
        assert secret_value not in dump_text
        assert "rotation doc says" not in dump_text
        assert "expires monthly" not in dump_text

    def test_dump_prose_under_sensitive_key_passes_through(self, tmp_path: Path) -> None:
        """The reported bug fix, on the eval CLI's dump surface: a
        default-sensitive-named key holding plain prose is no longer
        whole-masked (v0.2 softening, D-v02-7). The CLI has no
        ``sensitive_keys`` surface (deferred Q5), so this exercises the
        built-in-default + content-shape path only."""
        store_path = _seed_writer_store(tmp_path, [])
        writer = Memory(path=str(store_path), agent_id="default")
        writer.store("auth token TTL is 15 min", type=MemoryType.FACT, key="fact:auth-ttl")
        writer.close()

        reader = Memory(path=str(store_path), agent_id="default", read_only=True)
        reader.startup()
        try:
            dump_text, _count = eval_cmd._build_dump(reader, compile_key_patterns())
        finally:
            reader.close()

        assert "auth token TTL is 15 min" in dump_text
        assert "[REDACTED]" not in dump_text

    def test_history_log_versioned_object_written(self, tmp_path: Path) -> None:
        history = tmp_path / "hist.json"
        run = _default_run()
        eval_cmd.append_run(history, run)
        payload = json.loads(history.read_text(encoding="utf-8"))
        assert payload["schema_version"] == eval_cmd.HISTORY_SCHEMA_VERSION
        assert payload["runs"] == [run.to_dict()]

    def test_append_run_write_is_atomic_no_stray_temp_files(self, tmp_path: Path) -> None:
        """DB LOW (optional, item G): append_run writes via temp-file + os.replace;
        no partial/temp file should survive a successful write, and repeated
        appends must keep producing a single valid versioned log."""
        history = tmp_path / "hist.json"
        eval_cmd.append_run(history, _default_run(timestamp="2026-07-01T00:00:00+00:00"))
        eval_cmd.append_run(history, _default_run(timestamp="2026-07-15T00:00:00+00:00"))

        payload = json.loads(history.read_text(encoding="utf-8"))
        assert len(payload["runs"]) == 2
        leftover = [p for p in tmp_path.iterdir() if p.name != "hist.json"]
        assert leftover == []

    def test_history_run_records_estimator_and_embedding(self, tmp_path: Path) -> None:
        history = tmp_path / "hist.json"
        run = _default_run(estimator="heuristic", embedding="none")
        eval_cmd.append_run(history, run)
        payload = json.loads(history.read_text(encoding="utf-8"))
        row = payload["runs"][0]
        assert row["estimator"] == "heuristic"
        assert row["embedding"] == "none"

    def test_estimator_label_recorded(self) -> None:
        assert eval_cmd._estimator_label(HeuristicEstimator()) == "heuristic"

    def test_html_and_out_aliases_target_same_dest(self, tmp_path: Path) -> None:
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        out_report = tmp_path / "r.html"
        args = _parse_eval_args(
            [
                "--store",
                str(store_path),
                "--out",
                str(out_report),
                "--history",
                str(tmp_path / "h.json"),
            ]
        )
        assert args.html == str(out_report)
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK
        assert out_report.exists()

    def test_html_canonical_flag_same_dest(self, tmp_path: Path) -> None:
        args = _parse_eval_args(["--html", "r.html"])
        assert args.html == "r.html"

    def test_embedding_none_default_no_torch_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        class _FakeLocalEmbedder:
            def __init__(self, *args: object, **kwargs: object) -> None:
                calls.append(1)

        monkeypatch.setattr("tulving.LocalEmbedder", _FakeLocalEmbedder)
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        mem = eval_cmd._open_store(str(store_path), "default", "none")
        mem.close()
        assert calls == []

    def test_successful_scoring_records_correctness_and_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """QA gap: the happy-path scoring integration (run() -> AnswerScorer succeeds
        -> `model_used = args.model`, line 228 of eval_cmd.py) had no coverage --
        every existing integration test exercised only the degradation branches."""

        def succeeding_transport(
            _url: str, payload: dict[str, Any], _headers: dict[str, str], _timeout: int
        ) -> dict[str, Any]:
            system = payload["messages"][0]["content"]
            answer = "unknown" if "CONTEXT:\n(none)" in system else "hello"
            return {"choices": [{"message": {"content": answer}}]}

        code, history, report = _run_eval_with_transport(
            tmp_path, monkeypatch, succeeding_transport
        )

        assert code == _util.EXIT_OK
        payload = json.loads(history.read_text(encoding="utf-8"))
        row = payload["runs"][0]
        assert row["correctness"] == {"none": "0/1", "dump": "1/1", "curate": "1/1"}
        assert row["model"] == eval_cmd.DEFAULT_MODEL
        assert report.exists()

    def test_verdict_message_high_reduction(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """QA gap: the `reduction >= 2` verdict branch (eval_cmd.py:267-271) had no
        coverage -- only the 'store still small' branch was exercised."""

        def fake_measure(
            _mem: Any, _dump_text: str, _queries: list[str], _budget: int
        ) -> tuple[int, int, float, str]:
            return (1000, 100, 10.0, "heuristic")

        monkeypatch.setattr(eval_cmd, "_measure", fake_measure)
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        args = _parse_eval_args(
            [
                "--store",
                str(store_path),
                "--history",
                str(tmp_path / "h.json"),
                "--html",
                str(tmp_path / "r.html"),
            ]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK
        out = capsys.readouterr().out
        assert "fewer tokens" in out


# ---------------------------------------------------------------------------
# Mandatory regression tests
# ---------------------------------------------------------------------------


class TestMandatoryRegressions:
    def test_read_only_open_never_mutates_store(self, tmp_path: Path) -> None:
        """No eval-driven write touches the store, byte-for-byte (D-v02-6).

        FIXED (DB engineer, this batch): a prior run of this test discovered that
        ``SQLiteBackend._connection()`` re-issued ``PRAGMA auto_vacuum = INCREMENTAL``
        on EVERY new connection, incrementing the SQLite header's file-change-counter
        by one byte even when the mode already matched -- so a literal whole-file
        byte-hash comparison did not hold for ANY read-only ``Memory`` open, eval
        included. ``tulving/adapters/storage.py`` now skips that pragma entirely for
        an existing non-empty database (see ``tests/test_storage.py::TestPragmas::
        test_reopen_existing_nonempty_store_leaves_header_byte_identical``), so the
        original literal assertion holds again: the DB file is byte-identical.
        """
        store_path = _seed_writer_store(tmp_path, [("fact:a", "hello", MemoryType.FACT)])
        db_file = store_path / "tulving.db"
        # The writer's own close() leaves a tulving.lock file behind by design
        # (_AdvisoryLock.release() never unlinks it -- see memory.py). A read-only
        # handle never constructs an _AdvisoryLock at all, so that pre-existing
        # lock file must be untouched (not recreated, not rewritten) by eval.
        lock_before = (store_path / "tulving.lock").read_bytes()
        db_before = db_file.read_bytes()

        history = tmp_path / "h.json"
        report = tmp_path / "r.html"
        args = _parse_eval_args(
            ["--store", str(store_path), "--history", str(history), "--html", str(report)]
        )
        code = eval_cmd.run(args)
        assert code == _util.EXIT_OK

        assert db_file.read_bytes() == db_before
        assert (store_path / "tulving.lock").read_bytes() == lock_before

    def _run_eval_with_transport(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        transport: Callable[[str, dict[str, Any], dict[str, str], int], dict[str, Any]],
    ) -> tuple[int, Path, Path]:
        """Drive eval_cmd.run() end-to-end with a stubbed AnswerScorer transport."""
        return _run_eval_with_transport(tmp_path, monkeypatch, transport)

    def test_no_endpoint_degrades_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import urllib.error

        def raising_transport(
            _url: str, _payload: dict[str, Any], _headers: dict[str, str], _timeout: int
        ) -> dict[str, Any]:
            raise urllib.error.URLError("connection refused")

        code, history, report = self._run_eval_with_transport(
            tmp_path, monkeypatch, raising_transport
        )

        assert code == _util.EXIT_OK
        payload = json.loads(history.read_text(encoding="utf-8"))
        row = payload["runs"][0]
        assert row["correctness"] is None
        assert report.exists()
        captured = capsys.readouterr()
        assert captured.err.strip() != ""

    def test_non_json_response_degrades_loudly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SECURITY HIGH regression: a 200-OK non-JSON body (proxy page, truncated
        stream) raises json.JSONDecodeError from _urllib_transport's json.load() --
        a ValueError subclass that must NOT escape run()'s scoring except-clause,
        the outer except TulvingError, or guard_main. It must degrade loudly with a
        clean exit code, exactly like a network-refusal."""

        def non_json_transport(
            _url: str, _payload: dict[str, Any], _headers: dict[str, str], _timeout: int
        ) -> dict[str, Any]:
            json.loads("<html>not json</html>")  # raises json.JSONDecodeError
            raise AssertionError("unreachable")  # pragma: no cover

        code, history, report = self._run_eval_with_transport(
            tmp_path, monkeypatch, non_json_transport
        )

        assert code == _util.EXIT_OK
        payload = json.loads(history.read_text(encoding="utf-8"))
        row = payload["runs"][0]
        assert row["correctness"] is None
        assert report.exists()
        captured = capsys.readouterr()
        assert captured.err.strip() != ""

    def test_scoring_failure_message_is_redacted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """SECURITY MEDIUM regression: the scoring-failure emit() must redact the
        exception text like its sibling sites (:192, :253), not interpolate raw."""
        import urllib.error

        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456"

        def leaking_transport(
            _url: str, _payload: dict[str, Any], _headers: dict[str, str], _timeout: int
        ) -> dict[str, Any]:
            raise urllib.error.URLError(f"connection refused, leaked credential {secret}")

        code, _history, _report = self._run_eval_with_transport(
            tmp_path, monkeypatch, leaking_transport
        )

        assert code == _util.EXIT_OK
        captured = capsys.readouterr()
        assert secret not in captured.err
        assert "[REDACTED]" in captured.err

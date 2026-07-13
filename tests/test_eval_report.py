"""Tests for tulving.cli.eval_report — written BEFORE implementation.

Pure functions: `render(runs)` reads only the history log's `runs[]` (integers, floats,
ISO timestamps, "n/total" strings, store_path, model, estimator, embedding) — it never
sees memory content. Self-containment and escaping are security-adjacent and mandatory.
"""

from __future__ import annotations

import re
from typing import Any

from tulving.cli.eval_report import (
    compute_verdict,
    correctness_panel,
    render,
    stat_cards,
    svg_chart,
    table_rows,
)


def _run(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "timestamp": "2026-07-01T12:00:00+00:00",
        "store_path": "D:/MemoryStorage/store",
        "agent_id": "default",
        "store_size": 42,
        "budget": 1500,
        "dump_tokens": 5120,
        "curate_tokens": 640,
        "reduction": 8.0,
        "estimator": "tiktoken",
        "embedding": "none",
        "correctness": {"none": "0/3", "dump": "3/3", "curate": "3/3"},
        "model": "local-model",
        "tulving_version": "0.2.0",
        "retrieval": "kv-only",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Failure paths / robustness
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_reduction_precision_and_zero_handled(self) -> None:
        runs = [_run(curate_tokens=0, reduction=0.0, dump_tokens=0, correctness=None, model=None)]
        html = render(runs)  # must not raise ZeroDivisionError
        assert "<!doctype html>" in html.lower()

    def test_single_run_with_no_correctness_renders(self) -> None:
        runs = [_run(correctness=None, model=None)]
        html = render(runs)
        assert "<!doctype html>" in html.lower()


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


class TestBoundaryConditions:
    def test_single_run_shows_need_more_data_note(self) -> None:
        chart = svg_chart([_run()])
        assert "<polyline" not in chart

    def test_multi_run_draws_two_series(self) -> None:
        runs = [
            _run(timestamp="2026-07-01T12:00:00+00:00"),
            _run(timestamp="2026-07-15T12:00:00+00:00"),
        ]
        chart = svg_chart(runs)
        assert chart.count("<polyline") == 2

    def test_verdict_not_yet_when_reduction_below_two(self) -> None:
        label, tone, _note = compute_verdict([_run(reduction=1.2, store_size=5)])
        assert label == "Not yet"
        assert tone == "warn"

    def test_verdict_yes_when_reduction_and_correctness_good(self) -> None:
        label, tone, _note = compute_verdict(
            [_run(reduction=8.0, correctness={"none": "0/3", "dump": "2/3", "curate": "3/3"})]
        )
        assert label == "Yes"
        assert tone == "good"

    def test_verdict_almost_when_curate_trails_dump(self) -> None:
        label, tone, _note = compute_verdict(
            [_run(reduction=8.0, correctness={"none": "0/3", "dump": "3/3", "curate": "1/3"})]
        )
        assert label == "Almost"
        assert tone == "warn"

    def test_correctness_panel_absent_shows_guidance(self) -> None:
        panel = correctness_panel([_run(correctness=None, model=None)])
        assert "--probes" in panel
        assert "cbar-fill" not in panel


# ---------------------------------------------------------------------------
# Basic behavior
# ---------------------------------------------------------------------------


class TestBasicBehavior:
    def test_report_shows_store_basename_not_full_path(self) -> None:
        runs = [_run(store_path="D:/Secret/alice/store")]
        html = render(runs)
        assert ">store<" in html or "store" in html
        assert "alice" not in html
        assert "D:/Secret" not in html

    def test_estimator_drift_footnote_shown(self) -> None:
        runs = [_run(estimator="tiktoken"), _run(estimator="heuristic")]
        html = render(runs)
        assert "estimator" in html.lower()

    def test_estimator_drift_footnote_absent_for_single_estimator(self) -> None:
        runs = [_run(estimator="tiktoken"), _run(estimator="tiktoken")]
        html = render(runs)
        assert "span more than one estimator" not in html.lower()

    def test_stat_cards_reports_reduction(self) -> None:
        cards = stat_cards([_run(reduction=8.0)])
        assert "8" in cards

    def test_table_rows_one_row_per_run(self) -> None:
        rows = table_rows([_run(), _run(timestamp="2026-07-15T12:00:00+00:00")])
        assert rows.count("<tr") == 2


class TestRetrievalRegimeRendering:
    """The `retrieval` field's HTML surfacing: per-row mode + the mixed-regime
    footnote (parallel to `_estimator_footnote`) when the log mixes retrieval
    modes under one embedding label (docs/specs/cli-eval.md)."""

    def test_table_row_shows_retrieval_mode(self) -> None:
        rows = table_rows([_run(retrieval="semantic")])
        assert "semantic" in rows

        rows_kv = table_rows([_run(retrieval="kv-only")])
        assert "kv-only" in rows_kv

    def test_table_row_shows_unknown_for_legacy_row(self) -> None:
        legacy = _run()
        del legacy["retrieval"]
        rows = table_rows([legacy])
        assert "unknown" in rows

    def test_mixed_regime_footnote_absent_when_uniform(self) -> None:
        runs = [
            _run(embedding="local", retrieval="semantic"),
            _run(embedding="local", retrieval="semantic", timestamp="2026-07-15T12:00:00+00:00"),
        ]
        html_out = render(runs)
        assert "measured retrieval differently" not in html_out

    def test_mixed_regime_footnote_present_when_same_embedding_mixes_modes(self) -> None:
        """The bug's regression case: two runs both labeled embedding="local",
        one measured with a live semantic index, one degraded to kv-only
        (stale cache) -- the report must flag the mixed regime."""
        runs = [
            _run(embedding="local", retrieval="semantic"),
            _run(embedding="local", retrieval="kv-only", timestamp="2026-07-15T12:00:00+00:00"),
        ]
        html_out = render(runs)
        assert "measured retrieval differently" in html_out

    def test_mixed_regime_footnote_scoped_per_embedding_label(self) -> None:
        """A "none"/"local" split is NOT a retrieval-regime bug (embedding
        itself differs, by design) -- only mixing WITHIN one embedding label
        is the drift this footnote exists to catch."""
        runs = [
            _run(embedding="none", retrieval="kv-only"),
            _run(embedding="local", retrieval="semantic", timestamp="2026-07-15T12:00:00+00:00"),
        ]
        html_out = render(runs)
        assert "measured retrieval differently" not in html_out

    def test_mixed_regime_footnote_absent_for_single_legacy_only_group(self) -> None:
        """A group containing only `"unknown"` (all legacy rows) is one
        distinct value, same convention as `_estimator_footnote` -- no
        footnote, since there's no actual observed drift, just missing data."""
        legacy_a, legacy_b = _run(embedding="local"), _run(embedding="local")
        del legacy_a["retrieval"]
        del legacy_b["retrieval"]
        legacy_b["timestamp"] = "2026-07-15T12:00:00+00:00"
        html_out = render([legacy_a, legacy_b])
        assert "measured retrieval differently" not in html_out

    def test_old_rows_without_retrieval_render_without_crashing(self) -> None:
        legacy = _run()
        del legacy["retrieval"]
        html_out = render([legacy])
        assert "<!doctype html>" in html_out.lower()


class TestSecurity:
    def test_report_is_self_contained(self) -> None:
        runs = [_run(), _run(timestamp="2026-07-15T12:00:00+00:00")]
        html = render(runs)
        assert "<!doctype html>" in html.lower()
        assert "</html>" in html.lower()
        assert "http://" not in html
        assert "https://" not in html
        assert "<script" not in html.lower()
        assert "<link" not in html.lower()
        assert "src=" not in html.lower()

    def test_store_and_model_are_html_escaped(self) -> None:
        runs = [_run(store_path="D:/x/<b>store</b>", model="<script>alert(1)</script>")]
        html = render(runs)
        assert "<b>store</b>" not in html
        assert "<script>alert(1)</script>" not in html
        # Escaped forms are present somewhere (basename rendered, model rendered in panel).
        assert re.search(r"&lt;b&gt;store&lt;/b&gt;|&lt;script&gt;", html) is not None

    def test_retrieval_is_html_escaped(self) -> None:
        """reviewer LOW: `retrieval` reaches the report through a hand-editable
        JSON log (untrusted relative to the report's own code) exactly like
        `store_path`/`model` above -- a hostile value must render escaped,
        never as live markup."""
        runs = [_run(retrieval="<script>alert(1)</script>")]
        html = render(runs)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

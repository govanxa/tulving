"""Tulving CLI — self-contained HTML value-report renderer.

Pure functions over the eval history log's ``runs[]`` list (integers, floats, ISO
timestamps, ``"n/total"`` strings, ``store_path``, ``model``, ``estimator``,
``embedding``) -- this module never imports ``Memory`` and never sees memory
content. Behavioural spec: ``docs/testing_guides/examples/tulving_eval_report.py``
(inline CSS/SVG, theme-aware); this is a typed, escaped, tested refactor of it.

Security (D-v02-6 ruling 5): the store path renders as its BASENAME only (never the
full path, which can embed a username); every user-derived string interpolated into
the document passes through ``html.escape``.
"""

from __future__ import annotations

import html
import math
from datetime import UTC, datetime
from pathlib import PurePath
from typing import Any, Final

# SVG plot area within a 720x300 viewBox.
_PLOT_L: Final[float] = 70.0
_PLOT_R: Final[float] = 680.0
_PLOT_T: Final[float] = 30.0
_PLOT_B: Final[float] = 250.0


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def _nice_ceil(value: float) -> float:
    """Round up to a clean axis maximum (1/2/2.5/5 x 10^k)."""
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    for mult in (1, 2, 2.5, 5, 10):
        if value <= mult * base:
            return float(mult * base)
    return 10 * base


def _x_positions(n: int) -> list[float]:
    if n == 1:
        return [(_PLOT_L + _PLOT_R) / 2]
    step = (_PLOT_R - _PLOT_L) / (n - 1)
    return [_PLOT_L + i * step for i in range(n)]


def _y_for(value: float, ymax: float) -> float:
    return _PLOT_B - (value / ymax) * (_PLOT_B - _PLOT_T)


def _date_label(iso: str) -> str:
    """``'2026-07-01T..'`` -> ``'Jul 1'`` (cross-platform, no ``%-d``)."""
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return html.escape(iso[:10])
    return f"{dt:%b} {dt.day}"


def _date_full(iso: str) -> str:
    return html.escape(iso[:10])


def _display_store(store_path: str) -> str:
    """Render the store's BASENAME only, HTML-escaped (D-v02-6 ruling 5).

    The full path (which can embed a username) is stored in the log for the
    user's own reference but MUST NOT appear in the shareable report.
    """
    name = PurePath(store_path).name or store_path
    return html.escape(name)


def _num(score: str) -> int:
    return int(str(score).split("/")[0])


def _denom(score: str) -> int:
    return int(str(score).split("/")[1])


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def compute_verdict(runs: list[dict[str, Any]]) -> tuple[str, str, str]:
    """Return ``(label, tone, note)``; ``tone`` in ``{"good", "warn"}``."""
    latest = runs[-1]
    reduction = latest.get("reduction", 0) or 0
    corr = latest.get("correctness")

    if reduction < 2:
        return (
            "Not yet",
            "warn",
            f"store still small ({latest.get('store_size', 0)} memories) "
            "-- the win appears past ~20",
        )

    if corr:
        curate, dump, none = _num(corr["curate"]), _num(corr["dump"]), _num(corr["none"])
        if curate >= dump and curate > none:
            return ("Yes", "good", "and by more each run")
        if curate < dump:
            return ("Almost", "warn", "curate trails dump -- raise the token budget and re-run")
    return ("Yes", "good", "and by more each run")


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


def svg_chart(runs: list[dict[str, Any]]) -> str:
    """Dump-vs-curate token trend; ``< 2`` runs renders a "need more data" note."""
    n = len(runs)
    if n < 2:
        return (
            '<p class="desc" style="margin-top:8px">Only one run logged so far -- the '
            "trend chart appears once you have at least two. Run <code>tulving eval</code> "
            "again in a couple of weeks.</p>"
        )
    xs = _x_positions(n)
    dumps = [float(r.get("dump_tokens", 0) or 0) for r in runs]
    curates = [float(r.get("curate_tokens", 0) or 0) for r in runs]
    ymax = _nice_ceil(max(dumps + curates))

    def pts(values: list[float]) -> str:
        return " ".join(f"{x:.1f},{_y_for(v, ymax):.1f}" for x, v in zip(xs, values, strict=True))

    dump_pts, curate_pts = pts(dumps), pts(curates)
    area = "M" + " L".join(
        f"{x:.1f},{_y_for(v, ymax):.1f}" for x, v in zip(xs, curates, strict=True)
    )
    area += f" L{xs[-1]:.1f},{_PLOT_B} L{xs[0]:.1f},{_PLOT_B} Z"

    grid: list[str] = []
    for frac, val in ((0, ymax), (1 / 3, ymax * 2 / 3), (2 / 3, ymax / 3), (1, 0)):
        y = _PLOT_T + frac * (_PLOT_B - _PLOT_T)
        grid.append(f'<line class="grid-line" x1="66" y1="{y:.1f}" x2="690" y2="{y:.1f}"></line>')
        grid.append(
            f'<text class="axis-label" x="60" y="{y + 4:.1f}" text-anchor="end">{val:,.0f}</text>'
        )

    dots: list[str] = []
    for x, v in zip(xs, dumps, strict=True):
        dots.append(
            f'<circle class="dot dump" cx="{x:.1f}" cy="{_y_for(v, ymax):.1f}" r="4.5"></circle>'
        )
    for x, v in zip(xs, curates, strict=True):
        dots.append(
            f'<circle class="dot curate" cx="{x:.1f}" cy="{_y_for(v, ymax):.1f}" r="4.5"></circle>'
        )

    end_dump = (
        f'<text class="pt-label dump" x="{xs[-1] - 8:.1f}" '
        f'y="{_y_for(dumps[-1], ymax) - 9:.1f}" text-anchor="end">{dumps[-1]:,.0f}</text>'
    )
    end_cur = (
        f'<text class="pt-label curate" x="{xs[-1] - 8:.1f}" '
        f'y="{_y_for(curates[-1], ymax) + 18:.1f}" text-anchor="end">{curates[-1]:,.0f}</text>'
    )

    xlabels: list[str] = []
    for i, (x, r) in enumerate(zip(xs, runs, strict=True)):
        anchor = "start" if i == 0 else "end" if i == n - 1 else "middle"
        xlabels.append(
            f'<text class="axis-label" x="{x:.1f}" y="272" '
            f'text-anchor="{anchor}">{_date_label(str(r.get("timestamp", "")))}</text>'
        )

    return f"""<div class="chart-scroll">
      <svg class="chart" viewBox="0 0 720 300" role="img"
           aria-label="Token cost over time: dump rises while curate stays flat.">
        {"".join(grid)}
        <path d="{area}" fill="var(--accent-soft)"></path>
        <polyline class="series dump" points="{dump_pts}"></polyline>
        <polyline class="series curate" points="{curate_pts}"></polyline>
        {"".join(dots)}
        {end_dump}
        {end_cur}
        {"".join(xlabels)}
      </svg>
    </div>"""


def reduction_bars(runs: list[dict[str, Any]]) -> str:
    """Reduction ratio per run, latest highlighted."""
    reductions = [r.get("reduction", 0) or 0 for r in runs]
    rmax = max(reductions) or 1
    bars: list[str] = []
    for i, (r, red) in enumerate(zip(runs, reductions, strict=True)):
        latest = " latest" if i == len(runs) - 1 else ""
        height = max(3, red / rmax * 100)
        bars.append(
            f'<div class="redbar{latest}"><span class="cap">{red:g}x</span>'
            f'<div class="col" style="height:{height:.0f}%"></div>'
            f'<span class="date">{_date_label(str(r.get("timestamp", "")))}</span></div>'
        )
    body = "".join(bars)
    return f'<div class="redbars" role="img" aria-label="Reduction ratio over time.">{body}</div>'


def correctness_panel(runs: list[dict[str, Any]]) -> str:
    """None/dump/curate bars for the latest scored run; guidance when never scored."""
    scored = [r for r in runs if r.get("correctness")]
    if not scored:
        return (
            '<section class="panel"><div class="panel-head"><h2>Answer correctness</h2>'
            '<p class="desc">No correctness scored yet. Run <code>tulving eval</code> with '
            "<code>--probes</code> and your model to add it.</p></div></section>"
        )
    run = scored[-1]
    corr = run["correctness"]

    def bar(name: str, cls: str, score: str) -> str:
        pct = _num(score) / max(_denom(score), 1) * 100
        return (
            f'<div class="cbar-row"><span class="cbar-name">{name}</span>'
            f'<div class="cbar-track"><div class="cbar-fill {cls}" '
            f'style="width:{pct:.0f}%"></div></div>'
            f'<span class="cbar-score">{html.escape(str(score))}</span></div>'
        )

    model = html.escape(str(run.get("model") or "your model"))
    return f"""<section class="panel">
      <div class="panel-head">
        <h2>Answers stay correct -- the part that matters</h2>
        <p class="desc">Each condition faces the same probe set, scored by <strong>{model}</strong>.
           Latest scored run &middot; {_date_full(str(run.get("timestamp", "")))}.</p>
      </div>
      {bar("No memory", "none", corr["none"])}
      {bar("Full dump", "dump", corr["dump"])}
      {bar("Curate", "curate", corr["curate"])}
      <p class="cbar-note"><strong>Helping = curate ties the full dump and both beat
         no-memory.</strong> If curate ever trails dump, the token budget is too
         tight -- raise it and re-run.</p>
    </section>"""


def _retrieval_label(run: dict[str, Any]) -> str:
    """``"semantic"`` / ``"kv-only"`` / ``"unknown"`` (legacy rows predating
    the field), HTML-escaped. Mirrors ``_estimator_footnote``'s convention of
    defaulting an absent field to ``"unknown"`` rather than guessing a value
    this report was never told."""
    return html.escape(str(run.get("retrieval", "unknown")))


def table_rows(runs: list[dict[str, Any]]) -> str:
    """One dated ``<tr>`` per run."""
    rows: list[str] = []
    for i, r in enumerate(runs):
        latest = ' class="latest"' if i == len(runs) - 1 else ""
        corr = r.get("correctness")
        corr_cell = (
            f'<span class="ok">{html.escape(str(corr["curate"]))}</span>'
            if corr
            else '<span style="color:var(--ink-faint)">&mdash;</span>'
        )
        rows.append(
            f"<tr{latest}><td>{_date_full(str(r.get('timestamp', '')))}</td>"
            f"<td>{r.get('store_size', 0)}</td><td>{r.get('dump_tokens', 0):,}</td>"
            f"<td>{r.get('curate_tokens', 0):,}</td>"
            f'<td><span class="red">{(r.get("reduction", 0) or 0):g}x</span></td>'
            f"<td>{corr_cell}</td><td>{_retrieval_label(r)}</td></tr>"
        )
    return "".join(rows)


def stat_cards(runs: list[dict[str, Any]]) -> str:
    """The four headline stat cards for the latest run."""
    latest, first = runs[-1], runs[0]
    reduction = latest.get("reduction", 0) or 0
    dump, curate = latest.get("dump_tokens", 0), latest.get("curate_tokens", 0)
    saved = dump - curate
    pct = (1 - curate / dump) * 100 if dump else 0
    corr = latest.get("correctness")
    if corr:
        corr_val = (
            f'<p class="value good">{_num(corr["curate"])}'
            f'<span class="unit">/{_denom(corr["curate"])}</span></p>'
        )
        corr_meta = (
            f"curate vs <strong>dump {html.escape(str(corr['dump']))}</strong>; "
            f"none {html.escape(str(corr['none']))}"
        )
    else:
        corr_val = '<p class="value" style="color:var(--ink-faint)">&mdash;</p>'
        corr_meta = "run with <strong>--probes</strong> to score"
    grew = (
        f"up from <strong>{first.get('store_size', 0)}</strong>" if len(runs) > 1 else "first run"
    )

    saved_str = f"{saved:,}" if saved >= 0 else f"&minus;{abs(saved):,}"
    return f"""
      <div class="stat"><p class="label">Token reduction</p>
        <p class="value accent">{reduction:g}<span class="unit">x</span></p>
        <p class="meta">fewer tokens than dumping the whole store</p></div>
      <div class="stat"><p class="label">Saved per curate call</p>
        <p class="value">{saved_str}</p>
        <p class="meta"><strong>{pct:.0f}%</strong> of the dump cost</p></div>
      <div class="stat"><p class="label">Answer correctness</p>
        {corr_val}<p class="meta">{corr_meta}</p></div>
      <div class="stat"><p class="label">Store size</p>
        <p class="value">{latest.get("store_size", 0)}</p>
        <p class="meta">memories, {grew}</p></div>"""


def _estimator_footnote(runs: list[dict[str, Any]]) -> str:
    """Estimator-drift footnote (D-v02-6 ruling 5) -- required, not optional.

    Renders when the log's runs carry more than one distinct ``estimator`` value.
    Empty string when there is only one (or the field is absent everywhere).
    """
    estimators = {str(r.get("estimator", "unknown")) for r in runs}
    if len(estimators) <= 1:
        return ""
    names = ", ".join(html.escape(name) for name in sorted(estimators))
    return (
        f'<p class="note-footnote">Token counts span more than one estimator ({names}); '
        "install <code>tiktoken</code> for a stable trend.</p>"
    )


def _retrieval_footnote(runs: list[dict[str, Any]]) -> str:
    """Retrieval-regime-drift footnote -- the parallel note to
    ``_estimator_footnote`` for the same "never silently mix regimes" law
    (docs/specs/cli-eval.md).

    ``embedding`` records what was CONFIGURED; ``retrieval`` records what was
    ACTUALLY measured (``Memory.semantic_available`` at run time). A stale or
    disabled semantic index degrades a "local"/"openai" run to KV-only
    without changing its ``embedding`` label, so runs sharing one
    ``embedding`` value can legitimately mix ``retrieval`` regimes. Renders
    only when some ``embedding`` group in the log actually contains more than
    one distinct ``retrieval`` value; a group with a single value -- even
    ``"unknown"`` from rows that predate this field -- renders nothing, same
    convention as ``_estimator_footnote``.
    """
    by_embedding: dict[str, set[str]] = {}
    for r in runs:
        embedding = str(r.get("embedding", "unknown"))
        retrieval = str(r.get("retrieval", "unknown"))
        by_embedding.setdefault(embedding, set()).add(retrieval)
    mixed = sorted(embedding for embedding, modes in by_embedding.items() if len(modes) > 1)
    if not mixed:
        return ""
    names = ", ".join(html.escape(embedding) for embedding in mixed)
    return (
        f'<p class="note-footnote">Runs under the same embedding mode ({names}) measured '
        "retrieval differently (semantic vs. kv-only -- likely a stale or diverged "
        "vector-index cache on some runs); treat those runs' trend as mixed regimes, "
        "not a like-for-like comparison. Open the store with a writer and call "
        "<code>rebuild_index(re_embed=True)</code> to bring the cache current.</p>"
    )


# ---------------------------------------------------------------------------
# CSS (inline; theme-aware)
# ---------------------------------------------------------------------------

CSS: Final[str] = """
:root{--ground:#EAEEF3;--surface:#fff;--surface-2:#F4F7FB;--ink:#1A2230;--ink-soft:#545F6E;
--ink-faint:#8993A2;--line:#DBE2EC;--line-strong:#C6D0DD;--accent:#0E7C86;--accent-ink:#0A5960;
--accent-soft:rgba(14,124,134,.10);--dump:#C2703D;--none:#97A1AF;--good:#1E7A4E;
--good-soft:rgba(30,122,78,.12);--warn:#B26A00;--warn-soft:rgba(178,106,0,.12);
--shadow:0 1px 2px rgba(20,30,48,.05),0 8px 24px rgba(20,30,48,.06);
--font-sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--font-mono:"SF Mono","Cascadia Mono","JetBrains Mono",Consolas,ui-monospace,monospace;}
@media(prefers-color-scheme:dark){:root{--ground:#0F141B;--surface:#161D27;--surface-2:#1D2632;
--ink:#E7ECF3;--ink-soft:#A4AFBF;--ink-faint:#6C7889;--line:#29323F;--line-strong:#37424F;
--accent:#2FB6C0;--accent-ink:#7FDDE3;--accent-soft:rgba(47,182,192,.14);--dump:#E08A54;
--none:#6C7889;--good:#45C98B;--good-soft:rgba(69,201,139,.16);--warn:#E0A64B;
--warn-soft:rgba(224,166,75,.14);--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);}}
*{box-sizing:border-box}body{margin:0}
.page{background:var(--ground);color:var(--ink);font-family:var(--font-sans);line-height:1.55;
padding:clamp(18px,4vw,56px) clamp(14px,4vw,40px);min-height:100vh}
.wrap{max-width:960px;margin:0 auto}
.eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.16em;text-transform:uppercase;
color:var(--accent-ink);margin:0 0 14px;display:flex;align-items:center;gap:10px}
.eyebrow::after{content:"";height:1px;flex:1;background:var(--line-strong)}
.report-head{display:grid;grid-template-columns:1fr auto;gap:22px 28px;align-items:end;
padding-bottom:22px;border-bottom:1px solid var(--line);margin-bottom:26px}
h1{font-size:clamp(30px,5.4vw,50px);line-height:1.02;letter-spacing:-.02em;font-weight:780;margin:0;
text-wrap:balance}
.sub{margin:14px 0 0;color:var(--ink-soft);max-width:60ch;font-size:15px}
.sub code,.desc code,.note-item code{font-family:var(--font-mono);font-size:12.5px;
background:var(--surface-2);border:1px solid var(--line);border-radius:5px;padding:1px 6px;
color:var(--ink);white-space:nowrap}
.verdict{justify-self:end;text-align:center;border-radius:14px;padding:16px 22px;min-width:190px}
.verdict.good{background:var(--good-soft);
border:1px solid color-mix(in srgb,var(--good) 40%,transparent)}
.verdict.warn{background:var(--warn-soft);
border:1px solid color-mix(in srgb,var(--warn) 40%,transparent)}
.verdict .q{font-family:var(--font-mono);font-size:11px;letter-spacing:.12em;
text-transform:uppercase;color:var(--ink-soft);margin:0 0 6px}
.verdict .a{font-size:26px;font-weight:760;line-height:1}
.verdict.good .a{color:var(--good)}.verdict.warn .a{color:var(--warn)}
.verdict .note{font-size:12.5px;color:var(--ink-soft);margin:7px 0 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:16px 16px 15px;box-shadow:var(--shadow)}
.stat .label{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.09em;
text-transform:uppercase;color:var(--ink-faint);margin:0 0 10px}
.stat .value{font-family:var(--font-mono);font-size:30px;font-weight:640;letter-spacing:-.02em;
color:var(--ink);font-variant-numeric:tabular-nums;line-height:1;margin:0}
.stat .value.accent{color:var(--accent-ink)}.stat .value.good{color:var(--good)}
.stat .unit{font-size:15px;color:var(--ink-faint);font-weight:600}
.stat .meta{font-size:12.5px;color:var(--ink-soft);margin:8px 0 0}
.stat .meta strong{color:var(--ink)}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:14px;
padding:22px clamp(16px,3vw,26px);box-shadow:var(--shadow);margin-bottom:18px}
.panel-head{margin-bottom:18px}
.panel h2{font-size:18px;font-weight:680;letter-spacing:-.01em;margin:0}
.panel .desc{margin:6px 0 0;color:var(--ink-soft);font-size:14px;max-width:66ch}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px}
.legend .item{display:flex;align-items:center;gap:8px;font-family:var(--font-mono);font-size:12px;
color:var(--ink-soft)}
.legend .swatch{width:22px;height:3px;border-radius:2px}
.legend .swatch.dump{background:var(--dump)}.legend .swatch.curate{background:var(--accent)}
.chart-scroll{overflow-x:auto}.chart{width:100%;min-width:520px;height:auto;display:block}
.grid-line{stroke:var(--line);stroke-width:1}
.axis-label{font-family:var(--font-mono);font-size:11px;fill:var(--ink-faint);
font-variant-numeric:tabular-nums}
.series{fill:none;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
.series.dump{stroke:var(--dump)}.series.curate{stroke:var(--accent)}
.dot{stroke:var(--surface);stroke-width:2}.dot.dump{fill:var(--dump)}.dot.curate{fill:var(--accent)}
.pt-label{font-family:var(--font-mono);font-size:12px;font-weight:600;
font-variant-numeric:tabular-nums}
.pt-label.dump{fill:var(--dump)}.pt-label.curate{fill:var(--accent-ink)}
.redbars{display:flex;align-items:flex-end;gap:clamp(10px,4vw,34px);height:150px;padding:6px 4px 0}
.redbar{flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;height:100%;
justify-content:flex-end}
.redbar .col{width:100%;max-width:68px;border-radius:6px 6px 0 0;background:var(--accent-soft);
border:1px solid color-mix(in srgb,var(--accent) 35%,transparent);border-bottom:none}
.redbar.latest .col{background:var(--accent);border-color:var(--accent)}
.redbar .cap{font-family:var(--font-mono);font-size:13px;font-weight:640;color:var(--accent-ink);
font-variant-numeric:tabular-nums}
.redbar.latest .cap{color:var(--accent)}
.redbar .date{font-family:var(--font-mono);font-size:11px;color:var(--ink-faint)}
.cbar-row{display:grid;grid-template-columns:92px 1fr auto;align-items:center;
gap:14px;margin:0 0 12px}
.cbar-name{font-family:var(--font-mono);font-size:12.5px;text-transform:uppercase;
letter-spacing:.06em;color:var(--ink-soft)}
.cbar-track{height:26px;background:var(--surface-2);border-radius:7px;overflow:hidden;
border:1px solid var(--line)}
.cbar-fill{height:100%;border-radius:6px 0 0 6px}
.cbar-fill.none{background:var(--none)}.cbar-fill.dump{background:var(--dump)}
.cbar-fill.curate{background:var(--accent)}
.cbar-score{font-family:var(--font-mono);font-size:14px;font-weight:640;
font-variant-numeric:tabular-nums;color:var(--ink);min-width:52px;text-align:right}
.cbar-note{font-size:13px;color:var(--ink-soft);margin:16px 0 0;padding-top:14px;
border-top:1px dashed var(--line-strong)}.cbar-note strong{color:var(--ink)}
.table-scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:560px;font-size:14px}
thead th{font-family:var(--font-mono);font-size:10.5px;letter-spacing:.07em;
text-transform:uppercase;color:var(--ink-faint);font-weight:600;text-align:right;
padding:0 14px 12px;border-bottom:1px solid var(--line-strong)}
thead th:first-child{text-align:left}
tbody td{padding:12px 14px;text-align:right;border-bottom:1px solid var(--line);
font-variant-numeric:tabular-nums;font-family:var(--font-mono);color:var(--ink-soft)}
tbody td:first-child{text-align:left;color:var(--ink)}tbody tr:last-child td{border-bottom:none}
tbody tr.latest td{background:var(--accent-soft)}
tbody tr.latest td:first-child{font-weight:640;color:var(--ink)}
td .red{color:var(--accent-ink);font-weight:640}td .ok{color:var(--good);font-weight:640}
.notes{margin-top:26px}.notes h2{font-size:15px;margin:0 0 16px}
.note-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:16px 26px}
.note-item .k{font-family:var(--font-mono);font-size:11px;letter-spacing:.06em;
text-transform:uppercase;color:var(--accent-ink);margin:0 0 6px}
.note-item p{margin:0;font-size:13.5px;color:var(--ink-soft);line-height:1.6}
.note-footnote{margin-top:14px;font-size:12.5px;color:var(--ink-faint)}
footer{margin-top:30px;padding-top:18px;border-top:1px solid var(--line);
font-family:var(--font-mono);font-size:11.5px;color:var(--ink-faint);display:flex;
justify-content:space-between;gap:16px;flex-wrap:wrap}
@media(max-width:720px){.report-head{grid-template-columns:1fr}.verdict{justify-self:stretch}
.stats{grid-template-columns:repeat(2,1fr)}.note-grid{grid-template-columns:1fr}}
"""


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(runs: list[dict[str, Any]]) -> str:
    """Render the full self-contained ``<!doctype html>`` value report.

    Args:
        runs: The history log's ``runs[]`` list (integers/floats/strings only --
            never memory content).

    Returns:
        The complete HTML document as a string.
    """
    latest = runs[-1]
    label, tone, note = compute_verdict(runs)
    store_display = _display_store(str(latest.get("store_path", "your store")))
    span = (
        f"{_date_label(str(runs[0].get('timestamp', '')))} to "
        f"{_date_label(str(runs[-1].get('timestamp', '')))}"
        if len(runs) > 1
        else _date_label(str(runs[0].get("timestamp", "")))
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    check = (
        '<svg viewBox="0 0 24 24" fill="none" aria-hidden="true" style="width:22px;height:22px;'
        'vertical-align:-4px"><circle cx="12" cy="12" r="11" fill="currentColor" opacity=".14">'
        '</circle><path d="M7 12.5l3.2 3.2L17 8.5" stroke="currentColor" stroke-width="2.4" '
        'stroke-linecap="round" stroke-linejoin="round"></path></svg> '
    )
    badge_icon = check if tone == "good" else ""
    footnote = _estimator_footnote(runs)
    retrieval_footnote = _retrieval_footnote(runs)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Is Tulving helping? -- Value Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="page"><div class="wrap">

  <header class="report-head">
    <div>
      <p class="eyebrow">Tulving &middot; Value Report</p>
      <h1>Is Tulving helping?</h1>
      <p class="sub">Measured against <code>{store_display}</code>, read-only, across
         {len(runs)} run{"s" if len(runs) != 1 else ""} ({span}).</p>
    </div>
    <div class="verdict {tone}">
      <p class="q">Verdict</p>
      <p class="a">{badge_icon}{html.escape(label)}</p>
      <p class="note">{html.escape(note)}</p>
    </div>
  </header>

  <section class="stats" aria-label="Latest run summary">{stat_cards(runs)}
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>Context cost as your store grows</h2>
      <p class="desc">Dumping the whole store gets more expensive as it grows;
         <code>curate()</code> returns a budgeted slice that stays flat. The gap is
         what Tulving saves you.</p>
      <div class="legend">
        <span class="item"><span class="swatch dump"></span>Dump everything</span>
        <span class="item"><span class="swatch curate"></span>Tulving curate()</span>
      </div>
    </div>
    {svg_chart(runs)}
  </section>

  <section class="panel">
    <div class="panel-head">
      <h2>The savings compound over time</h2>
      <p class="desc">Reduction = dump tokens &divide; curate tokens. As the store
         grows and curate stays budgeted, the win gets bigger every run.</p>
    </div>
    {reduction_bars(runs)}
  </section>

  {correctness_panel(runs)}

  <section class="panel">
    <div class="panel-head"><h2>Run history</h2>
      <p class="desc">One dated row per run. Twice a month builds the trend above.</p></div>
    <div class="table-scroll"><table>
      <thead><tr><th>Run date</th><th>Store size</th><th>Dump tokens</th>
      <th>Curate tokens</th><th>Reduction</th><th>Correctness</th>
      <th>Retrieval</th></tr></thead>
      <tbody>{table_rows(runs)}</tbody>
    </table></div>
  </section>

  <section class="notes">
    <h2>How to read this</h2>
    <div class="note-grid">
      <div class="note-item"><p class="k">Where the numbers come from</p>
        <p><strong>Dump</strong> = tokens to put your whole store in the prompt.
           <strong>Curate</strong> = tokens <code>curate()</code> returns within its
           budget. <strong>Reduction</strong> = dump &divide; curate.</p></div>
      <div class="note-item"><p class="k">What "helping" looks like</p>
        <p>Curate correctness ties the full dump and both beat no-memory, while
           curate tokens stay flat as dump climbs. A curate dip means the budget is
           too tight.</p></div>
    </div>
    {footnote}
    {retrieval_footnote}
  </section>

  <footer>
    <span>Tulving -- external typed memory for AI agents</span>
    <span>Generated {generated} &middot; from eval history</span>
  </footer>

</div></div>
</body>
</html>"""

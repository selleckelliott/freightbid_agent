"""Visualize the Phase 6.5 compiled-vs-orchestrated capstone as a two-panel figure.

Reads ``benchmarks/compiled_dispatcher_stress_summary.json`` (decision quality + safety) and, when
present, ``benchmarks/context_cost_summary.json`` (cost/context), and renders for every world:

* **Collectible-profit regret, compiled vs source (%)** — the headline, colored PASS / WATCH / FAIL
  with the pass/watch reference bands shaded. Positive = the compiled model leaves money on the
  table relative to the source engine; a bar can still be FAIL while small if the world carries a
  safety-critical miss (the verdict color is the truth, the bands are just reference).
* **Agreement breakdown** — action / approval / warning agreement per world, which exposes *where*
  the compiled model diverges (e.g. warning agreement collapsing in tight-margin worlds even while
  action agreement holds) — i.e. why the source engine must stay authoritative there.

A footer ties in the cost/context benchmark: source-engine calls replaced per decision, decision
payload reduction, and rule-change recompile time.

    python -m benchmarks.run_compiled_dispatcher_stress --days 21
    python -m benchmarks.run_context_cost_benchmark --days 21
    python -m benchmarks.chart_compiled_dispatcher_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "compiled_dispatcher_stress_summary.json"
DEFAULT_COST = ROOT / "benchmarks" / "context_cost_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "compiled_dispatcher_results.png"

PASS = "PASS"
WATCH = "WATCH"
FAIL = "FAIL"
VERDICT_COLOR = {PASS: "#3fb950", WATCH: "#d29922", FAIL: "#f85149"}

ACTION_COLOR = "#58a6ff"
APPROVAL_COLOR = "#bc8cff"
WARNING_COLOR = "#39c5cf"
GREY = "#8b949e"


def _theme() -> None:
    matplotlib.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#c9d1d9",
        "grid.color": "#21262d",
        "text.color": "#c9d1d9",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def _regret_panel(ax, rows: List[Dict[str, Any]], pass_pct: float, watch_pct: float) -> None:
    y = list(range(len(rows)))
    # Reference bands: pass zone (<= pass_pct), watch zone (pass..watch), fail zone (> watch).
    ax.axvspan(-1e4, pass_pct, color=VERDICT_COLOR[PASS], alpha=0.06, zorder=0)
    ax.axvspan(pass_pct, watch_pct, color=VERDICT_COLOR[WATCH], alpha=0.07, zorder=0)
    ax.axvspan(watch_pct, 1e4, color=VERDICT_COLOR[FAIL], alpha=0.06, zorder=0)
    for yi, r in zip(y, rows):
        pct = r["regret_pct"]
        color = VERDICT_COLOR.get(r["verdict"], GREY)
        ax.barh(yi, pct, color=color, alpha=0.9, height=0.6, zorder=2,
                edgecolor="#0d1117", linewidth=0.6)
        ax.text(pct + (0.5 if pct >= 0 else -0.5), yi, f"{pct:+.1f}%",
                va="center", ha="left" if pct >= 0 else "right",
                fontsize=8, color="#c9d1d9")
    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    ax.axvline(pass_pct, color=VERDICT_COLOR[PASS], lw=1.0, ls=":", zorder=1)
    ax.axvline(watch_pct, color=VERDICT_COLOR[FAIL], lw=1.0, ls=":", zorder=1)
    xs = [r["regret_pct"] for r in rows] + [0.0, watch_pct]
    lo, hi = min(xs), max(xs)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.16 * span - 2.0, hi + 0.12 * span + 3.0)
    ax.set_title("Collectible-profit regret: compiled vs source")
    ax.set_xlabel("regret vs source engine (%)   ← better")
    ax.grid(axis="x", alpha=0.4)


def _agreement_panel(ax, rows: List[Dict[str, Any]]) -> None:
    h = 0.26
    for i, r in enumerate(rows):
        triples = (
            (r["action_agreement"], ACTION_COLOR, -h),
            (r["approval_agreement"], APPROVAL_COLOR, 0.0),
            (r["warning_agreement"], WARNING_COLOR, h),
        )
        for val, color, yo in triples:
            ax.barh(i + yo, val, height=h, color=color, alpha=0.9, zorder=2,
                    edgecolor="#0d1117", linewidth=0.4)
        ax.text(1.005, i + h, f"{r['warning_agreement']:.2f}", va="center", ha="left",
                fontsize=7, color=GREY)
    ax.axvline(1.0, color=GREY, lw=1.0, ls="--", zorder=1)
    ax.set_xlim(0.0, 1.16)
    ax.set_title("Agreement with source (action · approval · warning)")
    ax.set_xlabel("fraction of served loads in agreement   better →")
    ax.grid(axis="x", alpha=0.4)


def _label(r: Dict[str, Any]) -> str:
    lens = "base" if r.get("lens") in ("reference", None) else r["lens"]
    crit = r.get("safety_critical_misses", 0)
    flag = f"  ⚠{crit}" if crit else ""
    return f"{r['name']}  [{lens}]{flag}"


def _cost_footer(cost: Optional[Dict[str, Any]]) -> Optional[str]:
    if not cost:
        return None
    calls = cost.get("engine_calls_per_decision", {})
    pay = cost.get("decision_payload_bytes", {})
    rc = cost.get("recompile_test", {})
    ctx = cost.get("context_width", {})
    return (
        f"Cost/context (baseline world):  "
        f"{calls.get('calls_avoided_per_decision', '?')} source-engine port calls/decision replaced "
        f"by 1 predict   ·   decision payload {pay.get('reduction_x', '?')}x smaller "
        f"({pay.get('source_decision_row_mean', '?'):.0f}B→{pay.get('compiled_runtime_json_mean', '?'):.0f}B)"
        f"   ·   {ctx.get('compiled_feature_manifest_fields', '?')}-field context vs "
        f"{ctx.get('source_workflow_nodes', '?')}-node workflow"
        f"   ·   rule-change recompile ~{rc.get('time_to_recompile_seconds', '?'):.0f}s   ·   "
        f"wall-clock ~parity (in-memory engine)"
    )


def build_chart(data: Dict[str, Any], cost: Optional[Dict[str, Any]], out_png: Path) -> None:
    rows = list(data["conditions"])
    _theme()

    n = len(rows)
    height = max(4.8, 0.58 * n + 2.2)
    fig, (ax_reg, ax_agree) = plt.subplots(1, 2, figsize=(15.5, height), sharey=True)

    cfg = data.get("config", {})
    labels = [_label(r) for r in rows]
    y = list(range(n))

    pass_pct = float(cfg.get("regret_pass_pct", 2.0))
    watch_pct = float(cfg.get("regret_watch_pct", 5.0))
    trained = cfg.get("trained_on", [])
    n_train = len(trained) if isinstance(trained, list) else "?"

    title = (f"FreightBid Agent — Compiled-vs-Orchestrated Stress (Phase 6.5)   "
             f"({n} worlds · compiled frozen on {n_train} canonical worlds · "
             f"{cfg.get('days', '?')}d · shadow-only)")
    sub = data.get("headline", "")
    fig.suptitle(title, color="#e6edf3", fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.952, sub, ha="center", color=GREY, fontsize=9, wrap=True)
    footer = _cost_footer(cost)
    if footer:
        fig.text(0.5, 0.912, footer, ha="center", color="#9da7b1", fontsize=8.2)

    _regret_panel(ax_reg, rows, pass_pct, watch_pct)
    _agreement_panel(ax_agree, rows)

    for ax in (ax_reg, ax_agree):
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.invert_yaxis()  # baseline on top

    verdict_legend = [
        Patch(facecolor=VERDICT_COLOR[PASS], label=f"PASS (regret ≤ {pass_pct:g}%, no crit miss)"),
        Patch(facecolor=VERDICT_COLOR[WATCH], label=f"WATCH (≤ {watch_pct:g}% or minor miss)"),
        Patch(facecolor=VERDICT_COLOR[FAIL], label=f"FAIL (> {watch_pct:g}% or safety-critical miss)"),
    ]
    agree_legend = [
        Patch(facecolor=ACTION_COLOR, label="action"),
        Patch(facecolor=APPROVAL_COLOR, label="approval"),
        Patch(facecolor=WARNING_COLOR, label="warning"),
        Patch(facecolor="none", edgecolor="none", label="⚠N = N safety-critical misses"),
    ]
    leg1 = fig.legend(handles=verdict_legend, loc="lower center", ncol=3,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.30, 0.012), title="regret verdict")
    leg1.get_title().set_color(GREY)
    leg2 = fig.legend(handles=agree_legend, loc="lower center", ncol=4,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.74, 0.012), title="agreement breakdown")
    leg2.get_title().set_color(GREY)
    fig.add_artist(leg1)

    bottom = 0.24 if n <= 4 else 0.13
    fig.subplots_adjust(left=0.18, right=0.96, top=0.85, bottom=bottom, wspace=0.07)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Chart saved -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--cost", default=str(DEFAULT_COST))
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG))
    args = p.parse_args()
    path = Path(args.results)
    if not path.exists():
        sys.exit(f"Results not found: {path}\n"
                 f"Run: python -m benchmarks.run_compiled_dispatcher_stress")
    cost_path = Path(args.cost)
    cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else None
    build_chart(json.loads(path.read_text(encoding="utf-8")), cost, Path(args.out_png))


if __name__ == "__main__":
    main()

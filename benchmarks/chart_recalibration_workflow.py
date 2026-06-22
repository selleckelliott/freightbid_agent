"""Visualize the Phase 5.4 recalibration workflow as a two-panel diagnostic.

Reads ``benchmarks/recalibration_workflow_summary.json`` (written by
``run_recalibration_workflow.py``) and renders:

* **Reliability — before vs after** — for each world that drifted (pre severity WATCH/ALERT),
  the eval-window reliability curve of the frozen base model (**dashed**) and of the promoted
  recalibrated model (**solid**), against the ``y = x`` diagonal. The base curve bows below
  the diagonal (over-optimistic); the recalibrated curve is pulled back onto it.
* **ECE before vs after** — per world, the base (pre) and recalibrated (post) eval-window ECE
  as paired bars, with the WATCH / ALERT guide lines. Promoted worlds are marked ``✓``. The
  headline answer: which flagged worlds did the repair layer actually fix?

    python -m benchmarks.run_recalibration_workflow --days 21
    python -m benchmarks.chart_recalibration_workflow
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "recalibration_workflow_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "recalibration_workflow_comparison.png"

OK = "OK"
WATCH = "WATCH"
ALERT = "ALERT"

SEVERITY_COLOR = {OK: "#3fb950", WATCH: "#d29922", ALERT: "#f85149"}
PRE_COLOR = "#f85149"     # base / before
POST_COLOR = "#3fb950"    # recalibrated / after
GREY = "#8b949e"
DIAGONAL = "#6e7681"


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


def _curve(report: Optional[Dict[str, Any]], min_count: int = 1):
    """Bin centers vs observed rate, dropping sparse bins that are just sampling noise."""
    table = (report or {}).get("reliability_table") or []
    pts = [
        (b["mean_predicted"], b["observed_rate"])
        for b in table
        if b.get("count", 0) >= min_count
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return xs, ys


def _min_count(report: Optional[Dict[str, Any]]) -> int:
    """Drop bins holding <0.5% of the window (floor 25) so single-sample spikes vanish."""
    table = (report or {}).get("reliability_table") or []
    total = sum(b.get("count", 0) for b in table)
    return max(25, int(0.005 * total))


def _reliability_panel(ax, rows: List[Dict[str, Any]]) -> None:
    ax.plot([0, 1], [0, 1], ls="--", color=DIAGONAL, lw=1.4, zorder=1)
    drifted = [r for r in rows if r.get("severity_pre") in (WATCH, ALERT)]
    if not drifted:
        drifted = rows
    for r in drifted:
        xs_pre, ys_pre = _curve(r.get("pre"), _min_count(r.get("pre")))
        xs_post, ys_post = _curve(r.get("post"), _min_count(r.get("post")))
        if xs_pre:
            ax.plot(xs_pre, ys_pre, marker="o", ms=2.5, lw=1.4, ls="--",
                    color=PRE_COLOR, alpha=0.7, zorder=2)
        if xs_post:
            ax.plot(xs_post, ys_post, marker="o", ms=2.5, lw=2.0, ls="-",
                    color=POST_COLOR, alpha=0.85, zorder=3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Reliability on eval window — {len(drifted)} drifted worlds\n"
                 f"base over-optimistic (dashed) vs recalibrated (solid)")
    ax.set_xlabel("mean predicted P(win)")
    ax.set_ylabel("observed win rate")
    ax.grid(alpha=0.4)
    legend = [
        Line2D([0], [0], color=DIAGONAL, ls="--", lw=1.4, label="perfect calibration"),
        Line2D([0], [0], color=PRE_COLOR, ls="--", lw=1.6, marker="o", ms=3, label="base (pre)"),
        Line2D([0], [0], color=POST_COLOR, ls="-", lw=2.2, marker="o", ms=3, label="recalibrated (post)"),
    ]
    leg = ax.legend(handles=legend, loc="upper left", facecolor="#161b22",
                    edgecolor="#30363d", fontsize=8)
    for txt in leg.get_texts():
        txt.set_color("#c9d1d9")


def _ece_panel(ax, rows: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> None:
    n = len(rows)
    y = list(range(n))
    h = 0.36
    for yi, r in zip(y, rows):
        ece_pre = r.get("ece_pre") or 0.0
        ece_post = r.get("ece_post")
        pre_color = SEVERITY_COLOR.get(r.get("severity_pre"), GREY)
        ax.barh(yi + h / 2, ece_pre, color=pre_color, alpha=0.85, height=h, zorder=2,
                edgecolor="#0d1117", linewidth=0.5)
        if ece_post is not None:
            ax.barh(yi - h / 2, ece_post, color=POST_COLOR, alpha=0.9, height=h, zorder=2,
                    edgecolor="#0d1117", linewidth=0.5)
            mark = "  ✓" if r.get("promoted") else ""
            ax.text(ece_post + 0.004, yi - h / 2, f"{ece_post:.3f}{mark}", va="center",
                    ha="left", fontsize=7.5, color="#c9d1d9")
        ax.text(ece_pre + 0.004, yi + h / 2, f"{ece_pre:.3f}", va="center", ha="left",
                fontsize=7.5, color=GREY)
    watch = float(thresholds.get("ece_watch", 0.03))
    alert = float(thresholds.get("ece_alert", 0.07))
    ax.axvline(watch, color=SEVERITY_COLOR[WATCH], ls=":", lw=1.3, zorder=1)
    ax.axvline(alert, color=SEVERITY_COLOR[ALERT], ls=":", lw=1.3, zorder=1)
    ax.text(watch, -0.6, " WATCH", color=SEVERITY_COLOR[WATCH], fontsize=7.5, ha="left", va="top")
    ax.text(alert, -0.6, " ALERT", color=SEVERITY_COLOR[ALERT], fontsize=7.5, ha="left", va="top")
    ax.set_title("ECE by world — base (top) vs recalibrated (bottom)")
    ax.set_xlabel("ECE  (lower = better calibrated)   worse →")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.8, n - 0.2)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    rows = list(data["conditions"])
    thresholds = data.get("config", {}).get("thresholds", {})
    _theme()

    n = len(rows)
    height = max(4.5, 0.55 * n + 2.0)
    fig, (ax_rel, ax_ece) = plt.subplots(1, 2, figsize=(15, height))

    cfg = data.get("config", {})
    title = (f"FreightBid Agent — Recalibration Workflow   "
             f"({n} worlds · frozen base · {cfg.get('method', '?')} · "
             f"fit {cfg.get('fit_days', '?')}d / eval {cfg.get('eval_days', '?')}d)")
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.93, data.get("headline", ""), ha="center", color=GREY, fontsize=9.5)

    _reliability_panel(ax_rel, rows)
    _ece_panel(ax_ece, rows, thresholds)

    legend = [
        Patch(facecolor=SEVERITY_COLOR[ALERT], label="base ALERT"),
        Patch(facecolor=SEVERITY_COLOR[WATCH], label="base WATCH"),
        Patch(facecolor=SEVERITY_COLOR[OK], label="base OK"),
        Patch(facecolor=POST_COLOR, label="recalibrated (post)"),
    ]
    leg = fig.legend(handles=legend, loc="lower center", ncol=4, facecolor="#161b22",
                     edgecolor="#30363d", fontsize=9, bbox_to_anchor=(0.5, 0.005))
    for txt in leg.get_texts():
        txt.set_color("#c9d1d9")

    bottom = 0.2 if n <= 4 else 0.12
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=bottom, wspace=0.22)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Chart saved -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--results", default=str(DEFAULT_RESULTS))
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG))
    args = p.parse_args()
    path = Path(args.results)
    if not path.exists():
        sys.exit(f"Results not found: {path}\n"
                 f"Run: python -m benchmarks.run_recalibration_workflow")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

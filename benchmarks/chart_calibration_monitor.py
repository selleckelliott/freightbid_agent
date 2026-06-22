"""Visualize the Phase 5.3 calibration drift monitor as a two-panel diagnostic.

Reads ``benchmarks/calibration_monitor_summary.json`` (written by
``run_calibration_monitor.py``) and renders:

* **Reliability diagram** — for every world, the per-bin mean predicted ``P(win)`` vs the
  observed win rate, against the ``y = x`` perfect-calibration diagonal. The baseline
  (training) world hugs the diagonal; a drifted world bows **below** it (predicted > observed
  = over-optimistic). Curves are colored by severity.
* **ECE by world** — expected calibration error per world as a horizontal bar colored by
  severity (OK / WATCH / ALERT), with the WATCH and ALERT thresholds drawn as guide lines.
  The headline answer: which worlds can the model still be trusted on?

    python -m benchmarks.run_calibration_monitor --days 21
    python -m benchmarks.chart_calibration_monitor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "calibration_monitor_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "calibration_monitor_comparison.png"

OK = "OK"
WATCH = "WATCH"
ALERT = "ALERT"

SEVERITY_COLOR = {
    OK: "#3fb950",
    WATCH: "#d29922",
    ALERT: "#f85149",
}
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


def _reliability_panel(ax, rows: List[Dict[str, Any]]) -> None:
    ax.plot([0, 1], [0, 1], ls="--", color=DIAGONAL, lw=1.4, zorder=1,
            label="perfect calibration")
    for r in rows:
        table = r.get("reliability_table") or []
        if not table:
            continue
        xs = [b["mean_predicted"] for b in table]
        ys = [b["observed_rate"] for b in table]
        is_ref = not r.get("overrides")
        color = GREY if is_ref else SEVERITY_COLOR.get(r["severity"], GREY)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=2.4 if is_ref else 1.6,
                color=color, alpha=0.95 if is_ref else 0.8,
                zorder=3 if is_ref else 2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Reliability — predicted vs observed P(win)")
    ax.set_xlabel("mean predicted P(win)")
    ax.set_ylabel("observed win rate")
    ax.grid(alpha=0.4)


def _ece_panel(ax, rows: List[Dict[str, Any]], thresholds: Dict[str, Any]) -> None:
    y = list(range(len(rows)))
    for yi, r in zip(y, rows):
        ece = r.get("ece") or 0.0
        color = SEVERITY_COLOR.get(r["severity"], GREY)
        ax.barh(yi, ece, color=color, alpha=0.9, height=0.62, zorder=2,
                edgecolor="#0d1117", linewidth=0.6)
        flag = " (low-n)" if r.get("insufficient_data") else ""
        ax.text(ece + 0.004, yi, f"{ece:.3f}{flag}", va="center", ha="left",
                fontsize=8, color="#c9d1d9")
    watch = float(thresholds.get("ece_watch", 0.03))
    alert = float(thresholds.get("ece_alert", 0.07))
    ax.axvline(watch, color=SEVERITY_COLOR[WATCH], ls=":", lw=1.3, zorder=1)
    ax.axvline(alert, color=SEVERITY_COLOR[ALERT], ls=":", lw=1.3, zorder=1)
    ax.text(watch, len(rows) - 0.4, " WATCH", color=SEVERITY_COLOR[WATCH],
            fontsize=7.5, ha="left", va="top")
    ax.text(alert, len(rows) - 0.4, " ALERT", color=SEVERITY_COLOR[ALERT],
            fontsize=7.5, ha="left", va="top")
    ax.set_title("Expected calibration error by world")
    ax.set_xlabel("ECE  (lower = better calibrated)   worse →")
    ax.grid(axis="x", alpha=0.4)


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    rows = list(data["conditions"])
    thresholds = data.get("config", {}).get("thresholds", {})
    _theme()

    n = len(rows)
    height = max(4.5, 0.5 * n + 2.0)
    fig, (ax_rel, ax_ece) = plt.subplots(1, 2, figsize=(15, height))

    cfg = data.get("config", {})
    title = (f"FreightBid Agent — Calibration Drift Monitor   "
             f"({n} worlds · model trained+calibrated once on baseline · {cfg.get('days', '?')}d)")
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.93, data.get("headline", ""), ha="center", color=GREY, fontsize=10)

    _reliability_panel(ax_rel, rows)
    _ece_panel(ax_ece, rows, thresholds)

    labels = [f"{r['name']}  [{r['lens'] if r['lens'] != 'reference' else 'base'}]" for r in rows]
    ax_ece.set_yticks(list(range(n)))
    ax_ece.set_yticklabels(labels, fontsize=8.5)
    ax_ece.set_ylim(-0.7, n - 0.3)
    ax_ece.invert_yaxis()  # baseline on top

    severity_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=SEVERITY_COLOR[k],
               markersize=9, label=k)
        for k in (OK, WATCH, ALERT)
    ]
    severity_legend.append(
        Line2D([0], [0], color=GREY, lw=2.4, label="baseline (reference)")
    )
    leg = fig.legend(handles=severity_legend, loc="lower center", ncol=4,
                     facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                     bbox_to_anchor=(0.5, 0.005), title="severity")
    leg.get_title().set_color(GREY)

    bottom = 0.2 if n <= 4 else 0.13
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
                 f"Run: python -m benchmarks.run_calibration_monitor")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

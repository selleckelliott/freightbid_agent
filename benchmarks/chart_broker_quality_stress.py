"""Visualize the Phase 4.5 broker-quality stress sweep as a two-lens panel.

Reads ``benchmarks/broker_quality_stress_summary.json`` (written by
``run_broker_quality_stress.py``) and renders, for every broker-quality world:

* **EV uplift vs best fixed (%)** — the EV ``target`` policy's oracle-realized
  profit over the best fixed policy, colored by robustness verdict (HOLDS /
  NEUTRAL / REGRESSION) with the +/-1% neutral band shaded. The headline answer:
  does the EV recommender still beat fixed bidding under a degraded broker market?
* **Calibration drift vs baseline** — how far the baseline-trained model's
  predicted P(win) drifts from the world's true (oracle) P(win) on the selected
  bids, colored by which lens the condition stresses. Payment/coverage worlds
  move this without moving the EV verdict — the orthogonality the phase documents.

    python -m benchmarks.run_broker_quality_stress --days 21
    python -m benchmarks.chart_broker_quality_stress
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
DEFAULT_RESULTS = ROOT / "benchmarks" / "broker_quality_stress_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "broker_quality_stress_comparison.png"

HOLDS = "HOLDS"
NEUTRAL = "NEUTRAL"
REGRESSION = "REGRESSION"

VERDICT_COLOR = {
    HOLDS: "#3fb950",
    NEUTRAL: "#d29922",
    REGRESSION: "#f85149",
}
LENS_COLOR = {
    "ev": "#3fb950",
    "calibration": "#58a6ff",
    "both": "#d29922",
    "reference": "#6e7681",
}
LENS_LABEL = {
    "ev": "EV lens",
    "calibration": "calibration lens",
    "both": "both lenses",
    "reference": "reference (baseline)",
}
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


def _ev_panel(ax, rows: List[Dict[str, Any]], band: float) -> None:
    y = list(range(len(rows)))
    ax.axvspan(-band, band, color=GREY, alpha=0.12, zorder=0)
    for yi, r in zip(y, rows):
        pct = r["uplift_pct"]
        color = VERDICT_COLOR.get(r["verdict"], GREY)
        ax.barh(yi, pct, color=color, alpha=0.9, height=0.6, zorder=2,
                edgecolor="#0d1117", linewidth=0.6)
        ax.text(pct + (1.5 if pct >= 0 else -1.5), yi, f"{pct:+.1f}%",
                va="center", ha="left" if pct >= 0 else "right",
                fontsize=8, color="#c9d1d9")
    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    ax.set_title("EV uplift vs best fixed")
    ax.set_xlabel("target realized profit − best fixed (% of best fixed)   better →")
    ax.grid(axis="x", alpha=0.4)


def _calibration_panel(ax, rows: List[Dict[str, Any]]) -> None:
    y = list(range(len(rows)))
    for yi, r in zip(y, rows):
        drift = r.get("calibration_drift_vs_baseline") or 0.0
        color = LENS_COLOR.get(r["lens"], GREY)
        ax.plot([0, drift], [yi, yi], color=color, lw=2.2, alpha=0.85, zorder=2)
        ax.scatter([drift], [yi], color=color, s=46, zorder=3,
                   edgecolor="#0d1117", linewidth=0.8)
        ax.text(drift + (0.002 if drift >= 0 else -0.002), yi, f"{drift:+.3f}",
                va="center", ha="left" if drift >= 0 else "right",
                fontsize=8, color="#c9d1d9")
    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    ax.set_title("Model calibration drift vs baseline")
    ax.set_xlabel("Δ (predicted − true P(win)) vs baseline world")
    ax.grid(axis="x", alpha=0.4)


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    rows = list(data["conditions"])
    _theme()

    n = len(rows)
    height = max(4.5, 0.5 * n + 2.0)
    fig, (ax_ev, ax_cal) = plt.subplots(1, 2, figsize=(15, height), sharey=True)

    cfg = data.get("config", {})
    tally = data.get("tally", {})
    labels = [f"{r['name']}  [{r['lens'] if r['lens'] != 'reference' else 'base'}]" for r in rows]
    y = list(range(n))

    title = (f"FreightBid Agent — Broker-Quality Stress Test   "
             f"({n} worlds · model trained once on baseline · {cfg.get('days', '?')}d)")
    sub = data.get("headline", "")
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.925, sub, ha="center", color=GREY, fontsize=10)

    _ev_panel(ax_ev, rows, float(cfg.get("uplift_band_pct", 1.0)))
    _calibration_panel(ax_cal, rows)

    for ax in (ax_ev, ax_cal):
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.invert_yaxis()  # baseline on top

    verdict_legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=VERDICT_COLOR[HOLDS],
               markersize=9, label="HOLDS"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=VERDICT_COLOR[NEUTRAL],
               markersize=9, label="neutral"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=VERDICT_COLOR[REGRESSION],
               markersize=9, label="regression"),
    ]
    lens_legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LENS_COLOR[k],
               markersize=9, label=LENS_LABEL[k])
        for k in ("ev", "calibration", "both")
    ]
    leg1 = fig.legend(handles=verdict_legend, loc="lower center", ncol=3,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.30, 0.005), title="EV verdict")
    leg1.get_title().set_color(GREY)
    leg2 = fig.legend(handles=lens_legend, loc="lower center", ncol=3,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.74, 0.005), title="calibration lens")
    leg2.get_title().set_color(GREY)
    fig.add_artist(leg1)

    bottom = 0.22 if n <= 4 else 0.13
    fig.subplots_adjust(left=0.15, right=0.97, top=0.87, bottom=bottom, wspace=0.08)
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
                 f"Run: python -m benchmarks.run_broker_quality_stress")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

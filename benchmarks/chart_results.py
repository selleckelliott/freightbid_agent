"""Visualize benchmark results from run_scenarios.py output.

Workflow
--------
    # Step 1 – generate the JSON (skip if already done)
    python -m benchmarks.run_scenarios --out benchmarks/results.json

    # Step 2 – chart it
    python -m benchmarks.chart_results --results benchmarks/results.json

    # Or do both in one shot (slower – re-runs all scenarios)
    python -m benchmarks.chart_results

The script saves a PNG to --out-png (default: benchmarks/results_chart.png)
and optionally opens the chart in an interactive window (--show).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_JSON = ROOT / "benchmarks" / "results.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "results_chart.png"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        sys.exit(
            f"Results file not found: {path}\n"
            "Generate it with:\n"
            "  python -m benchmarks.run_scenarios --out benchmarks/results.json"
        )
    return json.loads(path.read_text())


def _extract(results: List[Dict[str, Any]]):
    feasibility    = [r["feasibility_rate"] for r in results]
    elapsed_ms     = [r["elapsed_ms"] for r in results]
    loads_in       = [r["n_loads_in"] for r in results]
    loads_ranked   = [r["n_loads_ranked"] for r in results]
    best_scores    = [r["best_score"]  for r in results if r["best_score"]  is not None]
    best_profits   = [r["best_profit"] for r in results if r["best_profit"] is not None]

    # Per-truck-origin averages
    origin_feas: Dict[str, List[float]] = {}
    for r in results:
        city = r["truck_origin"].split(",")[0].strip()
        origin_feas.setdefault(city, []).append(r["feasibility_rate"])
    origin_avg = {k: float(np.mean(v)) for k, v in origin_feas.items()}
    top_origins = sorted(origin_avg.items(), key=lambda x: -x[1])[:12]

    return feasibility, elapsed_ms, loads_in, loads_ranked, best_scores, best_profits, top_origins


# ---------------------------------------------------------------------------
# Chart builder
# ---------------------------------------------------------------------------

def build_chart(results: List[Dict[str, Any]], out_png: Path, show: bool) -> None:
    feasibility, elapsed_ms, loads_in, loads_ranked, best_scores, best_profits, top_origins = (
        _extract(results)
    )

    matplotlib.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor":   "#161b22",
        "axes.edgecolor":   "#30363d",
        "axes.labelcolor":  "#c9d1d9",
        "axes.titlecolor":  "#e6edf3",
        "xtick.color":      "#8b949e",
        "ytick.color":      "#8b949e",
        "grid.color":       "#21262d",
        "text.color":       "#c9d1d9",
        "figure.titlesize": 13,
        "axes.titlesize":   10,
        "axes.labelsize":   8,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
    })

    GREEN  = "#3fb950"
    BLUE   = "#58a6ff"
    ORANGE = "#f0883e"
    PURPLE = "#bc8cff"
    RED    = "#f85149"
    TEAL   = "#39d353"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"FreightBid Agent — Benchmark Results  ({len(results):,} scenarios)",
        color="#e6edf3", fontsize=14, fontweight="bold", y=0.98,
    )

    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32,
                          left=0.06, right=0.97, top=0.92, bottom=0.09)

    # --- (0,0) Feasibility rate histogram -----------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    no_feas = sum(1 for f in feasibility if f == 0)
    ax1.hist(feasibility, bins=25, color=GREEN, alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
    ax1.axvline(float(np.mean(feasibility)), color=ORANGE, linewidth=1.4, linestyle="--",
                label=f"mean {np.mean(feasibility):.2f}")
    ax1.set_title("Feasibility Rate Distribution")
    ax1.set_xlabel("Feasibility Rate (ranked / ingested)")
    ax1.set_ylabel("Scenarios")
    ax1.legend(fontsize=7, framealpha=0.3)
    ax1.text(0.97, 0.92, f"{no_feas} scenarios\nw/ 0 feasible",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=7, color=RED)
    ax1.grid(axis="y", alpha=0.4)
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- (0,1) Best score histogram -----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(best_scores, bins=30, color=BLUE, alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
    ax2.axvline(float(np.mean(best_scores)), color=ORANGE, linewidth=1.4, linestyle="--",
                label=f"mean {np.mean(best_scores):.1f}")
    ax2.axvline(float(np.median(best_scores)), color=PURPLE, linewidth=1.4, linestyle=":",
                label=f"median {np.median(best_scores):.1f}")
    ax2.set_title("Best Score Distribution")
    ax2.set_xlabel("Score (top-ranked load)")
    ax2.set_ylabel("Scenarios")
    ax2.legend(fontsize=7, framealpha=0.3)
    ax2.grid(axis="y", alpha=0.4)

    # --- (0,2) Best expected profit histogram --------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(best_profits, bins=30, color=TEAL, alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
    ax3.axvline(float(np.mean(best_profits)), color=ORANGE, linewidth=1.4, linestyle="--",
                label=f"mean ${np.mean(best_profits):.0f}")
    ax3.axvline(float(np.median(best_profits)), color=PURPLE, linewidth=1.4, linestyle=":",
                label=f"median ${np.median(best_profits):.0f}")
    ax3.set_title("Best Expected Profit Distribution")
    ax3.set_xlabel("Expected Profit ($)")
    ax3.set_ylabel("Scenarios")
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.legend(fontsize=7, framealpha=0.3)
    ax3.grid(axis="y", alpha=0.4)

    # --- (1,0) Elapsed ms histogram ------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(elapsed_ms, bins=40, color=PURPLE, alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
    ax4.axvline(float(np.mean(elapsed_ms)), color=ORANGE, linewidth=1.4, linestyle="--",
                label=f"mean {np.mean(elapsed_ms):.2f} ms")
    ax4.set_title("Per-Scenario Latency")
    ax4.set_xlabel("Elapsed Time (ms)")
    ax4.set_ylabel("Scenarios")
    ax4.legend(fontsize=7, framealpha=0.3)
    ax4.grid(axis="y", alpha=0.4)

    # --- (1,1) Loads in vs loads ranked scatter ------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    sc = ax5.scatter(
        loads_in, loads_ranked,
        c=feasibility, cmap="YlGn", alpha=0.55, s=10,
        vmin=0, vmax=max(feasibility) if feasibility else 1,
    )
    cbar = fig.colorbar(sc, ax=ax5, pad=0.02)
    cbar.ax.yaxis.set_tick_params(colors="#8b949e")
    cbar.set_label("Feasibility Rate", fontsize=7, color="#8b949e")
    # diagonal reference line
    mx = max(loads_in)
    ax5.plot([0, mx], [0, mx], color="#30363d", linewidth=0.8, linestyle="--")
    ax5.set_title("Loads Ingested vs Loads Ranked")
    ax5.set_xlabel("Loads Ingested")
    ax5.set_ylabel("Loads Ranked (feasible)")
    ax5.grid(alpha=0.3)

    # --- (1,2) Top truck-origin cities by avg feasibility --------------------
    ax6 = fig.add_subplot(gs[1, 2])
    cities = [t[0] for t in top_origins]
    avgs   = [t[1] for t in top_origins]
    colors = [GREEN if a >= np.mean(feasibility) else ORANGE for a in avgs]
    bars = ax6.barh(cities, avgs, color=colors, alpha=0.85, edgecolor="#0d1117", linewidth=0.4)
    ax6.axvline(float(np.mean(feasibility)), color="#8b949e", linewidth=1.0,
                linestyle="--", label=f"overall mean {np.mean(feasibility):.2f}")
    ax6.set_title("Top Truck Origins by Avg Feasibility")
    ax6.set_xlabel("Avg Feasibility Rate")
    ax6.invert_yaxis()
    ax6.legend(fontsize=7, framealpha=0.3)
    ax6.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax6.grid(axis="x", alpha=0.4)
    for bar, val in zip(bars, avgs):
        ax6.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0%}", va="center", fontsize=6, color="#c9d1d9")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved → {out_png}")

    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(DEFAULT_RESULTS_JSON),
                   help="Path to results.json from run_scenarios.py (default: benchmarks/results.json)")
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG),
                   help="Where to save the chart PNG (default: benchmarks/results_chart.png)")
    p.add_argument("--show", action="store_true",
                   help="Open the chart in an interactive window after saving.")
    args = p.parse_args()

    results = _load_results(Path(args.results))
    build_chart(results, Path(args.out_png), args.show)


if __name__ == "__main__":
    main()

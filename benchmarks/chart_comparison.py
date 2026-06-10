"""Visualize the three-way planner comparison dashboard.

Reads the JSON written by ``compare_planners.py`` (``--out``) and renders a
grouped-bar dashboard, one panel per metric, since the metrics live on very
different scales (a feasibility fraction, dollars, miles, load counts and
milliseconds can't share a single axis).

The first planner in the JSON (the heuristic) is treated as the baseline:
every other planner gets a "vs baseline" delta annotation under each panel.

Workflow
--------
    # Step 1 - generate the JSON (slower; re-runs every planner over all scenarios)
    python -m benchmarks.compare_planners --time-limit 0.2 --out benchmarks/compare_results.json

    # Step 2 - chart it
    python -m benchmarks.chart_comparison --results benchmarks/compare_results.json

The script saves a PNG to --out-png (default: benchmarks/compare_chart.png)
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_JSON = ROOT / "benchmarks" / "compare_results.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "compare_chart.png"

GREEN = "#3fb950"
RED = "#f85149"
GREY = "#8b949e"

# Stable colors per planner key; unknown keys fall back to the spare palette.
PLANNER_COLORS = {
    "heuristic": "#58a6ff",
    "ortools_distance": "#f0883e",
    "ortools_profit_aware": "#3fb950",
}
FALLBACK_COLORS = ["#bc8cff", "#39c5cf", "#d29922", "#ff7b72"]

# (json_key, panel title, value formatter, "lower is better")
_PANELS = [
    ("feasible_rate", "Feasible Rate", lambda v: f"{v:.1%}", False),
    ("avg_profit", "Avg Profit / Scenario", lambda v: f"${v:,.0f}", False),
    ("avg_deadhead_miles", "Avg Deadhead Miles", lambda v: f"{v:.1f} mi", True),
    ("avg_loads_selected", "Avg Loads Selected", lambda v: f"{v:.2f}", False),
    ("avg_runtime_ms", "Avg Runtime (ms, log)", lambda v: f"{v:.2f} ms", True),
]


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        sys.exit(
            f"Results file not found: {path}\n"
            "Generate it with:\n"
            "  python -m benchmarks.compare_planners --out benchmarks/compare_results.json"
        )
    data = json.loads(path.read_text())
    if "planners" not in data:
        sys.exit(
            "Results file uses the old two-planner schema. Regenerate it with:\n"
            "  python -m benchmarks.compare_planners --out benchmarks/compare_results.json"
        )
    return data


def _short_label(label: str) -> str:
    return label.replace("OR-Tools ", "")


def _delta_label(base: float, val: float, lower_is_better: bool) -> tuple[str, str]:
    """Return (text, color) describing one planner's change vs the baseline."""
    if base == 0:
        return "n/a", GREY
    pct = (val - base) / abs(base) * 100.0
    if abs(pct) < 0.05:
        return "±0.0%", GREY
    improved = (pct < 0) if lower_is_better else (pct > 0)
    color = GREEN if improved else RED
    if abs(pct) >= 1000:  # absurd percentages read better as multipliers
        return f"{val / base:,.0f}x", color
    return f"{pct:+.1f}%", color


def build_chart(data: Dict[str, Any], out_png: Path, show: bool) -> None:
    planners: List[Dict[str, Any]] = data["planners"]
    baseline = planners[0]
    scenarios = data.get("scenarios", 0)
    time_limit = data.get("ortools_time_limit_s")

    colors = []
    spare = iter(FALLBACK_COLORS)
    for p in planners:
        colors.append(PLANNER_COLORS.get(p["key"], next(spare, GREY)))

    matplotlib.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "axes.titlecolor": "#e6edf3",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "text.color": "#c9d1d9",
        "axes.titlesize": 11,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7,
    })

    fig = plt.figure(figsize=(17, 7))
    title = f"FreightBid Agent - Planner Comparison  ({scenarios:,} scenarios"
    if time_limit is not None:
        title += f", OR-Tools {time_limit:g}s/solve"
    title += ")"
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.99)

    gs = fig.add_gridspec(1, len(_PANELS), wspace=0.42,
                          left=0.05, right=0.98, top=0.84, bottom=0.20)

    bar_labels = [_short_label(p["label"]) for p in planners]

    for col, (key, panel_title, fmt, lower_is_better) in enumerate(_PANELS):
        ax = fig.add_subplot(gs[0, col])
        values = [float(p["metrics"][key]) for p in planners]

        log = key == "avg_runtime_ms"
        if log:
            ax.set_yscale("log")

        bars = ax.bar(bar_labels, values, color=colors, alpha=0.9,
                      edgecolor="#0d1117", linewidth=0.6, width=0.62)

        ax.set_title(panel_title)
        ax.grid(axis="y", alpha=0.4)
        ax.tick_params(axis="x", rotation=12)
        if not log:
            ax.set_ylim(0, max(values) * 1.28 if max(values) > 0 else 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    fmt(val), ha="center", va="bottom",
                    fontsize=7.5, color="#e6edf3", fontweight="bold")

        # Delta lines: each non-baseline planner vs the baseline.
        base_val = float(baseline["metrics"][key])
        for row, p in enumerate(planners[1:]):
            text, color = _delta_label(
                base_val, float(p["metrics"][key]), lower_is_better)
            ax.text(0.5, -0.20 - 0.085 * row,
                    f"{_short_label(p['label'])}: {text} vs {baseline['label']}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7, color=color, fontweight="bold")
        if lower_is_better:
            ax.text(0.5, -0.20 - 0.085 * (len(planners) - 1),
                    "(lower is better)", transform=ax.transAxes,
                    ha="center", va="top", fontsize=6, color=GREY)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Chart saved -> {out_png}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(DEFAULT_RESULTS_JSON),
                   help="Path to compare_results.json (default: benchmarks/compare_results.json)")
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG),
                   help="Where to save the chart PNG (default: benchmarks/compare_chart.png)")
    p.add_argument("--show", action="store_true",
                   help="Open the chart in an interactive window after saving.")
    args = p.parse_args()

    data = _load(Path(args.results))
    build_chart(data, Path(args.out_png), args.show)


if __name__ == "__main__":
    main()

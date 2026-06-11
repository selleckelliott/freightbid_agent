"""Chart the Phase 2.3 objective-tuning results as a Pareto frontier.

Reads the JSON written by ``tune_objective.py`` (``--out``) and renders a
profit-vs-deadhead scatter: every swept config, the Pareto-efficient
staircase, the heuristic / OR-Tools-distance baselines, named profile
labels, and the recommended knee config.

Workflow
--------
    python -m benchmarks.tune_objective --out benchmarks/tuning_results.json
    python -m benchmarks.chart_pareto --results benchmarks/tuning_results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_JSON = ROOT / "benchmarks" / "tuning_results.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "pareto_frontier.png"

GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
GREY = "#8b949e"
PURPLE = "#bc8cff"
CYAN = "#39c5cf"
YELLOW = "#d29922"

# One color per skip-profit floor so the second sweep axis stays readable.
FLOOR_COLORS = [BLUE, CYAN, PURPLE, YELLOW, ORANGE]


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        sys.exit(
            f"Results file not found: {path}\n"
            "Generate it with:\n"
            "  python -m benchmarks.tune_objective --out benchmarks/tuning_results.json"
        )
    return json.loads(path.read_text())


def build_chart(data: Dict[str, Any], out_png: Path, show: bool) -> None:
    configs = data["configs"]
    baselines = {b["key"]: b for b in data["baselines"]}
    min_feasible = data.get("min_feasible_rate", 0.85)

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
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
    })

    fig, ax = plt.subplots(figsize=(11.5, 8))
    title = ("FreightBid Agent - Profit vs Deadhead Pareto Frontier  "
             f"({data['scenario_count']:,} scenarios, "
             f"OR-Tools {data['ortools_time_limit_s']:g}s/solve)")
    fig.suptitle(title, color="#e6edf3", fontsize=13, fontweight="bold")

    floors = sorted({c["params"]["skip_profit_floor_dollars"] for c in configs})
    floor_color = {f: FLOOR_COLORS[i % len(FLOOR_COLORS)]
                   for i, f in enumerate(floors)}

    # All swept configs, colored by floor; infeasible ones hollow.
    for floor in floors:
        group = [c for c in configs
                 if c["params"]["skip_profit_floor_dollars"] == floor]
        ok = [c for c in group if c["metrics"]["feasible_rate"] >= min_feasible]
        low = [c for c in group if c["metrics"]["feasible_rate"] < min_feasible]
        if ok:
            ax.scatter(
                [c["metrics"]["avg_deadhead_miles"] for c in ok],
                [c["metrics"]["avg_profit"] for c in ok],
                s=52, color=floor_color[floor], alpha=0.9, zorder=3,
                label=f"floor ${floor:g}",
            )
        if low:
            ax.scatter(
                [c["metrics"]["avg_deadhead_miles"] for c in low],
                [c["metrics"]["avg_profit"] for c in low],
                s=52, facecolors="none", edgecolors=floor_color[floor],
                alpha=0.7, zorder=3,
                label=f"floor ${floor:g} (feasible < {min_feasible:.0%})",
            )

    # Pareto staircase: best achievable profit at each deadhead budget.
    front = sorted(
        (c for c in configs if c["pareto_efficient"]),
        key=lambda c: c["metrics"]["avg_deadhead_miles"],
    )
    if front:
        xs = [c["metrics"]["avg_deadhead_miles"] for c in front]
        ys = [c["metrics"]["avg_profit"] for c in front]
        ax.step(xs, ys, where="post", color=GREEN, linewidth=1.6,
                alpha=0.85, zorder=2, label="Pareto frontier")
        ax.scatter(xs, ys, s=130, facecolors="none", edgecolors=GREEN,
                   linewidths=1.6, zorder=4)

    # Annotate named profiles and the deadhead multiplier of frontier points.
    for c in configs:
        m = c["metrics"]
        if c.get("profile_name"):
            ax.annotate(
                c["profile_name"],
                (m["avg_deadhead_miles"], m["avg_profit"]),
                textcoords="offset points", xytext=(10, 8),
                fontsize=8, color="#e6edf3", fontweight="bold", zorder=5,
            )
        elif c["pareto_efficient"]:
            ax.annotate(
                f"{c['params']['deadhead_cost_multiplier']:g}x",
                (m["avg_deadhead_miles"], m["avg_profit"]),
                textcoords="offset points", xytext=(7, -11),
                fontsize=7, color=GREY, zorder=5,
            )

    rec = next((c for c in configs if c.get("recommended")), None)
    if rec:
        m = rec["metrics"]
        ax.annotate(
            "recommended (knee)",
            (m["avg_deadhead_miles"], m["avg_profit"]),
            textcoords="offset points", xytext=(28, -34),
            fontsize=9, color=GREEN, fontweight="bold", zorder=6,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4),
        )

    # Baselines for context.
    h = baselines["heuristic"]["metrics"]
    ax.scatter([h["avg_deadhead_miles"]], [h["avg_profit"]], marker="*",
               s=320, color="#e6edf3", edgecolors="#0d1117", zorder=5,
               label="Heuristic baseline")

    # Zoom to the decision-relevant band; far-off baselines get an arrow.
    profits = [c["metrics"]["avg_profit"] for c in configs] + [h["avg_profit"]]
    span = max(profits) - min(profits)
    y_lo, y_hi = min(profits) - 0.25 * span, max(profits) + 0.18 * span
    ax.set_ylim(y_lo, y_hi)

    if "ortools_distance" in baselines:
        d = baselines["ortools_distance"]["metrics"]
        if d["avg_profit"] >= y_lo:
            ax.scatter([d["avg_deadhead_miles"]], [d["avg_profit"]], marker="X",
                       s=150, color="#f85149", edgecolors="#0d1117", zorder=5,
                       label="OR-Tools Distance (v1)")
        else:
            ax.annotate(
                f"OR-Tools Distance (v1): ${d['avg_profit']:,.0f} "
                f"@ {d['avg_deadhead_miles']:.1f} mi  (off scale)",
                (d["avg_deadhead_miles"], y_lo),
                textcoords="offset points", xytext=(0, 26),
                fontsize=8, color="#f85149", ha="center", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2),
            )

    ax.set_xlabel("Average deadhead miles per scenario  (lower is better)")
    ax.set_ylabel("Average expected profit per scenario ($)  (higher is better)")
    ax.grid(alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Chart saved -> {out_png}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(DEFAULT_RESULTS_JSON),
                   help="Path to tuning_results.json")
    p.add_argument("--out-png", default=str(DEFAULT_OUT_PNG),
                   help="Where to save the chart PNG")
    p.add_argument("--show", action="store_true",
                   help="Open the chart interactively after saving.")
    args = p.parse_args()

    data = _load(Path(args.results))
    build_chart(data, Path(args.out_png), args.show)


if __name__ == "__main__":
    main()

"""Visualize the rolling-replay A/B (Phase 3.3).

Reads ``benchmarks/rolling_replay_summary.json`` (written by
``run_rolling_replay.py``) and renders a dashboard. Because both planners replay
the *same* world each episode, the most honest views are **paired**: the
per-episode difference (destination-aware − profit-aware) for cumulative profit
and deadhead, which removes world-to-world variance. Two distribution panels give
absolute scale and the profit/deadhead trade-off cloud.

    python -m benchmarks.run_rolling_replay --episodes 200
    python -m benchmarks.chart_rolling_replay
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "rolling_replay_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "rolling_replay_comparison.png"

PROFIT_KEY = "profit_aware"
DEST_KEY = "destination_aware"
PROFIT_COLOR = "#3fb950"
DEST_COLOR = "#bc8cff"
POS = "#3fb950"
NEG = "#f85149"
GREY = "#8b949e"


def _theme() -> None:
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
        "axes.labelsize": 9,
    })


def _paired(profit: List[Dict[str, Any]], dest: List[Dict[str, Any]], key: str):
    """Per-episode (dest − profit) deltas, aligned by world seed."""
    out = []
    for p, d in zip(profit, dest):
        if p["seed"] != d["seed"]:
            continue
        out.append(d[key] - p[key])
    return out


def _delta_hist(ax, deltas: List[float], title: str, lower_is_better: bool,
                unit: str) -> None:
    mean = statistics.fmean(deltas) if deltas else 0.0
    improved = (mean < 0) if lower_is_better else (mean > 0)
    color = POS if improved else NEG
    ax.hist(deltas, bins=24, color=color, alpha=0.65, edgecolor="#0d1117")
    ax.axvline(0, color=GREY, lw=1.2, ls="--")
    ax.axvline(mean, color=color, lw=2.0,
               label=f"mean {mean:+.1f} {unit}")
    ax.set_title(title)
    ax.set_xlabel(f"destination-aware − profit-aware ({unit})")
    ax.set_ylabel("episodes")
    ax.grid(axis="y", alpha=0.4)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=8)
    better = sum(1 for d in deltas if (d < 0) == lower_is_better and d != 0)
    worse = sum(1 for d in deltas if (d < 0) != lower_is_better and d != 0)
    tied = sum(1 for d in deltas if d == 0)
    ax.text(0.02, 0.97,
            f"better {better} · tied {tied} · worse {worse}  (of {len(deltas)})",
            transform=ax.transAxes, va="top", fontsize=7.5, color=GREY)


def _box(ax, profit_vals, dest_vals, title, unit) -> None:
    data = [profit_vals, dest_vals]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Profit-Aware", "Destination-Aware"])
    for patch, c in zip(bp["boxes"], [PROFIT_COLOR, DEST_COLOR]):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for med in bp["medians"]:
        med.set_color("#e6edf3")
    ax.set_title(title)
    ax.set_ylabel(unit)
    ax.grid(axis="y", alpha=0.4)


def _scatter(ax, profit, dest) -> None:
    ax.scatter([r["total_deadhead_miles"] for r in profit],
               [r["total_profit"] for r in profit],
               s=18, color=PROFIT_COLOR, alpha=0.6, label="Profit-Aware")
    ax.scatter([r["total_deadhead_miles"] for r in dest],
               [r["total_profit"] for r in dest],
               s=18, color=DEST_COLOR, alpha=0.6, label="Destination-Aware")
    ax.set_title("Profit vs Deadhead (per episode)")
    ax.set_xlabel("cumulative deadhead miles")
    ax.set_ylabel("cumulative profit ($)")
    ax.grid(alpha=0.35)
    ax.legend(facecolor="#161b22", edgecolor="#30363d", fontsize=8)


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    per = data["per_episode"]
    profit = per[PROFIT_KEY]
    dest = per.get(DEST_KEY, [])
    _theme()

    cfg = data.get("config", {})
    fig = plt.figure(figsize=(15, 9))
    title = (f"FreightBid Agent — Rolling Replay  "
             f"({cfg.get('episode_count', len(profit))} episodes × "
             f"{cfg.get('horizon_days', '?')}d horizon)")
    div = data.get(PROFIT_KEY, {}).get("divergence")
    if div:
        title += f"   ·   policy divergence {div['divergence_rate'] * 100:.1f}%"
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.24,
                          left=0.07, right=0.97, top=0.90, bottom=0.08)

    if dest:
        _delta_hist(fig.add_subplot(gs[0, 0]),
                    _paired(profit, dest, "total_profit"),
                    "Paired Δ cumulative profit", lower_is_better=False, unit="$")
        _delta_hist(fig.add_subplot(gs[0, 1]),
                    _paired(profit, dest, "total_deadhead_miles"),
                    "Paired Δ cumulative deadhead", lower_is_better=True, unit="mi")
        _box(fig.add_subplot(gs[1, 0]),
             [r["total_profit"] for r in profit],
             [r["total_profit"] for r in dest],
             "Cumulative profit distribution", "$")
        _scatter(fig.add_subplot(gs[1, 1]), profit, dest)
    else:
        _box(fig.add_subplot(gs[0, 0]),
             [r["total_profit"] for r in profit], [],
             "Cumulative profit distribution", "$")

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
        sys.exit(f"Results not found: {path}\nRun: python -m benchmarks.run_rolling_replay")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

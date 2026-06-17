"""Visualize the Phase 3.4 stress sweep as a robustness forest plot.

Reads ``benchmarks/stress_test_summary.json`` (written by ``run_stress_test.py``)
and renders, for every condition, the paired destination-aware − profit-aware
effect on cumulative **profit** (left) and **deadhead** (right) as a percentage
with 95% bootstrap confidence intervals. A vertical zero line marks "no effect";
markers are colored by the condition's robustness verdict (HOLDS / NEUTRAL /
REGRESSION). This is the one-glance answer to "does the destination advantage
generalise across markets?".

    python -m benchmarks.run_stress_test --episodes 30
    python -m benchmarks.chart_stress_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "stress_test_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "stress_test_comparison.png"

HOLDS = "HOLDS"
NEUTRAL = "NEUTRAL"
REGRESSION = "REGRESSION"
DEST_SKIPPED = "DEST_SKIPPED"

VERDICT_COLOR = {
    HOLDS: "#3fb950",
    NEUTRAL: "#d29922",
    REGRESSION: "#f85149",
    DEST_SKIPPED: "#6e7681",
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


def _pct_ci(
    paired: Optional[Dict[str, float]], base_mean: Optional[float]
) -> Optional[Tuple[float, float, float]]:
    """Convert a paired absolute delta (mean/ci_low/ci_high) to a percentage of
    the profit-aware baseline mean. Returns ``(pct, pct_low, pct_high)``."""
    if not paired or not base_mean:
        return None
    scale = 100.0 / abs(base_mean)
    return (
        paired["mean"] * scale,
        paired["ci_low"] * scale,
        paired["ci_high"] * scale,
    )


def _panel(
    ax,
    rows: List[Dict[str, Any]],
    key_paired: str,
    base_metric: str,
    title: str,
    good_is_negative: bool,
) -> None:
    names = [r["name"] for r in rows]
    y = list(range(len(rows)))

    for yi, r in zip(y, rows):
        base_mean = (
            r["profit_aware"]["metrics"][base_metric]["mean"]
            if r.get("profit_aware")
            else None
        )
        triple = _pct_ci(r.get(key_paired), base_mean)
        color = VERDICT_COLOR.get(r["verdict"], GREY)
        if triple is None:
            ax.scatter([0], [yi], marker="x", color=GREY, s=30, zorder=3)
            continue
        pct, lo, hi = triple
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, alpha=0.85, zorder=2)
        ax.scatter([pct], [yi], color=color, s=46, zorder=3,
                   edgecolor="#0d1117", linewidth=0.8)

    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.invert_yaxis()  # first condition (baseline) on top
    ax.set_title(title)
    ax.set_xlabel("destination-aware − profit-aware (% of baseline)")
    ax.grid(axis="x", alpha=0.4)
    arrow = "← better" if good_is_negative else "better →"
    ax.text(0.01 if good_is_negative else 0.99, 0.985, arrow,
            transform=ax.transAxes, ha="left" if good_is_negative else "right",
            va="top", fontsize=8, color=GREY)


def _tally(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out = {HOLDS: 0, NEUTRAL: 0, REGRESSION: 0, DEST_SKIPPED: 0}
    for r in rows:
        out[r["verdict"]] = out.get(r["verdict"], 0) + 1
    return out


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    rows = data["conditions"]
    _theme()

    n = len(rows)
    height = max(5.5, 0.5 * n + 2.0)
    fig, (ax_p, ax_d) = plt.subplots(
        1, 2, figsize=(15, height), sharey=True
    )

    cfg = data.get("config", {})
    tally = data.get("tally") or _tally(rows)
    evaluated = n - tally.get(DEST_SKIPPED, 0)
    title = (f"FreightBid Agent — Sequential Policy Stress Test   "
             f"({cfg.get('episodes_per_condition', '?')} episodes/condition · "
             f"{n} conditions)")
    sub = (f"destination-aware advantage HOLDS in {tally.get(HOLDS, 0)}/{evaluated}"
           f"  ·  neutral {tally.get(NEUTRAL, 0)}  ·  regresses {tally.get(REGRESSION, 0)}")
    fig.suptitle(title, color="#e6edf3", fontsize=14, fontweight="bold", y=0.995)
    fig.text(0.5, 0.925, sub, ha="center", color=GREY, fontsize=10)

    _panel(ax_p, rows, "paired_profit", "total_profit",
           "Δ cumulative profit", good_is_negative=False)
    _panel(ax_d, rows, "paired_deadhead", "total_deadhead_miles",
           "Δ cumulative deadhead", good_is_negative=True)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VERDICT_COLOR[HOLDS],
               markersize=9, label="HOLDS"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VERDICT_COLOR[NEUTRAL],
               markersize=9, label="neutral"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VERDICT_COLOR[REGRESSION],
               markersize=9, label="regression"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, facecolor="#161b22",
               edgecolor="#30363d", fontsize=9, bbox_to_anchor=(0.5, 0.005))

    fig.subplots_adjust(left=0.16, right=0.97, top=0.87, bottom=0.13, wspace=0.08)
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
        sys.exit(f"Results not found: {path}\nRun: python -m benchmarks.run_stress_test")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

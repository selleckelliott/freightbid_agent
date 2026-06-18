"""Visualize the Phase 4.3 EV bid recommender evaluation.

Reads ``benchmarks/bid_recommender_summary.json`` (written by
``run_bid_recommender_eval.py``) and renders a four-panel story:

* **Realized profit by policy** — oracle-weighted realized profit for each policy,
  with the clairvoyant oracle's best achievable EV as a reference line.
* **EV-regret vs oracle** — how much each policy leaves on the table versus the
  oracle (lower is better).
* **Selected-bid P(win): model vs oracle** — the model's win probability for the
  bid it actually picks tracks the oracle's true probability (calibration in
  aggregate, not just ranking).
* **Example load EV tradeoff** — for one held-out load, the full ask-vs-P(win)
  (left axis) and ask-vs-EV (right axis) curves, with the four ladder rungs marked.
  This is the visual heart of the method: a higher ask lifts profit-if-won but
  sinks win probability, so expected value peaks in the interior.

    python -m benchmarks.run_bid_recommender_eval --out benchmarks/bid_recommender_summary.json
    python -m benchmarks.chart_bid_recommender
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
DEFAULT_RESULTS = ROOT / "benchmarks" / "bid_recommender_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "bid_recommender_comparison.png"

POLICY_ORDER = [
    "conservative_fixed",
    "posted_rate",
    "stretch_fixed",
    "recommender_max_ev",
    "recommender_target",
]
RECOMMENDER_KEYS = {"recommender_max_ev", "recommender_target"}

# GitHub dark palette.
GREEN = "#3fb950"
GREY = "#6e7681"
BLUE = "#58a6ff"
AMBER = "#d29922"
RED = "#f85149"
FAINT = "#8b949e"

RUNG_COLOR = {
    "conservative": BLUE,
    "target": GREEN,
    "max_ev": AMBER,
    "stretch": RED,
}


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
        "legend.fontsize": 8,
    })


def _ordered_policies(policies: Dict[str, Any]) -> List[str]:
    return [k for k in POLICY_ORDER if k in policies]


def _short_label(policy: Dict[str, Any]) -> str:
    return policy["label"].replace(" (", "\n(")


def _bar_color(key: str) -> str:
    return GREEN if key in RECOMMENDER_KEYS else GREY


def _panel_realized(ax, data: Dict[str, Any]) -> None:
    policies = data["policies"]
    keys = _ordered_policies(policies)
    y = list(range(len(keys)))
    vals = [policies[k]["avg_realized_profit"] for k in keys]
    colors = [_bar_color(k) for k in keys]
    ax.barh(y, vals, color=colors, edgecolor="#0d1117", height=0.66, zorder=2)

    oracle = data.get("oracle_best_avg_ev")
    if oracle:
        ax.axvline(oracle, color=AMBER, lw=1.4, ls="--", zorder=3)
        ax.text(oracle, len(keys) - 0.5, f" oracle best EV ${oracle:,.0f}",
                color=AMBER, fontsize=8, va="center", ha="left")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" ${v:,.0f}", va="center", ha="left", fontsize=8.5,
                color="#e6edf3")

    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(policies[k]) for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("avg oracle-weighted realized profit ($)")
    ax.set_title("Realized profit by policy  (higher better)")
    ax.set_xlim(0, max(vals + [oracle or 0]) * 1.22)
    ax.grid(axis="x", alpha=0.4)


def _panel_regret(ax, data: Dict[str, Any]) -> None:
    policies = data["policies"]
    keys = _ordered_policies(policies)
    y = list(range(len(keys)))
    vals = [policies[k]["avg_ev_regret_vs_oracle"] for k in keys]
    colors = [_bar_color(k) for k in keys]
    ax.barh(y, vals, color=colors, edgecolor="#0d1117", height=0.66, zorder=2)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" ${v:,.0f}", va="center", ha="left", fontsize=8.5,
                color="#e6edf3")
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(policies[k]) for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("avg EV-regret vs oracle ($)")
    ax.set_title("EV-regret vs oracle  (lower better)")
    ax.set_xlim(0, max(vals) * 1.22 if vals else 1)
    ax.grid(axis="x", alpha=0.4)


def _panel_winprob(ax, data: Dict[str, Any]) -> None:
    policies = data["policies"]
    keys = _ordered_policies(policies)
    y = list(range(len(keys)))
    h = 0.36
    model = [policies[k]["avg_model_win_prob"] for k in keys]
    oracle = [policies[k]["avg_oracle_win_prob"] for k in keys]
    ax.barh([yi + h / 2 for yi in y], model, height=h, color=BLUE,
            edgecolor="#0d1117", label="model P(win)", zorder=2)
    ax.barh([yi - h / 2 for yi in y], oracle, height=h, color=AMBER,
            edgecolor="#0d1117", label="oracle P(win)", zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(policies[k]) for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("avg P(win) of the selected bid")
    ax.set_title("Selected-bid win probability: model vs oracle")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", facecolor="#161b22", edgecolor="#30363d")
    ax.grid(axis="x", alpha=0.4)


def _panel_example(ax, data: Dict[str, Any]) -> None:
    ex = data.get("example_load")
    if not ex:
        ax.text(0.5, 0.5, "no example load", ha="center", va="center",
                transform=ax.transAxes, color=FAINT)
        ax.set_axis_off()
        return
    curve = ex["curve"]
    rpm = curve["ask_rpm"]

    # Left axis: win probability.
    ax.plot(rpm, curve["model_win_prob"], color=BLUE, lw=2.0, label="P(win) model")
    ax.plot(rpm, curve["oracle_win_prob"], color=BLUE, lw=1.4, ls="--", alpha=0.7,
            label="P(win) oracle")
    ax.set_ylabel("P(win)", color=BLUE)
    ax.tick_params(axis="y", colors=BLUE)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("ask ($/mi)")

    # Right axis: expected value.
    ax2 = ax.twinx()
    ax2.plot(rpm, curve["model_ev"], color=GREEN, lw=2.0, label="EV model")
    ax2.plot(rpm, curve["oracle_ev"], color=GREEN, lw=1.4, ls="--", alpha=0.7,
             label="EV oracle")
    ax2.set_ylabel("expected value ($)", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.spines["top"].set_visible(False)

    # Mark the ladder rungs on the model-EV curve. Coincident rungs (e.g. when
    # target == max_ev) are grouped so their labels never pile up.
    rec_label = ex.get("recommended_label")
    groups: "dict[float, list]" = {}
    for rung in ex.get("rungs", []):
        groups.setdefault(round(rung["ask_rpm"], 3), []).append(rung)
    for gi, (x, members) in enumerate(groups.items()):
        labels = [m["label"] for m in members]
        is_rec = rec_label in labels
        ev = members[0]["expected_value"]
        color = GREEN if is_rec else RUNG_COLOR.get(labels[0], FAINT)
        ax2.axvline(x, color=color, lw=1.0, ls=":", alpha=0.55, zorder=1)
        ax2.scatter([x], [ev], color=color, s=120 if is_rec else 55, zorder=5,
                    edgecolor="#0d1117", linewidth=1.0, marker="*" if is_rec else "o")
        tag = "/".join(labels) + (" \u2605" if is_rec else "")
        ax2.annotate(tag, (x, ev), textcoords="offset points",
                     xytext=(0, 10 + 13 * (gi % 2)), ha="center", fontsize=7.5,
                     color=color)

    miles = ex.get("loaded_miles")
    title = (f"Example load {ex.get('load_id', '?')} "
             f"({miles:,.0f} mi) — EV tradeoff")
    ax.set_title(title)

    lines = [
        Line2D([0], [0], color=BLUE, lw=2, label="P(win) model"),
        Line2D([0], [0], color=BLUE, lw=1.4, ls="--", label="P(win) oracle"),
        Line2D([0], [0], color=GREEN, lw=2, label="EV model"),
        Line2D([0], [0], color=GREEN, lw=1.4, ls="--", label="EV oracle"),
    ]
    ax.legend(handles=lines, loc="upper right", facecolor="#161b22",
              edgecolor="#30363d", framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    _theme()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    h = data.get("headline", {})
    n = data.get("n_test_loads", "?")
    title = "FreightBid Agent — Expected-Value Bid Recommender"
    sub = (
        f"{n} held-out loads  ·  target realized ${h.get('target_realized_profit', 0):,.0f}"
        f" vs best fixed ${h.get('best_fixed_realized_profit', 0):,.0f}"
        f" ({h.get('target_uplift_pct_vs_best_fixed', 0):+.1f}%)"
        f"  ·  EV-regret ${h.get('target_ev_regret_vs_oracle', 0):,.0f}"
        f" vs oracle best ${data.get('oracle_best_avg_ev', 0):,.0f}"
    ).replace("$", r"\$")  # escape so matplotlib doesn't parse $...$ as mathtext
    fig.suptitle(title, color="#e6edf3", fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.94, sub, ha="center", color=FAINT, fontsize=10)

    _panel_realized(axes[0][0], data)
    _panel_regret(axes[0][1], data)
    _panel_winprob(axes[1][0], data)
    _panel_example(axes[1][1], data)

    fig.subplots_adjust(left=0.13, right=0.93, top=0.88, bottom=0.08,
                        wspace=0.42, hspace=0.32)
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
                 f"Run: python -m benchmarks.run_bid_recommender_eval")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

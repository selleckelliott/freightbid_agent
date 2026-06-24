"""Chart the Phase 8.3 fleet dispatch benchmark (greedy vs fleet-aware).

Reads the committed ``benchmarks/fleet_dispatch_summary.json`` and renders
``benchmarks/fleet_dispatch_comparison.png``: a per-condition grouped view of
greedy vs fleet-aware fleet profit (with bootstrap CIs) on top, and the
paired fleet-aware-minus-greedy profit delta (the coordination value, with its
CI) on the bottom, annotated with each condition's verdict.

    python -m benchmarks.chart_fleet_dispatch
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "benchmarks" / "fleet_dispatch_summary.json"
DEFAULT_OUT = ROOT / "benchmarks" / "fleet_dispatch_comparison.png"

_VERDICT_COLOR = {"HOLDS": "#2e7d32", "NEUTRAL": "#f9a825", "REGRESSION": "#c62828"}


def _err(stat: Dict[str, float]) -> List[float]:
    return [
        max(0.0, stat["mean"] - stat["ci_low"]),
        max(0.0, stat["ci_high"] - stat["mean"]),
    ]


def main() -> None:
    summary = json.loads(DEFAULT_IN.read_text(encoding="utf-8"))
    conditions: List[Dict[str, Any]] = summary["conditions"]
    names = [c["name"] for c in conditions]
    x = range(len(names))

    greedy_mean = [c["greedy"]["metrics"]["total_profit"]["mean"] for c in conditions]
    fleet_mean = [c["fleet_aware"]["metrics"]["total_profit"]["mean"] for c in conditions]
    greedy_err = list(zip(*[_err(c["greedy"]["metrics"]["total_profit"]) for c in conditions]))
    fleet_err = list(zip(*[_err(c["fleet_aware"]["metrics"]["total_profit"]) for c in conditions]))

    paired_mean = [c["paired_profit"]["mean"] for c in conditions]
    paired_err = list(zip(*[_err(c["paired_profit"]) for c in conditions]))
    verdicts = [c["verdict"] for c in conditions]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 8.5), gridspec_kw={"height_ratios": [3, 2]}
    )

    width = 0.38
    ax_top.bar(
        [i - width / 2 for i in x], greedy_mean, width, yerr=greedy_err,
        label="greedy (uncoordinated)", color="#90a4ae", capsize=3,
    )
    ax_top.bar(
        [i + width / 2 for i in x], fleet_mean, width, yerr=fleet_err,
        label="fleet-aware (CP-SAT coordinated)", color="#1565c0", capsize=3,
    )
    cfg = summary.get("config", {})
    ax_top.set_ylabel("Fleet total profit ($)")
    ax_top.set_title(
        f"Phase 8.3 - Fleet dispatch: greedy vs fleet-aware  "
        f"(fleet={cfg.get('fleet_size', '?')}, "
        f"{cfg.get('episode_count', '?')} episodes/condition, "
        f"{cfg.get('horizon_days', '?')}d horizon)"
    )
    ax_top.legend(loc="upper right")
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.set_xticks(list(x))
    ax_top.set_xticklabels([])

    colors = [_VERDICT_COLOR.get(v, "#607d8b") for v in verdicts]
    ax_bot.bar(list(x), paired_mean, 0.6, yerr=paired_err, color=colors, capsize=3)
    ax_bot.axhline(0.0, color="black", linewidth=0.8)
    ax_bot.set_ylabel("Paired profit delta ($)\nfleet-aware - greedy")
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.set_xticks(list(x))
    ax_bot.set_xticklabels(names, rotation=20, ha="right")
    for i, v in zip(x, verdicts):
        ymax = paired_mean[i] + paired_err[1][i]
        ax_bot.annotate(
            v, (i, ymax), textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize=8, color=_VERDICT_COLOR.get(v, "#607d8b"),
        )

    fig.tight_layout()
    fig.savefig(DEFAULT_OUT, dpi=120)
    print(f"Wrote {DEFAULT_OUT}")


if __name__ == "__main__":
    main()

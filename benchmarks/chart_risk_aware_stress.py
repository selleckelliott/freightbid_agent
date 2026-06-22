"""Visualize the Phase 5.5 risk-aware stress sweep as a two-panel capstone figure.

Reads ``benchmarks/risk_aware_stress_summary.json`` (written by
``run_risk_aware_stress.py``) and renders, for every broker-quality world:

* **Collectible-profit uplift, full risk-aware vs raw EV (%)** — the headline verdict,
  colored HOLDS / NEUTRAL / REGRESSION with the +/-1% neutral band shaded. Worlds where the
  Phase 5.4 recalibrator was promoted are flagged. The headline question: does the *full*
  risk-aware stack beat plain EV bidding on realized collectible profit under stress?
* **Where the uplift comes from** — the same gain decomposed into its two Phase 5 levers:
  the **payment-risk** contribution (risk-adjusted EV vs raw EV) and the **recalibration**
  contribution (full vs risk-adjusted EV). This separates "pricing default/pay-delay risk
  helped" from "repairing win-probability drift helped", world by world.

    python -m benchmarks.run_risk_aware_stress --days 21
    python -m benchmarks.chart_risk_aware_stress
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
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "benchmarks" / "risk_aware_stress_summary.json"
DEFAULT_OUT_PNG = ROOT / "benchmarks" / "risk_aware_stress_comparison.png"

HOLDS = "HOLDS"
NEUTRAL = "NEUTRAL"
REGRESSION = "REGRESSION"

VERDICT_COLOR = {
    HOLDS: "#3fb950",
    NEUTRAL: "#d29922",
    REGRESSION: "#f85149",
}
# The two Phase 5 levers the headline uplift decomposes into.
PAYMENT_COLOR = "#58a6ff"   # risk-adjusted EV vs raw EV  (Phase 5.1 + 5.2)
RECAL_COLOR = "#bc8cff"     # full vs risk-adjusted EV    (Phase 5.3 + 5.4)
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


def _headline_panel(ax, rows: List[Dict[str, Any]], band: float) -> None:
    y = list(range(len(rows)))
    ax.axvspan(-band, band, color=GREY, alpha=0.12, zorder=0)
    for yi, r in zip(y, rows):
        pct = r["uplift_vs_raw_ev"]
        color = VERDICT_COLOR.get(r["verdict"], GREY)
        ax.barh(yi, pct, color=color, alpha=0.9, height=0.6, zorder=2,
                edgecolor="#0d1117", linewidth=0.6)
        ax.text(pct + (0.6 if pct >= 0 else -0.6), yi, f"{pct:+.1f}%",
                va="center", ha="left" if pct >= 0 else "right",
                fontsize=8, color="#c9d1d9")
    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    # Pad both ends so value labels clear the spine / y-tick labels.
    xs = [r["uplift_vs_raw_ev"] for r in rows] + [0.0]
    lo, hi = min(xs), max(xs)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.16 * span - 3.0, hi + 0.10 * span + 3.0)
    ax.set_title("Collectible-profit uplift: full risk-aware vs raw EV")
    ax.set_xlabel("realized collectible profit vs raw EV (%)   better →")
    ax.grid(axis="x", alpha=0.4)


def _contribution_panel(ax, rows: List[Dict[str, Any]]) -> None:
    """Decompose each world's gain into payment-risk and recalibration levers."""
    n = len(rows)
    h = 0.38
    for i, r in enumerate(rows):
        pay = r["risk_adj_uplift_vs_raw"]
        recal = r["full_uplift_vs_risk_adj"]
        ax.barh(i - h / 2, pay, height=h, color=PAYMENT_COLOR, alpha=0.9,
                zorder=2, edgecolor="#0d1117", linewidth=0.5)
        ax.barh(i + h / 2, recal, height=h, color=RECAL_COLOR, alpha=0.9,
                zorder=2, edgecolor="#0d1117", linewidth=0.5)
        for val, yo in ((pay, i - h / 2), (recal, i + h / 2)):
            ax.text(val + (0.5 if val >= 0 else -0.5), yo, f"{val:+.1f}",
                    va="center", ha="left" if val >= 0 else "right",
                    fontsize=7, color=GREY)
    ax.axvline(0, color=GREY, lw=1.3, ls="--", zorder=1)
    ax.set_title("Where the uplift comes from (vs the weaker arm)")
    ax.set_xlabel("payment-risk (risk-adj − raw) · recalibration (full − risk-adj)  (%)")
    ax.grid(axis="x", alpha=0.4)


def _label(r: Dict[str, Any]) -> str:
    lens = "base" if r["lens"] == "reference" else r["lens"]
    flag = "  ⟳recal" if r.get("recalibrator_promoted") else ""
    return f"{r['name']}  [{lens}]{flag}"


def build_chart(data: Dict[str, Any], out_png: Path) -> None:
    rows = list(data["conditions"])
    _theme()

    n = len(rows)
    height = max(4.5, 0.55 * n + 2.0)
    fig, (ax_head, ax_contrib) = plt.subplots(1, 2, figsize=(15, height), sharey=True)

    cfg = data.get("config", {})
    labels = [_label(r) for r in rows]
    y = list(range(n))

    title = (f"FreightBid Agent — Risk-Aware Stress Test (Phase 5.5)   "
             f"({n} worlds · models frozen on baseline · {cfg.get('days', '?')}d · "
             f"realized collectible profit)")
    sub = data.get("headline", "")
    fig.suptitle(title, color="#e6edf3", fontsize=13.5, fontweight="bold", y=0.995)
    fig.text(0.5, 0.93, sub, ha="center", color=GREY, fontsize=9.5, wrap=True)

    _headline_panel(ax_head, rows, float(cfg.get("uplift_band_pct", 1.0)))
    _contribution_panel(ax_contrib, rows)

    for ax in (ax_head, ax_contrib):
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.set_ylim(-0.7, n - 0.3)
        ax.invert_yaxis()  # baseline on top

    verdict_legend = [
        Patch(facecolor=VERDICT_COLOR[HOLDS], label="HOLDS (> +1%)"),
        Patch(facecolor=VERDICT_COLOR[NEUTRAL], label="neutral (±1%)"),
        Patch(facecolor=VERDICT_COLOR[REGRESSION], label="regression (< −1%)"),
    ]
    lever_legend = [
        Patch(facecolor=PAYMENT_COLOR, label="payment-risk (risk-adj − raw)"),
        Patch(facecolor=RECAL_COLOR, label="recalibration (full − risk-adj)"),
        Line2D([0], [0], marker=r"$⟳$", color="none", markerfacecolor=GREY,
               markeredgecolor=GREY, markersize=10, label="recalibrator promoted"),
    ]
    leg1 = fig.legend(handles=verdict_legend, loc="lower center", ncol=3,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.30, 0.005), title="full-vs-raw verdict")
    leg1.get_title().set_color(GREY)
    leg2 = fig.legend(handles=lever_legend, loc="lower center", ncol=3,
                      facecolor="#161b22", edgecolor="#30363d", fontsize=9,
                      bbox_to_anchor=(0.73, 0.005), title="uplift levers")
    leg2.get_title().set_color(GREY)
    fig.add_artist(leg1)

    bottom = 0.22 if n <= 4 else 0.14
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=bottom, wspace=0.08)
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
                 f"Run: python -m benchmarks.run_risk_aware_stress")
    build_chart(json.loads(path.read_text(encoding="utf-8")), Path(args.out_png))


if __name__ == "__main__":
    main()

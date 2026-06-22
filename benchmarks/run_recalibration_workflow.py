"""Recalibration workflow sweep (Phase 5.4).

Repairs the win-probability drift the Phase 5.3 monitor flags — *without retraining the base
model*. The calibrated Phase 4.2 winnability model is trained **once on the baseline world**
(exactly the served / Phase 5.3 artifact) and held frozen. For every broker-quality world in
``config/broker_quality_stress.yaml`` the world is carved by calendar day into an early
**fit** window (``fit_days``) and a later, disjoint **eval** window (``eval_days``); a
lightweight post-hoc recalibrator (``ml/calibration/recalibrator.py``) is fit on the fit
window's ``(raw P(win), realized bid_won)`` pairs and judged only on the eval window.

A recalibrator is **promoted** only if, on the held-out eval window, it improves ECE without
raising severity or worsening Brier past ``max_brier_worsening`` (see
``ml.calibration.recalibration_workflow.decide_promotion``) — otherwise the frozen base model
is kept. The reserve / win-curve worlds Phase 5.3 flagged ALERT (``tight_brokers``,
``high_contention``, ``degraded_corner``) should be repaired to WATCH/OK; the baseline and the
payment/coverage worlds need no repair and should not promote.

This is operational recalibration, not retraining, and it changes **no** recommender behavior
(the recalibrated adapter is not wired into the live container here). Writes
``benchmarks/recalibration_workflow_summary.json`` (committed, lean) for the chart + README.

Examples
--------
    # quick smoke (tiny seeded builds, 2 worlds, short windows)
    python -m benchmarks.run_recalibration_workflow --fast

    # canonical sweep
    python -m benchmarks.run_recalibration_workflow --days 21 \
        --out benchmarks/recalibration_workflow_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from benchmarks.run_bid_recommender_eval import _build_frame
from benchmarks.run_broker_quality_stress import Condition, load_conditions, world_cfg
from benchmarks.run_calibration_monitor import _calibrated_model
from ml.calibration.recalibration_workflow import (
    RecalibrationConfig,
    RecalibrationResult,
    load_recalibration_config,
    recalibrate,
    time_split,
)
from ml.config import MLConfig, load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.monitoring.calibration_drift import ALERT, OK, WATCH, CalibrationThresholds
from ml.monitoring.calibration_report import load_calibration_config, report_to_dict
from ml.training.winnability_dataset import LABEL, resolve_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "recalibration.yaml"
DEFAULT_MONITOR_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "recalibration_workflow_summary.json"

# Smoke subset: the baseline reference + one decisive reserve/win-curve drifter.
_FAST_CONDITIONS = ("baseline", "tight_brokers")


def _raw_probs(model, frame):
    """Frozen base model's P(win) and realized bid_won for a window frame."""
    return model.predict_proba(frame), frame[LABEL].to_numpy()


def _record(cond: Condition, result: RecalibrationResult) -> Dict[str, Any]:
    pre, post = result.pre, result.post
    record: Dict[str, Any] = {
        "name": cond.name,
        "lens": cond.lens(),
        "rationale": cond.rationale,
        "overrides": cond.overrides,
        "promoted": result.promoted,
        "reason": result.reason,
        "method": result.method,
        "n_fit": result.n_fit,
        "n_eval": result.n_eval,
        "recalibrator": result.recalibrator,
        "pre": report_to_dict(pre),
        "post": report_to_dict(post) if post is not None else None,
        "severity_pre": pre.severity,
        "severity_post": post.severity if post is not None else None,
        "ece_pre": pre.ece,
        "ece_post": post.ece if post is not None else None,
        "ece_improvement": (
            round(pre.ece - post.ece, 6)
            if (post is not None and pre.ece is not None and post.ece is not None)
            else None
        ),
    }
    return record


# Held-out operational draw: a fresh snapshot seed the frozen base model never trained on.
# The base model is fit on the *default*-seed baseline (the served artifact); every world's
# fit/eval windows come from this offset draw, so the baseline world is no longer scored on
# its own in-sample training days (which would float every world up to a spurious WATCH).
SEED_OFFSET = 1000


def run_recalibration(
    cfg: MLConfig,
    thresholds: CalibrationThresholds,
    config: RecalibrationConfig,
    conditions: List[Condition],
    *,
    days: int,
    seed_offset: int = SEED_OFFSET,
) -> List[Dict[str, Any]]:
    """Train + calibrate once on baseline, then fit/evaluate/promote a recalibrator per world.

    The base model is trained on the baseline *training* draw and frozen. Each world's
    recalibration fit/eval windows are carved from a **separate, later operational draw**
    (``synthetic_data.seed + seed_offset``) that the base model never saw, so even the
    no-shift baseline world is evaluated purely out-of-sample — exactly like Phase 5.3's
    held-out test split — rather than on its own training days.
    """
    # Baseline (the reference world) runs first.
    conditions = sorted(conditions, key=lambda c: (bool(c.overrides), c.name))
    records: List[Dict[str, Any]] = []
    op_seed = cfg.synthetic_data.seed + seed_offset

    with TemporaryDirectory() as base_tmp:
        base_world = world_cfg(cfg, Path(base_tmp))
        build_winnability_dataset(base_world, days=days)
        base_frame, _ = _build_frame(base_world)
        model = _calibrated_model(base_frame, base_world.winnability.random_seed)

        for i, cond in enumerate(conditions, 1):
            t0 = time.perf_counter()
            with TemporaryDirectory() as ctmp:
                world = world_cfg(cfg, Path(ctmp), cond.overrides or None)
                build_winnability_dataset(world, days=days, seed=op_seed)
                frame, _ = _build_frame(world)

                fit_df, eval_df = time_split(frame, config.fit_days, config.eval_days)
                raw_fit, y_fit = _raw_probs(model, fit_df)
                raw_eval, y_eval = _raw_probs(model, eval_df)
                result = recalibrate(
                    raw_fit, y_fit, raw_eval, y_eval, thresholds, config, label=cond.name
                )

            record = _record(cond, result)
            records.append(record)
            _print_line(i, len(conditions), record, time.perf_counter() - t0)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEVERITY_TAG = {OK: "OK   ", WATCH: "WATCH", ALERT: "ALERT"}


def _print_line(i: int, n: int, record: Dict[str, Any], elapsed: float) -> None:
    pre = _SEVERITY_TAG.get(record["severity_pre"], record["severity_pre"] or "  -  ")
    post = _SEVERITY_TAG.get(record["severity_post"], record["severity_post"] or "  -  ")
    ece_pre = record.get("ece_pre")
    ece_post = record.get("ece_post")
    pre_s = "  n/a" if ece_pre is None else f"{ece_pre:.4f}"
    post_s = "  n/a" if ece_post is None else f"{ece_post:.4f}"
    flag = "PROMOTED" if record["promoted"] else f"kept ({record['reason']})"
    print(
        f"[{i:2d}/{n}] {record['name']:<19} {pre}->{post}  "
        f"ece {pre_s}->{post_s}   {flag:<26} ({elapsed:.0f}s)"
    )


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    promoted = [r for r in records if r["promoted"]]
    alert_pre = [r for r in records if r["severity_pre"] == ALERT]
    alert_repaired = [
        r for r in alert_pre if r["promoted"] and r["severity_post"] in (OK, WATCH)
    ]
    return {
        "promoted_count": len(promoted),
        "alert_pre_count": len(alert_pre),
        "alert_repaired_count": len(alert_repaired),
        "headline": (
            f"Recalibration: repaired {len(alert_repaired)}/{len(alert_pre)} ALERT worlds "
            f"(post WATCH/OK), {len(promoted)} promoted across {len(records)} worlds; "
            f"baseline + payment/coverage left unchanged."
        ),
    }


def _format_table(records: List[Dict[str, Any]]) -> str:
    header = (
        f"{'world':<19} {'pre':<6} {'post':<6} {'ece_pre':>8} {'ece_post':>9} "
        f"{'improve':>8}  {'decision':<28}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        pre = _SEVERITY_TAG.get(r["severity_pre"], "  -  ")
        post = _SEVERITY_TAG.get(r["severity_post"], "  -  ")
        ep = "  n/a" if r["ece_pre"] is None else f"{r['ece_pre']:.4f}"
        eq = "  n/a" if r["ece_post"] is None else f"{r['ece_post']:.4f}"
        imp = "  n/a" if r["ece_improvement"] is None else f"{r['ece_improvement']:+.4f}"
        decision = "PROMOTED" if r["promoted"] else f"kept ({r['reason']})"
        lines.append(
            f"{r['name']:<19} {pre:<6} {post:<6} {ep:>8} {eq:>9} {imp:>8}  {decision:<28}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Recalibration policy config (recalibration: block).")
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG),
                        help="Calibration-monitor config for the shared severity thresholds.")
    parser.add_argument("--conditions", default=None,
                        help="Override the stress-conditions YAML (default: the monitor's).")
    parser.add_argument("--days", type=int, default=21,
                        help="Synthetic horizon per world (must cover fit_days + eval_days).")
    parser.add_argument("--fit-days", type=int, default=None, help="Override config fit_days.")
    parser.add_argument("--eval-days", type=int, default=None, help="Override config eval_days.")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Override fit-window + eval-window min_samples.")
    parser.add_argument("--method", default=None, help="Override recalibrator method.")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + short windows + a 2-world subset (smoke).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated condition names to run (default: all).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    thresholds, extras = load_calibration_config(args.monitor_config)
    config = load_recalibration_config(args.config)

    # CLI overrides.
    if args.fit_days is not None:
        config = replace(config, fit_days=args.fit_days)
    if args.eval_days is not None:
        config = replace(config, eval_days=args.eval_days)
    if args.method is not None:
        config = replace(config, method=args.method)
    if args.min_samples is not None:
        config = replace(config, min_samples=args.min_samples)
        thresholds = replace(thresholds, min_samples=args.min_samples)

    conditions_path = Path(args.conditions or extras["conditions"])
    if not conditions_path.is_absolute():
        conditions_path = ROOT / conditions_path
    conditions = load_conditions(conditions_path)

    cfg = load_ml_config()
    days = args.days
    if args.fast:
        days = 6
        config = replace(config, fit_days=2, eval_days=3, min_samples=40)
        thresholds = replace(thresholds, min_samples=40)
        if not args.only:
            conditions = [c for c in conditions if c.name in _FAST_CONDITIONS]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        conditions = [c for c in conditions if c.name in wanted]
        missing = wanted - {c.name for c in conditions}
        if missing:
            raise SystemExit(f"unknown condition(s): {sorted(missing)}")
    if not conditions:
        raise SystemExit("no conditions selected")
    if days < config.fit_days + config.eval_days:
        raise SystemExit(
            f"--days {days} is shorter than fit_days+eval_days "
            f"({config.fit_days}+{config.eval_days}); widen --days or the windows."
        )

    print("=" * 80)
    print(f"Recalibration workflow: {len(conditions)} worlds x frozen-base P(win) "
          f"({config.method}; fit {config.fit_days}d / eval {config.eval_days}d; {days}d worlds)")
    print("=" * 80)

    start = time.time()
    records = run_recalibration(cfg, thresholds, config, conditions, days=days)
    elapsed = time.time() - start
    summary_stats = _summarize(records)

    summary: Dict[str, Any] = {
        "config": {
            "fast": args.fast,
            "days": days,
            "method": config.method,
            "fit_days": config.fit_days,
            "eval_days": config.eval_days,
            "recalibration": config.as_dict(),
            "thresholds": thresholds.as_dict(),
            "condition_count": len(conditions),
            "trained_on": "baseline",
            "base_calibration": "isotonic(validation)",
            "winnability_seed": cfg.winnability.random_seed,
            "operational_seed": cfg.synthetic_data.seed + SEED_OFFSET,
            "operational_seed_offset": SEED_OFFSET,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "promoted_count": summary_stats["promoted_count"],
        "alert_pre_count": summary_stats["alert_pre_count"],
        "alert_repaired_count": summary_stats["alert_repaired_count"],
        "headline": summary_stats["headline"],
        "conditions": records,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + _format_table(records))
    print("\n" + "=" * 80)
    print(summary["headline"])
    print("=" * 80)
    print(f"Wrote {out_path} ({elapsed / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

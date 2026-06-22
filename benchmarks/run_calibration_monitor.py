"""Calibration drift monitor sweep (Phase 5.3).

Promotes the Phase 4.5 observation — the baseline-trained winnability model turns
over-optimistic under reserve / win-curve shift — into a *standing* monitor. The
winnability model is trained **once on the baseline world** and calibrated on that world's
validation slice (mirroring the served Phase 4.2 artifact), then held fixed; every condition
in ``config/broker_quality_stress.yaml`` is a broker-quality-shifted world the *same* model
is scored on. For each world the monitor compares the model's predicted ``P(win)`` against
the realized ``bid_won`` on the held-out **test** split and asks: are the probabilities still
trustworthy?

This is **detection only** (Phase 5.4 repairs) and changes **no recommender behavior** — it
reads predictions and outcomes, nothing else. Reuses the Phase 4.5 world machinery
(``Condition`` / ``world_cfg`` / Common Random Numbers) so calibration drift is measured on
the same shifts the EV sweep already characterized.

Per world it reports ECE, signed bias (``mean predicted − observed win rate``), Brier, log
loss, a reliability table, and an ``OK`` / ``WATCH`` / ``ALERT`` severity (config-driven
thresholds), plus drift vs the baseline world. The reserve / win-curve worlds
(``tight_brokers``, ``sharp_win_curve``, ``high_contention``) move the true win curve and
should trip ``WATCH`` / ``ALERT``; the baseline (training) world is the in-distribution
reference and should stay ``OK``.

Writes ``benchmarks/calibration_monitor_summary.json`` (committed, lean) for the chart and
the README.

Examples
--------
    # quick smoke (tiny seeded builds, 2 worlds, capped rows)
    python -m benchmarks.run_calibration_monitor --fast

    # canonical sweep
    python -m benchmarks.run_calibration_monitor --days 21 \
        --out benchmarks/calibration_monitor_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from benchmarks.run_bid_recommender_eval import _build_frame, _train_model
from benchmarks.run_broker_quality_stress import (
    Condition,
    load_conditions,
    world_cfg,
)
from ml.config import MLConfig, load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.monitoring.calibration_drift import (
    ALERT,
    OK,
    WATCH,
    CalibrationReport,
    CalibrationThresholds,
    calibration_drift,
    evaluate_calibration,
    worst_severity,
)
from ml.monitoring.calibration_report import (
    format_table,
    headline,
    load_calibration_config,
    report_to_dict,
)
from ml.training.winnability_dataset import LABEL, resolve_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "calibration_monitor_summary.json"

# Smoke subset: the baseline reference + one decisive reserve/win-curve drifter.
_FAST_CONDITIONS = ("baseline", "tight_brokers")
SUPPORTED_TARGETS = ("winnability",)


def _calibrated_model(frame, seed: int):
    """Quick-train winnability on the train split, then isotonic-calibrate on validation.

    Matches how the served Phase 4.2 model is calibrated, so the baseline world is a clean
    in-distribution reference. Calibration is skipped only on a degenerate validation slice.
    """
    model = _train_model(frame, seed)
    val = frame[frame["split"] == "validation"].reset_index(drop=True)
    if len(val) >= 2 and val[LABEL].nunique() > 1:
        model = model.make_calibrated(val, val[LABEL].to_numpy(), method="isotonic")
    return model


def _test_slice(frame, max_rows: Optional[int]):
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    if max_rows is not None:
        test = test.iloc[:max_rows]
    return test


def _world_report(
    model, frame, thresholds: CalibrationThresholds, name: str, max_rows: Optional[int]
) -> CalibrationReport:
    test = _test_slice(frame, max_rows)
    y_prob = model.predict_proba(test)
    y_true = test[LABEL].to_numpy()
    return evaluate_calibration(y_true, y_prob, thresholds, label=name)


def _record(cond: Condition, report: CalibrationReport, drift: Dict[str, Any]) -> Dict[str, Any]:
    record = report_to_dict(report)
    record.update(
        {
            "name": cond.name,
            "lens": cond.lens(),
            "rationale": cond.rationale,
            "overrides": cond.overrides,
            "bias_drift": drift["bias_drift"],
            "ece_drift": drift["ece_drift"],
        }
    )
    return record


def run_monitor(
    cfg: MLConfig,
    thresholds: CalibrationThresholds,
    conditions: List[Condition],
    *,
    days: int,
    max_rows: Optional[int],
) -> List[Dict[str, Any]]:
    """Train + calibrate once on baseline, then monitor the fixed model on every world."""
    # Baseline (the training world) runs first so drift is anchored.
    conditions = sorted(conditions, key=lambda c: (bool(c.overrides), c.name))
    records: List[Dict[str, Any]] = []
    baseline_report: Optional[CalibrationReport] = None

    with TemporaryDirectory() as base_tmp:
        base_world = world_cfg(cfg, Path(base_tmp))
        build_winnability_dataset(base_world, days=days)
        base_frame, _ = _build_frame(base_world)
        model = _calibrated_model(base_frame, base_world.winnability.random_seed)

        for i, cond in enumerate(conditions, 1):
            t0 = time.perf_counter()
            if not cond.overrides:
                report = _world_report(model, base_frame, thresholds, cond.name, max_rows)
                baseline_report = report
                drift = calibration_drift(report, None)
            else:
                with TemporaryDirectory() as ctmp:
                    world = world_cfg(cfg, Path(ctmp), cond.overrides)
                    build_winnability_dataset(world, days=days)
                    frame, _ = _build_frame(world)
                    report = _world_report(model, frame, thresholds, cond.name, max_rows)
                drift = calibration_drift(report, baseline_report)
            record = _record(cond, report, drift)
            records.append(record)
            _print_line(i, len(conditions), record, time.perf_counter() - t0)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_SEVERITY_TAG = {OK: "OK    ", WATCH: "WATCH ", ALERT: "ALERT "}


def _print_line(i: int, n: int, record: Dict[str, Any], elapsed: float) -> None:
    tag = _SEVERITY_TAG.get(record["severity"], record["severity"])
    ece = record.get("ece")
    drift = record.get("ece_drift")
    ece_s = "  n/a" if ece is None else f"{ece:.4f}"
    drift_s = "  base" if not record["overrides"] else f"{drift:+.4f}" if drift is not None else "   n/a"
    low = " low-n" if record.get("insufficient_data") else ""
    print(
        f"[{i:2d}/{n}] {tag} {record['name']:<19} {record['lens']:<11} "
        f"ece {ece_s}  ece-drift {drift_s}{low}   ({elapsed:.0f}s)"
    )


def _tally(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tally = {OK: 0, WATCH: 0, ALERT: 0}
    for r in records:
        tally[r["severity"]] = tally.get(r["severity"], 0) + 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Monitor config (thresholds + target + conditions).")
    parser.add_argument("--conditions", default=None,
                        help="Override the stress-conditions YAML from the config.")
    parser.add_argument("--days", type=int, default=21,
                        help="Synthetic horizon per world (fast mode forces 6).")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap test predictions scored per world (fast forces 6000).")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + a 2-world subset for a quick smoke.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated condition names to run (default: all).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    thresholds, extras = load_calibration_config(args.config)
    target = extras["target"]
    if target not in SUPPORTED_TARGETS:
        raise SystemExit(
            f"target {target!r} not supported yet (have: {', '.join(SUPPORTED_TARGETS)}). "
            "The monitor core is target-agnostic; only winnability is wired into this sweep."
        )

    conditions_path = Path(args.conditions or extras["conditions"])
    if not conditions_path.is_absolute():
        conditions_path = ROOT / conditions_path
    conditions = load_conditions(conditions_path)

    cfg = load_ml_config()

    days = args.days
    max_rows = args.max_rows
    if args.fast:
        days = 6
        max_rows = max_rows or 6000
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

    print("=" * 78)
    print(f"Calibration drift monitor: {len(conditions)} worlds x P(win)-vs-bid_won "
          f"(model trained+calibrated once on baseline, {days}d worlds)")
    print("=" * 78)

    start = time.time()
    records = run_monitor(cfg, thresholds, conditions, days=days, max_rows=max_rows)
    elapsed = time.time() - start
    tally = _tally(records)

    summary: Dict[str, Any] = {
        "config": {
            "fast": args.fast,
            "days": days,
            "max_rows": max_rows,
            "target": target,
            "condition_count": len(conditions),
            "trained_on": "baseline",
            "calibration": "isotonic(validation)",
            "thresholds": thresholds.as_dict(),
            "winnability_seed": cfg.winnability.random_seed,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "tally": tally,
        "worst_severity": worst_severity([r["severity"] for r in records]),
        "headline": headline(tally, len(records)),
        "conditions": records,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + format_table(records))
    print("\n" + "=" * 78)
    print(summary["headline"])
    print("=" * 78)
    print(f"Wrote {out_path} ({elapsed / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

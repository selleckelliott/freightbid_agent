"""Broker-quality stress sweep (Phase 4.5).

Stress-tests the Phase 4.2/4.3 **bid-winnability + EV recommendation** layer the way
Phase 3.4 stress-tested destination-aware *dispatch*. The research question:

    Does the EV bid recommender stay useful when the broker market degrades —
    slower pay, more unknown credit, riskier brokers, more no-rate loads, more
    contention, loads disappearing faster?

The calibrated winnability model is trained **once on the baseline world** and then
held fixed; every condition in ``config/broker_quality_stress.yaml`` is a
broker-quality-shifted world the *same* model is evaluated on — distribution shift at
inference, never retraining (mirrors Phase 3.4). All conditions reuse the baseline
seeds (Common Random Numbers), so a condition's only difference is its perturbed knob.

Two honest lenses are reported per condition (see README Phase 4.5):

* **EV lens** — ``uplift_pct`` of the EV ``target`` policy's oracle-realized profit over
  the best fixed policy, tagged HOLDS (>= +1%) / NEUTRAL / REGRESSION (<= -1%). Moved by
  knobs that change the hidden reserve or the win curve.
* **Calibration lens** — the gap between the model's predicted P(win) and the world's
  true (oracle) P(win) on the selected bids, and its drift versus the baseline world.
  Moved by broker payment/coverage knobs, which are **orthogonal to realized bid profit**
  (``realized = P(win) x (ask - cost)`` has no payment term) — so they stress trust in
  the model without moving the EV verdict. Documenting this is the point.

Writes ``benchmarks/broker_quality_stress_summary.json`` (committed, lean) for the
chart script and the README.

Examples
--------
    # quick smoke (tiny seeded builds, 2 conditions, capped loads)
    python -m benchmarks.run_broker_quality_stress --fast

    # canonical sweep
    python -m benchmarks.run_broker_quality_stress --days 21 \
        --out benchmarks/broker_quality_stress_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

import yaml

from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from application.config_loader import load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from benchmarks.run_bid_recommender_eval import _build_frame, _train_model, evaluate_policies
from ml.config import BrokersConfig, MLConfig, OutcomesConfig, SyntheticDataConfig, load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.training.winnability_dataset import load_snapshots_and_trials, resolve_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONDITIONS = ROOT / "config" / "broker_quality_stress.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "broker_quality_stress_summary.json"

# Verdicts (mirrors the +/-1% band idiom of the Phase 3.4 sweep).
VERDICT_HOLDS = "HOLDS"
VERDICT_NEUTRAL = "NEUTRAL"
VERDICT_REGRESSION = "REGRESSION"
UPLIFT_BAND_PCT = 1.0

# Override sections a condition may carry, mapped to their config dataclass. Only
# real dataclass fields are accepted within each (a typo guard, like the 3.4 sweep).
_SECTIONS = {
    "synthetic_data": SyntheticDataConfig,
    "brokers": BrokersConfig,
    "outcomes": OutcomesConfig,
}
_SECTION_FIELDS = {name: {f.name for f in fields(dc)} for name, dc in _SECTIONS.items()}
# IO paths are redirected to a temp dir per world; a condition must not set them.
_SECTION_FIELDS["outcomes"] -= {"snapshot_path", "outcomes_path", "trials_path"}
_META_KEYS = {"name", "rationale"}

# Knobs that move the EV verdict (change the hidden reserve or the win curve); every
# other override only shifts model features/labels -> the calibration lens.
_EV_KNOBS = {
    ("synthetic_data", "unposted_rate_fraction"),
    ("outcomes", "reservation_contention_drop"),
    ("outcomes", "reservation_center_mult"),
    ("outcomes", "win_logistic_scale_rpm"),
}
# Smoke subset: baseline + one clean single-section EV shift.
_FAST_CONDITIONS = ("baseline", "no_rate_heavy")


class Condition:
    """One stress world: a name, rationale, and per-section knob overrides."""

    __slots__ = ("name", "rationale", "overrides")

    def __init__(self, name: str, rationale: str, overrides: Dict[str, Dict[str, Any]]):
        self.name = name
        self.rationale = rationale
        self.overrides = overrides

    def lens(self) -> str:
        """Which reporting lens this condition primarily stresses (derived)."""
        touched = {(sec, k) for sec, kv in self.overrides.items() for k in kv}
        if not touched:
            return "reference"
        ev = bool(touched & _EV_KNOBS)
        other = bool(touched - _EV_KNOBS)
        if ev and other:
            return "both"
        return "ev" if ev else "calibration"


def load_conditions(path: Path) -> List[Condition]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    conditions: List[Condition] = []
    for entry in doc["conditions"]:
        name = entry.get("name")
        if not name:
            raise ValueError("every condition needs a 'name'")
        unknown_top = set(entry) - _META_KEYS - set(_SECTIONS)
        if unknown_top:
            raise ValueError(f"condition {name!r} has unknown keys: {sorted(unknown_top)}")
        overrides: Dict[str, Dict[str, Any]] = {}
        for section in _SECTIONS:
            block = entry.get(section)
            if block is None:
                continue
            if not isinstance(block, dict):
                raise ValueError(f"condition {name!r} section {section!r} must be a mapping")
            unknown = set(block) - _SECTION_FIELDS[section]
            if unknown:
                raise ValueError(
                    f"condition {name!r} section {section!r} has unknown knobs: {sorted(unknown)}"
                )
            if block:
                overrides[section] = dict(block)
        conditions.append(Condition(name, str(entry.get("rationale", "")).strip(), overrides))
    return conditions


def world_cfg(cfg: MLConfig, tmp: Path, overrides: Optional[Dict[str, Dict[str, Any]]] = None) -> MLConfig:
    """Baseline config with section overrides applied and IO redirected to ``tmp``.

    The frozen dataclasses are perturbed via ``dataclasses.replace`` so the same seeds
    carry through (Common Random Numbers); only the listed knobs change.
    """
    overrides = overrides or {}
    synthetic = replace(cfg.synthetic_data, **overrides.get("synthetic_data", {}))
    brokers = replace(cfg.brokers, **overrides.get("brokers", {}))
    outcomes = replace(
        cfg.outcomes,
        **overrides.get("outcomes", {}),
        snapshot_path=str(tmp / "snap.jsonl"),
        outcomes_path=str(tmp / "out.jsonl"),
        trials_path=str(tmp / "trials.jsonl"),
    )
    return replace(cfg, synthetic_data=synthetic, brokers=brokers, outcomes=outcomes)


def _verdict(uplift_pct: float) -> str:
    if uplift_pct >= UPLIFT_BAND_PCT:
        return VERDICT_HOLDS
    if uplift_pct <= -UPLIFT_BAND_PCT:
        return VERDICT_REGRESSION
    return VERDICT_NEUTRAL


def _condition_record(cond: Condition, summary: Dict, baseline_gap: Optional[float]) -> Dict[str, Any]:
    headline = summary["headline"]
    target = summary["policies"].get("recommender_target", {})
    model_p = target.get("avg_model_win_prob", 0.0)
    oracle_p = target.get("avg_oracle_win_prob", 0.0)
    gap = round(model_p - oracle_p, 4)
    uplift = headline["target_uplift_pct_vs_best_fixed"]
    return {
        "name": cond.name,
        "lens": cond.lens(),
        "verdict": _verdict(uplift),
        "rationale": cond.rationale,
        "overrides": cond.overrides,
        "n_loads": summary["n_test_loads"],
        "target_realized_profit": headline["target_realized_profit"],
        "best_fixed_realized_profit": headline["best_fixed_realized_profit"],
        "uplift_pct": uplift,
        "target_ev_regret_vs_oracle": headline["target_ev_regret_vs_oracle"],
        "model_win_prob": round(model_p, 4),
        "oracle_win_prob": round(oracle_p, 4),
        "calibration_gap": gap,
        "calibration_drift_vs_baseline": (
            None if baseline_gap is None else round(gap - baseline_gap, 4)
        ),
    }


def run_sweep(cfg: MLConfig, bid_cfg, conditions: List[Condition], *,
              days: int, max_loads: Optional[int]) -> List[Dict[str, Any]]:
    """Train once on baseline, then evaluate the fixed model on every condition."""
    records: List[Dict[str, Any]] = []
    baseline_gap: Optional[float] = None

    with TemporaryDirectory() as base_tmp:
        base_world = world_cfg(cfg, Path(base_tmp))
        build_winnability_dataset(base_world, days=days)
        frame, base_snaps = _build_frame(base_world)
        model = _train_model(frame, base_world.winnability.random_seed)
        adapter = ModelWinnabilityAdapter(model)
        recommender = EVBidRecommender(adapter, bid_cfg)
        base_outcomes = resolve_path(base_world.outcomes.outcomes_path)

        for i, cond in enumerate(conditions, 1):
            c_start = time.perf_counter()
            if not cond.overrides:
                # Reuse the already-built baseline world (it *is* the training world).
                summary = evaluate_policies(
                    adapter, recommender, base_world, bid_cfg, base_snaps,
                    base_outcomes, max_loads=max_loads,
                )
            else:
                with TemporaryDirectory() as ctmp:
                    world = world_cfg(cfg, Path(ctmp), cond.overrides)
                    build_winnability_dataset(world, days=days)
                    snaps, _ = load_snapshots_and_trials(world)
                    summary = evaluate_policies(
                        adapter, recommender, world, bid_cfg, snaps,
                        resolve_path(world.outcomes.outcomes_path), max_loads=max_loads,
                    )
            record = _condition_record(cond, summary, baseline_gap)
            if cond.lens() == "reference":
                baseline_gap = record["calibration_gap"]
                record["calibration_drift_vs_baseline"] = 0.0
            records.append(record)
            _print_condition_line(i, len(conditions), record, time.perf_counter() - c_start)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_VERDICT_TAG = {VERDICT_HOLDS: "HOLDS   ", VERDICT_NEUTRAL: "neutral ", VERDICT_REGRESSION: "REGRESS "}


def _tally(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tally = {VERDICT_HOLDS: 0, VERDICT_NEUTRAL: 0, VERDICT_REGRESSION: 0}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return tally


def _print_condition_line(i: int, n: int, record: Dict[str, Any], elapsed: float) -> None:
    tag = _VERDICT_TAG.get(record["verdict"], record["verdict"])
    drift = record["calibration_drift_vs_baseline"]
    drift_s = "  base" if drift in (0.0, None) and record["lens"] == "reference" else f"{drift:+.3f}"
    print(f"[{i:2d}/{n}] {tag} {record['name']:<19} {record['lens']:<11} "
          f"uplift {record['uplift_pct']:+6.1f}%  cal-drift {drift_s}   ({elapsed:.0f}s)")


def _headline(tally: Dict[str, int], total: int) -> str:
    return (f"EV beats best fixed in {tally.get(VERDICT_HOLDS, 0)}/{total} broker-quality "
            f"worlds (neutral {tally.get(VERDICT_NEUTRAL, 0)}, "
            f"regression {tally.get(VERDICT_REGRESSION, 0)}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conditions", default=str(DEFAULT_CONDITIONS),
                        help="Stress conditions YAML (default: broker_quality_stress.yaml).")
    parser.add_argument("--days", type=int, default=21,
                        help="Synthetic horizon per world (fast mode forces 6).")
    parser.add_argument("--max-loads", type=int, default=None,
                        help="Cap held-out loads scored per condition (fast forces 150).")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + a 2-condition subset for a quick smoke.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated condition names to run (default: all).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")
    conditions = load_conditions(Path(args.conditions))

    days = args.days
    max_loads = args.max_loads
    if args.fast:
        days = 6
        max_loads = max_loads or 150
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
    # Ensure the baseline (training world) runs first so calibration drift is anchored.
    conditions.sort(key=lambda c: (bool(c.overrides), c.name))

    print("=" * 78)
    print(f"Broker-quality stress: {len(conditions)} worlds x EV-vs-fixed eval "
          f"(model trained once on baseline, {days}d worlds)")
    print("=" * 78)

    start = time.time()
    records = run_sweep(cfg, bid_cfg, conditions, days=days, max_loads=max_loads)
    elapsed = time.time() - start
    tally = _tally(records)

    summary: Dict[str, Any] = {
        "config": {
            "fast": args.fast,
            "days": days,
            "max_loads": max_loads,
            "condition_count": len(conditions),
            "trained_on": "baseline",
            "uplift_band_pct": UPLIFT_BAND_PCT,
            "winnability_seed": cfg.winnability.random_seed,
            "win_logistic_scale_rpm_baseline": cfg.outcomes.win_logistic_scale_rpm,
            "cost_per_loaded_mile": bid_cfg.cost_per_loaded_mile,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "tally": tally,
        "headline": _headline(tally, len(records)),
        "conditions": records,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(summary["headline"])
    print("=" * 78)
    print(f"Wrote {out_path} ({elapsed / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

"""Sequential policy stress-test sweep (Phase 3.4).

Replays the Phase 3.3 rolling A/B (profit-aware vs destination-aware) under each
*condition* in ``config/stress_conditions.yaml`` — a perturbation of the baseline
world / economics / HOS — and reports, per condition, a paired robustness verdict
(HOLDS / NEUTRAL / REGRESSION). The research question: does the destination-aware
advantage generalise across shifted markets, or is it an artifact of one synthetic
world?

Every condition shares the same base seed (Common Random Numbers), so a
condition's only difference from the baseline is the perturbed parameter — a
variance-reduced one-factor-at-a-time comparison. The baseline world parameters
come from ``config/rolling_replay.yaml`` (single source of truth), so the
``baseline`` condition reproduces the shipped Phase 3.3 result.

The model artifact is the *same* one trained on the baseline distribution
(``ml/artifacts/destination_desirability_model.joblib``); stress is distribution
shift at inference, not retraining. If the artifact is absent the sweep still runs
the profit-aware trajectory for every condition and marks each ``DEST_SKIPPED``.

Outputs ``benchmarks/stress_test_summary.json`` (committed) for the chart script
and the README.

Examples
--------
    # quick smoke (3 episodes/condition, short solves)
    python -m benchmarks.run_stress_test --episodes 3 --time-limit 0.2

    # canonical sweep (30 episodes/condition)
    python -m benchmarks.run_stress_test --episodes 30 \
        --out benchmarks/stress_test_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from adapters.inbound.api.container import build_container
from simulation.experiment import (
    VERDICT_DEST_SKIPPED,
    VERDICT_HOLDS,
    VERDICT_NEUTRAL,
    VERDICT_REGRESSION,
    ConditionSpec,
    WorldDefaults,
    run_condition,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = ROOT / "config" / "rolling_replay.yaml"
DEFAULT_CONDITIONS = ROOT / "config" / "stress_conditions.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "stress_test_summary.json"
MODEL_PATH = ROOT / "ml" / "artifacts" / "destination_desirability_model.joblib"

# Keys a stress-conditions entry may carry beyond the override fields.
_META_KEYS = {"name", "rationale"}
_SPEC_FIELDS = {
    "name",
    "rationale",
    "loads_per_snapshot_mean",
    "unposted_rate_fraction",
    "horizon_days",
    "competition_take_rate",
    "fuel_cost_per_mile",
    "deadhead_fuel_multiplier",
    "daily_drive_hours",
    "force_equipment",
}


def _parse_start(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_world(config_path: Path) -> tuple[WorldDefaults, int, float, float]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    world = cfg["world"]
    episodes_cfg = cfg["episodes"]
    truck_cfg = cfg["truck"]
    planners_cfg = cfg["planners"]
    base = WorldDefaults(
        start_date=_parse_start(world["start_date"]),
        snapshots_per_day=int(world["snapshots_per_day"]),
        loads_per_snapshot_mean=float(world["loads_per_snapshot_mean"]),
        unposted_rate_fraction=float(world["unposted_rate_fraction"]),
        max_post_age_hours=float(world["max_post_age_hours"]),
        horizon_days=int(world["horizon_days"]),
        radius_mi=float(truck_cfg["radius_mi"]),
        daily_drive_hours=float(truck_cfg["daily_drive_hours"]),
    )
    base_seed = int(episodes_cfg["base_seed"])
    time_limit = float(planners_cfg.get("solver_time_limit_seconds", 0.3))
    destination_weight = float(planners_cfg.get("destination_weight", 1.0))
    return base, base_seed, time_limit, destination_weight


def _load_conditions(path: Path) -> List[ConditionSpec]:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = doc["conditions"]
    specs: List[ConditionSpec] = []
    for entry in entries:
        unknown = set(entry) - _SPEC_FIELDS
        if unknown:
            raise ValueError(
                f"condition {entry.get('name', '?')!r} has unknown keys: {sorted(unknown)}"
            )
        kwargs = {k: v for k, v in entry.items() if k in _SPEC_FIELDS}
        kwargs.setdefault("rationale", "")
        specs.append(ConditionSpec(**kwargs))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_BASE_CONFIG),
                        help="Baseline world config (default: rolling_replay.yaml).")
    parser.add_argument("--conditions", default=str(DEFAULT_CONDITIONS),
                        help="Stress conditions config (default: stress_conditions.yaml).")
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per condition (Common Random Numbers).")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="OR-Tools solver time limit per decision (seconds).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated condition names to run (default: all).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--keep-per-episode", action="store_true",
                        help="Retain per-episode rows in the summary (default: "
                             "drop them to keep the committed artifact lean; the "
                             "run is deterministic from --episodes + base seed).")
    args = parser.parse_args()

    base, base_seed, cfg_time_limit, destination_weight = _load_world(
        Path(args.config)
    )
    time_limit = args.time_limit if args.time_limit is not None else cfg_time_limit
    specs = _load_conditions(Path(args.conditions))
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        specs = [s for s in specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise SystemExit(f"unknown condition(s): {sorted(missing)}")

    container = build_container(ROOT / "config")
    model_path: Optional[Path] = MODEL_PATH if MODEL_PATH.exists() else None
    if model_path is None:
        print(f"(!) No model artifact at {MODEL_PATH} — destination-aware "
              f"trajectory skipped; every condition will be DEST_SKIPPED.")

    print("=" * 78)
    print(f"Stress sweep: {len(specs)} conditions x {args.episodes} episodes "
          f"x 2 planners (OR-Tools {time_limit:.2f}s/decision)")
    print("=" * 78)

    results: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for i, spec in enumerate(specs, 1):
        c_start = time.perf_counter()
        result = run_condition(
            spec,
            container=container,
            base=base,
            episode_count=args.episodes,
            base_seed=base_seed,
            solver_time_limit=time_limit,
            destination_weight=destination_weight,
            model_path=model_path,
        )
        elapsed = time.perf_counter() - c_start
        row = result.to_dict()
        if not args.keep_per_episode:
            row.pop("per_episode", None)
        results.append(row)
        _print_condition_line(i, len(specs), result, elapsed)

    total_elapsed = time.perf_counter() - t_start
    tally = _tally(results)

    summary: Dict[str, Any] = {
        "config": {
            "episodes_per_condition": args.episodes,
            "condition_count": len(specs),
            "snapshots_per_day": base.snapshots_per_day,
            "baseline_loads_per_snapshot_mean": base.loads_per_snapshot_mean,
            "radius_mi": base.radius_mi,
            "solver_time_limit_seconds": time_limit,
            "base_seed": base_seed,
            "destination_weight": destination_weight,
            "model_artifact_present": model_path is not None,
        },
        "elapsed_seconds": total_elapsed,
        "tally": tally,
        "conditions": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _print_headline(tally, len(results))
    print(f"\nWrote {out_path} ({total_elapsed / 60.0:.1f} min).")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_VERDICT_TAG = {
    VERDICT_HOLDS: "HOLDS    ",
    VERDICT_NEUTRAL: "neutral  ",
    VERDICT_REGRESSION: "REGRESS  ",
    VERDICT_DEST_SKIPPED: "skipped  ",
}


def _tally(results: List[Dict[str, Any]]) -> Dict[str, int]:
    tally = {
        VERDICT_HOLDS: 0,
        VERDICT_NEUTRAL: 0,
        VERDICT_REGRESSION: 0,
        VERDICT_DEST_SKIPPED: 0,
    }
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return tally


def _print_condition_line(
    i: int, n: int, result, elapsed: float
) -> None:
    tag = _VERDICT_TAG.get(result.verdict, result.verdict)
    delta = result.headline_delta
    if delta:
        detail = (f"profit {delta['profit_pct']:+5.1f}%  "
                  f"deadhead {delta['deadhead_pct']:+5.1f}%")
    else:
        detail = "(destination-aware skipped)"
    print(f"[{i:2d}/{n}] {tag} {result.name:<22} {detail}   ({elapsed:.0f}s)")


def _print_headline(tally: Dict[str, int], total: int) -> None:
    print("\n" + "=" * 78)
    evaluated = total - tally.get(VERDICT_DEST_SKIPPED, 0)
    print(f"Robustness: destination-aware advantage HOLDS in "
          f"{tally.get(VERDICT_HOLDS, 0)}/{evaluated}, "
          f"neutral in {tally.get(VERDICT_NEUTRAL, 0)}, "
          f"regresses in {tally.get(VERDICT_REGRESSION, 0)}"
          + (f", skipped {tally[VERDICT_DEST_SKIPPED]}"
             if tally.get(VERDICT_DEST_SKIPPED) else ""))
    print("=" * 78)


if __name__ == "__main__":
    main()

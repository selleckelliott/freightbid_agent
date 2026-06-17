"""Rolling replanning simulation runner (Phase 3.3).

Replays many independent synthetic worlds, one per episode, for the profit-aware
planner and (artifact-gated) the destination-aware planner, then reports the
*cumulative* sequential outcomes with cross-episode confidence intervals.

For every episode both planners face the **same world** (same seed, same snapshot
stream) and the **same** starting truck — only the dispatch policy differs. While
the profit-aware trajectory runs, the destination-aware planner rides along as a
*shadow*: at each decision it is asked what it would pick from the identical
board + truck state, without executing, which yields the policy-divergence
(``decision_overlap_rate``) metric.

This runner is a thin CLI over ``simulation.experiment.run_condition`` (the same
engine the Phase 3.4 stress sweep uses) for the single *baseline* condition, so
the two stay reconciled by construction. Outputs
``benchmarks/rolling_replay_summary.json`` (committed) for the chart script and
the README.

Examples
--------
    # quick smoke (5 episodes, short solves)
    python -m benchmarks.run_rolling_replay --episodes 5

    # canonical run
    python -m benchmarks.run_rolling_replay --episodes 200 \
        --out benchmarks/rolling_replay_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from adapters.inbound.api.container import build_container
from simulation.experiment import ConditionSpec, WorldDefaults, run_condition

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "rolling_replay.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "rolling_replay_summary.json"
MODEL_PATH = ROOT / "ml" / "artifacts" / "destination_desirability_model.joblib"


def _parse_start(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count.")
    parser.add_argument("--horizon-days", type=int, default=None)
    parser.add_argument("--time-limit", type=float, default=None,
                        help="OR-Tools solver time limit per decision (seconds).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--model-path",
        default=str(MODEL_PATH),
        help="Destination model artifact (default: the canonical ml/artifacts path). "
        "When absent the destination-aware planner is skipped.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    world = cfg["world"]
    episodes_cfg = cfg["episodes"]
    truck_cfg = cfg["truck"]
    planners_cfg = cfg["planners"]

    episode_count = args.episodes or int(episodes_cfg["count"])
    horizon_days = args.horizon_days or int(world["horizon_days"])
    base_seed = int(episodes_cfg["base_seed"])
    time_limit = args.time_limit if args.time_limit is not None else float(
        planners_cfg.get("solver_time_limit_seconds", 0.3)
    )
    start_dt = _parse_start(world["start_date"])

    base = WorldDefaults(
        start_date=start_dt,
        snapshots_per_day=int(world["snapshots_per_day"]),
        loads_per_snapshot_mean=float(world["loads_per_snapshot_mean"]),
        unposted_rate_fraction=float(world["unposted_rate_fraction"]),
        max_post_age_hours=float(world["max_post_age_hours"]),
        horizon_days=horizon_days,
        radius_mi=float(truck_cfg["radius_mi"]),
        daily_drive_hours=float(truck_cfg["daily_drive_hours"]),
    )

    container = build_container(ROOT / "config")
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"(destination-aware planner skipped: no model artifact at {model_path})")

    print("=" * 78)
    print(f"Rolling replay: {episode_count} episodes x {horizon_days}d horizon "
          f"(OR-Tools {time_limit:.2f}s/decision)")
    print("=" * 78)

    t_start = time.perf_counter()
    result = run_condition(
        ConditionSpec(name="baseline", rationale="Phase 3.3 canonical world"),
        container=container,
        base=base,
        episode_count=episode_count,
        base_seed=base_seed,
        solver_time_limit=time_limit,
        destination_weight=float(planners_cfg.get("destination_weight", 1.0)),
        model_path=model_path,
    )
    elapsed = time.perf_counter() - t_start

    summary: Dict[str, Any] = {
        "config": {
            "episode_count": episode_count,
            "horizon_days": horizon_days,
            "snapshots_per_day": base.snapshots_per_day,
            "loads_per_snapshot_mean": base.loads_per_snapshot_mean,
            "radius_mi": base.radius_mi,
            "daily_drive_hours": base.daily_drive_hours,
            "solver_time_limit_seconds": time_limit,
            "base_seed": base_seed,
        },
        "elapsed_seconds": elapsed,
        "profit_aware": result.profit_aware,
        "per_episode": result.per_episode,
    }
    _print_block("OR-Tools Profit-Aware", summary["profit_aware"])

    if result.destination_aware is not None:
        summary["destination_aware"] = result.destination_aware
        summary["delta_destination_vs_profit"] = result.headline_delta
        summary["onward_diagnostic"] = result.onward_diagnostic
        _print_block("OR-Tools Destination-Aware", summary["destination_aware"])
        _print_delta(result.headline_delta)
        _print_divergence(result.divergence)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path} ({elapsed:.1f}s).")


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _ci(stat: Dict[str, float]) -> str:
    return f"{stat['mean']:.1f} [{stat['ci_low']:.1f}, {stat['ci_high']:.1f}]"


def _print_block(label: str, summ: Dict[str, Any]) -> None:
    m = summ["metrics"]
    print(f"\n{label}  (n={summ['episode_count']} episodes)")
    print(f"  Cumulative profit:   ${_ci(m['total_profit'])}")
    print(f"  Cumulative deadhead: {_ci(m['total_deadhead_miles'])} mi")
    print(f"  Idle hours:          {_ci(m['idle_hours'])}")
    print(f"  Loads completed:     {_ci(m['loads_completed'])}")
    print(f"  Profit / day:        ${_ci(m['profit_per_day'])}")
    print(f"  Deadhead / load:     {_ci(m['deadhead_per_load'])} mi")


def _print_delta(delta: Optional[Dict[str, float]]) -> None:
    if not delta:
        return
    print("\nDelta (Destination-Aware vs Profit-Aware):")
    print(f"  Profit:           {delta['profit_pct']:+.1f}%")
    print(f"  Deadhead:         {delta['deadhead_pct']:+.1f}% (lower is better)")
    print(f"  Deadhead / load:  {delta['deadhead_per_load_pct']:+.1f}% (lower is better)")
    print(f"  Idle hours:       {delta['idle_hours_pct']:+.1f}%")
    print(f"  Loads:            {delta['loads_pct']:+.1f}%")


def _print_divergence(divergence: Optional[Dict[str, Any]]) -> None:
    if not divergence:
        return
    print("\nPolicy divergence (shadow on identical inputs):")
    print(f"  Decision overlap rate: {divergence['decision_overlap_rate'] * 100:.1f}%")
    print(f"  Divergence rate:       {divergence['divergence_rate'] * 100:.1f}% "
          f"(over {divergence['shadow_decision_count']} decisions)")


if __name__ == "__main__":
    main()

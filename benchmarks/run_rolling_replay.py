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

Outputs ``benchmarks/rolling_replay_summary.json`` (committed) for the chart
script and the README.

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
import random
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from adapters.inbound.api.container import build_container
from application.destination_desirability_service import (
    DestinationDesirabilityService,
)
from application.ortools_destination_aware_planner import (
    ORToolsDestinationAwarePlanner,
)
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from domain.models.truck_state import TruckState
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from ml.markets import MARKET_PROFILES, MarketProfile
from simulation.metrics import RollingReplayMetrics, summarize_episodes
from simulation.rolling_replay import ReplayConfig, ReplayEpisode, run_episode
from simulation.snapshot_board import (
    EQUIPMENT_ML_TO_DOMAIN,
    ROUND_TRIP_ML_EQUIPMENT,
    SnapshotBoard,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "rolling_replay.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "rolling_replay_summary.json"
MODEL_PATH = ROOT / "ml" / "artifacts" / "destination_desirability_model.joblib"

PROFIT_KEY = "profit_aware"
DEST_KEY = "destination_aware"


def _parse_start(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sample_start(rng: random.Random) -> Tuple[MarketProfile, str]:
    """Sample a start hub (weighted by outbound density) + a round-trip trailer.

    Starting at a strong hub keeps the board populated; the equipment is drawn
    from the hub's own mix (restricted to the three ML codes that round-trip
    losslessly to the domain trailer vocabulary), so the chosen class is actually
    posted there.
    """
    market = _weighted(rng, MARKET_PROFILES, [m.outbound_density for m in MARKET_PROFILES])
    mix = [(e, w) for e, w in market.equipment_mix if e in ROUND_TRIP_ML_EQUIPMENT]
    ml_equipment = _weighted(rng, [e for e, _ in mix], [w for _, w in mix])
    return market, ml_equipment


def _weighted(rng: random.Random, items, weights):
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


def _build_truck(
    market: MarketProfile, ml_equipment: str, start_time: datetime, daily_drive_hours: float
) -> TruckState:
    return TruckState(
        truck_id=1,
        current_city=market.name,
        current_state=market.state,
        latitude=market.lat,
        longitude=market.lon,
        available_at=start_time,
        trailer_type=EQUIPMENT_ML_TO_DOMAIN[ml_equipment],
        max_load_capacity=50000.0,
        current_load_id=None,
        home_city=market.name,
        home_state=market.state,
        remaining_capacity=50000.0,
        driver_hours_left=daily_drive_hours,
        speed=50.0,
        heading=0.0,
        timestamp=start_time,
    )


def _metrics_row(m: RollingReplayMetrics) -> Dict[str, Any]:
    return {
        "total_profit": m.total_profit,
        "total_deadhead_miles": m.total_deadhead_miles,
        "total_loaded_miles": m.total_loaded_miles,
        "loads_completed": m.loads_completed,
        "idle_hours": m.idle_hours,
        "profit_per_day": m.profit_per_day,
        "deadhead_per_load": m.deadhead_per_load,
        "utilization_rate": m.utilization_rate,
        "feasible_decision_rate": m.feasible_decision_rate,
    }


def _onward_diagnostic(episodes: List[ReplayEpisode]) -> Optional[Dict[str, float]]:
    """Predicted vs realized onward-deadhead over the destination trajectory.

    An in-distribution sanity check (correlation + signed bias), not a held-out
    MAE: the realized value is read from a later snapshot, so it mixes model
    error with world noise.
    """
    pred: List[float] = []
    real: List[float] = []
    for ep in episodes:
        for d in ep.decisions:
            if d.predicted_onward is not None and d.realized_onward is not None:
                pred.append(d.predicted_onward)
                real.append(d.realized_onward)
    if len(pred) < 2:
        return None
    signed_bias = statistics.fmean(p - r for p, r in zip(pred, real))
    mae = statistics.fmean(abs(p - r) for p, r in zip(pred, real))
    try:
        corr = statistics.correlation(pred, real)
    except statistics.StatisticsError:
        corr = 0.0
    return {
        "pairs": len(pred),
        "correlation": corr,
        "signed_bias_miles": signed_bias,
        "mae_miles": mae,
        "mean_predicted": statistics.fmean(pred),
        "mean_realized": statistics.fmean(real),
    }


def _headline_delta(dest: Dict[str, Any], profit: Dict[str, Any]) -> Dict[str, float]:
    def pct(metric: str) -> float:
        new = dest["metrics"][metric]["mean"]
        old = profit["metrics"][metric]["mean"]
        return ((new - old) / abs(old) * 100.0) if old else 0.0

    return {
        "profit_pct": pct("total_profit"),
        "deadhead_pct": pct("total_deadhead_miles"),
        "deadhead_per_load_pct": pct("deadhead_per_load"),
        "idle_hours_pct": pct("idle_hours"),
        "loads_pct": pct("loads_completed"),
    }


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
    episode_end = start_dt + timedelta(days=horizon_days)

    container = build_container(ROOT / "config")
    replay_config = ReplayConfig(
        radius_mi=float(truck_cfg["radius_mi"]),
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
        daily_drive_hours=float(truck_cfg["daily_drive_hours"]),
    )

    ortools_kwargs = dict(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        solver_time_limit_seconds=time_limit,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )
    profit_planner = ORToolsProfitAwarePlanner(
        objective_weights=container.config.ortools_objective_weights, **ortools_kwargs
    )

    dest_planner = None
    dest_service = None
    if MODEL_PATH.exists():
        dest_service = DestinationDesirabilityService.from_artifact(MODEL_PATH)
        dest_planner = ORToolsDestinationAwarePlanner(
            objective_weights=container.config.ortools_objective_weights,
            destination_service=dest_service,
            destination_weight=float(planners_cfg.get("destination_weight", 1.0)),
            **ortools_kwargs,
        )
    else:
        print(f"(destination-aware planner skipped: no model artifact at {MODEL_PATH})")

    print("=" * 78)
    print(f"Rolling replay: {episode_count} episodes x {horizon_days}d horizon "
          f"(OR-Tools {time_limit:.2f}s/decision)")
    print("=" * 78)

    profit_metrics: List[RollingReplayMetrics] = []
    dest_metrics: List[RollingReplayMetrics] = []
    profit_episodes: List[ReplayEpisode] = []
    dest_episodes: List[ReplayEpisode] = []
    per_episode: Dict[str, List[Dict[str, Any]]] = {PROFIT_KEY: [], DEST_KEY: []}

    t_start = time.perf_counter()
    for e in range(episode_count):
        episode_seed = base_seed + e
        params = GeneratorParams(
            start_date=start_dt,
            days=horizon_days + 1,  # +1 buffer so loads exist through the final day
            snapshots_per_day=int(world["snapshots_per_day"]),
            loads_per_snapshot_mean=float(world["loads_per_snapshot_mean"]),
            unposted_rate_fraction=float(world["unposted_rate_fraction"]),
            max_post_age_hours=float(world["max_post_age_hours"]),
            seed=episode_seed,
        )
        records = generate_history(params)
        sample_rng = random.Random(episode_seed * 7919 + 17)
        market, ml_equipment = _sample_start(sample_rng)
        truck = _build_truck(
            market, ml_equipment, start_dt, replay_config.daily_drive_hours
        )

        profit_ep = run_episode(
            profit_planner, SnapshotBoard(records), truck,
            episode_end=episode_end, ml_equipment=ml_equipment,
            config=replay_config, episode_seed=episode_seed,
            planner_label="OR-Tools Profit-Aware",
            shadow_planner=dest_planner,
        )
        profit_metrics.append(profit_ep.metrics)
        profit_episodes.append(profit_ep)
        per_episode[PROFIT_KEY].append(
            {"seed": episode_seed, "ml_equipment": ml_equipment,
             "start_market": market.name, **_metrics_row(profit_ep.metrics)}
        )

        if dest_planner is not None:
            dest_ep = run_episode(
                dest_planner, SnapshotBoard(records), truck,
                episode_end=episode_end, ml_equipment=ml_equipment,
                config=replay_config, episode_seed=episode_seed,
                planner_label="OR-Tools Destination-Aware",
                destination_service=dest_service,
            )
            dest_metrics.append(dest_ep.metrics)
            dest_episodes.append(dest_ep)
            per_episode[DEST_KEY].append(
                {"seed": episode_seed, "ml_equipment": ml_equipment,
                 "start_market": market.name, **_metrics_row(dest_ep.metrics)}
            )

        if (e + 1) % max(1, episode_count // 10) == 0:
            print(f"  ... {e + 1}/{episode_count} episodes")

    elapsed = time.perf_counter() - t_start

    summary: Dict[str, Any] = {
        "config": {
            "episode_count": episode_count,
            "horizon_days": horizon_days,
            "snapshots_per_day": int(world["snapshots_per_day"]),
            "loads_per_snapshot_mean": float(world["loads_per_snapshot_mean"]),
            "radius_mi": replay_config.radius_mi,
            "daily_drive_hours": replay_config.daily_drive_hours,
            "solver_time_limit_seconds": time_limit,
            "base_seed": base_seed,
        },
        "elapsed_seconds": elapsed,
        "profit_aware": summarize_episodes(profit_metrics),
        "per_episode": per_episode,
    }
    _print_block("OR-Tools Profit-Aware", summary["profit_aware"])

    if dest_metrics:
        summary["destination_aware"] = summarize_episodes(dest_metrics)
        summary["delta_destination_vs_profit"] = _headline_delta(
            summary["destination_aware"], summary["profit_aware"]
        )
        summary["onward_diagnostic"] = _onward_diagnostic(dest_episodes)
        _print_block("OR-Tools Destination-Aware", summary["destination_aware"])
        _print_delta(summary["delta_destination_vs_profit"])
        _print_divergence(summary["profit_aware"].get("divergence"))

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


def _print_delta(delta: Dict[str, float]) -> None:
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

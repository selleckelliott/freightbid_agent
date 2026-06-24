"""Fleet dispatch A/B experiment harness (Phase 8.3).

Runs the Phase 8 headline comparison under one named *condition*: the same fleet,
the same shared synthetic world (common random numbers), dispatched two ways —

* **greedy** (``GreedyFleetPolicy``): each free truck, in id order, grabs its own
  best feasible load (conflict-free but *uncoordinated*);
* **fleet-aware** (``AssignmentFleetPolicy``): a global CP-SAT max-profit matching
  over the whole truck x load field.

Both arms share the *identical* ``ProfitPairScorer``, so the measured gap is the
value of **global coordination**, not of avoiding double-booking (the greedy
baseline already avoids that). Per condition we report each arm's fleet summary
(profit, deadhead, utilisation, **balance** = per-truck profit dispersion/Gini,
**destination HHI**, contention), the paired profit/deadhead deltas, and a
robustness verdict reusing the Phase 3.4 ``classify_verdict`` rule.

A condition reuses the existing ``ConditionSpec`` axes (density, unposted rate,
competition, fuel/deadhead cost, HOS, equipment, horizon). ``force_equipment``
doubles as the **homogeneous-vs-heterogeneous** switch: pinned -> every truck
shares one trailer (maximum contention on the shared board); unset -> each truck
draws its own start hub + trailer (the realistic heterogeneous fleet).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Dict, List, Optional

from application.evaluate_loads import EvaluateLoadsService
from application.fleet.assignment_fleet_policy import AssignmentFleetPolicy
from application.fleet.greedy_fleet_policy import GreedyFleetPolicy
from application.fleet.pair_scorer import ProfitPairScorer
from domain.models.truck_state import TruckState
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from simulation.experiment import (
    ConditionSpec,
    WorldDefaults,
    build_truck,
    classify_verdict,
    sample_start,
)
from simulation.fleet_metrics import FleetEpisodeMetrics, summarize_fleet_episodes
from simulation.fleet_simulator import FleetReplayConfig, run_fleet_episode
from simulation.metrics import paired_comparison
from simulation.snapshot_board import SnapshotBoard

GREEDY_KEY = "greedy"
FLEET_KEY = "fleet_aware"


@dataclass
class FleetConditionResult:
    name: str
    rationale: str
    fleet_size: int
    homogeneous: bool
    episode_count: int
    effective: Dict[str, Any]
    verdict: str
    greedy: Dict[str, Any]
    fleet_aware: Dict[str, Any]
    headline_delta: Dict[str, float]
    paired_profit: Dict[str, float]
    paired_deadhead: Dict[str, float]
    per_episode: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "fleet_size": self.fleet_size,
            "homogeneous": self.homogeneous,
            "episode_count": self.episode_count,
            "effective": self.effective,
            "verdict": self.verdict,
            "greedy": self.greedy,
            "fleet_aware": self.fleet_aware,
            "headline_delta": self.headline_delta,
            "paired_profit": self.paired_profit,
            "paired_deadhead": self.paired_deadhead,
            "per_episode": self.per_episode,
        }


def build_fleet(
    rng: random.Random,
    *,
    fleet_size: int,
    start_time,
    daily_drive_hours: float,
    force_equipment: Optional[str],
) -> List[TruckState]:
    """Sample a K-truck fleet (CRN-shared between arms via ``rng``).

    Heterogeneous (``force_equipment is None``): each truck draws its own start
    hub + round-trip trailer. Homogeneous: the trailer is pinned for every truck,
    concentrating the whole fleet on one equipment market (maximum contention).
    """
    trucks: List[TruckState] = []
    for k in range(fleet_size):
        market, ml_equipment = sample_start(rng, force_equipment)
        truck = build_truck(market, ml_equipment, start_time, daily_drive_hours)
        trucks.append(replace(truck, truck_id=k + 1))
    return trucks


def _build_scorer(container: Any, spec: ConditionSpec) -> ProfitPairScorer:
    """Profit scorer for the condition; rebuilds the evaluator on a cost override."""
    avg_speed = container.config.average_speed_mph
    load_unload = container.config.planning_constraints.average_load_unload_hours
    if spec.has_cost_override():
        base_cost = container.config.cost_model
        new_cost = replace(
            base_cost,
            fuel_cost_per_mile=(
                spec.fuel_cost_per_mile
                if spec.fuel_cost_per_mile is not None
                else base_cost.fuel_cost_per_mile
            ),
            deadhead_fuel_multiplier=(
                spec.deadhead_fuel_multiplier
                if spec.deadhead_fuel_multiplier is not None
                else base_cost.deadhead_fuel_multiplier
            ),
        )
        evaluator = EvaluateLoadsService(
            distance_provider=container.evaluator.distance_provider,
            toll_estimator=container.evaluator.toll_estimator,
            cost_model=new_cost,
            average_speed_mph=avg_speed,
            load_unload_hours=load_unload,
        )
    else:
        evaluator = container.evaluator
    return ProfitPairScorer(evaluator, container.config.planning_constraints)


def fleet_metrics_row(m: FleetEpisodeMetrics) -> Dict[str, Any]:
    return {
        "total_profit": m.total_profit,
        "total_deadhead_miles": m.total_deadhead_miles,
        "total_loaded_miles": m.total_loaded_miles,
        "loads_completed": m.loads_completed,
        "loads_per_truck": m.loads_per_truck,
        "mean_utilization_rate": m.mean_utilization_rate,
        "min_utilization_rate": m.min_utilization_rate,
        "profit_dispersion": m.profit_dispersion,
        "profit_gini": m.profit_gini,
        "destination_hhi": m.destination_hhi,
        "contention_events": m.contention_events,
        "deadhead_per_load": m.deadhead_per_load,
    }


def _headline_delta(
    fleet_summary: Dict[str, Any], greedy_summary: Dict[str, Any]
) -> Dict[str, float]:
    def pct(metric: str) -> float:
        new = fleet_summary["metrics"][metric]["mean"]
        old = greedy_summary["metrics"][metric]["mean"]
        return ((new - old) / abs(old) * 100.0) if old else 0.0

    return {
        "profit_pct": pct("total_profit"),
        "deadhead_pct": pct("total_deadhead_miles"),
        "deadhead_per_load_pct": pct("deadhead_per_load"),
        "loads_pct": pct("loads_completed"),
        "mean_utilization_pct": pct("mean_utilization_rate"),
        "profit_dispersion_pct": pct("profit_dispersion"),
    }


def run_fleet_condition(
    spec: ConditionSpec,
    *,
    container: Any,
    base: WorldDefaults,
    fleet_size: int,
    episode_count: int,
    base_seed: int,
    solver_time_limit: float = 1.0,
    max_candidates_per_truck: Optional[int] = None,
) -> FleetConditionResult:
    """Replay ``episode_count`` worlds for greedy + fleet-aware under ``spec``."""
    horizon_days = (
        spec.horizon_days if spec.horizon_days is not None else base.horizon_days
    )
    density = (
        spec.loads_per_snapshot_mean
        if spec.loads_per_snapshot_mean is not None
        else base.loads_per_snapshot_mean
    )
    unposted = (
        spec.unposted_rate_fraction
        if spec.unposted_rate_fraction is not None
        else base.unposted_rate_fraction
    )
    daily = (
        spec.daily_drive_hours
        if spec.daily_drive_hours is not None
        else base.daily_drive_hours
    )
    start_dt = base.start_date
    episode_end = start_dt + timedelta(days=horizon_days)

    avg_speed = container.config.average_speed_mph
    load_unload = container.config.planning_constraints.average_load_unload_hours
    fleet_config = FleetReplayConfig(
        radius_mi=base.radius_mi,
        average_speed_mph=avg_speed,
        load_unload_hours=load_unload,
        daily_drive_hours=daily,
    )
    scorer = _build_scorer(container, spec)
    greedy = GreedyFleetPolicy(scorer)
    fleet_aware = AssignmentFleetPolicy(
        scorer,
        max_candidates_per_truck=max_candidates_per_truck,
        solver_time_limit_seconds=solver_time_limit,
    )

    base_cost = container.config.cost_model
    effective = {
        "fleet_size": fleet_size,
        "loads_per_snapshot_mean": density,
        "unposted_rate_fraction": unposted,
        "competition_take_rate": spec.competition_take_rate,
        "fuel_cost_per_mile": (
            spec.fuel_cost_per_mile
            if spec.fuel_cost_per_mile is not None
            else base_cost.fuel_cost_per_mile
        ),
        "deadhead_fuel_multiplier": (
            spec.deadhead_fuel_multiplier
            if spec.deadhead_fuel_multiplier is not None
            else base_cost.deadhead_fuel_multiplier
        ),
        "daily_drive_hours": daily,
        "force_equipment": spec.force_equipment,
        "horizon_days": horizon_days,
    }

    greedy_metrics: List[FleetEpisodeMetrics] = []
    fleet_metrics: List[FleetEpisodeMetrics] = []
    per_episode: Dict[str, List[Dict[str, Any]]] = {GREEDY_KEY: [], FLEET_KEY: []}

    for e in range(episode_count):
        episode_seed = base_seed + e
        params = GeneratorParams(
            start_date=start_dt,
            days=horizon_days + 1,
            snapshots_per_day=base.snapshots_per_day,
            loads_per_snapshot_mean=density,
            unposted_rate_fraction=unposted,
            max_post_age_hours=base.max_post_age_hours,
            seed=episode_seed,
        )
        records = generate_history(params)
        fleet_rng = random.Random(episode_seed * 7919 + 17)
        trucks = build_fleet(
            fleet_rng,
            fleet_size=fleet_size,
            start_time=start_dt,
            daily_drive_hours=daily,
            force_equipment=spec.force_equipment,
        )

        g_ep = run_fleet_episode(
            greedy,
            SnapshotBoard(
                records,
                competition_take_rate=spec.competition_take_rate,
                competition_seed=episode_seed,
            ),
            trucks,
            episode_end=episode_end,
            config=fleet_config,
            episode_seed=episode_seed,
            policy_label="Greedy",
        )
        f_ep = run_fleet_episode(
            fleet_aware,
            SnapshotBoard(
                records,
                competition_take_rate=spec.competition_take_rate,
                competition_seed=episode_seed,
            ),
            trucks,
            episode_end=episode_end,
            config=fleet_config,
            episode_seed=episode_seed,
            policy_label="Fleet-Aware",
        )
        greedy_metrics.append(g_ep.metrics)
        fleet_metrics.append(f_ep.metrics)
        per_episode[GREEDY_KEY].append(
            {"seed": episode_seed, **fleet_metrics_row(g_ep.metrics)}
        )
        per_episode[FLEET_KEY].append(
            {"seed": episode_seed, **fleet_metrics_row(f_ep.metrics)}
        )

    greedy_summary = summarize_fleet_episodes(greedy_metrics)
    fleet_summary = summarize_fleet_episodes(fleet_metrics)
    paired_profit = paired_comparison(
        [m.total_profit for m in greedy_metrics],
        [m.total_profit for m in fleet_metrics],
        lower_is_better=False,
    )
    paired_deadhead = paired_comparison(
        [m.total_deadhead_miles for m in greedy_metrics],
        [m.total_deadhead_miles for m in fleet_metrics],
        lower_is_better=True,
    )
    verdict = classify_verdict(paired_profit, paired_deadhead["mean"])

    return FleetConditionResult(
        name=spec.name,
        rationale=spec.rationale,
        fleet_size=fleet_size,
        homogeneous=spec.force_equipment is not None,
        episode_count=episode_count,
        effective=effective,
        verdict=verdict,
        greedy=greedy_summary,
        fleet_aware=fleet_summary,
        headline_delta=_headline_delta(fleet_summary, greedy_summary),
        paired_profit=paired_profit,
        paired_deadhead=paired_deadhead,
        per_episode=per_episode,
    )

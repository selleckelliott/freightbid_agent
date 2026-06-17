"""Stress-test experiment harness (Phase 3.4).

Runs one rolling A/B (profit-aware vs destination-aware) under a single named
*condition* — a perturbation of the baseline synthetic world / economics / HOS —
and reports a paired robustness verdict. The same machinery drives the Phase 3.3
baseline (one condition) and the Phase 3.4 sweep (many conditions), so the two
stay reconciled by construction.

A condition can shift any of the stress axes:

* market density (``loads_per_snapshot_mean``),
* unposted-rate fraction,
* load-view competition (``competition_take_rate`` -> view-biased board thinning),
* fuel / deadhead cost (rebuilds the ``CostModel`` -> evaluator -> objective
  weights -> planners, so the planners optimise the same economics the simulator
  scores them on),
* HOS strictness (``daily_drive_hours``),
* equipment mix (``force_equipment``),
* horizon length (``horizon_days``).

Every condition shares the same base seed (common random numbers): condition
episode *e* faces the same underlying world seed, so only the perturbed parameter
moves between conditions — a variance-reduced comparison.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from application.destination_desirability_service import (
    DestinationDesirabilityService,
)
from application.evaluate_loads import EvaluateLoadsService
from application.ortools_destination_aware_planner import (
    ORToolsDestinationAwarePlanner,
)
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from domain.models.truck_state import TruckState
from domain.policies.ortools_objective_weights import (
    CENTS_PER_DOLLAR,
    ORToolsObjectiveWeights,
)
from ml.data.synthetic_history_generator import GeneratorParams, generate_history
from ml.markets import MARKET_PROFILES, MarketProfile
from simulation.metrics import (
    RollingReplayMetrics,
    paired_comparison,
    summarize_episodes,
)
from simulation.rolling_replay import ReplayConfig, ReplayEpisode, run_episode
from simulation.snapshot_board import (
    EQUIPMENT_ML_TO_DOMAIN,
    ROUND_TRIP_ML_EQUIPMENT,
    SnapshotBoard,
)

PROFIT_KEY = "profit_aware"
DEST_KEY = "destination_aware"

VERDICT_HOLDS = "HOLDS"
VERDICT_NEUTRAL = "NEUTRAL"
VERDICT_REGRESSION = "REGRESSION"
VERDICT_DEST_SKIPPED = "DEST_SKIPPED"


# ---------------------------------------------------------------------------
# Specs and results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldDefaults:
    """Baseline world/truck parameters a condition perturbs (from the 3.3 config)."""

    start_date: datetime
    snapshots_per_day: int
    loads_per_snapshot_mean: float
    unposted_rate_fraction: float
    max_post_age_hours: float
    horizon_days: int
    radius_mi: float
    daily_drive_hours: float


@dataclass(frozen=True)
class ConditionSpec:
    """One stress condition: ``None`` fields fall back to the baseline."""

    name: str
    rationale: str = ""
    loads_per_snapshot_mean: Optional[float] = None
    unposted_rate_fraction: Optional[float] = None
    horizon_days: Optional[int] = None
    competition_take_rate: float = 0.0
    fuel_cost_per_mile: Optional[float] = None
    deadhead_fuel_multiplier: Optional[float] = None
    daily_drive_hours: Optional[float] = None
    force_equipment: Optional[str] = None

    def has_cost_override(self) -> bool:
        return (
            self.fuel_cost_per_mile is not None
            or self.deadhead_fuel_multiplier is not None
        )


@dataclass
class ConditionResult:
    name: str
    rationale: str
    episode_count: int
    effective: Dict[str, Any]
    verdict: str
    profit_aware: Dict[str, Any]
    destination_aware: Optional[Dict[str, Any]] = None
    headline_delta: Optional[Dict[str, float]] = None
    paired_profit: Optional[Dict[str, float]] = None
    paired_deadhead: Optional[Dict[str, float]] = None
    divergence: Optional[Dict[str, Any]] = None
    onward_diagnostic: Optional[Dict[str, float]] = None
    per_episode: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rationale": self.rationale,
            "episode_count": self.episode_count,
            "effective": self.effective,
            "verdict": self.verdict,
            "profit_aware": self.profit_aware,
            "destination_aware": self.destination_aware,
            "headline_delta": self.headline_delta,
            "paired_profit": self.paired_profit,
            "paired_deadhead": self.paired_deadhead,
            "divergence": self.divergence,
            "onward_diagnostic": self.onward_diagnostic,
            "per_episode": self.per_episode,
        }


def classify_verdict(
    paired_profit: Dict[str, float], mean_deadhead_delta: float
) -> str:
    """Robustness label from the paired profit CI and mean paired deadhead delta.

    * ``REGRESSION`` — the paired profit delta CI sits entirely below zero.
    * ``HOLDS`` — profit is no worse (CI low >= 0) *and* deadhead is no worse
      (mean delta <= 0): at least as profitable, with less repositioning.
    * ``NEUTRAL`` — anything in between (CI straddles zero, or profit holds but
      deadhead rises).
    """
    if paired_profit["ci_high"] < 0:
        return VERDICT_REGRESSION
    if paired_profit["ci_low"] >= 0 and mean_deadhead_delta <= 0:
        return VERDICT_HOLDS
    return VERDICT_NEUTRAL


# ---------------------------------------------------------------------------
# World sampling (shared with the Phase 3.3 runner)
# ---------------------------------------------------------------------------

def _weighted(rng: random.Random, items, weights):
    total = sum(weights)
    r = rng.random() * total
    upto = 0.0
    for item, w in zip(items, weights):
        upto += w
        if upto >= r:
            return item
    return items[-1]


def sample_start(
    rng: random.Random, force_equipment: Optional[str] = None
) -> Tuple[MarketProfile, str]:
    """Sample a start hub (weighted by outbound density) + a round-trip trailer.

    With ``force_equipment`` the trailer is pinned (the equipment-mix stress
    axis); otherwise it is drawn from the hub's own posted mix, restricted to the
    three ML codes that round-trip losslessly to the domain trailer vocabulary.
    """
    market = _weighted(
        rng, MARKET_PROFILES, [m.outbound_density for m in MARKET_PROFILES]
    )
    if force_equipment is not None:
        return market, force_equipment
    mix = [(e, w) for e, w in market.equipment_mix if e in ROUND_TRIP_ML_EQUIPMENT]
    ml_equipment = _weighted(rng, [e for e, _ in mix], [w for _, w in mix])
    return market, ml_equipment


def build_truck(
    market: MarketProfile,
    ml_equipment: str,
    start_time: datetime,
    daily_drive_hours: float,
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


def metrics_row(m: RollingReplayMetrics) -> Dict[str, Any]:
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


def headline_delta(
    dest: Dict[str, Any], profit: Dict[str, Any]
) -> Dict[str, float]:
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


def onward_diagnostic(episodes: List[ReplayEpisode]) -> Optional[Dict[str, float]]:
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
    try:
        corr = statistics.correlation(pred, real)
    except statistics.StatisticsError:
        corr = 0.0
    return {
        "pairs": len(pred),
        "correlation": corr,
        "signed_bias_miles": statistics.fmean(p - r for p, r in zip(pred, real)),
        "mae_miles": statistics.fmean(abs(p - r) for p, r in zip(pred, real)),
        "mean_predicted": statistics.fmean(pred),
        "mean_realized": statistics.fmean(real),
    }


# ---------------------------------------------------------------------------
# Cost-stack assembly (cost overrides rebuild evaluator + weights + planners)
# ---------------------------------------------------------------------------

def _build_planners(
    container: Any,
    spec: ConditionSpec,
    base: WorldDefaults,
    solver_time_limit: float,
    destination_weight: float,
    model_path: Optional[Path],
):
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
        base_w = container.config.ortools_objective_weights
        weights = ORToolsObjectiveWeights.from_cost_model(
            new_cost,
            average_speed_mph=avg_speed,
            profit_multiplier=base_w.profit_cents_multiplier / CENTS_PER_DOLLAR,
            skip_profit_floor_dollars=base_w.skip_profit_floor_dollars,
        )
    else:
        evaluator = container.evaluator
        weights = container.config.ortools_objective_weights

    ortools_kwargs = dict(
        distance_provider=evaluator.distance_provider,
        evaluate_loads_service=evaluator,
        constraints=container.config.planning_constraints,
        solver_time_limit_seconds=solver_time_limit,
        average_speed_mph=avg_speed,
        load_unload_hours=load_unload,
    )
    profit_planner = ORToolsProfitAwarePlanner(
        objective_weights=weights, **ortools_kwargs
    )

    dest_planner = None
    dest_service = None
    if model_path is not None and Path(model_path).exists():
        dest_service = DestinationDesirabilityService.from_artifact(Path(model_path))
        dest_planner = ORToolsDestinationAwarePlanner(
            objective_weights=weights,
            destination_service=dest_service,
            destination_weight=destination_weight,
            **ortools_kwargs,
        )

    daily = (
        spec.daily_drive_hours
        if spec.daily_drive_hours is not None
        else base.daily_drive_hours
    )
    replay_config = ReplayConfig(
        radius_mi=base.radius_mi,
        average_speed_mph=avg_speed,
        load_unload_hours=load_unload,
        daily_drive_hours=daily,
    )
    return replay_config, profit_planner, dest_planner, dest_service


# ---------------------------------------------------------------------------
# The condition run
# ---------------------------------------------------------------------------

def run_condition(
    spec: ConditionSpec,
    *,
    container: Any,
    base: WorldDefaults,
    episode_count: int,
    base_seed: int,
    solver_time_limit: float,
    destination_weight: float = 1.0,
    model_path: Optional[Path] = None,
) -> ConditionResult:
    """Replay ``episode_count`` worlds for both planners under ``spec``."""
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
    start_dt = base.start_date
    episode_end = start_dt + timedelta(days=horizon_days)

    replay_config, profit_planner, dest_planner, dest_service = _build_planners(
        container, spec, base, solver_time_limit, destination_weight, model_path
    )
    base_cost = container.config.cost_model
    effective = {
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
        "daily_drive_hours": replay_config.daily_drive_hours,
        "force_equipment": spec.force_equipment,
        "horizon_days": horizon_days,
    }

    profit_metrics: List[RollingReplayMetrics] = []
    dest_metrics: List[RollingReplayMetrics] = []
    dest_episodes: List[ReplayEpisode] = []
    per_episode: Dict[str, List[Dict[str, Any]]] = {PROFIT_KEY: [], DEST_KEY: []}

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
        sample_rng = random.Random(episode_seed * 7919 + 17)
        market, ml_equipment = sample_start(sample_rng, spec.force_equipment)
        truck = build_truck(
            market, ml_equipment, start_dt, replay_config.daily_drive_hours
        )

        profit_ep = run_episode(
            profit_planner,
            SnapshotBoard(
                records,
                competition_take_rate=spec.competition_take_rate,
                competition_seed=episode_seed,
            ),
            truck,
            episode_end=episode_end,
            ml_equipment=ml_equipment,
            config=replay_config,
            episode_seed=episode_seed,
            planner_label="OR-Tools Profit-Aware",
            shadow_planner=dest_planner,
        )
        profit_metrics.append(profit_ep.metrics)
        per_episode[PROFIT_KEY].append(
            {
                "seed": episode_seed,
                "ml_equipment": ml_equipment,
                "start_market": market.name,
                **metrics_row(profit_ep.metrics),
            }
        )

        if dest_planner is not None:
            dest_ep = run_episode(
                dest_planner,
                SnapshotBoard(
                    records,
                    competition_take_rate=spec.competition_take_rate,
                    competition_seed=episode_seed,
                ),
                truck,
                episode_end=episode_end,
                ml_equipment=ml_equipment,
                config=replay_config,
                episode_seed=episode_seed,
                planner_label="OR-Tools Destination-Aware",
                destination_service=dest_service,
            )
            dest_metrics.append(dest_ep.metrics)
            dest_episodes.append(dest_ep)
            per_episode[DEST_KEY].append(
                {
                    "seed": episode_seed,
                    "ml_equipment": ml_equipment,
                    "start_market": market.name,
                    **metrics_row(dest_ep.metrics),
                }
            )

    profit_summary = summarize_episodes(profit_metrics)
    result = ConditionResult(
        name=spec.name,
        rationale=spec.rationale,
        episode_count=episode_count,
        effective=effective,
        verdict=VERDICT_DEST_SKIPPED,
        profit_aware=profit_summary,
        divergence=profit_summary.get("divergence"),
        per_episode=per_episode,
    )

    if dest_metrics:
        dest_summary = summarize_episodes(dest_metrics)
        paired_profit = paired_comparison(
            [m.total_profit for m in profit_metrics],
            [m.total_profit for m in dest_metrics],
            lower_is_better=False,
        )
        paired_deadhead = paired_comparison(
            [m.total_deadhead_miles for m in profit_metrics],
            [m.total_deadhead_miles for m in dest_metrics],
            lower_is_better=True,
        )
        result.destination_aware = dest_summary
        result.headline_delta = headline_delta(dest_summary, profit_summary)
        result.paired_profit = paired_profit
        result.paired_deadhead = paired_deadhead
        result.onward_diagnostic = onward_diagnostic(dest_episodes)
        result.verdict = classify_verdict(paired_profit, paired_deadhead["mean"])

    return result

"""Fleet rolling-horizon replay loop (Phase 8.2).

The multi-truck analogue of :func:`simulation.rolling_replay.run_episode`. Where
the single-truck loop advances one ``TruckSimulator`` over a shared
``SnapshotBoard``, :func:`run_fleet_episode` advances **K** simulators over **one**
shared board, asking a :class:`~ports.fleet_dispatch.FleetDispatchPolicy` to
coordinate which truck takes which load at each decision epoch.

Event-driven clock. At each epoch time ``t``:

1. gather the trucks that are *free* (``available_at <= t``) and HOS-reset them;
2. build each free truck's visible board from its own position and equipment
   (the board is shared, so consumption is shared — no load is double-booked);
3. ``policy.assign(...)`` returns a conflict-free set of assignments;
4. execute each assignment on its truck's simulator and ``mark_consumed`` the load;
5. idle the unassigned free trucks, then jump the clock to the next event — the
   earliest of (a busy truck freeing up) or (the next board snapshot).

Because every event time is strictly greater than ``t``, the clock advances
monotonically and the loop terminates (with a guard as a backstop). With **one**
truck there is no contention, so the trajectory reduces *exactly* to the
single-truck rolling loop — the K=1 invariant that bridges Phase 8 back to Phase 3.

All per-pair financials still flow from ``EvaluateLoadsService`` via the policy's
scorer, so fleet metrics reconcile with every prior phase by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from application.ortools_destination_aware_planner import domain_equipment_to_ml
from domain.models.load import Load
from domain.models.truck_state import TruckState
from ports.fleet_dispatch import FleetDispatchPolicy
from simulation.fleet_metrics import FleetEpisodeMetrics
from simulation.metrics import RollingReplayMetrics
from simulation.snapshot_board import SnapshotBoard
from simulation.truck_simulator import TruckSimulator


@dataclass
class FleetReplayConfig:
    radius_mi: float = 250.0
    average_speed_mph: float = 50.0
    load_unload_hours: float = 1.5
    daily_drive_hours: float = 11.0
    max_epochs: int = 100000


@dataclass
class FleetDecision:
    decision_time: datetime
    truck_id: int
    load_id: int
    profit: float
    revenue: float
    cost: float
    deadhead_miles: float
    loaded_miles: float
    driver_hours: float
    destination_state: str
    default_probability: Optional[float] = None


@dataclass
class FleetEpisode:
    episode_seed: int
    policy_label: str
    decisions: List[FleetDecision] = field(default_factory=list)
    metrics: FleetEpisodeMetrics = field(
        default_factory=lambda: FleetEpisodeMetrics(horizon_hours=0.0)
    )


@runtime_checkable
class FleetRiskScorer(Protocol):
    """Artifact-gated payment-risk hook (Phase 5 model adapts to this in 8.3)."""

    def default_probability(self, load: Load) -> float: ...


def run_fleet_episode(
    policy: FleetDispatchPolicy,
    board: SnapshotBoard,
    trucks: Sequence[TruckState],
    *,
    episode_end: datetime,
    config: FleetReplayConfig,
    episode_seed: int = 0,
    policy_label: str = "",
    risk_scorer: Optional[FleetRiskScorer] = None,
) -> FleetEpisode:
    if not trucks:
        return FleetEpisode(episode_seed=episode_seed, policy_label=policy_label)

    sims: Dict[int, TruckSimulator] = {}
    ml_equipment: Dict[int, str] = {}
    per_truck_horizon: Dict[int, float] = {}
    for t in trucks:
        sims[t.truck_id] = TruckSimulator(
            t,
            average_speed_mph=config.average_speed_mph,
            load_unload_hours=config.load_unload_hours,
            daily_drive_hours=config.daily_drive_hours,
        )
        ml_equipment[t.truck_id] = domain_equipment_to_ml(t.trailer_type)
        per_truck_horizon[t.truck_id] = (
            episode_end - t.available_at
        ).total_seconds() / 3600.0

    fleet_start = min(t.available_at for t in trucks)
    fleet_horizon = (episode_end - fleet_start).total_seconds() / 3600.0

    decisions: List[FleetDecision] = []
    destination_counts: Dict[str, int] = {}
    contention_events = 0

    clock = fleet_start
    guard = 0
    while clock < episode_end and guard < config.max_epochs:
        guard += 1
        free_ids = sorted(
            tid for tid, sim in sims.items() if sim.truck_state.available_at <= clock
        )
        candidates: Dict[int, List[Load]] = {}
        for tid in free_ids:
            sim = sims[tid]
            sim.apply_hos_reset_if_needed()
            ts = sim.truck_state
            candidates[tid] = board.visible_loads_at(
                clock, ts.latitude, ts.longitude, config.radius_mi, ml_equipment[tid]
            )

        if _has_contention(candidates):
            contention_events += 1

        assignments = policy.assign(
            [sims[tid].truck_state for tid in free_ids], candidates, clock
        )

        assigned_ids: set[int] = set()
        load_by_id = {
            load.load_id: load
            for loads in candidates.values()
            for load in loads
        }
        # Execute in a deterministic (truck_id, load_id) order.
        for a in sorted(assignments, key=lambda x: (x.truck_id, x.load_id)):
            load = load_by_id.get(a.load_id)
            if load is None:  # defensive: policy must choose from candidates
                continue
            sim = sims[a.truck_id]
            result = sim.execute_load(load, a.stop)
            board.mark_consumed(a.load_id)
            assigned_ids.add(a.truck_id)
            p_default = (
                risk_scorer.default_probability(load)
                if risk_scorer is not None
                else None
            )
            decisions.append(
                FleetDecision(
                    decision_time=clock,
                    truck_id=a.truck_id,
                    load_id=a.load_id,
                    profit=result.profit,
                    revenue=result.revenue,
                    cost=result.cost,
                    deadhead_miles=result.deadhead_miles,
                    loaded_miles=result.loaded_miles,
                    driver_hours=result.driver_hours,
                    destination_state=load.destination_state,
                    default_probability=p_default,
                )
            )
            destination_counts[load.destination_state] = (
                destination_counts.get(load.destination_state, 0) + 1
            )

        next_clock = _next_event_time(clock, sims, board)
        if next_clock is None or next_clock >= episode_end:
            break
        for tid in free_ids:
            if tid not in assigned_ids:
                sims[tid].idle_to(next_clock)
        clock = next_clock

    metrics = _aggregate_fleet(
        decisions=decisions,
        sims=sims,
        per_truck_horizon=per_truck_horizon,
        fleet_horizon=fleet_horizon,
        destination_counts=destination_counts,
        contention_events=contention_events,
        risk_used=risk_scorer is not None,
    )
    return FleetEpisode(
        episode_seed=episode_seed,
        policy_label=policy_label,
        decisions=decisions,
        metrics=metrics,
    )


def _has_contention(candidates: Dict[int, List[Load]]) -> bool:
    """True if any single load is visible to two or more free trucks at once."""
    seen: Dict[int, int] = {}
    for loads in candidates.values():
        for load in loads:
            seen[load.load_id] = seen.get(load.load_id, 0) + 1
            if seen[load.load_id] >= 2:
                return True
    return False


def _next_event_time(
    clock: datetime,
    sims: Dict[int, TruckSimulator],
    board: SnapshotBoard,
) -> Optional[datetime]:
    """Earliest of: a busy truck freeing up, or the next board snapshot."""
    events: List[datetime] = []
    future_free = [
        sim.truck_state.available_at
        for sim in sims.values()
        if sim.truck_state.available_at > clock
    ]
    if future_free:
        events.append(min(future_free))
    nxt_snap = board.next_snapshot_after(clock)
    if nxt_snap is not None:
        events.append(nxt_snap)
    return min(events) if events else None


def _aggregate_fleet(
    *,
    decisions: List[FleetDecision],
    sims: Dict[int, TruckSimulator],
    per_truck_horizon: Dict[int, float],
    fleet_horizon: float,
    destination_counts: Dict[str, int],
    contention_events: int,
    risk_used: bool,
) -> FleetEpisodeMetrics:
    truck_metrics: List[RollingReplayMetrics] = []
    for tid in sorted(sims):
        m = RollingReplayMetrics(
            horizon_hours=per_truck_horizon[tid],
            idle_hours=sims[tid].idle_hours,
        )
        for d in (d for d in decisions if d.truck_id == tid):
            m.decision_count += 1
            m.feasible_decision_count += 1
            m.loads_completed += 1
            m.total_profit += d.profit
            m.total_revenue += d.revenue
            m.total_cost += d.cost
            m.total_deadhead_miles += d.deadhead_miles
            m.total_loaded_miles += d.loaded_miles
        truck_metrics.append(m)

    expected_collectible: Optional[float] = None
    expected_exposure: Optional[float] = None
    if risk_used:
        expected_collectible = sum(
            d.profit * (1.0 - (d.default_probability or 0.0)) for d in decisions
        )
        expected_exposure = sum(
            d.revenue * (d.default_probability or 0.0) for d in decisions
        )

    return FleetEpisodeMetrics(
        horizon_hours=fleet_horizon,
        truck_metrics=truck_metrics,
        destination_market_counts=dict(destination_counts),
        contention_events=contention_events,
        expected_collectible_profit=expected_collectible,
        expected_default_exposure=expected_exposure,
    )

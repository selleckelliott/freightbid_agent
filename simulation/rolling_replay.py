"""Rolling-horizon (MPC-style) replay loop (Phase 3.3).

One :func:`run_episode` call replays a single synthetic world for a single
planner: observe the visible board, let the planner choose, execute only the
first chosen load, advance the truck, replan — until the horizon ends. The same
``board`` (rebuilt from the same ``episode_seed``) and the same initial truck are
handed to every planner, so the only thing that varies between runs is the
dispatch policy.

The loop also supports two diagnostics without changing the trajectory:

* a **shadow planner** — at each decision point the shadow is asked what it would
  pick from the *identical* (board, truck) inputs, *without executing*; the
  agreement flag feeds ``decision_overlap_rate`` (policy divergence);
* a **destination service** — the Phase 3.1 model's predicted onward-deadhead for
  the load actually selected, reconciled post-hoc against the realized onward
  deadhead (the deadhead of the next executed decision).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from domain.models.load import Load
from domain.models.truck_state import TruckState

from application.ortools_destination_aware_planner import (
    _to_board_load,
    domain_equipment_to_ml,
)
from simulation.metrics import RollingReplayMetrics
from simulation.snapshot_board import SnapshotBoard
from simulation.truck_simulator import TruckSimulator


@dataclass
class ReplayConfig:
    radius_mi: float = 250.0
    average_speed_mph: float = 50.0
    load_unload_hours: float = 1.5
    daily_drive_hours: float = 11.0
    max_decisions: int = 10000


@dataclass
class ReplayDecision:
    decision_time: datetime
    candidate_count: int
    selected_load_id: Optional[int]
    feasible: bool
    revenue: float = 0.0
    cost: float = 0.0
    expected_profit: float = 0.0
    actual_profit: float = 0.0
    deadhead_miles: float = 0.0
    loaded_miles: float = 0.0
    driver_hours: float = 0.0
    predicted_onward: Optional[float] = None
    realized_onward: Optional[float] = None
    shadow_selected_load_id: Optional[int] = None
    agreement: Optional[bool] = None


@dataclass
class ReplayEpisode:
    episode_seed: int
    ml_equipment: str
    planner_label: str
    decisions: List[ReplayDecision] = field(default_factory=list)
    metrics: RollingReplayMetrics = field(default_factory=RollingReplayMetrics)


def _predict_onward(
    service: Any, selected: Load, board_loads: List[Load]
) -> float:
    """Phase 3.1 predicted onward-deadhead for ``selected`` (mirrors the planner)."""
    board = [_to_board_load(l) for l in board_loads if l.load_id != selected.load_id]
    return service.predict_next_deadhead(
        destination_lat=selected.destination_latitude,
        destination_lon=selected.destination_longitude,
        destination_state=selected.destination_state,
        arrival_time=selected.delivery_time,
        equipment_type=domain_equipment_to_ml(selected.equipment_type),
        visible_loads=board,
        load_age_hours=0.0,
        mode="TL",
    )


def run_episode(
    planner: Any,
    board: SnapshotBoard,
    initial_truck: TruckState,
    *,
    episode_end: datetime,
    ml_equipment: str,
    config: ReplayConfig,
    episode_seed: int = 0,
    planner_label: str = "",
    destination_service: Optional[Any] = None,
    shadow_planner: Optional[Any] = None,
) -> ReplayEpisode:
    sim = TruckSimulator(
        initial_truck,
        average_speed_mph=config.average_speed_mph,
        load_unload_hours=config.load_unload_hours,
        daily_drive_hours=config.daily_drive_hours,
    )
    horizon_hours = (
        episode_end - initial_truck.available_at
    ).total_seconds() / 3600.0
    decisions: List[ReplayDecision] = []

    guard = 0
    while sim.truck_state.available_at < episode_end and guard < config.max_decisions:
        guard += 1
        sim.apply_hos_reset_if_needed()
        ts = sim.truck_state
        t = ts.available_at
        board_loads = board.visible_loads_at(
            t, ts.latitude, ts.longitude, config.radius_mi, ml_equipment
        )
        if not board_loads:
            nxt = board.next_snapshot_after(t)
            if nxt is None or nxt >= episode_end:
                break
            sim.idle_to(nxt)
            continue

        plan = planner.build_plan(board_loads, ts)
        selected_id = plan.stops[0].load_id if plan.stops else None

        shadow_selected_id: Optional[int] = None
        agreement: Optional[bool] = None
        if shadow_planner is not None:
            shadow_plan = shadow_planner.build_plan(board_loads, ts)
            shadow_selected_id = (
                shadow_plan.stops[0].load_id if shadow_plan.stops else None
            )
            # Only score divergence at genuine choice points: if *both* planners
            # decline the whole board (e.g. the truck is HOS-depleted and idling),
            # that is a forced idle, not a policy decision, so it is excluded from
            # the overlap denominator. Skip-vs-serve still counts as a disagreement.
            if selected_id is not None or shadow_selected_id is not None:
                agreement = selected_id == shadow_selected_id

        if not plan.stops:
            # Planner declined the entire board: record the (skip) decision so it
            # counts toward divergence, then idle to the next snapshot.
            decisions.append(
                ReplayDecision(
                    decision_time=t,
                    candidate_count=len(board_loads),
                    selected_load_id=None,
                    feasible=False,
                    shadow_selected_load_id=shadow_selected_id,
                    agreement=agreement,
                )
            )
            nxt = board.next_snapshot_after(t)
            if nxt is None or nxt >= episode_end:
                break
            sim.idle_to(nxt)
            continue

        stop = plan.stops[0]
        selected = next((l for l in board_loads if l.load_id == selected_id), None)
        if selected is None:  # defensive: id must be in the candidate set
            break

        predicted_onward = None
        if destination_service is not None:
            predicted_onward = _predict_onward(
                destination_service, selected, board_loads
            )

        result = sim.execute_load(selected, stop)
        board.mark_consumed(selected.load_id)
        decisions.append(
            ReplayDecision(
                decision_time=t,
                candidate_count=len(board_loads),
                selected_load_id=selected.load_id,
                feasible=True,
                revenue=result.revenue,
                cost=result.cost,
                expected_profit=stop.profit,
                actual_profit=result.profit,
                deadhead_miles=result.deadhead_miles,
                loaded_miles=result.loaded_miles,
                driver_hours=result.driver_hours,
                predicted_onward=predicted_onward,
                shadow_selected_load_id=shadow_selected_id,
                agreement=agreement,
            )
        )

    # Realized onward-deadhead: the deadhead the truck actually drove to reach the
    # *next* executed load (censored/None for the final load).
    taken = [d for d in decisions if d.selected_load_id is not None]
    for cur, nxt in zip(taken, taken[1:]):
        cur.realized_onward = nxt.deadhead_miles

    metrics = _aggregate(decisions, sim.idle_hours, horizon_hours)
    return ReplayEpisode(
        episode_seed=episode_seed,
        ml_equipment=ml_equipment,
        planner_label=planner_label,
        decisions=decisions,
        metrics=metrics,
    )


def _aggregate(
    decisions: List[ReplayDecision], idle_hours: float, horizon_hours: float
) -> RollingReplayMetrics:
    m = RollingReplayMetrics(horizon_hours=horizon_hours, idle_hours=idle_hours)
    for d in decisions:
        m.decision_count += 1
        if d.agreement is not None:
            m.shadow_decision_count += 1
            if d.agreement:
                m.shadow_agreements += 1
        if d.selected_load_id is None:
            continue
        m.feasible_decision_count += 1
        m.loads_completed += 1
        m.total_profit += d.actual_profit
        m.total_revenue += d.revenue
        m.total_cost += d.cost
        m.total_deadhead_miles += d.deadhead_miles
        m.total_loaded_miles += d.loaded_miles
        if d.predicted_onward is not None:
            m.selected_predicted_onward_sum += d.predicted_onward
            m.selected_predicted_onward_count += 1
    return m

"""Fleet dispatch policy port (Phase 8.1).

The strategy boundary for *coordinating a fleet*: given the trucks that are free
right now and, per truck, the loads that truck can see on the shared board,
return a conflict-free set of ``Assignment``s. Mirrors the structural
``application.planner.Planner`` interface — implementations conform by shape, so
the benchmark harness swaps an uncoordinated greedy baseline and a globally
optimised assignment policy interchangeably.

Contract every implementation must honour:

* **at most one load per truck** — a truck appears in at most one assignment;
* **at most one truck per load** — no load is double-booked across the fleet;
* **feasible, in-budget pairs only** — infeasible (wrong equipment, over the
  deadhead cap, HOS-blown, window-missed, below the profit floor) pairs are never
  assigned, matching the single-truck engine's own feasibility rule;
* **deterministic** — the same inputs yield the same assignments, in a stable
  order (ascending ``truck_id``).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Protocol, runtime_checkable

from domain.models.fleet import Assignment
from domain.models.load import Load
from domain.models.truck_state import TruckState


@runtime_checkable
class FleetDispatchPolicy(Protocol):
    def assign(
        self,
        trucks: List[TruckState],
        candidates_by_truck_id: Dict[int, List[Load]],
        now: datetime,
    ) -> List[Assignment]: ...

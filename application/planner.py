from typing import List, Protocol, runtime_checkable

from domain.models.load import Load
from domain.models.plan import Plan
from domain.models.truck_state import TruckState


@runtime_checkable
class Planner(Protocol):
    """Single-truck planner interface.

    Both ``PlanBuilderService`` (greedy heuristic) and ``ORToolsPlanner``
    (OR-Tools optimization) conform to this structural interface so the
    benchmark harness can compare them interchangeably.
    """

    def build_plan(
        self,
        loads: List[Load],
        truck_state: TruckState,
        plan_id: int = 1,
    ) -> Plan: ...

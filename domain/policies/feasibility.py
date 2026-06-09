from domain.models.load_evaluation import LoadEvaluation
from domain.models.truck_state import TruckState
from domain.policies.constraints import CostModel, PlanningConstraints


def _normalize(value: str) -> str:
    return (value or "").strip().lower()


def feasibility_checker(
    load_evaluation: LoadEvaluation,
    truck_state: TruckState,
    constraints: PlanningConstraints,
    cost_model: CostModel | None = None,
) -> tuple[bool, str]:
    load = load_evaluation.load

    if load.weight > truck_state.remaining_capacity:
        return False, "Load weight exceeds truck remaining capacity"

    if _normalize(load.equipment_type) != _normalize(truck_state.trailer_type):
        return False, "Truck does not have required trailer type"

    if load_evaluation.deadhead_miles > constraints.max_deadhead_miles:
        return False, "Deadhead miles exceed maximum allowed"

    if load.miles > constraints.max_load_miles:
        return False, "Load miles exceed maximum allowed"

    if load_evaluation.total_miles > constraints.max_total_miles:
        return False, "Total miles exceed maximum allowed"

    if load_evaluation.driver_hours > truck_state.driver_hours_left:
        return False, "Not enough driver hours left for this load"

    if load_evaluation.driver_hours > constraints.max_driver_hours:
        return False, "Driver hours exceed maximum allowed"

    if (
        load_evaluation.pickup_eta is not None
        and load_evaluation.pickup_eta > load.pickup_window_end
    ):
        return False, "Cannot reach pickup before window closes"

    if (
        load_evaluation.delivery_eta is not None
        and load_evaluation.delivery_eta > load.delivery_window_end
    ):
        return False, "Cannot deliver before delivery window closes"

    if cost_model is not None:
        if load_evaluation.total_cost > constraints.max_total_cost:
            return False, "Total cost exceeds maximum allowed"
        if load_evaluation.expected_profit < constraints.min_expected_profit:
            return False, "Expected profit below minimum required"

    return True, "Load is feasible"
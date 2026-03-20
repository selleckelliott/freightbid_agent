from domain.models.load_evaluation import LoadEvaluation
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints


def feasibility_checker(load_evaluation: LoadEvaluation, truck_state: TruckState, constraints: PlanningConstraints) -> tuple[bool, str]:
    # Check if the truck can handle the load's weight and dimensions
    if load_evaluation.load.weight > truck_state.max_load_capacity:
        return False, "Load weight exceeds truck capacity"
    
    # Check if the deadhead miles for the load exceed the maximum allowed
    if load_evaluation.deadhead_miles > constraints.max_deadhead_miles:
        return False, "Deadhead miles exceed maximum allowed"
    
    # Check if the load miles for the load exceed the maximum allowed
    if load_evaluation.load.miles > constraints.max_load_miles:
        return False, "Load miles exceed maximum allowed"
    
    # Check if the truck has the right trailer type for the load
    if load_evaluation.load.equipment_type != truck_state.trailer_type:
        return False, "Truck does not have required trailer type"
    
    # Check if load hours exceed the driver's available hours
    if load_evaluation.driver_hours > truck_state.driver_hours_left:
        return False, "Not enough driver hours left for this load"
    
    return True, "Load is feasible"
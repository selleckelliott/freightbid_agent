def is_load_feasible(load, truck_state, constraints) -> (bool | str):
    # Check if the truck can handle the load's weight and dimensions
    if load.weight > truck_state.max_load_capacity:
        return False, "Load weight exceeds truck capacity"
    
    # Check if the deadhead miles for the load exceed the maximum allowed
    if load.deadhead_miles > constraints.max_deadhead_miles:
        return False, "Deadhead miles exceed maximum allowed"
    
    # Check if the load miles for the load exceed the maximum allowed
    if load.load_miles > constraints.max_load_miles:
        return False, "Load miles exceed maximum allowed"
    
    # Check if the truck has the right trailer type for the load
    if load.trailer_type_required and load.trailer_type_required != truck_state.trailer_type:
        return False, "Truck does not have required trailer type"
    
    return True, "Load is feasible"
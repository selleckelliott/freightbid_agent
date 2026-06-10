"""Integer objective rates for the profit-aware OR-Tools planner (Phase 2.2).

OR-Tools works best with integer costs, so every rate here is expressed in
*cents*. The rates are derived from the business ``CostModel`` rather than
hand-tuned magic numbers, which makes the solver objective a faithful,
integer-scaled mirror of negative expected plan profit:

    minimize   sum(repositioning_arc_miles * deadhead_cost_cents_per_mile)
             + sum(skipped load static profit * profit multiplier)

Minimising that total is equivalent to maximising
``selected static profit - deadhead cost`` — the expected plan profit.
"""
from __future__ import annotations

from dataclasses import dataclass

from domain.policies.constraints import CostModel

CENTS_PER_DOLLAR = 100


@dataclass(frozen=True)
class ORToolsObjectiveWeights:
    """Cents-denominated weights for the profit-aware objective.

    ``deadhead_cost_cents_per_mile``
        Objective cost of one empty/repositioning mile.
    ``profit_cents_multiplier``
        Cents of skip-penalty per dollar of a load's static (deadhead-free)
        expected profit. 100 keeps penalties on the same cents scale as the
        arc costs.
    """

    deadhead_cost_cents_per_mile: int
    profit_cents_multiplier: int = CENTS_PER_DOLLAR

    @classmethod
    def from_cost_model(
        cls,
        cost_model: CostModel,
        average_speed_mph: float,
        deadhead_cost_multiplier: float = 1.0,
        profit_multiplier: float = 1.0,
    ) -> "ORToolsObjectiveWeights":
        """Derive the per-mile deadhead rate from the business cost model.

        An empty mile costs fuel (with the deadhead fuel multiplier),
        maintenance, and the pro-rated driver + opportunity time spent
        driving it at the fleet's average speed. The optional multipliers
        are ablation knobs; both default to the true cost model (1.0).
        """
        if average_speed_mph <= 0:
            raise ValueError("average_speed_mph must be positive")
        dollars_per_mile = (
            cost_model.fuel_cost_per_mile * cost_model.deadhead_fuel_multiplier
            + cost_model.maintenance_cost_per_mile
            + (
                cost_model.driver_cost_per_hour
                + cost_model.time_opportunity_cost_per_hour
            )
            / average_speed_mph
        )
        return cls(
            deadhead_cost_cents_per_mile=max(
                1,
                round(dollars_per_mile * CENTS_PER_DOLLAR * deadhead_cost_multiplier),
            ),
            profit_cents_multiplier=max(
                1, round(CENTS_PER_DOLLAR * profit_multiplier)
            ),
        )

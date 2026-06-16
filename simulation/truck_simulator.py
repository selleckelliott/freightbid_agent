"""Thin truck-advance model for the rolling replay.

The simulator owns **no cost math**. The planners already perform a
position-aware financial replay (deadhead, pickup/delivery ETAs, revenue, cost,
profit) and expose the realized numbers on ``plan.stops[0]`` via
``EvaluateLoadsService``. The simulator's whole job is therefore to:

* read the realized financials straight off the executed stop
  (:meth:`execute_load`), so rolling metrics reconcile exactly with the one-shot
  benchmark;
* advance the truck to the delivered load's destination, set ``available_at`` to
  the stop's ``delivery_eta`` and decrement the driver's remaining drive hours by
  the same duty time the evaluator charged;
* model a simple daily Hours-of-Service reset (:meth:`apply_hos_reset_if_needed`)
  so the truck doesn't run dry forever after day one;
* accrue idle time when no load is taken (:meth:`idle_to`).

Driver hours are recomputed from the stop's mileage with the *same*
``average_speed_mph`` / ``load_unload_hours`` the evaluator uses, so the HOS
ledger stays consistent with the planner's own feasibility accounting (the
``PlanStop`` carries no duty-hours field of its own).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime

from domain.models.load import Load
from domain.models.plan import PlanStop
from domain.models.truck_state import TruckState


@dataclass
class CompletedLoadResult:
    """Realized outcome of executing one load (lifted from the plan stop)."""

    load_id: int
    revenue: float
    cost: float
    profit: float
    deadhead_miles: float
    loaded_miles: float
    driver_hours: float
    pickup_time: datetime
    delivery_time: datetime


class TruckSimulator:
    def __init__(
        self,
        initial_state: TruckState,
        *,
        average_speed_mph: float = 50.0,
        load_unload_hours: float = 1.5,
        daily_drive_hours: float = 11.0,
    ):
        self._state = initial_state
        self._avg_speed = average_speed_mph
        self._load_unload_hours = load_unload_hours
        self._daily_drive_hours = daily_drive_hours
        self._last_reset_date: date = initial_state.available_at.date()
        self.idle_hours: float = 0.0

    @property
    def truck_state(self) -> TruckState:
        return self._state

    # ------------------------------------------------------------- hours of service
    def apply_hos_reset_if_needed(self) -> bool:
        """Restore the daily drive-hour cap when the clock enters a new duty day.

        A deliberately simple v1 HOS model: drive hours are replenished to
        ``daily_drive_hours`` at each calendar-day boundary (an implicit
        overnight reset). Enough to stop the truck from idling forever once the
        first day's hours are spent, without the full 14h/11h/70h rule set.
        """
        today = self._state.available_at.date()
        if today > self._last_reset_date:
            self._last_reset_date = today
            self._state = replace(
                self._state, driver_hours_left=self._daily_drive_hours
            )
            return True
        return False

    def _driver_hours_for(self, stop: PlanStop) -> float:
        miles = stop.deadhead_miles + stop.load_miles
        return miles / self._avg_speed + self._load_unload_hours

    # ----------------------------------------------------------------- execution
    def execute_load(self, load: Load, stop: PlanStop) -> CompletedLoadResult:
        """Advance the truck through ``stop`` (the first stop of a built plan)."""
        driver_hours = self._driver_hours_for(stop)
        result = CompletedLoadResult(
            load_id=load.load_id,
            revenue=stop.revenue,
            cost=stop.cost,
            profit=stop.profit,
            deadhead_miles=stop.deadhead_miles,
            loaded_miles=stop.load_miles,
            driver_hours=driver_hours,
            pickup_time=stop.pickup_eta,
            delivery_time=stop.delivery_eta,
        )
        self._state = replace(
            self._state,
            latitude=load.destination_latitude,
            longitude=load.destination_longitude,
            current_city=load.destination_city,
            current_state=load.destination_state,
            available_at=stop.delivery_eta,
            timestamp=stop.delivery_eta,
            driver_hours_left=max(0.0, self._state.driver_hours_left - driver_hours),
        )
        return result

    def idle_to(self, next_time: datetime) -> float:
        """Advance the clock with no load taken; returns idle hours accrued."""
        if next_time <= self._state.available_at:
            return 0.0
        delta = (next_time - self._state.available_at).total_seconds() / 3600.0
        self.idle_hours += delta
        self._state = replace(
            self._state, available_at=next_time, timestamp=next_time
        )
        return delta

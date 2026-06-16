"""Tests for the thin truck-advance simulator."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from domain.models.load import Load
from domain.models.plan import PlanStop
from domain.models.truck_state import TruckState
from simulation.truck_simulator import TruckSimulator

T0 = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)


def make_truck(available_at=T0, driver_hours_left=11.0):
    return TruckState(
        truck_id=1,
        current_city="Denver",
        current_state="CO",
        latitude=39.74,
        longitude=-104.99,
        available_at=available_at,
        trailer_type="Reefer",
        max_load_capacity=50000.0,
        current_load_id=None,
        home_city="Denver",
        home_state="CO",
        remaining_capacity=50000.0,
        driver_hours_left=driver_hours_left,
        speed=50.0,
        heading=0.0,
        timestamp=available_at,
    )


def make_load(load_id=1):
    return Load(
        load_id=load_id,
        weight=8000.0,
        created_at=T0,
        origin_city="Denver",
        origin_state="CO",
        origin_latitude=39.74,
        origin_longitude=-104.99,
        destination_city="Dallas",
        destination_state="TX",
        destination_latitude=32.78,
        destination_longitude=-96.80,
        pickup_window_start=T0 + timedelta(hours=4),
        pickup_window_end=T0 + timedelta(hours=8),
        delivery_window_start=T0 + timedelta(hours=18),
        delivery_window_end=T0 + timedelta(hours=20),
        miles=300.0,
        total_rate=1200.0,
        equipment_type="Reefer",
    )


def make_stop(load, deadhead=20.0, load_miles=300.0):
    return PlanStop(
        load_id=load.load_id,
        pickup_eta=load.pickup_window_start,
        delivery_eta=load.delivery_window_end,
        deadhead_miles=deadhead,
        load_miles=load_miles,
        revenue=load.total_rate,
        cost=700.0,
        profit=500.0,
    )


def test_execute_load_advances_truck_to_destination():
    sim = TruckSimulator(make_truck(), average_speed_mph=50.0, load_unload_hours=1.5)
    load = make_load()
    result = sim.execute_load(load, make_stop(load, deadhead=20.0, load_miles=300.0))

    state = sim.truck_state
    assert state.latitude == load.destination_latitude
    assert state.longitude == load.destination_longitude
    assert state.current_city == "Dallas"
    assert state.available_at == load.delivery_window_end
    # driver hours decremented by (deadhead+loaded)/speed + load/unload
    expected_hours = (20.0 + 300.0) / 50.0 + 1.5
    assert result.driver_hours == expected_hours
    assert state.driver_hours_left == 11.0 - expected_hours
    assert result.profit == 500.0


def test_execute_load_lifts_financials_from_stop():
    sim = TruckSimulator(make_truck())
    load = make_load()
    stop = make_stop(load)
    result = sim.execute_load(load, stop)
    assert result.revenue == stop.revenue
    assert result.cost == stop.cost
    assert result.profit == stop.profit
    assert result.deadhead_miles == stop.deadhead_miles
    assert result.loaded_miles == stop.load_miles


def test_hos_reset_on_new_calendar_day():
    sim = TruckSimulator(make_truck(driver_hours_left=2.0), daily_drive_hours=11.0)
    # same day -> no reset
    assert sim.apply_hos_reset_if_needed() is False
    assert sim.truck_state.driver_hours_left == 2.0
    # advance into the next day -> reset restores the daily cap
    sim._state = replace(sim._state, available_at=T0 + timedelta(days=1))
    assert sim.apply_hos_reset_if_needed() is True
    assert sim.truck_state.driver_hours_left == 11.0
    # idempotent within that day
    assert sim.apply_hos_reset_if_needed() is False


def test_idle_accrues_hours_and_advances_clock():
    sim = TruckSimulator(make_truck())
    accrued = sim.idle_to(T0 + timedelta(hours=3))
    assert accrued == 3.0
    assert sim.idle_hours == 3.0
    assert sim.truck_state.available_at == T0 + timedelta(hours=3)
    # idling to an earlier/equal time is a no-op
    assert sim.idle_to(T0) == 0.0
    assert sim.idle_hours == 3.0

"""Tests for label construction (Phase 3.1)."""
from datetime import datetime, timedelta, timezone

import pytest

from ml.data.labeling import LabelConfig, build_label
from ml.data.load_history_schema import LoadSnapshotRecord
from ml.geo import haversine_miles

BASE = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
CFG = LabelConfig(search_window_hours=8.0, min_rate_per_mile=1.75, max_deadhead_cap_miles=300.0)

# Completed load delivers here.
DEST_LAT, DEST_LON = 40.0, -111.0


def _rec(
    load_id,
    *,
    equip="F",
    olat=40.0,
    olon=-111.0,
    pickup=BASE,
    arrival=BASE,
    miles=100.0,
    rate=500.0,
):
    return LoadSnapshotRecord(
        snapshot_time=BASE,
        load_id=load_id,
        origin_city="O",
        origin_state="UT",
        origin_lat=olat,
        origin_lon=olon,
        destination_city="D",
        destination_state="UT",
        destination_lat=DEST_LAT,
        destination_lon=DEST_LON,
        pickup_start=pickup,
        pickup_end=pickup + timedelta(hours=2),
        dropoff_start=arrival,
        dropoff_end=arrival + timedelta(hours=2),
        equipment_type=equip,
        loaded_miles=miles,
        posted_at=BASE,
        total_rate=rate,
    )


def test_label_equals_nearest_viable_origin_distance():
    completed = _rec("C", arrival=BASE)
    near = _rec("N", olat=40.5, olon=-111.0, pickup=BASE + timedelta(hours=1))
    far = _rec("F", olat=41.5, olon=-111.0, pickup=BASE + timedelta(hours=2))

    expected = haversine_miles(DEST_LAT, DEST_LON, 40.5, -111.0)
    assert build_label(completed, [near, far], CFG) == pytest.approx(expected)


def test_no_viable_future_load_returns_cap():
    completed = _rec("C")
    assert build_label(completed, [], CFG) == CFG.max_deadhead_cap_miles


def test_equipment_mismatch_is_ignored():
    completed = _rec("C", equip="F")
    mismatch = _rec("M", equip="HS", olat=40.1, pickup=BASE + timedelta(hours=1))
    assert build_label(completed, [mismatch], CFG) == CFG.max_deadhead_cap_miles


def test_search_window_is_respected():
    completed = _rec("C", arrival=BASE)
    inside = _rec("I", olat=40.5, pickup=BASE + timedelta(hours=7))
    outside = _rec("O", olat=40.1, pickup=BASE + timedelta(hours=20))

    # Only the in-window load counts, even though the out-of-window one is closer.
    expected = haversine_miles(DEST_LAT, DEST_LON, 40.5, -111.0)
    assert build_label(completed, [inside, outside], CFG) == pytest.approx(expected)


def test_low_rate_and_unrated_loads_are_not_viable():
    completed = _rec("C")
    cheap = _rec("L", olat=40.1, pickup=BASE + timedelta(hours=1), rate=100.0)  # rpm 1.0
    unrated = _rec("U", olat=40.1, pickup=BASE + timedelta(hours=1), rate=None)
    assert build_label(completed, [cheap, unrated], CFG) == CFG.max_deadhead_cap_miles


def test_label_is_capped_at_max():
    completed = _rec("C", arrival=BASE)
    # Viable but very far origin -> distance exceeds the cap.
    far = _rec("X", olat=10.0, olon=-111.0, pickup=BASE + timedelta(hours=1))
    assert build_label(completed, [far], CFG) == CFG.max_deadhead_cap_miles

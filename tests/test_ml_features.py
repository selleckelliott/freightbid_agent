"""Tests for the decision-time feature builder (Phase 3.1)."""
from datetime import datetime, timedelta, timezone

import math

import pytest

from ml.config import FeatureConfig
from ml.data.load_history_schema import LoadSnapshotRecord
from ml.features.destination_features import DestinationQuery, build_features, daypart

SNAP = datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)
DENVER = (39.7392, -104.9903)


def _board_load(load_id, olat, olon, *, equip="Flatbed", miles=200.0, rate=500.0):
    return LoadSnapshotRecord(
        snapshot_time=SNAP,
        load_id=load_id,
        origin_city="O",
        origin_state="CO",
        origin_lat=olat,
        origin_lon=olon,
        destination_city="D",
        destination_state="CO",
        destination_lat=0.0,
        destination_lon=0.0,
        pickup_start=SNAP,
        pickup_end=SNAP + timedelta(hours=2),
        dropoff_start=SNAP + timedelta(hours=6),
        dropoff_end=SNAP + timedelta(hours=8),
        equipment_type=equip,
        loaded_miles=miles,
        posted_at=SNAP - timedelta(hours=2),
        total_rate=rate,
    )


def _query(equip="Flatbed"):
    return DestinationQuery(
        destination_lat=DENVER[0],
        destination_lon=DENVER[1],
        destination_state="CO",
        equipment_type=equip,
        arrival_dt=datetime(2026, 1, 2, 14, 0, tzinfo=timezone.utc),
        load_age_hours_value=3.0,
    )


def _board():
    return [
        _board_load("b1", 39.80, -104.99, equip="Flatbed", miles=200.0, rate=500.0),
        _board_load("b2", 40.00, -104.99, equip="Reefer", miles=300.0, rate=750.0),
        _board_load("b3", 34.05, -118.24, equip="Flatbed", miles=400.0, rate=1000.0),
    ]


def test_density_and_equipment_counts():
    feats = build_features(_query(), _board(), FeatureConfig())
    # Two CO loads are within every ring; the LA load is excluded.
    assert feats["loads_within_50"] == 2
    assert feats["loads_within_150"] == 2
    # Only the Flatbed CO load matches equipment within 50 mi.
    assert feats["equip_match_within_50"] == 1
    assert feats["posted_loads_near"] == 2
    assert feats["median_rate_per_mile_near"] == pytest.approx(2.5)


def test_zone_and_time_features():
    feats = build_features(_query(), _board(), FeatureConfig())
    assert feats["destination_zone"] == "Denver"
    assert feats["destination_state"] == "CO"
    assert feats["arrival_hour"] == 14
    assert feats["is_weekend"] == 0
    assert feats["sin_hour"] == pytest.approx(math.sin(2 * math.pi * 14 / 24))


def test_no_required_numeric_nulls():
    feats = build_features(_query(), _board(), FeatureConfig())
    for key, value in feats.items():
        if isinstance(value, (int, float)):
            assert value is not None
            assert not math.isnan(value)


def test_candidate_excludes_itself_from_board():
    board = _board()
    # Same destination as the board context, but carrying b1's id.
    me = DestinationQuery(
        destination_lat=DENVER[0],
        destination_lon=DENVER[1],
        destination_state="CO",
        equipment_type="Flatbed",
        arrival_dt=datetime(2026, 1, 2, 14, 0, tzinfo=timezone.utc),
        load_id="b1",
    )
    feats = build_features(me, board, FeatureConfig())
    # b1 shares this load's id and must not be counted; only b2 remains within 50.
    assert feats["loads_within_50"] == 1


def test_empty_board_yields_zero_density():
    feats = build_features(_query(), [], FeatureConfig())
    assert feats["loads_within_50"] == 0
    assert feats["median_rate_per_mile_near"] == 0.0


def test_daypart_buckets():
    assert daypart(8) == "morning"
    assert daypart(13) == "midday"
    assert daypart(19) == "evening"
    assert daypart(2) == "overnight"

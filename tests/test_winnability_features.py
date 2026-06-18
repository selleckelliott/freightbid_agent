"""Tests for the Phase 4.2 winnability feature builder."""
from datetime import datetime, timedelta, timezone
from math import isnan

from ml.data.load_history_schema import LoadSnapshotRecord
from ml.features.winnability_features import (
    CATEGORICAL_COLUMNS,
    NON_FEATURE_COLUMNS,
    BidQuery,
    ask_ratio_bucket,
    build_winnability_features,
    feature_columns,
    market_rate_for,
)
from ml.markets import market_by_name

# Dallas hub coordinates -> origin market avg rpm 2.40 (see ml/markets.py).
_DALLAS_LAT, _DALLAS_LON = 32.7767, -96.7970
_MARKET_RATE = market_by_name("Dallas").avg_rate_per_mile


def _record(total_rate, *, loaded_miles=500.0) -> LoadSnapshotRecord:
    t = datetime(2026, 3, 2, 14, 0, tzinfo=timezone.utc)  # a Monday midday
    return LoadSnapshotRecord(
        snapshot_time=t,
        load_id="L1",
        origin_city="Dallas",
        origin_state="TX",
        origin_lat=_DALLAS_LAT,
        origin_lon=_DALLAS_LON,
        destination_city="Denver",
        destination_state="CO",
        destination_lat=39.7392,
        destination_lon=-104.9903,
        pickup_start=t + timedelta(hours=4),
        pickup_end=t + timedelta(hours=8),
        dropoff_start=t + timedelta(hours=20),
        dropoff_end=t + timedelta(hours=24),
        equipment_type="HS",
        loaded_miles=loaded_miles,
        posted_at=t - timedelta(hours=3),
        total_rate=total_rate,
        weight=9000.0,
        length=40.0,
        mode="TL",
        load_views="low",
        broker_id="BRK0001",
        broker_name="Summit Logistics",
        broker_credit_bucket="A",
        broker_days_to_pay=20,
        broker_bonded=True,
        broker_quick_pay_available=False,
        broker_age_days=900,
        commodity="steel",
        tarp_required=True,
        appointment_required=False,
    )


def test_market_rate_matches_origin_market():
    assert market_rate_for(_DALLAS_LAT, _DALLAS_LON) == _MARKET_RATE


def test_ask_to_market_ratio_is_bid_over_market():
    rec = _record(total_rate=1400.0)
    feats = build_winnability_features(rec, bid_rpm=_MARKET_RATE)  # ratio == 1.0
    assert abs(feats["ask_to_market_ratio"] - 1.0) < 1e-9
    feats_high = build_winnability_features(rec, bid_rpm=_MARKET_RATE * 1.25)
    assert abs(feats_high["ask_to_market_ratio"] - 1.25) < 1e-9


def test_posted_ratio_present_when_rate_posted():
    rec = _record(total_rate=500.0 * 2.0)  # posted rpm = 2.0
    feats = build_winnability_features(rec, bid_rpm=2.2)
    assert feats["has_posted_rate"] == 1.0
    assert abs(feats["ask_to_posted_ratio"] - 1.1) < 1e-9


def test_posted_ratio_is_nan_when_no_rate():
    rec = _record(total_rate=None)
    feats = build_winnability_features(rec, bid_rpm=2.4)
    assert feats["has_posted_rate"] == 0.0
    assert isnan(feats["ask_to_posted_ratio"])


def test_no_latent_fields_in_feature_dict():
    rec = _record(total_rate=1200.0)
    feats = build_winnability_features(rec, bid_rpm=2.4)
    latent = {
        "reservation_rpm",
        "contention_intensity",
        "true_pay_days",
        "true_default_prob",
        "rate_bias",
        "won",
    }
    assert not (set(feats) & latent)


def test_feature_columns_excludes_bookkeeping():
    rec = _record(total_rate=1200.0)
    cols = list(build_winnability_features(rec, bid_rpm=2.4)) + list(NON_FEATURE_COLUMNS)
    selected = feature_columns(cols)
    assert not (set(selected) & set(NON_FEATURE_COLUMNS))
    # Categorical feature columns survive selection.
    for c in CATEGORICAL_COLUMNS:
        assert c in selected


def test_bidquery_matches_record_features():
    rec = _record(total_rate=1000.0)
    query = BidQuery(
        snapshot_time=rec.snapshot_time,
        origin_lat=rec.origin_lat,
        origin_lon=rec.origin_lon,
        equipment_type=rec.equipment_type,
        loaded_miles=rec.loaded_miles,
        posted_at=rec.posted_at,
        mode=rec.mode,
        weight=rec.weight,
        length=rec.length,
        load_views=rec.load_views,
        total_rate=rec.total_rate,
        commodity=rec.commodity,
        tarp_required=rec.tarp_required,
        appointment_required=rec.appointment_required,
        broker_credit_bucket=rec.broker_credit_bucket,
        broker_days_to_pay=rec.broker_days_to_pay,
        broker_bonded=rec.broker_bonded,
        broker_quick_pay_available=rec.broker_quick_pay_available,
        broker_age_days=rec.broker_age_days,
    )
    a = build_winnability_features(rec, bid_rpm=2.3)
    b = build_winnability_features(query, bid_rpm=2.3)
    assert a == b  # no train/serve skew


def test_ask_ratio_bucket_assigns_and_clamps():
    edges = [0.0, 0.90, 0.975, 1.025, 1.10, 1.20, 100.0]
    assert ask_ratio_bucket(0.85, edges) == 0
    assert ask_ratio_bucket(0.95, edges) == 1
    assert ask_ratio_bucket(1.0, edges) == 2
    assert ask_ratio_bucket(1.05, edges) == 3
    assert ask_ratio_bucket(1.15, edges) == 4
    assert ask_ratio_bucket(1.25, edges) == 5
    # NaN degrades into the final bin instead of erroring.
    assert ask_ratio_bucket(float("nan"), edges) == 5

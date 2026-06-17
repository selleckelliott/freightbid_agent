"""Tests for the Phase 4.1 outcome simulator (ml/data/outcome_simulator.py)."""
import random
import statistics
from datetime import datetime, timedelta, timezone

from ml.brokers import (
    BrokerPoolParams,
    build_broker_pool,
    observable_broker_columns,
    sample_broker_for_origin,
)
from ml.data.load_history_schema import LoadSnapshotRecord
from ml.data.outcome_simulator import (
    OutcomeConfig,
    sample_bid_trials,
    simulate_outcomes,
    win_prob,
)

_BASE = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _pool():
    return build_broker_pool(BrokerPoolParams())


def _record(pool, i, *, total_rate=1500.0, load_views="low"):
    broker = sample_broker_for_origin(random.Random(i), pool, "Dallas")
    cols = observable_broker_columns(broker)
    return LoadSnapshotRecord(
        snapshot_time=_BASE + timedelta(hours=i),
        load_id=f"L{i}",
        origin_city="Dallas", origin_state="TX", origin_lat=32.78, origin_lon=-96.80,
        destination_city="Denver", destination_state="CO",
        destination_lat=39.74, destination_lon=-104.99,
        pickup_start=_BASE, pickup_end=_BASE + timedelta(hours=6),
        dropoff_start=_BASE, dropoff_end=_BASE + timedelta(hours=12),
        equipment_type="HS", loaded_miles=600.0, posted_at=_BASE - timedelta(hours=2),
        total_rate=total_rate, mode="TL", load_views=load_views, **cols,
    )


def test_win_prob_decreases_in_bid():
    res = 2.5
    probs = [win_prob(res, bid, 0.06) for bid in (2.0, 2.3, 2.5, 2.7, 3.0)]
    assert all(probs[i] > probs[i + 1] for i in range(len(probs) - 1))
    # At the reservation the broker is indifferent (~50/50).
    assert abs(win_prob(res, res, 0.06) - 0.5) < 1e-6


def test_simulation_is_deterministic():
    pool = _pool()
    cfg = OutcomeConfig()
    recs = [_record(pool, i) for i in range(1, 50)]
    a = simulate_outcomes(recs, pool, cfg)
    b = simulate_outcomes(recs, pool, cfg)
    assert [o.to_json_dict() for o in a] == [o.to_json_dict() for o in b]


def test_higher_contention_covers_faster():
    pool = _pool()
    cfg = OutcomeConfig()
    hi = [_record(pool, i, load_views="high") for i in range(1, 250)]
    lo = [_record(pool, i, load_views="be_the_first") for i in range(1, 250)]
    out_hi = simulate_outcomes(hi, pool, cfg)
    out_lo = simulate_outcomes(lo, pool, cfg)
    # High views => more contention => shorter mean time-to-cover.
    assert statistics.mean(o.contention_intensity for o in out_hi) > statistics.mean(
        o.contention_intensity for o in out_lo
    )
    assert statistics.mean(o.time_to_cover_hours for o in out_hi) < statistics.mean(
        o.time_to_cover_hours for o in out_lo
    )


def test_unposted_loads_require_negotiation():
    pool = _pool()
    cfg = OutcomeConfig()
    posted = _record(pool, 1, total_rate=1500.0)
    unposted = _record(pool, 2, total_rate=None)
    out = simulate_outcomes([posted, unposted], pool, cfg)
    by_id = {o.load_id: o for o in out}
    assert by_id["L1"].negotiation_required is False
    assert by_id["L1"].negotiated_rate is None
    assert by_id["L2"].negotiation_required is True
    assert by_id["L2"].negotiated_rate is not None


def test_reservation_respects_floor():
    pool = _pool()
    cfg = OutcomeConfig(reservation_floor_rpm=1.0)
    recs = [_record(pool, i) for i in range(1, 200)]
    out = simulate_outcomes(recs, pool, cfg)
    assert all(o.reservation_rpm >= cfg.reservation_floor_rpm for o in out)


def test_payment_outcomes_consistent():
    pool = _pool()
    cfg = OutcomeConfig()
    recs = [_record(pool, i) for i in range(1, 400)]
    out = simulate_outcomes(recs, pool, cfg)
    for o in out:
        assert o.payment_outcome in ("paid", "late", "default")
        # Defaulted loads have no realized pay-days; everything else does.
        assert (o.realized_pay_days is None) == (o.payment_outcome == "default")


def test_bid_trials_winrate_falls_as_ask_rises():
    pool = _pool()
    cfg = OutcomeConfig()
    recs = [_record(pool, i) for i in range(1, 400)]
    out = simulate_outcomes(recs, pool, cfg)
    trials = sample_bid_trials(recs, out, cfg)
    mults = cfg.bid_trial_rpm_multipliers
    wins = {m: [0, 0] for m in mults}
    for idx, t in enumerate(trials):
        m = mults[idx % len(mults)]
        wins[m][0] += int(t.won)
        wins[m][1] += 1
    rates = [wins[m][0] / wins[m][1] for m in mults]
    # Cheapest ask wins most often; priciest least often.
    assert rates[0] > rates[-1]

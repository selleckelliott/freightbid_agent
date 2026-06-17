"""Tests for the Phase 4.1 broker pool (ml/brokers.py)."""
import statistics

from ml.brokers import (
    HIDDEN_BROKER_FIELDS,
    OBSERVABLE_BROKER_COLUMNS,
    BrokerPoolParams,
    build_broker_pool,
    broker_index,
    observable_broker_columns,
    sample_broker_for_origin,
)
import random


def _pool(**overrides):
    return build_broker_pool(BrokerPoolParams(**overrides))


def test_pool_is_deterministic():
    a = _pool(pool_size=60)
    b = _pool(pool_size=60)
    assert [x.__dict__ for x in a] == [x.__dict__ for x in b]


def test_unknown_credit_fraction_honored():
    params = BrokerPoolParams(pool_size=100, unknown_credit_fraction=0.18)
    pool = build_broker_pool(params)
    unknown = [b for b in pool if b.credit_bucket == "unknown"]
    assert len(unknown) == round(0.18 * 100)
    # Unknown credit hides the bucket AND days-to-pay, but the latent truth remains.
    for b in unknown:
        assert b.days_to_pay is None
        assert not b.is_credit_known
        assert b.true_pay_days > 0  # ground truth still exists


def test_quality_correlations_have_expected_sign():
    pool = _pool(pool_size=400)
    bonded = [b.true_pay_days for b in pool if b.bonded]
    not_bonded = [b.true_pay_days for b in pool if not b.bonded]
    # Bonded brokers pay sooner and default less, on average.
    assert statistics.mean(bonded) < statistics.mean(not_bonded)
    bonded_def = [b.true_default_prob for b in pool if b.bonded]
    not_bonded_def = [b.true_default_prob for b in pool if not b.bonded]
    assert statistics.mean(bonded_def) < statistics.mean(not_bonded_def)
    # A-credit brokers pay sooner than C-credit brokers.
    a_days = [b.true_pay_days for b in pool if b.credit_bucket == "A"]
    c_days = [b.true_pay_days for b in pool if b.credit_bucket == "C"]
    assert statistics.mean(a_days) < statistics.mean(c_days)


def test_observable_columns_exclude_latents():
    pool = _pool(pool_size=10)
    cols = observable_broker_columns(pool[0])
    assert set(cols) == set(OBSERVABLE_BROKER_COLUMNS)
    assert not (set(cols) & set(HIDDEN_BROKER_FIELDS))


def test_broker_index_round_trips():
    pool = _pool(pool_size=20)
    idx = broker_index(pool)
    assert idx[pool[5].broker_id] is pool[5]


def test_sample_for_origin_prefers_home_market():
    pool = _pool(pool_size=120)
    # Pick the market that the most brokers call home.
    from collections import Counter
    home_counts = Counter(b.home_market for b in pool)
    market, _ = home_counts.most_common(1)[0]
    base_share = home_counts[market] / len(pool)
    rng = random.Random(7)
    picks = [sample_broker_for_origin(rng, pool, market) for _ in range(3000)]
    sampled_share = sum(1 for b in picks if b.home_market == market) / len(picks)
    assert sampled_share > base_share  # home_bias tilts sampling toward local brokers

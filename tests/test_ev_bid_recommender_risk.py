"""Phase 5.1 — risk-adjusted EV objective tests.

The recommender's objective shifts from "expected profit if won" to "expected
*collectible* profit after default and payment-delay risk". These tests pin the formula,
the safer-broker preference, the honest "every ask loses money" warning, and — most
importantly — that the feature is **inert** when the flag is off or no payment model is
wired (byte-identical Phase 4.3b behavior). They drive the recommender through stub
:class:`WinnabilityPort` / :class:`PaymentRiskPort` implementations, no artifact needed.
"""
from __future__ import annotations

import dataclasses
import math
from datetime import datetime, timedelta
from typing import Callable, Optional

import pytest

from application.bid_recommender import BidRecommenderService
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from domain.models.bid_recommendation import MAX_EV, TARGET
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from ml.features.winnability_features import BidQuery
from ports.payment_risk import PaymentEstimate, PaymentRiskPort
from ports.winnability import WinnabilityPort

_DALLAS = (32.78, -96.80)


def _cfg(**overrides) -> BidRecommenderConfig:
    base = load_bid_recommender_config("config")
    return dataclasses.replace(base, **overrides) if overrides else base


def _risk_cfg(**overrides) -> BidRecommenderConfig:
    return _cfg(risk_adjusted_ev_enabled=True, **overrides)


def _query(total_rate: Optional[float] = None, loaded_miles: float = 300.0) -> BidQuery:
    t = datetime(2026, 1, 5, 12, 0, 0)
    return BidQuery(
        snapshot_time=t,
        origin_lat=_DALLAS[0],
        origin_lon=_DALLAS[1],
        equipment_type="F",
        loaded_miles=loaded_miles,
        posted_at=t - timedelta(hours=3),
        total_rate=total_rate,
    )


class _ConstantPort(WinnabilityPort):
    def __init__(self, p: float) -> None:
        self._p = p

    def win_probabilities(self, query, bid_rpms):
        return [self._p for _ in bid_rpms]


class _CurvePort(WinnabilityPort):
    def __init__(self, fn: Callable[[float], float]) -> None:
        self._fn = fn

    def win_probabilities(self, query, bid_rpms):
        return [max(0.0, min(1.0, self._fn(r))) for r in bid_rpms]


def _interior_curve(r: float) -> float:
    return 2.392 - 0.756 * r


class _StubPayment(PaymentRiskPort):
    """Returns a fixed :class:`PaymentEstimate` regardless of the query."""

    def __init__(self, estimate: Optional[PaymentEstimate]) -> None:
        self._estimate = estimate

    def estimate(self, query):
        return self._estimate


# --- 1. Same raw EV, prefer the safer broker -----------------------------------
def test_risk_adjusted_ev_prefers_safer_broker():
    q = _query(total_rate=720.0)
    safe = PaymentEstimate(p_default=0.05, p_collect=0.95, expected_pay_days=28.0)
    risky = PaymentEstimate(p_default=0.45, p_collect=0.55, expected_pay_days=28.0)

    safe_scoring = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(safe)
    ).score(q)
    risky_scoring = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(risky)
    ).score(q)

    # Raw EV is identical (payment never touches it) — only the risk-adjusted EV moves.
    assert [c.ask_amount for c in safe_scoring.candidates] == [
        c.ask_amount for c in risky_scoring.candidates
    ]
    assert [c.expected_value for c in safe_scoring.candidates] == [
        c.expected_value for c in risky_scoring.candidates
    ]
    for cs, cr in zip(safe_scoring.candidates, risky_scoring.candidates):
        assert cs.risk_adjusted_ev > cr.risk_adjusted_ev

    safe_rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(safe)
    ).recommend(q, load_id=1)
    risky_rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(risky)
    ).recommend(q, load_id=1)
    assert safe_rec.payment_risk_available and risky_rec.payment_risk_available
    safe_opt = safe_rec.option(safe_rec.recommended_label)
    risky_opt = risky_rec.option(risky_rec.recommended_label)
    assert safe_opt.risk_adjusted_ev > risky_opt.risk_adjusted_ev


# --- 2. Flag off => byte-identical, even with a payment port attached -----------
def test_flag_off_is_byte_identical():
    q = _query(total_rate=720.0)
    est = PaymentEstimate(0.30, 0.70, 50.0)
    base = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(q, load_id=1)
    with_port = EVBidRecommender(
        _CurvePort(_interior_curve), _cfg(), payment=_StubPayment(est)
    ).recommend(q, load_id=1)
    assert with_port == base
    assert with_port.payment_risk_available is False
    assert all(o.risk_adjusted_ev is None for o in with_port.options)


# --- 3. Payment model missing => byte-identical, even with the flag on ----------
def test_payment_missing_is_byte_identical():
    q = _query(total_rate=720.0)
    base = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(q, load_id=1)

    none_port = EVBidRecommender(
        _CurvePort(_interior_curve), _risk_cfg(), payment=_StubPayment(None)
    ).recommend(q, load_id=1)
    no_port = EVBidRecommender(
        _CurvePort(_interior_curve), _risk_cfg(), payment=None
    ).recommend(q, load_id=1)

    assert none_port == base
    assert no_port == base


# --- 4. Pay-days unavailable => no delay penalty, default risk still applies -----
def test_pay_days_none_drops_delay_but_keeps_default_discount():
    q = _query(total_rate=720.0)
    est = PaymentEstimate(p_default=0.10, p_collect=0.90, expected_pay_days=None)
    rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(est)
    ).recommend(q, load_id=1)
    o = rec.option(rec.recommended_label)
    assert o.expected_pay_days is None
    assert o.delay_penalty == 0.0
    # p_collect discount still bites: collectible profit < raw profit => lower EV.
    assert o.risk_adjusted_ev < o.expected_value
    assert o.expected_collected_revenue == pytest.approx(round(o.ask_amount * 0.90, 2))


# --- 5. Pay-days within terms => no delay penalty -------------------------------
def test_pay_days_within_free_window_has_no_delay_penalty():
    q = _query(total_rate=720.0)
    est = PaymentEstimate(p_default=0.05, p_collect=0.95, expected_pay_days=20.0)
    rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(est)
    ).recommend(q, load_id=1)
    o = rec.option(rec.recommended_label)
    assert o.expected_pay_days == 20.0
    assert o.delay_penalty == 0.0


# --- 6. p_default is clamped to [0, 1] -----------------------------------------
def test_p_default_is_clamped():
    q = _query(total_rate=720.0)
    high = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(
            PaymentEstimate(p_default=1.5, p_collect=-0.5, expected_pay_days=40.0)
        )
    ).recommend(q, load_id=1).option(MAX_EV)
    assert high.p_default == 1.0
    assert high.p_collect == 0.0
    assert high.expected_collected_revenue == 0.0

    low = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(
            PaymentEstimate(p_default=-0.2, p_collect=1.2, expected_pay_days=20.0)
        )
    ).recommend(q, load_id=1).option(MAX_EV)
    assert low.p_default == 0.0
    assert low.p_collect == 1.0


# --- 7. High default probability drives risk-adjusted EV negative ---------------
def test_high_default_makes_risk_adjusted_ev_negative():
    q = _query(total_rate=720.0)
    est = PaymentEstimate(p_default=0.95, p_collect=0.05, expected_pay_days=45.0)
    rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(), payment=_StubPayment(est)
    ).recommend(q, load_id=1)
    assert rec.options
    assert all(o.risk_adjusted_ev < 0 for o in rec.options)


# --- 8. All-negative => still recommends, but flags it (surface, don't block) ----
def test_all_negative_still_recommends_with_warning():
    q = _query(total_rate=720.0)
    risky = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(),
        payment=_StubPayment(PaymentEstimate(0.95, 0.05, 45.0)),
    ).recommend(q, load_id=1)
    assert risky.payment_risk_available is True
    assert risky.risk_adjusted_ev_positive is False
    assert risky.risk_adjusted_warning == "All candidate asks have negative risk-adjusted EV."
    # Not suppressed — a best (least-negative) option is still returned.
    assert risky.option(risky.recommended_label) is not None
    assert risky.recommended_ask > 0

    safe = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(),
        payment=_StubPayment(PaymentEstimate(0.02, 0.98, 25.0)),
    ).recommend(q, load_id=1)
    assert safe.risk_adjusted_ev_positive is True
    assert safe.risk_adjusted_warning is None


# --- 9. Risk fields are finite or None, never NaN ------------------------------
def test_risk_fields_are_finite_or_none():
    q = _query(total_rate=720.0)
    rec = EVBidRecommender(
        _ConstantPort(0.6), _risk_cfg(),
        payment=_StubPayment(PaymentEstimate(0.10, 0.90, 40.0)),
    ).recommend(q, load_id=1)
    for o in rec.options:
        for v in (
            o.risk_adjusted_ev, o.p_default, o.p_collect, o.delay_penalty,
            o.expected_collected_revenue, o.risk_adjusted_profit_if_won,
            o.expected_pay_days,
        ):
            assert v is not None and math.isfinite(v)

    off = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(q, load_id=1)
    for o in off.options:
        assert o.risk_adjusted_ev is None
        assert o.p_default is None
        assert o.ranking_ev == o.expected_value  # property collapses to raw EV


# --- 10. Formula matches a hand computation ------------------------------------
def test_risk_adjusted_formula_matches_by_hand():
    q = _query(total_rate=720.0)
    est = PaymentEstimate(p_default=0.20, p_collect=0.80, expected_pay_days=45.0)
    rec = EVBidRecommender(
        _ConstantPort(0.5), _risk_cfg(), payment=_StubPayment(est)
    ).recommend(q, load_id=1)
    o = rec.option(MAX_EV)

    p_collect = 0.80
    exp_rev = o.ask_amount * p_collect
    delay = exp_rev * 0.18 * (45.0 - 30.0) / 365.0
    ra_profit = exp_rev - o.estimated_cost - delay
    ra_ev = 0.5 * ra_profit

    assert o.p_default == 0.20
    assert o.p_collect == 0.80
    assert o.expected_collected_revenue == pytest.approx(round(exp_rev, 2))
    assert o.delay_penalty == pytest.approx(round(delay, 2))
    assert o.risk_adjusted_profit_if_won == pytest.approx(round(ra_profit, 2))
    assert o.risk_adjusted_ev == pytest.approx(round(ra_ev, 2))


# --- 11. Surfacing: BidRange carries risk fields on, None off -------------------
def _evaluation_with_cost(cost: float, miles: float) -> LoadEvaluation:
    t = datetime(2026, 1, 5, 12, 0, 0)
    load = Load(
        load_id=11,
        weight=10000.0,
        created_at=t,
        origin_city="Dallas",
        origin_state="TX",
        origin_latitude=_DALLAS[0],
        origin_longitude=_DALLAS[1],
        destination_city="Houston",
        destination_state="TX",
        destination_latitude=29.76,
        destination_longitude=-95.37,
        pickup_window_start=t,
        pickup_window_end=t + timedelta(hours=6),
        delivery_window_start=t + timedelta(hours=8),
        delivery_window_end=t + timedelta(hours=14),
        miles=miles,
        total_rate=miles * 2.40,
        equipment_type="Flatbed",
    )
    return LoadEvaluation(
        load=load,
        deadhead_miles=0.0,
        total_miles=miles,
        driver_hours=miles / 50.0,
        expected_revenue=load.total_rate,
        total_cost=cost,
    )


def _service(payment: Optional[PaymentRiskPort], enabled: bool) -> BidRecommenderService:
    from domain.policies.constraints import BiddingConstraints
    from domain.policies.scoring_weights import BidPolicy

    cfg = _risk_cfg() if enabled else _cfg()
    ev = EVBidRecommender(_ConstantPort(0.6), cfg, payment=payment)
    return BidRecommenderService(
        BidPolicy(),
        BiddingConstraints(
            min_bid_amount=100.0,
            max_bid_amount=25000.0,
            min_rate_per_mile=1.00,
            max_rate_per_mile=6.00,
            bidding_time_limit_seconds=5,
        ),
        ev_recommender=ev,
    )


def test_surfacing_populated_when_on_and_none_when_off():
    evaluation = _evaluation_with_cost(cost=1.39 * 300.0, miles=300.0)

    on = _service(_StubPayment(PaymentEstimate(0.15, 0.85, 40.0)), enabled=True).recommend(
        evaluation
    )
    assert on.payment_risk_available is True
    assert on.risk_adjusted_ev_at_target is not None
    assert on.p_default_at_target == 0.15
    assert on.p_collect_at_target == 0.85
    assert on.expected_pay_days_at_target == 40.0
    assert on.risk_adjusted_ev_positive in (True, False)

    off = _service(_StubPayment(PaymentEstimate(0.15, 0.85, 40.0)), enabled=False).recommend(
        evaluation
    )
    assert off.payment_risk_available is None
    assert off.risk_adjusted_ev_at_target is None
    assert off.p_default_at_target is None
    assert off.risk_adjusted_warning is None

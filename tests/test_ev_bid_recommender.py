"""Phase 4.3 — EV bid recommender tests.

These exercise the recommender through stub :class:`WinnabilityPort` implementations
(no model artifact needed): the EV math, the market-anchored candidate guardrails, the
ladder selection, the extrapolation guard, determinism, leakage discipline, and the
no-model cost-plus-margin fallback (which must reproduce the existing
``BidRecommenderService`` target).
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta
from math import isnan
from typing import Callable, List, Optional, Sequence

import pytest

from adapters.outbound.winnability.noop_adapter import NoopWinnabilityAdapter
from application.bid_recommender import BidRecommenderService
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from domain.models.bid_recommendation import (
    CONSERVATIVE,
    LADDER_LABELS,
    MAX_EV,
    STRETCH,
    TARGET,
)
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.policies.constraints import BiddingConstraints
from domain.policies.scoring_weights import BidPolicy
from ml.features.winnability_features import BidQuery, market_rate_for
from ports.winnability import WinnabilityPort

_DALLAS = (32.78, -96.80)


def _cfg() -> BidRecommenderConfig:
    return load_bid_recommender_config("config")


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
    """``P(win)`` from an explicit (usually decreasing) function of ask rpm."""

    def __init__(self, fn: Callable[[float], float]) -> None:
        self._fn = fn
        self.seen_queries: List[BidQuery] = []
        self.seen_rpms: List[List[float]] = []

    def win_probabilities(self, query, bid_rpms):
        self.seen_queries.append(query)
        self.seen_rpms.append(list(bid_rpms))
        return [max(0.0, min(1.0, self._fn(r))) for r in bid_rpms]


def _interior_curve(r: float) -> float:
    # ~0.85 at 2.04/mi down to ~0.20 at 2.88/mi -> an interior EV optimum near 2.28.
    return 2.392 - 0.756 * r


# --- 1. EV identity ------------------------------------------------------------
def test_expected_value_equals_probability_times_profit():
    rec = EVBidRecommender(_ConstantPort(0.5), _cfg()).recommend(
        _query(total_rate=720.0), load_id=1
    )
    assert rec.winnability_available
    assert rec.options
    for o in rec.options:
        assert o.expected_value == pytest.approx(round(0.5 * o.profit_if_won, 2))


# --- 2. Margin-vs-win tradeoff: a shallow win curve pushes max-EV to a higher ask ---
def test_max_ev_prefers_higher_ask_when_win_curve_is_shallow():
    shallow = EVBidRecommender(
        _CurvePort(lambda r: 0.95 - 0.02 * (r - 2.0)), _cfg()
    ).recommend(_query(total_rate=720.0), load_id=2)
    steep = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(
        _query(total_rate=720.0), load_id=2
    )
    # A flatter win curve lets profit dominate, so max-EV sits at a higher ask.
    assert shallow.option(MAX_EV).ask_rpm > steep.option(MAX_EV).ask_rpm
    mx = shallow.option(MAX_EV)
    assert mx.expected_value == max(o.expected_value for o in shallow.options)


# --- 3. Guardrails: every returned candidate clears the profit + margin floors ---
def test_guardrails_enforced_on_every_option():
    cfg = _cfg()
    miles = 300.0
    rec = EVBidRecommender(_ConstantPort(0.6), cfg).recommend(
        _query(total_rate=720.0, loaded_miles=miles), load_id=3
    )
    breakeven_rpm = cfg.cost_per_loaded_mile
    for o in rec.options:
        assert o.profit_if_won >= cfg.min_profit_dollars - 1e-6
        assert o.ask_rpm >= breakeven_rpm + cfg.min_margin_rpm - 1e-6
        assert o.ask_rpm <= cfg.max_rate_per_mile + 1e-6


# --- 4. Missing posted rate falls back to the market anchor --------------------
def test_missing_posted_rate_uses_market_anchor():
    port_a = _CurvePort(_interior_curve)
    port_b = _CurvePort(_interior_curve)
    rec_none = EVBidRecommender(port_a, _cfg()).recommend(_query(total_rate=None), load_id=4)
    rec_posted = EVBidRecommender(port_b, _cfg()).recommend(
        _query(total_rate=720.0), load_id=4  # posted == market (2.40/mi)
    )
    assert rec_none.winnability_available and rec_none.options
    # Posted == market, so the anchored ladder is identical to the no-posted case.
    assert [o.ask_rpm for o in rec_none.options] == [o.ask_rpm for o in rec_posted.options]


# --- 5. Ladder labels are a valid, ordered subset ------------------------------
def test_ladder_labels_valid_and_ordered():
    rec = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(
        _query(total_rate=720.0), load_id=5
    )
    labels = [o.label for o in rec.options]
    assert set(labels) <= set(LADDER_LABELS)
    assert len(labels) == len(set(labels))  # no duplicate rungs
    assert rec.recommended_label in (TARGET, MAX_EV)
    assert rec.option(rec.recommended_label) is not None
    assert rec.recommended_ask == rec.option(rec.recommended_label).ask_amount

    cons, tgt, stretch = rec.option(CONSERVATIVE), rec.option(TARGET), rec.option(STRETCH)
    if cons and tgt:
        assert cons.ask_rpm <= tgt.ask_rpm
    if tgt and stretch:
        assert tgt.ask_rpm <= stretch.ask_rpm


# --- 6. Extrapolation guard: out-of-support asks are flagged + excluded ---------
def test_extrapolated_candidates_excluded_from_ladder():
    cfg = _cfg()
    # Posted 3.5/mi -> ask_to_market_ratio 1.46, outside the trained [0.85, 1.25] band.
    q = _query(total_rate=300.0 * 3.5)
    # An *increasing* curve would otherwise drag the pick to the highest (out-of-support) ask.
    rec = EVBidRecommender(_CurvePort(lambda r: min(0.99, 0.3 + 0.2 * r)), cfg).recommend(
        q, load_id=6
    )
    assert rec.options
    market = market_rate_for(*_DALLAS)
    for o in rec.options:
        assert not o.extrapolated
        ratio = o.ask_rpm / market
        assert cfg.trained_ask_ratio_min - 1e-9 <= ratio <= cfg.trained_ask_ratio_max + 1e-9
    assert all(abs(o.ask_rpm - 3.5) > 1e-6 for o in rec.options)


# --- 7. Determinism ------------------------------------------------------------
def test_recommendation_is_deterministic():
    r1 = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(
        _query(total_rate=720.0), load_id=7
    )
    r2 = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).recommend(
        _query(total_rate=720.0), load_id=7
    )
    assert r1 == r2


# --- 8. No leakage: the port only ever sees the BidQuery + the ask grid ---------
def test_recommender_passes_no_oracle_to_the_port():
    port = _CurvePort(_interior_curve)
    q = _query(total_rate=720.0)
    EVBidRecommender(port, _cfg()).recommend(q, load_id=8)
    assert port.seen_queries and port.seen_queries[0] is q
    latent = {"reservation_rpm", "contention_intensity", "true_pay_days", "true_default_prob"}
    seen_fields = {f.name for f in dataclasses.fields(port.seen_queries[0])}
    assert not (seen_fields & latent)
    assert all(isinstance(r, float) for r in port.seen_rpms[0])


# --- 9. No-model fallback == today's cost-plus-margin behavior ------------------
def test_noop_adapter_reproduces_margin_recommender():
    cfg = _cfg()
    miles = 300.0
    cost = cfg.cost_per_loaded_mile * miles
    rec = EVBidRecommender(NoopWinnabilityAdapter(), cfg).recommend(
        _query(total_rate=720.0, loaded_miles=miles), load_id=9
    )
    assert rec.winnability_available is False
    assert [o.label for o in rec.options] == [TARGET]
    assert isnan(rec.options[0].expected_value)

    service = BidRecommenderService(
        BidPolicy(),
        BiddingConstraints(
            min_bid_amount=100.0,
            max_bid_amount=25000.0,
            min_rate_per_mile=1.00,
            max_rate_per_mile=6.00,
            bidding_time_limit_seconds=5,
        ),
    )
    evaluation = _evaluation_with_cost(cost, miles)
    expected_target = round(service.recommend(evaluation).target_bid, 2)
    assert rec.recommended_ask == pytest.approx(expected_target)


# --- 10. Public score() exposes the full candidate curve (benchmark/chart reuse) ---
def test_score_returns_full_candidate_curve():
    from domain.models.bid_recommendation import CandidateScoring

    # High posted rate -> at least one out-of-support (extrapolated) candidate exists.
    q = _query(total_rate=300.0 * 3.5)
    scoring = EVBidRecommender(_CurvePort(_interior_curve), _cfg()).score(q)
    assert isinstance(scoring, CandidateScoring)
    assert scoring.candidates
    assert any(c.extrapolated for c in scoring.candidates)  # 3.5/mi is out of support
    for c in scoring.candidates:
        assert c.expected_value == pytest.approx(round(c.win_probability * c.profit_if_won, 2))


def test_score_returns_none_without_model():
    rec = EVBidRecommender(NoopWinnabilityAdapter(), _cfg())
    assert rec.score(_query(total_rate=720.0)) is None


def _evaluation_with_cost(cost: float, miles: float) -> LoadEvaluation:
    """Build a minimal ``LoadEvaluation`` carrying ``total_cost`` so the no-model
    fallback can be reconciled against the existing ``BidRecommenderService``."""
    t = datetime(2026, 1, 5, 12, 0, 0)
    load = Load(
        load_id=9,
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

"""Phase 4.3b — EV surfacing through the live BidRecommenderService.

These pin the *additive* contract: with no EV recommender wired the service is
byte-identical to the pre-4.3 cost-plus-margin recommender; with one wired it annotates
the same ``BidRange`` with the EV ladder **without changing** ``min``/``target``/``max``.
A fake :class:`WinnabilityPort` (no model artifact needed) drives the EV path, and the
graceful "enabled but artifact missing" container wiring is covered too.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

from adapters.inbound.api.container import _build_ev_recommender
from adapters.outbound.winnability.noop_adapter import NoopWinnabilityAdapter
from application.bid_recommender import BidRecommenderService, bid_query_from_load
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from domain.models.bid_recommendation import MAX_EV, TARGET, BidOption
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.policies.constraints import BiddingConstraints
from domain.policies.scoring_weights import BidPolicy
from ml.features.winnability_features import BidQuery
from ports.winnability import WinnabilityPort

_DALLAS = (32.78, -96.80)
_T = datetime(2026, 1, 5, 12, 0, 0)


def _cfg() -> BidRecommenderConfig:
    return load_bid_recommender_config("config")


def _constraints() -> BiddingConstraints:
    return BiddingConstraints(
        min_bid_amount=100.0,
        max_bid_amount=25000.0,
        min_rate_per_mile=1.00,
        max_rate_per_mile=6.00,
        bidding_time_limit_seconds=5,
    )


def _evaluation(cost: float = 417.0, miles: float = 300.0, load_id: int = 1) -> LoadEvaluation:
    load = Load(
        load_id=load_id,
        weight=10000.0,
        created_at=_T,
        origin_city="Dallas",
        origin_state="TX",
        origin_latitude=_DALLAS[0],
        origin_longitude=_DALLAS[1],
        destination_city="Houston",
        destination_state="TX",
        destination_latitude=29.76,
        destination_longitude=-95.37,
        pickup_window_start=_T,
        pickup_window_end=_T + timedelta(hours=6),
        delivery_window_start=_T + timedelta(hours=8),
        delivery_window_end=_T + timedelta(hours=14),
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


class _CurvePort(WinnabilityPort):
    def __init__(self, fn: Callable[[float], float]) -> None:
        self._fn = fn

    def win_probabilities(self, query: BidQuery, bid_rpms):
        return [max(0.0, min(1.0, self._fn(r))) for r in bid_rpms]


def _interior_curve(r: float) -> float:
    # ~0.85 at 2.04/mi down to ~0.20 at 2.88/mi -> an interior EV optimum.
    return 2.392 - 0.756 * r


def _ev_service() -> BidRecommenderService:
    ev = EVBidRecommender(_CurvePort(_interior_curve), _cfg())
    return BidRecommenderService(BidPolicy(), _constraints(), ev_recommender=ev)


# --- 1. No EV recommender => byte-identical, EV fields all None -----------------
def test_no_ev_recommender_leaves_ev_fields_none():
    bid = BidRecommenderService(BidPolicy(), _constraints()).recommend(_evaluation())
    assert bid.winnability_available is None
    assert bid.win_probability_at_target is None
    assert bid.expected_value_at_target is None
    assert bid.ev_recommended_bid is None
    assert bid.ev_recommended_label is None
    assert bid.ev_recommended_rate_per_mile is None
    assert bid.ladder is None


# --- 2. Model on surfaces the EV ladder WITHOUT moving the margin bid -----------
def test_model_on_surfaces_ev_ladder_additively():
    evaluation = _evaluation()
    baseline = BidRecommenderService(BidPolicy(), _constraints()).recommend(evaluation)
    enriched = _ev_service().recommend(evaluation, decided_at=_T + timedelta(hours=3))

    # The legacy cost-plus-margin bid is untouched (additive-only).
    assert (enriched.min_bid, enriched.target_bid, enriched.max_bid) == (
        baseline.min_bid,
        baseline.target_bid,
        baseline.max_bid,
    )
    assert enriched.breakeven == baseline.breakeven
    assert enriched.rationale == baseline.rationale

    # EV fields are now populated and finite.
    assert enriched.winnability_available is True
    assert enriched.ev_recommended_label in (TARGET, MAX_EV)
    assert enriched.ev_recommended_bid is not None and enriched.ev_recommended_bid > 0
    assert 0.0 < enriched.win_probability_at_target <= 1.0
    assert enriched.expected_value_at_target is not None
    assert enriched.ev_recommended_rate_per_mile > 0
    assert enriched.ladder and all(isinstance(o, BidOption) for o in enriched.ladder)
    assert all(0.0 <= o.win_probability <= 1.0 for o in enriched.ladder)


# --- 3. Graceful fallback: no winnability signal => EV fields None, no NaN -------
def test_noop_port_marks_unavailable_without_nan():
    ev = EVBidRecommender(NoopWinnabilityAdapter(), _cfg())
    service = BidRecommenderService(BidPolicy(), _constraints(), ev_recommender=ev)
    bid = service.recommend(_evaluation(), decided_at=_T)
    assert bid.winnability_available is False
    assert bid.win_probability_at_target is None
    assert bid.expected_value_at_target is None
    assert bid.ev_recommended_bid is None
    assert bid.ladder is None
    # Margin bid still present and ordered.
    assert bid.min_bid <= bid.target_bid <= bid.max_bid


# --- 4. decided_at defaults to the load's posting time when omitted ------------
def test_decided_at_defaults_to_created_at():
    bid = _ev_service().recommend(_evaluation())  # no decided_at
    assert bid.winnability_available is True
    assert bid.ev_recommended_bid is not None


# --- 5. bid_query_from_load copies observable fields, defaults the rest ---------
def test_bid_query_from_load_maps_observable_fields():
    evaluation = _evaluation(load_id=42)
    load = evaluation.load
    decided = _T + timedelta(hours=2)
    q = bid_query_from_load(load, decided)

    assert q.snapshot_time == decided
    assert q.posted_at == load.created_at
    assert (q.origin_lat, q.origin_lon) == (load.origin_latitude, load.origin_longitude)
    assert q.equipment_type == load.equipment_type
    assert q.loaded_miles == load.miles
    assert q.weight == load.weight
    assert q.total_rate == load.total_rate
    # Broker board columns + competition are not on the live Load -> stay default/unknown.
    assert q.broker_credit_bucket is None
    assert q.broker_days_to_pay is None
    assert q.load_views == "low"
    assert q.mode == "TL"
    assert q.length == 0.0


# --- 6. Config flag parsing ----------------------------------------------------
def test_config_reads_enabled_flag(tmp_path):
    (tmp_path / "bid_recommender.yaml").write_text(
        "model:\n  enabled: true\n  artifact_path: foo.joblib\n", encoding="utf-8"
    )
    cfg = load_bid_recommender_config(tmp_path)
    assert cfg.enabled is True
    assert cfg.model_path == "foo.joblib"
    # Default committed config ships disabled.
    assert load_bid_recommender_config("config").enabled is False


# --- 7. Container wiring is graceful when enabled but the artifact is missing ---
def test_build_ev_recommender_none_when_disabled():
    assert _build_ev_recommender(BidRecommenderConfig(enabled=False)) is None


def test_build_ev_recommender_none_when_artifact_missing():
    cfg = BidRecommenderConfig(enabled=True, model_path="ml/artifacts/__does_not_exist__.joblib")
    assert _build_ev_recommender(cfg) is None

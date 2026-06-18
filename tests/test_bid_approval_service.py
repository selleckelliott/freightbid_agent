"""Phase 4.4 — BidApprovalService: recommendation capture, happy path, lazy expiry.

Uses the real container wiring (evaluator + recommender + in-memory repos) but swaps in a
:class:`FixedClock` so TTL expiry is deterministic. With the default config the winnability
model is off, so the EV snapshot fields are ``None``.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from adapters.outbound.clock import FixedClock
from application.bid_approval_service import (
    BidApprovalService,
    BidDraftNotFound,
    LoadNotFoundForBid,
)
from domain.enums.bid_approval_status import BidApprovalStatus as S
from domain.models.bid_draft import InvalidBidTransition

_NOW = datetime(2026, 6, 18, 12, 0, 0)


@pytest.fixture
def service(container, sample_loads):
    container.load_repo.add_many(sample_loads)
    clock = FixedClock(_NOW)
    svc = BidApprovalService(
        bid_repo=container.bid_repo,
        load_repo=container.load_repo,
        truck_repo=container.truck_repo,
        evaluator=container.evaluator,
        bid_recommender=container.bid_recommender,
        clock=clock,
        config=container.bid_approval_config,
    )
    return svc, clock, container, sample_loads


def test_create_draft_captures_recommendation(service, sample_truck):
    svc, _clock, container, loads = service
    load = loads[0]
    expected = container.bid_recommender.recommend(
        container.evaluator.evaluate_one(load, sample_truck),
        decided_at=sample_truck.available_at,
    )
    draft = svc.create_draft(sample_truck, load.load_id, actor_id="alice")
    assert draft.status is S.DRAFTED
    assert draft.recommended_amount == round(expected.target_bid, 2)
    assert draft.current_amount == draft.recommended_amount
    assert draft.truck_id == sample_truck.truck_id
    # default container has the model off -> EV snapshot is null
    assert draft.winnability_available is None
    assert draft.expected_value is None
    assert draft.audit[0].actor_id == "alice"


def test_create_draft_unknown_load_raises(service, sample_truck):
    svc, *_ = service
    with pytest.raises(LoadNotFoundForBid):
        svc.create_draft(sample_truck, 999999, actor_id="alice")


def test_default_actor_comes_from_config(service, sample_truck):
    svc, _clock, container, loads = service
    draft = svc.create_draft(sample_truck, loads[0].load_id)  # no actor_id
    assert draft.audit[0].actor_id == container.bid_approval_config.default_actor


def test_happy_path_edit_approve_submit(service, sample_truck):
    svc, _clock, _container, loads = service
    draft = svc.create_draft(sample_truck, loads[0].load_id, actor_id="alice")
    edited = svc.edit_draft(draft.bid_id, draft.recommended_amount + 25, reason="hot lane")
    assert edited.status is S.EDITED
    assert edited.delta_from_recommended == 25.0
    approved = svc.approve_draft(draft.bid_id, actor_id="bob")
    assert approved.status is S.APPROVED
    submitted = svc.submit_mock_draft(draft.bid_id, actor_id="bob")
    assert submitted.status is S.SUBMITTED_MOCK
    assert submitted.submission_ref is not None


def test_get_unknown_raises(service):
    svc, *_ = service
    with pytest.raises(BidDraftNotFound):
        svc.get_draft(4242)


def test_lazy_expiry_on_get(service, sample_truck):
    svc, clock, container, loads = service
    draft = svc.create_draft(sample_truck, loads[0].load_id, actor_id="alice")
    ttl = container.bid_approval_config.draft_ttl_minutes
    clock.value = _NOW + timedelta(minutes=ttl + 1)
    refreshed = svc.get_draft(draft.bid_id)
    assert refreshed.status is S.EXPIRED
    assert refreshed.audit[-1].actor_id == "system"


def test_lazy_expiry_on_list(service, sample_truck):
    svc, clock, container, loads = service
    svc.create_draft(sample_truck, loads[0].load_id, actor_id="alice")
    clock.value = _NOW + timedelta(minutes=container.bid_approval_config.draft_ttl_minutes + 1)
    drafts = svc.list_drafts()
    assert all(d.status is S.EXPIRED for d in drafts)


def test_lazy_expiry_blocks_subsequent_action(service, sample_truck):
    svc, clock, container, loads = service
    draft = svc.create_draft(sample_truck, loads[0].load_id, actor_id="alice")
    clock.value = _NOW + timedelta(minutes=container.bid_approval_config.draft_ttl_minutes + 1)
    with pytest.raises(InvalidBidTransition):
        svc.approve_draft(draft.bid_id, actor_id="bob")


def test_list_status_filter(service, sample_truck):
    svc, _clock, _container, loads = service
    d1 = svc.create_draft(sample_truck, loads[0].load_id, actor_id="a")
    d2 = svc.create_draft(sample_truck, loads[1].load_id, actor_id="a")
    svc.approve_draft(d2.bid_id, actor_id="a")
    drafted = svc.list_drafts(S.DRAFTED)
    approved = svc.list_drafts(S.APPROVED)
    assert [d.bid_id for d in drafted] == [d1.bid_id]
    assert [d.bid_id for d in approved] == [d2.bid_id]

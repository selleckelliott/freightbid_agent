"""Human-in-the-loop bid-approval workflow service (Phase 4.4).

Orchestrates the :class:`~domain.models.bid_draft.BidDraft` lifecycle over a repository
plus the existing evaluator/recommender and an injected clock. The **domain** owns the
state machine; this service owns persistence and **lazy expiry**: it refreshes a draft's
TTL from the clock on every read and action (no scheduler), after which the domain guard
naturally rejects any action on an expired draft.

``create_draft`` re-runs the recommender server-side for an explicit (truck, load) — it
does not rely on a prior ``/rank`` call — and seeds the draft with the recommendation
snapshot (including the optional EV fields, which are ``None`` when the model is off).
"""
from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from application.bid_recommender import BidRecommenderService
from application.config_loader import BidApprovalConfig
from application.evaluate_loads import EvaluateLoadsService
from domain.enums.bid_approval_status import BidApprovalStatus
from domain.models.bid_draft import BidDraft
from domain.models.truck_state import TruckState
from ports.bid_repository import BidApprovalRepositoryPort
from ports.clock import ClockPort
from ports.load_repository import LoadRepositoryPort
from ports.truck_repository import TruckRepositoryPort


class BidDraftNotFound(Exception):
    """Raised when no draft exists for the given id (→ HTTP 404)."""

    def __init__(self, bid_id: int) -> None:
        super().__init__(f"bid draft {bid_id} not found")
        self.bid_id = bid_id


class LoadNotFoundForBid(Exception):
    """Raised when ``create_draft`` is asked for a load that isn't ingested (→ HTTP 404)."""

    def __init__(self, load_id: int) -> None:
        super().__init__(f"load {load_id} not found")
        self.load_id = load_id


class BidApprovalService:
    def __init__(
        self,
        bid_repo: BidApprovalRepositoryPort,
        load_repo: LoadRepositoryPort,
        truck_repo: TruckRepositoryPort,
        evaluator: EvaluateLoadsService,
        bid_recommender: BidRecommenderService,
        clock: ClockPort,
        config: BidApprovalConfig,
    ) -> None:
        self._bids = bid_repo
        self._loads = load_repo
        self._trucks = truck_repo
        self._evaluator = evaluator
        self._recommender = bid_recommender
        self._clock = clock
        self._config = config

    # ------------------------------------------------------------------ create
    def create_draft(
        self, truck: TruckState, load_id: int, actor_id: Optional[str] = None
    ) -> BidDraft:
        """Evaluate (truck, load) → recommend → store a fresh ``DRAFTED`` draft."""
        load = self._loads.get(load_id)
        if load is None:
            raise LoadNotFoundForBid(load_id)
        self._trucks.upsert(truck)
        now = self._clock.now()
        evaluation = self._evaluator.evaluate_one(load, truck)
        bid = self._recommender.recommend(evaluation, decided_at=truck.available_at)
        draft = BidDraft.create(
            bid_id=self._bids.next_id(),
            load_id=load.load_id,
            truck_id=truck.truck_id,
            recommended_amount=bid.target_bid,
            recommended_rate_per_mile=bid.rate_per_mile_at_target,
            rationale=bid.rationale,
            now=now,
            expires_at=now + timedelta(minutes=self._config.draft_ttl_minutes),
            actor_id=self._actor(actor_id),
            winnability_available=bid.winnability_available,
            win_probability=bid.win_probability_at_target,
            expected_value=bid.expected_value_at_target,
            ev_recommended_label=bid.ev_recommended_label,
            ev_recommended_bid=bid.ev_recommended_bid,
        )
        return self._bids.add(draft)

    # ------------------------------------------------------- reads (lazy expiry)
    def get_draft(self, bid_id: int) -> BidDraft:
        return self._get_refreshed(bid_id, self._clock.now())

    def list_drafts(self, status: Optional[BidApprovalStatus] = None) -> List[BidDraft]:
        now = self._clock.now()
        drafts = self._bids.list_all()
        for draft in drafts:
            self._refresh_expiry(draft, now)
        if status is not None:
            drafts = [d for d in drafts if d.status == status]
        return sorted(drafts, key=lambda d: d.bid_id)

    # ----------------------------------------------------------------- actions
    def edit_draft(
        self,
        bid_id: int,
        amount: float,
        reason: Optional[str] = None,
        actor_id: Optional[str] = None,
    ) -> BidDraft:
        now = self._clock.now()
        draft = self._get_refreshed(bid_id, now)
        draft.edit(amount, reason, self._actor(actor_id), now)
        return self._bids.update(draft)

    def approve_draft(
        self, bid_id: int, actor_id: Optional[str] = None, note: Optional[str] = None
    ) -> BidDraft:
        now = self._clock.now()
        draft = self._get_refreshed(bid_id, now)
        draft.approve(self._actor(actor_id), now, note=note)
        return self._bids.update(draft)

    def reject_draft(
        self, bid_id: int, actor_id: Optional[str] = None, note: Optional[str] = None
    ) -> BidDraft:
        now = self._clock.now()
        draft = self._get_refreshed(bid_id, now)
        draft.reject(self._actor(actor_id), now, note=note)
        return self._bids.update(draft)

    def submit_mock_draft(
        self, bid_id: int, actor_id: Optional[str] = None, note: Optional[str] = None
    ) -> BidDraft:
        now = self._clock.now()
        draft = self._get_refreshed(bid_id, now)
        draft.submit_mock(self._actor(actor_id), now, note=note)
        return self._bids.update(draft)

    # --------------------------------------------------------------- internals
    def _get_refreshed(self, bid_id: int, now) -> BidDraft:
        draft = self._bids.get(bid_id)
        if draft is None:
            raise BidDraftNotFound(bid_id)
        self._refresh_expiry(draft, now)
        return draft

    def _refresh_expiry(self, draft: BidDraft, now) -> None:
        if draft.is_expired(now):
            draft.expire(now)
            self._bids.update(draft)

    def _actor(self, actor_id: Optional[str]) -> str:
        return actor_id or self._config.default_actor

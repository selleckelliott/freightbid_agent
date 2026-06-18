"""Human-in-the-loop bid draft aggregate (Phase 4.4).

A reviewable bid recommendation that a dispatcher drives through an explicit lifecycle
(:class:`~domain.enums.bid_approval_status.BidApprovalStatus`). The aggregate owns its
**state machine**: each action validates the transition, appends an immutable
:class:`BidAuditEvent`, and stamps ``updated_at``. Illegal transitions raise
:class:`InvalidBidTransition`.

Pure domain — no IO, no framework; the clock is injected as ``now`` on every call.
``recommended_amount`` is immutable (the model's original ask) while ``current_amount``
moves on edit, so the recommended→adjusted delta is always recoverable. Expiry is **not**
self-enforced here: the application service refreshes it from the clock on every read/
action (no scheduler), then the status guard rejects any action on an expired draft.

``submitted_mock`` is a *simulated* terminal state for workflow validation only — it
never represents a real broker/Truckstop submission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from domain.enums.bid_approval_status import BidApprovalStatus

# Action names — the audit + transition vocabulary.
ACTION_CREATE = "create"
ACTION_EDIT = "edit"
ACTION_APPROVE = "approve"
ACTION_REJECT = "reject"
ACTION_SUBMIT_MOCK = "submit_mock"
ACTION_EXPIRE = "expire"

SYSTEM_ACTOR = "system"

_S = BidApprovalStatus
# Legal source states per action.
_EDITABLE = frozenset({_S.DRAFTED, _S.EDITED, _S.APPROVED})
_APPROVABLE = frozenset({_S.DRAFTED, _S.EDITED})
_REJECTABLE = frozenset({_S.DRAFTED, _S.EDITED, _S.APPROVED})
_SUBMITTABLE = frozenset({_S.APPROVED})


class InvalidBidTransition(Exception):
    """Raised when an action is illegal for a draft's current status."""


@dataclass(frozen=True)
class BidAuditEvent:
    """One immutable entry in a draft's audit trail (who / when / why / amount delta)."""

    at: datetime
    action: str
    actor_id: str
    from_status: Optional[BidApprovalStatus]
    to_status: BidApprovalStatus
    note: Optional[str] = None
    amount_before: Optional[float] = None
    amount_after: Optional[float] = None


@dataclass
class BidDraft:
    """A bid recommendation under human review, plus its audit trail."""

    bid_id: int
    load_id: int
    truck_id: int
    status: BidApprovalStatus
    recommended_amount: float
    recommended_rate_per_mile: float
    current_amount: float
    rationale: str
    created_at: datetime
    expires_at: datetime
    updated_at: datetime
    # Optional recommendation snapshot (EV surfacing, 4.3b) — None when the model is off.
    winnability_available: Optional[bool] = None
    win_probability: Optional[float] = None
    expected_value: Optional[float] = None
    ev_recommended_label: Optional[str] = None
    ev_recommended_bid: Optional[float] = None
    # Edit / submission bookkeeping.
    edit_reason: Optional[str] = None
    submission_ref: Optional[str] = None
    audit: List[BidAuditEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ factory
    @classmethod
    def create(
        cls,
        *,
        bid_id: int,
        load_id: int,
        truck_id: int,
        recommended_amount: float,
        recommended_rate_per_mile: float,
        rationale: str,
        now: datetime,
        expires_at: datetime,
        actor_id: str,
        winnability_available: Optional[bool] = None,
        win_probability: Optional[float] = None,
        expected_value: Optional[float] = None,
        ev_recommended_label: Optional[str] = None,
        ev_recommended_bid: Optional[float] = None,
    ) -> "BidDraft":
        """Build a fresh ``DRAFTED`` draft seeded from a recommendation, with the opening
        ``create`` audit event already recorded."""
        amount = round(float(recommended_amount), 2)
        draft = cls(
            bid_id=bid_id,
            load_id=load_id,
            truck_id=truck_id,
            status=BidApprovalStatus.DRAFTED,
            recommended_amount=amount,
            recommended_rate_per_mile=recommended_rate_per_mile,
            current_amount=amount,
            rationale=rationale,
            created_at=now,
            expires_at=expires_at,
            updated_at=now,
            winnability_available=winnability_available,
            win_probability=win_probability,
            expected_value=expected_value,
            ev_recommended_label=ev_recommended_label,
            ev_recommended_bid=ev_recommended_bid,
        )
        draft.audit.append(
            BidAuditEvent(
                at=now,
                action=ACTION_CREATE,
                actor_id=actor_id,
                from_status=None,
                to_status=BidApprovalStatus.DRAFTED,
                note="drafted from recommendation",
                amount_before=None,
                amount_after=amount,
            )
        )
        return draft

    # ------------------------------------------------------------------ derived
    @property
    def delta_from_recommended(self) -> float:
        """Signed dollar delta of the (possibly edited) ask vs the model's recommendation."""
        return round(self.current_amount - self.recommended_amount, 2)

    @property
    def delta_percent(self) -> float:
        """``delta_from_recommended`` as a percent of the recommendation (zero-guarded)."""
        if self.recommended_amount == 0:
            return 0.0
        return round(
            (self.current_amount - self.recommended_amount) / self.recommended_amount * 100.0,
            2,
        )

    def is_expired(self, now: datetime) -> bool:
        """True when a non-terminal draft has passed its TTL (the service acts on this)."""
        return not self.status.is_terminal and now >= self.expires_at

    # -------------------------------------------------------------- transitions
    def edit(self, amount: float, reason: Optional[str], actor_id: str, now: datetime) -> None:
        """Adjust the ask → ``edited``. Invalidates a prior approval (must re-approve)."""
        if amount is None or float(amount) <= 0:
            raise InvalidBidTransition("edit amount must be a positive number")
        self._guard(_EDITABLE, ACTION_EDIT)
        before = self.current_amount
        self.current_amount = round(float(amount), 2)
        self.edit_reason = reason
        self._transition(
            ACTION_EDIT,
            BidApprovalStatus.EDITED,
            actor_id,
            now,
            note=reason,
            amount_before=before,
            amount_after=self.current_amount,
        )

    def approve(self, actor_id: str, now: datetime, note: Optional[str] = None) -> None:
        self._guard(_APPROVABLE, ACTION_APPROVE)
        self._transition(ACTION_APPROVE, BidApprovalStatus.APPROVED, actor_id, now, note=note)

    def reject(self, actor_id: str, now: datetime, note: Optional[str] = None) -> None:
        self._guard(_REJECTABLE, ACTION_REJECT)
        self._transition(ACTION_REJECT, BidApprovalStatus.REJECTED, actor_id, now, note=note)

    def submit_mock(self, actor_id: str, now: datetime, note: Optional[str] = None) -> None:
        """Simulated terminal submission — stamps a mock ref. No external IO whatsoever."""
        self._guard(_SUBMITTABLE, ACTION_SUBMIT_MOCK)
        self.submission_ref = f"MOCK-{self.bid_id}-{int(now.timestamp())}"
        self._transition(
            ACTION_SUBMIT_MOCK, BidApprovalStatus.SUBMITTED_MOCK, actor_id, now, note=note
        )

    def expire(self, now: datetime) -> None:
        """System-driven expiry of a non-terminal draft (idempotent on terminal drafts)."""
        if self.status.is_terminal:
            return
        self._transition(
            ACTION_EXPIRE, BidApprovalStatus.EXPIRED, SYSTEM_ACTOR, now, note="ttl elapsed"
        )

    # ---------------------------------------------------------------- internals
    def _guard(self, allowed: frozenset, action: str) -> None:
        if self.status not in allowed:
            raise InvalidBidTransition(
                f"cannot {action} a bid in status '{self.status.value}'"
            )

    def _transition(
        self,
        action: str,
        to_status: BidApprovalStatus,
        actor_id: str,
        now: datetime,
        *,
        note: Optional[str] = None,
        amount_before: Optional[float] = None,
        amount_after: Optional[float] = None,
    ) -> None:
        event = BidAuditEvent(
            at=now,
            action=action,
            actor_id=actor_id,
            from_status=self.status,
            to_status=to_status,
            note=note,
            amount_before=amount_before,
            amount_after=amount_after,
        )
        self.status = to_status
        self.updated_at = now
        self.audit.append(event)

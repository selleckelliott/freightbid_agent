"""Phase 4.4 — domain state machine for the human-in-the-loop bid draft.

These pin the lifecycle rules in isolation (no service/API): legal transitions, illegal
transitions raising :class:`InvalidBidTransition`, edit deltas, the audit trail, and
clock-driven expiry. The domain is pure — ``now`` is injected on every call.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from domain.enums.bid_approval_status import BidApprovalStatus as S
from domain.models.bid_draft import (
    ACTION_CREATE,
    SYSTEM_ACTOR,
    BidDraft,
    InvalidBidTransition,
)

_NOW = datetime(2026, 6, 18, 12, 0, 0)


def _draft(now: datetime = _NOW, ttl: int = 30, recommended: float = 460.0) -> BidDraft:
    return BidDraft.create(
        bid_id=1,
        load_id=7,
        truck_id=101,
        recommended_amount=recommended,
        recommended_rate_per_mile=1.92,
        rationale="cost-plus-margin",
        now=now,
        expires_at=now + timedelta(minutes=ttl),
        actor_id="dispatcher",
    )


def test_create_starts_drafted_with_opening_audit():
    d = _draft()
    assert d.status is S.DRAFTED
    assert d.current_amount == d.recommended_amount == 460.0
    assert d.delta_from_recommended == 0.0
    assert d.delta_percent == 0.0
    assert len(d.audit) == 1
    ev = d.audit[0]
    assert ev.action == ACTION_CREATE
    assert ev.from_status is None
    assert ev.to_status is S.DRAFTED
    assert ev.amount_after == 460.0


def test_edit_moves_to_edited_and_records_delta():
    d = _draft()
    d.edit(500.0, "tight market", "alice", _NOW + timedelta(minutes=1))
    assert d.status is S.EDITED
    assert d.current_amount == 500.0
    assert d.delta_from_recommended == 40.0
    assert d.delta_percent == pytest.approx(8.7, abs=0.01)
    assert d.edit_reason == "tight market"
    last = d.audit[-1]
    assert last.action == "edit"
    assert last.actor_id == "alice"
    assert last.amount_before == 460.0
    assert last.amount_after == 500.0


def test_edit_legal_from_drafted_edited_and_approved():
    # drafted -> edited
    d = _draft()
    d.edit(470.0, None, "a", _NOW)
    assert d.status is S.EDITED
    # edited -> edited (re-edit)
    d.edit(480.0, None, "a", _NOW)
    assert d.status is S.EDITED
    # approved -> edited (invalidates approval)
    d.approve("a", _NOW)
    assert d.status is S.APPROVED
    d.edit(490.0, None, "a", _NOW)
    assert d.status is S.EDITED


def test_edit_rejects_nonpositive_amount():
    d = _draft()
    with pytest.raises(InvalidBidTransition):
        d.edit(0.0, "bad", "a", _NOW)
    with pytest.raises(InvalidBidTransition):
        d.edit(-5.0, "bad", "a", _NOW)


def test_approve_legal_from_drafted_and_edited_only():
    d = _draft()
    d.approve("a", _NOW)
    assert d.status is S.APPROVED
    # approving an already-approved draft is illegal
    with pytest.raises(InvalidBidTransition):
        d.approve("a", _NOW)


def test_reject_legal_from_active_states():
    for setup in ("drafted", "edited", "approved"):
        d = _draft()
        if setup == "edited":
            d.edit(470.0, None, "a", _NOW)
        elif setup == "approved":
            d.approve("a", _NOW)
        d.reject("a", _NOW, note="not worth it")
        assert d.status is S.REJECTED


def test_submit_mock_only_from_approved_and_stamps_ref():
    d = _draft()
    # cannot submit from drafted
    with pytest.raises(InvalidBidTransition):
        d.submit_mock("a", _NOW)
    d.approve("a", _NOW)
    d.submit_mock("a", _NOW)
    assert d.status is S.SUBMITTED_MOCK
    assert d.submission_ref is not None
    assert d.submission_ref.startswith("MOCK-1-")


def test_edit_after_approve_requires_reapprove_before_submit():
    d = _draft()
    d.approve("a", _NOW)
    d.edit(480.0, "second look", "a", _NOW)
    assert d.status is S.EDITED
    with pytest.raises(InvalidBidTransition):
        d.submit_mock("a", _NOW)
    d.approve("a", _NOW)
    d.submit_mock("a", _NOW)
    assert d.status is S.SUBMITTED_MOCK


@pytest.mark.parametrize("terminal_setup", ["rejected", "submitted_mock", "expired"])
def test_terminal_states_block_all_actions(terminal_setup):
    d = _draft()
    if terminal_setup == "rejected":
        d.reject("a", _NOW)
    elif terminal_setup == "submitted_mock":
        d.approve("a", _NOW)
        d.submit_mock("a", _NOW)
    else:
        d.expire(_NOW + timedelta(minutes=31))
    assert d.status.is_terminal
    with pytest.raises(InvalidBidTransition):
        d.edit(500.0, None, "a", _NOW)
    with pytest.raises(InvalidBidTransition):
        d.approve("a", _NOW)
    with pytest.raises(InvalidBidTransition):
        d.reject("a", _NOW)
    with pytest.raises(InvalidBidTransition):
        d.submit_mock("a", _NOW)


def test_is_expired_and_expire_uses_system_actor():
    d = _draft(ttl=30)
    assert d.is_expired(_NOW + timedelta(minutes=10)) is False
    assert d.is_expired(_NOW + timedelta(minutes=31)) is True
    d.expire(_NOW + timedelta(minutes=31))
    assert d.status is S.EXPIRED
    assert d.audit[-1].actor_id == SYSTEM_ACTOR


def test_approved_draft_can_still_expire():
    d = _draft(ttl=30)
    d.approve("a", _NOW)
    assert d.is_expired(_NOW + timedelta(minutes=31)) is True
    d.expire(_NOW + timedelta(minutes=31))
    assert d.status is S.EXPIRED


def test_expire_is_idempotent_on_terminal():
    d = _draft()
    d.reject("a", _NOW)
    d.expire(_NOW + timedelta(minutes=31))  # no-op
    assert d.status is S.REJECTED


def test_delta_percent_zero_guarded():
    d = _draft(recommended=0.0)
    d.edit(50.0, None, "a", _NOW)
    assert d.delta_percent == 0.0


def test_audit_trail_accumulates_full_path():
    d = _draft()
    d.edit(500.0, "r", "alice", _NOW)
    d.approve("bob", _NOW)
    d.submit_mock("bob", _NOW)
    transitions = [(e.from_status, e.to_status) for e in d.audit]
    assert transitions == [
        (None, S.DRAFTED),
        (S.DRAFTED, S.EDITED),
        (S.EDITED, S.APPROVED),
        (S.APPROVED, S.SUBMITTED_MOCK),
    ]

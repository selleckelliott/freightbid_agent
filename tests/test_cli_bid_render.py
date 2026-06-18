"""Phase 4.4 — deterministic `rich` rendering for bid drafts + the review queue.

Renders into a recording ``Console`` and asserts on the exported text, mirroring the
existing CLI render tests. Covers the status/amount/delta header, the audit timeline, the
``submitted_mock`` "simulated only" note, and the optional EV snapshot line.
"""
from __future__ import annotations

from rich.console import Console

from adapters.inbound.cli.render import render_bid_draft, render_bid_queue


def _draft(**overrides) -> dict:
    draft = {
        "bid_id": 1,
        "load_id": 7,
        "truck_id": 101,
        "status": "edited",
        "recommended_amount": 460.0,
        "recommended_rate_per_mile": 1.92,
        "current_amount": 490.0,
        "delta_from_recommended": 30.0,
        "delta_percent": 6.52,
        "rationale": "cost-plus-margin target",
        "created_at": "2026-06-18T12:00:00",
        "expires_at": "2026-06-18T12:30:00",
        "updated_at": "2026-06-18T12:01:00",
        "edit_reason": "hot lane",
        "submission_ref": None,
        "winnability_available": None,
        "win_probability": None,
        "expected_value": None,
        "ev_recommended_label": None,
        "ev_recommended_bid": None,
        "audit": [
            {
                "at": "2026-06-18T12:00:00",
                "action": "create",
                "actor_id": "alice",
                "from_status": None,
                "to_status": "drafted",
                "note": "drafted from recommendation",
                "amount_before": None,
                "amount_after": 460.0,
            },
            {
                "at": "2026-06-18T12:01:00",
                "action": "edit",
                "actor_id": "alice",
                "from_status": "drafted",
                "to_status": "edited",
                "note": "hot lane",
                "amount_before": 460.0,
                "amount_after": 490.0,
            },
        ],
    }
    draft.update(overrides)
    return draft


def _render(fn, data) -> str:
    console = Console(record=True, width=200)
    fn(console, data)
    return console.export_text()


def test_render_bid_draft_shows_header_amounts_and_audit():
    out = _render(render_bid_draft, _draft())
    assert "Bid #1" in out
    assert "edited" in out
    assert "Recommended $460" in out
    assert "Current $490" in out
    assert "Delta $30" in out
    assert "Edit reason: hot lane" in out
    # audit timeline
    assert "Audit trail" in out
    assert "create" in out
    assert "drafted -> edited" in out


def test_render_bid_draft_flags_simulated_submission():
    data = _draft(status="submitted_mock", submission_ref="MOCK-1-1781")
    out = _render(render_bid_draft, data)
    assert "Submission (SIMULATED): MOCK-1-1781" in out
    assert "simulated submission for workflow validation only" in out


def test_render_bid_draft_ev_line_when_model_on():
    data = _draft(
        winnability_available=True,
        win_probability=0.62,
        expected_value=180.0,
        ev_recommended_label="target",
        ev_recommended_bid=475.0,
    )
    out = _render(render_bid_draft, data)
    assert "EV: win 62%" in out
    assert "pick target $475" in out


def test_render_bid_draft_no_ev_line_when_model_off():
    out = _render(render_bid_draft, _draft())
    assert "EV: win" not in out


def test_render_bid_queue_table():
    out = _render(render_bid_queue, {"bids": [_draft(), _draft(bid_id=2, status="approved")]})
    assert "Bid drafts (2)" in out
    assert "edited" in out
    assert "approved" in out

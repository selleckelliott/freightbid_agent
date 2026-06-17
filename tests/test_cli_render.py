"""Pins the shared CLI render helpers so the demo SVGs stay faithful to the CLI."""
from rich.console import Console

from adapters.inbound.cli.render import (
    PLAN_COLUMNS,
    RANK_COLUMNS,
    build_plan_table,
    build_rank_table,
    render_plan,
    render_rank,
)

RANK_DATA = {
    "truck_id": 101,
    "ranked": [
        {
            "load_id": 1,
            "score": 1.23,
            "expected_profit": 512.0,
            "rate_per_mile": 2.45,
            "deadhead_miles": 18.0,
            "pickup_eta": "2026-05-27T18:00:00Z",
            "rationale": "score=1.23 strong lane",
            "bid": {"target_bid": 980.0, "rationale": "anchored to cost+margin"},
        },
        {
            "load_id": 2,
            "score": 0.81,
            "expected_profit": 300.0,
            "rate_per_mile": 1.90,
            "deadhead_miles": 42.0,
            "pickup_eta": "2026-05-28T09:00:00Z",
            "rationale": "score=0.81 longer deadhead",
            "bid": {"target_bid": 720.0, "rationale": "thin margin"},
        },
    ],
}

PLAN_DATA = {
    "plan_id": 7,
    "truck_id": 101,
    "horizon_hours": 48,
    "feasible": True,
    "expected_revenue": 1900.0,
    "expected_cost": 1100.0,
    "expected_profit": 800.0,
    "expected_deadhead_miles": 60.0,
    "rationale": "2 stops within HOS",
    "stops": [
        {
            "load_id": 1,
            "pickup_eta": "2026-05-27T18:00:00Z",
            "delivery_eta": "2026-05-27T23:00:00Z",
            "deadhead_miles": 18.0,
            "load_miles": 240.0,
            "revenue": 980.0,
            "cost": 560.0,
            "profit": 420.0,
        }
    ],
}


def test_build_rank_table_shape():
    table = build_rank_table(RANK_DATA)
    assert [c.header for c in table.columns] == RANK_COLUMNS
    assert table.row_count == len(RANK_DATA["ranked"])


def test_build_plan_table_shape():
    table = build_plan_table(PLAN_DATA)
    assert [c.header for c in table.columns] == PLAN_COLUMNS
    assert table.row_count == len(PLAN_DATA["stops"])


def test_render_rank_emits_table_and_rationale():
    console = Console(record=True, width=200)
    render_rank(console, RANK_DATA)
    text = console.export_text()
    assert "Top 2 loads for truck 101" in text
    assert "Load 1:" in text
    assert "Bid:" in text


def test_render_plan_emits_totals():
    console = Console(record=True, width=200)
    render_plan(console, PLAN_DATA)
    text = console.export_text()
    assert "Plan #7" in text
    assert "Totals" in text

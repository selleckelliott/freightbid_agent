import json
from pathlib import Path

from fastapi.testclient import TestClient

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container

ROOT = Path(__file__).resolve().parents[1]


def _client():
    container = build_container(ROOT / "config")
    return TestClient(create_app(container))


def test_health():
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_full_ingest_rank_plan_flow():
    c = _client()
    loads_payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    rank_payload = json.loads((ROOT / "sample_data" / "rank_request.json").read_text())

    r = c.post("/loads", json=loads_payload)
    assert r.status_code == 200
    assert r.json()["accepted"] == len(loads_payload["loads"])

    r = c.post("/rank", json=rank_payload)
    assert r.status_code == 200
    data = r.json()
    assert data["truck_id"] == 101
    assert 1 <= len(data["ranked"]) <= 10
    top = data["ranked"][0]
    assert top["bid"]["min_bid"] <= top["bid"]["target_bid"] <= top["bid"]["max_bid"]
    assert "score=" in top["rationale"]

    r = c.post("/plan", json=rank_payload)
    assert r.status_code == 200
    plan = r.json()
    assert plan["feasible"]
    assert len(plan["stops"]) >= 1
    assert abs(plan["expected_profit"] - (plan["expected_revenue"] - plan["expected_cost"])) < 1e-6


def test_rank_bid_ev_fields_null_when_model_disabled():
    """Default container ships winnability disabled => EV fields serialize as null,
    leaving the legacy bid object unchanged (Phase 4.3b additive contract)."""
    c = _client()
    loads_payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    rank_payload = json.loads((ROOT / "sample_data" / "rank_request.json").read_text())
    c.post("/loads", json=loads_payload)

    bid = c.post("/rank", json=rank_payload).json()["ranked"][0]["bid"]
    assert bid["min_bid"] <= bid["target_bid"] <= bid["max_bid"]
    for field in (
        "winnability_available",
        "win_probability_at_target",
        "expected_value_at_target",
        "ev_recommended_label",
        "ev_recommended_bid",
        "ev_recommended_rate_per_mile",
        "ladder",
    ):
        assert bid[field] is None

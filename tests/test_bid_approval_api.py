"""Phase 4.4 — bid approval workflow over the HTTP API.

End-to-end create -> edit -> approve -> submit-mock through ``TestClient``, plus the error
contracts: 404 (unknown draft / unknown load), 409 (illegal transition), 400 (bad status
filter). EV snapshot fields are null under the default model-off container.
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container

ROOT = Path(__file__).resolve().parents[1]


def _client():
    container = build_container(ROOT / "config")
    return TestClient(create_app(container))


def _truck_payload():
    return json.loads((ROOT / "sample_data" / "truck.json").read_text())


def _ingest(c) -> int:
    loads_payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    c.post("/loads", json=loads_payload)
    return loads_payload["loads"][0]["load_id"]


def test_create_returns_drafted_with_null_ev():
    c = _client()
    load_id = _ingest(c)
    r = c.post("/bids", json={"truck": _truck_payload(), "load_id": load_id, "actor_id": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "drafted"
    assert body["current_amount"] == body["recommended_amount"]
    assert body["delta_from_recommended"] == 0.0
    assert body["winnability_available"] is None
    assert body["expected_value"] is None
    assert body["audit"][0]["action"] == "create"


def test_full_lifecycle_create_edit_approve_submit():
    c = _client()
    load_id = _ingest(c)
    bid_id = c.post(
        "/bids", json={"truck": _truck_payload(), "load_id": load_id, "actor_id": "alice"}
    ).json()["bid_id"]

    recommended = c.get(f"/bids/{bid_id}").json()["recommended_amount"]
    r = c.patch(
        f"/bids/{bid_id}",
        json={"amount": recommended + 30, "reason": "hot lane", "actor_id": "alice"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "edited"
    assert r.json()["delta_from_recommended"] == 30.0
    assert r.json()["edit_reason"] == "hot lane"

    r = c.post(f"/bids/{bid_id}/approve", json={"actor_id": "bob"})
    assert r.status_code == 200
    assert r.json()["status"] == "approved"

    r = c.post(f"/bids/{bid_id}/submit-mock", json={"actor_id": "bob"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "submitted_mock"
    assert body["submission_ref"].startswith("MOCK-")
    actions = [e["action"] for e in body["audit"]]
    assert actions == ["create", "edit", "approve", "submit_mock"]


def test_create_unknown_load_404():
    c = _client()
    _ingest(c)
    r = c.post("/bids", json={"truck": _truck_payload(), "load_id": 987654})
    assert r.status_code == 404


def test_get_unknown_draft_404():
    c = _client()
    assert c.get("/bids/9999").status_code == 404


def test_illegal_transition_returns_409():
    c = _client()
    load_id = _ingest(c)
    bid_id = c.post("/bids", json={"truck": _truck_payload(), "load_id": load_id}).json()["bid_id"]
    # submit-mock straight from drafted is illegal
    r = c.post(f"/bids/{bid_id}/submit-mock")
    assert r.status_code == 409
    assert "drafted" in r.json()["detail"]


def test_queue_and_status_filter():
    c = _client()
    load_id = _ingest(c)
    b1 = c.post("/bids", json={"truck": _truck_payload(), "load_id": load_id}).json()["bid_id"]
    b2 = c.post("/bids", json={"truck": _truck_payload(), "load_id": load_id}).json()["bid_id"]
    c.post(f"/bids/{b2}/reject")

    all_bids = c.get("/bids").json()["bids"]
    assert len(all_bids) == 2
    drafted = c.get("/bids", params={"status": "drafted"}).json()["bids"]
    assert [b["bid_id"] for b in drafted] == [b1]
    rejected = c.get("/bids", params={"status": "rejected"}).json()["bids"]
    assert [b["bid_id"] for b in rejected] == [b2]


def test_bad_status_filter_400():
    c = _client()
    r = c.get("/bids", params={"status": "bogus"})
    assert r.status_code == 400

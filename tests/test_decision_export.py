"""Phase 7.3 — durable decision & audit export.

Covers: the :class:`DecisionRecord` / :class:`Provenance` shapes + the pure CSV flattener, warning
derivation, the **PII-redaction invariant** for an exported broker reference, the stdlib
JSONL/CSV/bundle writers (accepting both domain objects and JSON dicts), the :class:`DecisionLog`
service, the additive read-only ``GET /decisions`` endpoint (provenance + records, status filter, 400,
side-effect-free), a guard that no model id is fabricated under the model-off container, and the CLI
``export`` command end-to-end against a faked API.
"""
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import adapters.inbound.cli.main as cli_main
from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container
from application.ingestion.real_broker_schema import broker_reference_from_mapping
from application.services.decision_exporter import (
    SOURCE_POLICY_VERSION,
    DecisionExporter,
    DecisionLog,
    ExportBundleReport,
    build_decision_record,
    gather_provenance,
)
from domain.models.bid_draft import BidDraft
from domain.models.decision_record import (
    CSV_COLUMNS,
    DECISION_RECORD_SCHEMA_VERSION,
    WARNING_NEGATIVE_EXPECTED_VALUE,
    DecisionRecord,
    Provenance,
)

ROOT = Path(__file__).resolve().parents[1]
_NOW = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _prov(now: datetime = _NOW, **over) -> Provenance:
    return gather_provenance(
        source_policy_version=over.get("source_policy_version", SOURCE_POLICY_VERSION),
        model_artifact_ids=over.get("model_artifact_ids", {}),
        feature_manifest_hash=over.get("feature_manifest_hash", None),
        config_payload=over.get("config_payload", {"k": "v"}),
        now=now,
        git_commit_value="deadbeefcafe",
        git_describe_value="v0.7.3-test",
    )


def _draft(bid_id: int = 1, expected_value=None, win_available=None) -> BidDraft:
    return BidDraft.create(
        bid_id=bid_id,
        load_id=1000 + bid_id,
        truck_id=7,
        recommended_amount=1450.0,
        recommended_rate_per_mile=2.9,
        rationale="cost-plus-margin",
        now=_NOW,
        expires_at=_NOW + timedelta(minutes=30),
        actor_id="alice",
        winnability_available=win_available,
        expected_value=expected_value,
    )


# --------------------------------------------------------------------------- #
# Provenance + record shape
# --------------------------------------------------------------------------- #
def test_provenance_to_dict_shape():
    d = _prov().to_dict()
    assert d["source_policy_version"] == SOURCE_POLICY_VERSION
    assert d["git_commit"] == "deadbeefcafe"
    assert d["git_describe"] == "v0.7.3-test"
    assert d["schema_version"] == DECISION_RECORD_SCHEMA_VERSION
    assert d["model_artifact_ids"] == {}
    assert d["generated_at"] == _NOW.isoformat()
    assert isinstance(d["config_hash"], str) and len(d["config_hash"]) == 16


def test_record_to_dict_nested_shape():
    rec = build_decision_record(_draft(), provenance=_prov())
    d = rec.to_dict()
    assert d["decision_id"] == 1
    assert d["load_id"] == 1001
    assert d["status"] == "drafted"
    assert set(d["recommendation"]) >= {
        "recommended_amount", "current_amount", "delta_from_recommended",
        "delta_percent", "win_probability", "expected_value",
    }
    assert d["broker_reference"] is None
    assert d["provenance"]["source_policy_version"] == SOURCE_POLICY_VERSION
    assert len(d["audit"]) == 1 and d["audit"][0]["action"] == "create"


def test_to_row_keys_match_csv_columns():
    rec = build_decision_record(_draft(), provenance=_prov())
    assert tuple(rec.to_row().keys()) == CSV_COLUMNS


def test_to_row_flattens_audit_and_warnings():
    draft = _draft(expected_value=-9.0)
    draft.approve("bob", _NOW + timedelta(minutes=1))
    rec = build_decision_record(draft, provenance=_prov())
    row = rec.to_row()
    assert row["audit_event_count"] == 2
    assert row["last_action"] == "approve"
    assert row["last_actor"] == "bob"
    assert row["warnings"] == WARNING_NEGATIVE_EXPECTED_VALUE
    assert row["source_policy_version"] == SOURCE_POLICY_VERSION


# --------------------------------------------------------------------------- #
# Warning derivation
# --------------------------------------------------------------------------- #
def test_negative_expected_value_derives_warning():
    rec = build_decision_record(_draft(expected_value=-1.0), provenance=_prov())
    assert rec.warnings == [WARNING_NEGATIVE_EXPECTED_VALUE]


def test_non_negative_expected_value_has_no_warning():
    assert build_decision_record(_draft(expected_value=50.0), provenance=_prov()).warnings == []
    assert build_decision_record(_draft(expected_value=None), provenance=_prov()).warnings == []


def test_explicit_warnings_override_derivation():
    rec = build_decision_record(
        _draft(expected_value=-1.0), provenance=_prov(), warnings=["payment_risk"]
    )
    assert rec.warnings == ["payment_risk"]


# --------------------------------------------------------------------------- #
# PII redaction invariant on an exported broker reference
# --------------------------------------------------------------------------- #
def test_exported_broker_reference_carries_no_raw_pii():
    raw = {
        "broker_id": "BRK-9",
        "name": "Acme Logistics",
        "contact_name": "Jane Doe",
        "contact_email": "jane@acme.example",
        "contact_phone": "+1-555-867-5309",
        "address": "123 Main St, Dallas TX",
        "credit_bucket": "a",
        "days_to_pay": "30",
        "bonded": "yes",
        "age_days": "900",
    }
    ref = broker_reference_from_mapping(raw)
    rec = build_decision_record(_draft(), provenance=_prov(), broker_reference=ref)
    blob = json.dumps(rec.to_dict())
    for secret in ("jane@acme.example", "Jane Doe", "123 Main St", "+1-555-867-5309", "5558675309"):
        assert secret not in blob
    assert rec.broker_reference["broker_id"] == "BRK-9"
    assert rec.broker_reference["credit_bucket"] == "A"  # decision field kept (normalized)
    assert "contact" in rec.broker_reference  # presence-flagged / tokenized contact


# --------------------------------------------------------------------------- #
# Exporter writers
# --------------------------------------------------------------------------- #
def test_write_jsonl_is_deterministic(tmp_path):
    recs = [build_decision_record(_draft(i), provenance=_prov()) for i in (1, 2, 3)]
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert DecisionExporter.write_jsonl(recs, p1) == 3
    DecisionExporter.write_jsonl(recs, p2)
    lines = p1.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and all(json.loads(ln)["decision_id"] for ln in lines)
    assert p1.read_bytes() == p2.read_bytes()  # stable, sorted-key output


def test_write_csv_header_and_rows(tmp_path):
    recs = [build_decision_record(_draft(1, expected_value=-3.0), provenance=_prov())]
    path = tmp_path / "decisions.csv"
    assert DecisionExporter.write_csv(recs, path) == 1
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert list(rows[0].keys()) == list(CSV_COLUMNS)
    assert rows[0]["decision_id"] == "1"
    assert rows[0]["warnings"] == WARNING_NEGATIVE_EXPECTED_VALUE


def test_exporter_accepts_objects_and_dicts(tmp_path):
    recs = [build_decision_record(_draft(1), provenance=_prov())]
    dicts = [r.to_dict() for r in recs]
    a, b = tmp_path / "obj.jsonl", tmp_path / "dict.jsonl"
    DecisionExporter.write_jsonl(recs, a)
    DecisionExporter.write_jsonl(dicts, b)
    assert a.read_bytes() == b.read_bytes()


def test_write_bundle_creates_files_and_manifest(tmp_path):
    prov = _prov()
    d1 = build_decision_record(_draft(1), provenance=prov)
    approved = _draft(2)
    approved.approve("bob", _NOW + timedelta(minutes=1))
    d2 = build_decision_record(approved, provenance=prov)
    report = DecisionExporter.write_bundle([d1, d2], tmp_path / "audit", provenance=prov)

    assert isinstance(report, ExportBundleReport)
    out = tmp_path / "audit"
    assert (out / "decisions.jsonl").exists()
    assert (out / "decisions.csv").exists()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["decision_count"] == 2
    assert manifest["status_counts"] == {"drafted": 1, "approved": 1}
    assert manifest["schema_version"] == DECISION_RECORD_SCHEMA_VERSION
    assert manifest["exported_at"] == _NOW.isoformat()
    assert manifest["provenance"]["git_commit"] == "deadbeefcafe"
    assert report.to_dict()["status_counts"] == {"drafted": 1, "approved": 1}


def test_write_bundle_accepts_provenance_dict(tmp_path):
    prov = _prov()
    recs = [build_decision_record(_draft(1), provenance=prov)]
    report = DecisionExporter.write_bundle(
        [r.to_dict() for r in recs], tmp_path / "b", provenance=prov.to_dict()
    )
    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    assert manifest["provenance"]["source_policy_version"] == SOURCE_POLICY_VERSION


def test_write_bundle_empty(tmp_path):
    report = DecisionExporter.write_bundle([], tmp_path / "empty", provenance=_prov())
    assert report.decision_count == 0
    assert list(csv.DictReader((tmp_path / "empty" / "decisions.csv").open())) == []
    assert (tmp_path / "empty" / "decisions.jsonl").read_text() == ""


# --------------------------------------------------------------------------- #
# DecisionLog service
# --------------------------------------------------------------------------- #
def test_decision_log_builds_records_from_live_drafts(container, sample_truck, sample_loads):
    container.load_repo.add_many(sample_loads)
    draft = container.bid_approval_service.create_draft(sample_truck, sample_loads[0].load_id)
    records = container.decision_log.records()
    assert len(records) == 1
    assert isinstance(records[0], DecisionRecord)
    assert records[0].decision_id == draft.bid_id
    assert records[0].load_id == sample_loads[0].load_id


# --------------------------------------------------------------------------- #
# GET /decisions
# --------------------------------------------------------------------------- #
def _client() -> TestClient:
    return TestClient(create_app(build_container(ROOT / "config")))


def _truck_payload():
    return json.loads((ROOT / "sample_data" / "truck.json").read_text())


def _ingest_and_draft(c) -> int:
    loads_payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    c.post("/loads", json=loads_payload)
    load_id = loads_payload["loads"][0]["load_id"]
    return c.post(
        "/bids", json={"truck": _truck_payload(), "load_id": load_id, "actor_id": "alice"}
    ).json()["bid_id"]


def test_get_decisions_empty_has_provenance_and_no_records():
    data = _client().get("/decisions").json()
    assert data["decisions"] == []
    assert data["provenance"]["source_policy_version"] == SOURCE_POLICY_VERSION
    # model-off container fabricates no model ids / manifest hash
    assert data["provenance"]["model_artifact_ids"] == {}
    assert data["provenance"]["feature_manifest_hash"] is None


def test_get_decisions_returns_record():
    c = _client()
    bid_id = _ingest_and_draft(c)
    data = c.get("/decisions").json()
    assert len(data["decisions"]) == 1
    rec = data["decisions"][0]
    assert rec["decision_id"] == bid_id
    assert rec["status"] == "drafted"
    assert rec["audit"][0]["action"] == "create"


def test_get_decisions_status_filter():
    c = _client()
    bid_id = _ingest_and_draft(c)
    c.post(f"/bids/{bid_id}/approve", json={"actor_id": "alice"})
    approved = c.get("/decisions", params={"status": "approved"}).json()
    assert [r["status"] for r in approved["decisions"]] == ["approved"]
    assert c.get("/decisions", params={"status": "rejected"}).json()["decisions"] == []


def test_get_decisions_bad_status_400():
    assert _client().get("/decisions", params={"status": "nope"}).status_code == 400


def test_get_decisions_is_side_effect_free():
    c = _client()
    bid_id = _ingest_and_draft(c)
    before = c.get(f"/bids/{bid_id}").json()
    c.get("/decisions")
    c.get("/decisions")
    after = c.get(f"/bids/{bid_id}").json()
    assert before == after  # building decision records never mutates a draft


# --------------------------------------------------------------------------- #
# CLI export (faked API)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.is_error = False

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, params=None):
        return _FakeResp(self._payload)


def test_cli_export_writes_bundle(tmp_path, monkeypatch):
    c = _client()
    _ingest_and_draft(c)
    payload = c.get("/decisions").json()
    monkeypatch.setattr(cli_main, "_client", lambda api: _FakeClient(payload))

    out = tmp_path / "audit"
    result = CliRunner().invoke(cli_main.app, ["export", str(out), "--format", "bundle"])
    assert result.exit_code == 0, result.output
    assert (out / "decisions.jsonl").exists()
    assert (out / "decisions.csv").exists()
    assert json.loads((out / "manifest.json").read_text())["decision_count"] == 1


def test_cli_export_jsonl_single_file(tmp_path, monkeypatch):
    c = _client()
    _ingest_and_draft(c)
    payload = c.get("/decisions").json()
    monkeypatch.setattr(cli_main, "_client", lambda api: _FakeClient(payload))

    out = tmp_path / "decisions.jsonl"
    result = CliRunner().invoke(cli_main.app, ["export", str(out), "--format", "jsonl"])
    assert result.exit_code == 0, result.output
    assert len(out.read_text(encoding="utf-8").splitlines()) == 1

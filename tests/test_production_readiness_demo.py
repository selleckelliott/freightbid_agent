"""Tests for the Phase 7.5 production-readiness capstone demo (run_production_readiness_demo).

Covers the deterministic end-to-end contract: every stage passes (verdict PASS), the structural
``facts`` are pinned (board ingress, the broker PII-redaction invariant, the source recommendation,
the human-approval terminal states, and the audit export + provenance), the demo is **deterministic**
across two independent runs, the export bundle is actually written, the transcript renders the story,
and ``main`` writes the committed artifacts (and writes nothing under ``--dry-run``).
"""
import json
from pathlib import Path

from benchmarks.run_production_readiness_demo import (
    DEFAULT_BROKERS,
    DEFAULT_LOADS,
    DEFAULT_TRUCK,
    VERDICT_PASS,
    build_app_client,
    load_broker_rows,
    main,
    render_transcript,
    run_production_readiness_demo,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _inputs():
    truck = json.loads(DEFAULT_TRUCK.read_text(encoding="utf-8"))
    loads = json.loads(DEFAULT_LOADS.read_text(encoding="utf-8"))
    brokers = load_broker_rows(DEFAULT_BROKERS)
    return truck, loads, brokers


def _run(export_dir: Path):
    client, _ = build_app_client(CONFIG)
    truck, loads, brokers = _inputs()
    return run_production_readiness_demo(
        client,
        config_dir=CONFIG,
        truck=truck,
        loads=loads,
        broker_rows=brokers,
        export_dir=export_dir,
    )


# --------------------------------------------------------------------------- #
# Every stage passes
# --------------------------------------------------------------------------- #
def test_all_nine_stages_pass(tmp_path):
    result = _run(tmp_path / "export")
    assert result["ok"] is True
    assert result["verdict"] == VERDICT_PASS
    assert result["stages_total"] == 9
    assert result["stages_passed"] == 9
    keys = [s["key"] for s in result["stages"]]
    assert keys == [
        "preflight",
        "readiness",
        "board_ingress",
        "broker_contract",
        "operating_ingest",
        "recommendation",
        "approval",
        "audit_export",
        "final_readiness",
    ]
    for stage in result["stages"]:
        assert stage["ok"] is True, stage


# --------------------------------------------------------------------------- #
# Determinism — two independent runs return byte-identical content
# --------------------------------------------------------------------------- #
def test_demo_is_deterministic(tmp_path):
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    # No volatile fields (timestamps/git/runtime) are returned by the demo itself.
    assert a == b
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------------------- #
# Structural facts: board ingress, recommendation, approval, audit
# --------------------------------------------------------------------------- #
def test_board_ingress_facts(tmp_path):
    board = _run(tmp_path / "export")["facts"]["board"]
    assert board == {
        "source": "sandbox",
        "available": True,
        "fetched": 12,
        "accepted": 12,
        "rejected": 0,
    }


def test_operating_ingest_and_recommendation_facts(tmp_path):
    facts = _run(tmp_path / "export")["facts"]
    assert facts["operating_ingest_accepted"] == 4
    rec = facts["recommendation"]
    assert rec["ranked_count"] == 2
    assert rec["top_load_id"] == 1
    assert rec["recommended_bid"] == 459.72
    # cost-plus-margin with models off -> a real, positive per-mile ask
    assert rec["rate_per_mile_at_target"] > 1.0


def test_approval_terminal_states(tmp_path):
    approval = _run(tmp_path / "export")["facts"]["approval"]
    assert approval["approved_final_status"] == "submitted_mock"
    assert approval["rejected_final_status"] == "rejected"
    assert approval["approved_load_id"] == 1
    assert approval["rejected_bid_id"] != approval["approved_bid_id"]


def test_audit_export_facts(tmp_path):
    audit = _run(tmp_path / "export")["facts"]["audit_export"]
    assert audit["decision_count"] == 2
    assert audit["status_counts"] == {"submitted_mock": 1, "rejected": 1}
    assert audit["files"] == ["decisions.csv", "decisions.jsonl", "manifest.json"]
    assert audit["source_policy_version"] == "phase-5.5-full-risk-aware"
    # default clone: rule-based source engine, no model artifacts back the decision
    assert audit["model_artifact_ids"] == {}
    assert "git_commit" in audit["provenance_keys"]
    assert "config_hash" in audit["provenance_keys"]


# --------------------------------------------------------------------------- #
# Broker contract + PII-redaction invariant
# --------------------------------------------------------------------------- #
def test_broker_pii_redaction_invariant(tmp_path):
    broker = _run(tmp_path / "export")["facts"]["broker_contract"]
    assert broker["brokers_validated"] == 3
    # The raw fixture rows DO carry PII, and the redacted references DO NOT — the whole point.
    assert broker["raw_had_pii"] is True
    assert broker["pii_leak"] is False
    # Contact presence is preserved as flags; the linkable fields become pseudonymous tokens.
    assert broker["sample_has_email"] is True
    assert broker["sample_has_phone"] is True
    assert broker["sample_email_token_present"] is True
    assert broker["sample_phone_token_present"] is True


def test_export_reports_no_contact_pii(tmp_path):
    audit = _run(tmp_path / "export")["facts"]["audit_export"]
    assert audit["pii_leak_in_export"] is False
    assert audit["records_carrying_broker"] == 0


# --------------------------------------------------------------------------- #
# The export bundle is actually written to disk
# --------------------------------------------------------------------------- #
def test_export_bundle_written(tmp_path):
    export_dir = tmp_path / "export"
    _run(export_dir)
    jsonl = export_dir / "decisions.jsonl"
    csv = export_dir / "decisions.csv"
    manifest = export_dir / "manifest.json"
    assert jsonl.exists() and csv.exists() and manifest.exists()
    lines = [l for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 2
    man = json.loads(manifest.read_text(encoding="utf-8"))
    assert man["decision_count"] == 2
    assert man["status_counts"] == {"submitted_mock": 1, "rejected": 1}


# --------------------------------------------------------------------------- #
# Transcript rendering
# --------------------------------------------------------------------------- #
def test_transcript_renders_story_and_is_deterministic(tmp_path):
    result = _run(tmp_path / "export")
    t1 = render_transcript(result)
    t2 = render_transcript(result)
    assert t1 == t2
    assert "Production-Readiness Capstone Demo (Phase 7.5)" in t1
    assert "VERDICT: PASS - 9/9 stages green." in t1
    assert "Source-engine recommendation" in t1
    # every stage shows an OK marker
    assert t1.count("OK  ") == 9


# --------------------------------------------------------------------------- #
# main() writes the committed artifacts; --dry-run writes nothing
# --------------------------------------------------------------------------- #
def test_main_writes_summary_and_transcript(tmp_path):
    out = tmp_path / "summary.json"
    transcript = tmp_path / "transcript.md"
    export = tmp_path / "export"
    code = main(
        [
            "--out", str(out),
            "--transcript", str(transcript),
            "--export-dir", str(export),
        ]
    )
    assert code == 0
    assert out.exists() and transcript.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdict"] == VERDICT_PASS
    assert payload["stages_total"] == 9
    # the committed summary carries the run metadata wrapper
    assert "generated_at_utc" in payload
    assert "runtime_seconds" in payload
    assert (export / "decisions.jsonl").exists()


def test_main_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "summary.json"
    transcript = tmp_path / "transcript.md"
    export = tmp_path / "export"
    code = main(
        [
            "--out", str(out),
            "--transcript", str(transcript),
            "--export-dir", str(export),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not out.exists()
    assert not transcript.exists()
    assert not export.exists()

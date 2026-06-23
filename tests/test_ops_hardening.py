"""Phase 7.4 — deployment & operations hardening tests.

Covers the readiness probe, the local config-validation preflight, the artifact-availability report,
and the end-to-end smoke runner, plus the additive ``GET /ready`` surface (and that ``/health`` and the
existing endpoints are unchanged), the ops CLI commands, and the hardened Dockerfile.
"""
import dataclasses
import json
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import adapters.inbound.cli.main as cli_main
from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import build_container
from application.config_loader import BidRecommenderConfig
from application.services.decision_exporter import SOURCE_POLICY_VERSION
from application.services.ops_checks import (
    STATUS_DEGRADED,
    STATUS_READY,
    artifact_availability,
    readiness_report,
    run_smoke,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"


def _container():
    return build_container(CONFIG)


def _client() -> TestClient:
    return TestClient(create_app(build_container(CONFIG)))


def _truck():
    return json.loads((ROOT / "sample_data" / "truck.json").read_text())


def _loads():
    return json.loads((ROOT / "sample_data" / "loads.json").read_text())


# --------------------------------------------------------------------------- #
# config validation (local — no server)
# --------------------------------------------------------------------------- #
def test_validate_config_ok():
    report = validate_config(CONFIG)
    assert report["ok"] is True
    assert [c["name"] for c in report["checks"]] == [
        "app_config",
        "bid_recommender",
        "bid_approval",
        "compiled_dispatcher",
        "load_board",
        "objective_profiles",
    ]
    assert all(c["ok"] for c in report["checks"])
    assert all(c["error"] is None for c in report["checks"])


def test_validate_config_missing_dir_fails_without_raising(tmp_path):
    report = validate_config(tmp_path)  # empty dir: required files absent
    assert report["ok"] is False
    app_config = next(c for c in report["checks"] if c["name"] == "app_config")
    assert app_config["ok"] is False
    assert app_config["error"]  # a typed, non-empty message rather than a crash


# --------------------------------------------------------------------------- #
# artifact availability
# --------------------------------------------------------------------------- #
def test_artifact_availability_reports_absent_gitignored_models():
    c = _container()
    report = artifact_availability(
        c.bid_recommender_config,
        c.compiled_dispatcher_config,
        c.compiled_dispatcher_shadow.availability(),
    )
    assert set(report) == {"winnability", "payment_risk", "compiled_dispatcher"}
    # the default config disables every optional model; `present` reflects disk state (the artifacts
    # are gitignored, so absent in a fresh clone) and must be reported as a plain bool either way
    for name in ("winnability", "payment_risk", "compiled_dispatcher"):
        assert report[name]["enabled"] is False
        assert isinstance(report[name]["present"], bool)
    assert report["compiled_dispatcher"]["shadow_available"] is False


# --------------------------------------------------------------------------- #
# readiness
# --------------------------------------------------------------------------- #
def test_readiness_default_is_ready():
    report = readiness_report(_container())
    assert report["status"] == STATUS_READY
    assert report["warnings"] == []
    assert report["checks"]["engine"]["ok"] is True
    assert report["checks"]["load_board"]["source"] == "sandbox"
    assert report["checks"]["load_board"]["available"] is True
    assert report["source_policy_version"] == SOURCE_POLICY_VERSION


def test_readiness_degraded_when_enabled_model_artifact_missing():
    c = _container()
    c.bid_recommender_config = BidRecommenderConfig(
        enabled=True, model_path="ml/artifacts/does_not_exist.joblib"
    )
    report = readiness_report(c)
    assert report["status"] == STATUS_DEGRADED
    assert any("winnability" in w for w in report["warnings"])


def test_readiness_degraded_when_board_unavailable():
    c = _container()
    # a sandbox board with count 0 reports unavailable (side-effect-free)
    from adapters.outbound.load_board.sandbox import SandboxLoadBoardAdapter

    c.load_board = SandboxLoadBoardAdapter(seed=1, count=0)
    report = readiness_report(c)
    assert report["status"] == STATUS_DEGRADED
    assert any("load board" in w for w in report["warnings"])


# --------------------------------------------------------------------------- #
# GET /ready  (additive, read-only) + /health regression
# --------------------------------------------------------------------------- #
def test_get_ready_endpoint_shape():
    r = _client().get("/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in (STATUS_READY, STATUS_DEGRADED)
    assert body["checks"]["engine"]["ok"] is True
    assert body["source_policy_version"] == SOURCE_POLICY_VERSION
    assert "git_describe" in body and "git_commit" in body
    assert set(body["checks"]["artifacts"]) == {
        "winnability",
        "payment_risk",
        "compiled_dispatcher",
    }


def test_get_ready_is_side_effect_free():
    c = _client()
    assert c.get("/loads").json() == []
    first = c.get("/ready").json()
    second = c.get("/ready").json()
    assert first == second  # deterministic, no volatile fields
    assert c.get("/loads").json() == []  # readiness never ingests


def test_health_unchanged():
    assert _client().get("/health").json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# end-to-end smoke runner
# --------------------------------------------------------------------------- #
def test_run_smoke_end_to_end_passes():
    report = run_smoke(_client(), truck=_truck(), loads=_loads())
    assert report["ok"] is True
    assert [s["name"] for s in report["steps"]] == [
        "health",
        "ready",
        "pull",
        "ingest",
        "rank",
        "bid_draft",
        "decisions",
    ]
    assert all(s["ok"] for s in report["steps"])


def test_run_smoke_reports_failure_without_raising():
    # no loads ingested -> ingest + rank + bid + decisions fail, but never crash
    report = run_smoke(_client(), truck=_truck(), loads={"loads": []})
    assert report["ok"] is False
    names = {s["name"] for s in report["steps"]}
    assert {"health", "ready", "pull", "ingest", "rank"} <= names
    assert next(s for s in report["steps"] if s["name"] == "rank")["ok"] is False


# --------------------------------------------------------------------------- #
# CLI commands
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


def test_cli_validate_config_ok():
    result = CliRunner().invoke(cli_main.app, ["validate-config"])
    assert result.exit_code == 0, result.output
    assert "All config valid" in result.output


def test_cli_validate_config_bad_dir(tmp_path):
    result = CliRunner().invoke(
        cli_main.app, ["validate-config", "--config-dir", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_cli_ready_renders(monkeypatch):
    payload = _client().get("/ready").json()
    monkeypatch.setattr(cli_main, "_client", lambda api: _FakeClient(payload))
    result = CliRunner().invoke(cli_main.app, ["ready"])
    assert result.exit_code == 0, result.output
    assert "Readiness" in result.output
    assert "load board" in result.output


def test_cli_smoke_test_pass(monkeypatch):
    monkeypatch.setattr(
        cli_main.ops_checks,
        "run_smoke",
        lambda *a, **k: {"ok": True, "steps": [{"name": "health", "ok": True, "detail": "x"}]},
    )
    monkeypatch.setattr(cli_main, "_client", lambda api: _FakeClient({}))
    result = CliRunner().invoke(cli_main.app, ["smoke-test"])
    assert result.exit_code == 0, result.output
    assert "Smoke test passed" in result.output


def test_cli_smoke_test_fail(monkeypatch):
    monkeypatch.setattr(
        cli_main.ops_checks,
        "run_smoke",
        lambda *a, **k: {"ok": False, "steps": [{"name": "rank", "ok": False, "detail": "0 ranked"}]},
    )
    monkeypatch.setattr(cli_main, "_client", lambda api: _FakeClient({}))
    result = CliRunner().invoke(cli_main.app, ["smoke-test"])
    assert result.exit_code == 1
    assert "Smoke test failed" in result.output


# --------------------------------------------------------------------------- #
# Dockerfile hardening (regression guard)
# --------------------------------------------------------------------------- #
def test_dockerfile_is_hardened():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "HEALTHCHECK" in dockerfile
    assert "/health" in dockerfile
    assert "USER freight" in dockerfile  # runs as a non-root user

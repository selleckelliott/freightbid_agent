"""Phase 6.4 — shadow-mode compiled-dispatcher tests.

The compiled dispatcher runs **beside** the source engine and may *only* observe. These tests pin
the safety boundary (fail-closed, source never mutated, no bid authority), the comparison math
(action/bid/approval/warning/EV deltas), the fail-closed reasons (disabled / no_artifact /
manifest_mismatch / invalid_output / prediction_error), the container wiring, and the additive
``/rank`` surface (``compiled_shadow`` null when off; ``ranked`` byte-identical when on).

Most cases run on tiny **crafted** predictions (instant). A single module-scoped trained model backs
the "compiled model actually serves beside the source" integration checks.
"""
import inspect
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adapters.inbound.api.app import create_app
from adapters.inbound.api.container import (
    _build_compiled_dispatcher_shadow,
    build_container,
)
from adapters.outbound.compiled_dispatcher.noop_compiled_dispatcher import (
    NoopCompiledDispatcher,
)
from adapters.outbound.compiled_dispatcher.sklearn_compiled_dispatcher import (
    SklearnCompiledDispatcher,
)
from application.config_loader import (
    CompiledDispatcherConfig,
    load_bid_recommender_config,
)
from application.services.shadow_compiled_dispatcher_service import (
    ShadowCompiledDispatcherService,
    _jaccard,
    _validate_prediction,
)
from ml.models.compiled_dispatcher_model import (
    CompiledDispatcherModel,
    CompiledDispatcherPrediction,
    default_feature_manifest,
    feature_manifest_hash,
)
from ports.compiled_dispatcher import (
    KNOWN_WARNINGS,
    REASON_DISABLED,
    REASON_INVALID_OUTPUT,
    REASON_MANIFEST_MISMATCH,
    REASON_NO_ARTIFACT,
    REASON_PREDICTION_ERROR,
    CompiledDispatcherAvailability,
    CompiledDispatcherPort,
    CompiledDispatcherShadowComparison,
    CompiledDispatcherUnavailable,
    source_prediction_from_targets,
)

ROOT = Path(__file__).resolve().parents[1]
SMOKE_WORLDS = {"baseline", "slow_pay", "degraded_corner"}


# --------------------------------------------------------------------------- #
# Tiny builders
# --------------------------------------------------------------------------- #
def _targets(decision="bid", bid=500.0, rpm=2.5, ev=120.0, warnings=(), load_id="L1"):
    return dict(
        recommended_load_id=load_id,
        decision=decision,
        recommended_bid_amount=bid,
        recommended_bid_rpm=rpm,
        risk_adjusted_ev=ev,
        warnings=list(warnings),
    )


def _pred(decision="bid", bid=500.0, rpm=2.5, ratio=1.0, ev=120.0,
          approval=False, warnings=(), load_id="L1"):
    return CompiledDispatcherPrediction(
        recommended_load_id=load_id,
        decision=decision,
        recommended_bid=bid,
        recommended_bid_rpm=rpm,
        bid_ratio=ratio,
        risk_adjusted_ev=ev,
        approval_required=approval,
        warnings=list(warnings),
    )


class _StubPort(CompiledDispatcherPort):
    """A controllable port: serves a fixed prediction, or fails in a chosen way."""

    def __init__(self, prediction=None, *, available=True, reason=None,
                 raise_unavailable=None, raise_exc=None):
        self._prediction = prediction
        self._available = available
        self._reason = reason
        self._raise_unavailable = raise_unavailable
        self._raise_exc = raise_exc

    def availability(self) -> CompiledDispatcherAvailability:
        return CompiledDispatcherAvailability(
            available=self._available, reason=self._reason
        )

    def predict(self, features):
        if self._raise_exc is not None:
            raise self._raise_exc
        if self._raise_unavailable is not None:
            raise CompiledDispatcherUnavailable(self._raise_unavailable)
        return self._prediction


class _SpyPort(_StubPort):
    """A stub that also exposes bid-authority methods, to prove the service never touches them."""

    def __init__(self, prediction):
        super().__init__(prediction)
        self.draft_called = False
        self.approve_called = False
        self.submit_called = False

    def create_draft(self, *a, **k):  # pragma: no cover - must never be called
        self.draft_called = True

    def approve(self, *a, **k):  # pragma: no cover - must never be called
        self.approve_called = True

    def submit_mock(self, *a, **k):  # pragma: no cover - must never be called
        self.submit_called = True


def _service(port):
    return ShadowCompiledDispatcherService(port)


# --------------------------------------------------------------------------- #
# Module-scoped trained model (the real "shadow beside source" path)
# --------------------------------------------------------------------------- #
def _generate_tiny_rows():
    from benchmarks.run_broker_quality_stress import load_conditions
    from ml.calibration.recalibration_workflow import RecalibrationConfig
    from ml.config import load_ml_config
    from ml.data.build_compiled_dispatcher_dataset import build_dataset
    from ml.monitoring.calibration_drift import CalibrationThresholds
    from ml.workflows.teacher_trace_generator import generate_traces

    cfg = load_ml_config()
    cfg = replace(cfg, synthetic_data=replace(
        cfg.synthetic_data, loads_per_snapshot_mean=16.0, snapshots_per_day=4))
    bid_cfg = load_bid_recommender_config("config")
    thresholds = replace(CalibrationThresholds(), min_samples=40)
    recal = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [c for c in load_conditions(Path("config/broker_quality_stress.yaml"))
                  if c.name in SMOKE_WORLDS]
    traces = generate_traces(
        cfg, bid_cfg, thresholds, recal, conditions, days=6, max_loads_per_world=120,
    )
    return build_dataset(traces).rows


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    rows = _generate_tiny_rows()
    model = CompiledDispatcherModel(random_state=63).fit(rows)
    path = model.save(tmp_path_factory.mktemp("compiled") / "compiled_dispatcher_model.joblib")
    return path, rows


# --------------------------------------------------------------------------- #
# Comparison math (crafted, no model)
# --------------------------------------------------------------------------- #
def test_available_comparison_reports_all_fields():
    src = source_prediction_from_targets(_targets(decision="bid", bid=500.0, ev=120.0))
    compiled = _pred(decision="bid", bid=450.0, ev=110.0)
    cmp = _service(_StubPort(compiled)).compare(src, {"market_rate": 2.4})

    assert cmp.compiled_available is True
    assert cmp.shadow_only is True
    assert cmp.fallback_reason is None
    assert cmp.source_action == "bid" and cmp.compiled_action == "bid"
    assert cmp.action_agrees is True
    assert cmp.source_bid == 500.0 and cmp.compiled_bid == 450.0
    assert cmp.bid_delta == -50.0
    assert cmp.bid_delta_percent == pytest.approx(-10.0)
    assert cmp.ev_delta == pytest.approx(-10.0)
    assert cmp.approval_agrees is True
    assert cmp.compiled_latency_ms is not None and cmp.compiled_latency_ms >= 0.0


def test_warning_agreement_is_jaccard():
    src = source_prediction_from_targets(_targets(warnings=["payment_risk"]))
    compiled = _pred(warnings=["payment_risk", "calibration_alert"])
    cmp = _service(_StubPort(compiled)).compare(src, {})
    assert cmp.warning_agreement == pytest.approx(0.5)
    assert cmp.source_warnings == ["payment_risk"]
    assert cmp.compiled_warnings == ["payment_risk", "calibration_alert"]


def test_empty_warning_sets_agree_fully():
    src = source_prediction_from_targets(_targets(warnings=[]))
    cmp = _service(_StubPort(_pred(warnings=[]))).compare(src, {})
    assert cmp.warning_agreement == 1.0
    assert _jaccard([], []) == 1.0


def test_action_and_approval_disagreement():
    src = source_prediction_from_targets(
        _targets(decision="bid", warnings=[])
    )
    compiled = _pred(decision="approval_required", approval=True, warnings=["payment_risk"])
    cmp = _service(_StubPort(compiled)).compare(src, {})
    assert cmp.action_agrees is False
    assert cmp.approval_agrees is False
    assert cmp.source_approval_required is False
    assert cmp.compiled_approval_required is True


def test_no_bid_rows_have_null_bid_delta():
    src = source_prediction_from_targets(
        _targets(decision="no_bid", bid=None, rpm=None, ev=None, warnings=["no_feasible_bid"])
    )
    compiled = _pred(decision="no_bid", bid=None, rpm=None, ev=None, warnings=["no_feasible_bid"])
    cmp = _service(_StubPort(compiled)).compare(src, {})
    assert cmp.action_agrees is True
    assert cmp.bid_delta is None and cmp.bid_delta_percent is None
    assert cmp.ev_delta is None


def test_comparison_to_dict_has_the_full_contract():
    src = source_prediction_from_targets(_targets())
    cmp = _service(_StubPort(_pred())).compare(src, {})
    expected_keys = {
        "compiled_available", "shadow_only",
        "source_action", "compiled_action", "action_agrees",
        "source_bid", "compiled_bid", "bid_delta", "bid_delta_percent",
        "source_approval_required", "compiled_approval_required", "approval_agrees",
        "source_warnings", "compiled_warnings", "warning_agreement",
        "source_risk_adjusted_ev", "compiled_risk_adjusted_ev", "ev_delta",
        "compiled_latency_ms", "fallback_reason",
    }
    assert set(cmp.to_dict().keys()) == expected_keys


# --------------------------------------------------------------------------- #
# Fail-closed reasons
# --------------------------------------------------------------------------- #
def test_disabled_port_falls_back_with_source_preserved():
    src = source_prediction_from_targets(_targets(bid=500.0))
    cmp = _service(NoopCompiledDispatcher(REASON_DISABLED)).compare(src, {})
    assert cmp.compiled_available is False
    assert cmp.fallback_reason == REASON_DISABLED
    assert cmp.shadow_only is True
    # source side still populated; compiled side blanked
    assert cmp.source_action == "bid" and cmp.source_bid == 500.0
    assert cmp.compiled_action is None and cmp.compiled_bid is None
    assert cmp.action_agrees is None and cmp.warning_agreement is None


def test_missing_artifact_reason():
    cmp = _service(NoopCompiledDispatcher(REASON_NO_ARTIFACT)).compare(
        source_prediction_from_targets(_targets()), {}
    )
    assert cmp.compiled_available is False
    assert cmp.fallback_reason == REASON_NO_ARTIFACT


def test_unavailable_raised_in_predict_is_caught():
    port = _StubPort(raise_unavailable=REASON_MANIFEST_MISMATCH, available=True)
    cmp = _service(port).compare(source_prediction_from_targets(_targets()), {})
    assert cmp.compiled_available is False
    assert cmp.fallback_reason == REASON_MANIFEST_MISMATCH


def test_prediction_exception_is_caught():
    port = _StubPort(raise_exc=ValueError("boom"), available=True)
    cmp = _service(port).compare(source_prediction_from_targets(_targets()), {})
    assert cmp.compiled_available is False
    assert cmp.fallback_reason == REASON_PREDICTION_ERROR


@pytest.mark.parametrize("bad", [
    _pred(decision="garbage"),
    _pred(decision="bid", warnings=["unknown_warning"]),
    _pred(decision="bid", bid=-10.0),
])
def test_invalid_compiled_output_falls_back(bad):
    cmp = _service(_StubPort(bad)).compare(source_prediction_from_targets(_targets()), {})
    assert cmp.compiled_available is False
    assert cmp.fallback_reason == REASON_INVALID_OUTPUT


def test_invalid_output_validator_accepts_known_warnings():
    assert _validate_prediction(_pred(warnings=sorted(KNOWN_WARNINGS))) is None
    assert _validate_prediction(_pred(decision="nope")) == REASON_INVALID_OUTPUT


# --------------------------------------------------------------------------- #
# Safety: source never mutated; no bid authority
# --------------------------------------------------------------------------- #
def test_source_object_unchanged_after_comparison_available():
    src = source_prediction_from_targets(_targets(bid=500.0, warnings=["payment_risk"]))
    before = deepcopy(asdict(src))
    _service(_StubPort(_pred(bid=450.0))).compare(src, {"market_rate": 2.4})
    assert asdict(src) == before


def test_source_object_unchanged_after_comparison_fallback():
    src = source_prediction_from_targets(_targets(bid=500.0, warnings=["payment_risk"]))
    before = deepcopy(asdict(src))
    _service(NoopCompiledDispatcher()).compare(src, {})
    assert asdict(src) == before


def test_service_never_invokes_bid_authority_methods():
    spy = _SpyPort(_pred())
    _service(spy).compare(source_prediction_from_targets(_targets()), {})
    assert spy.draft_called is False
    assert spy.approve_called is False
    assert spy.submit_called is False


def test_service_holds_no_bid_repository_or_approval_handle():
    svc = _service(NoopCompiledDispatcher())
    # The only collaborator is the read-only port — no approval workflow reachable.
    sig = inspect.signature(ShadowCompiledDispatcherService.__init__)
    assert list(sig.parameters)[1:] == ["port"]
    for value in vars(svc).values():
        assert not hasattr(value, "create_draft")
        assert not hasattr(value, "approve")
        assert not hasattr(value, "submit_mock")


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
def test_noop_adapter_is_unavailable_and_fails_closed():
    noop = NoopCompiledDispatcher(REASON_DISABLED)
    assert noop.availability().available is False
    assert noop.availability().reason == REASON_DISABLED
    with pytest.raises(CompiledDispatcherUnavailable):
        noop.predict({})


def test_sklearn_adapter_serves_and_round_trips(trained_artifact):
    path, rows = trained_artifact
    expected = feature_manifest_hash(default_feature_manifest())
    adapter = SklearnCompiledDispatcher.from_artifact(path, expected_manifest_hash=expected)
    assert adapter.availability().available is True
    assert adapter.feature_manifest_hash == expected
    pred = adapter.predict(rows[0]["features"])
    assert isinstance(pred, CompiledDispatcherPrediction)
    assert pred.decision in {"bid", "no_bid", "approval_required"}


def test_sklearn_adapter_manifest_mismatch_fails_closed(trained_artifact):
    path, _rows = trained_artifact
    adapter = SklearnCompiledDispatcher.from_artifact(path, expected_manifest_hash="deadbeef")
    assert adapter.availability().available is False
    assert adapter.availability().reason == REASON_MANIFEST_MISMATCH
    with pytest.raises(CompiledDispatcherUnavailable):
        adapter.predict({"market_rate": 2.4})


def test_sklearn_adapter_translates_feature_manifest_error(trained_artifact):
    path, _rows = trained_artifact
    adapter = SklearnCompiledDispatcher.from_artifact(path)  # no expected hash => manifest_ok
    # A feature dict missing manifest fields makes the model raise FeatureManifestError,
    # which the adapter must translate into a fail-closed CompiledDispatcherUnavailable.
    with pytest.raises(CompiledDispatcherUnavailable):
        adapter.predict({"market_rate": 2.4})


# --------------------------------------------------------------------------- #
# Compiled model actually runs beside the source decision
# --------------------------------------------------------------------------- #
def test_shadow_runs_beside_source_over_real_rows(trained_artifact):
    path, rows = trained_artifact
    expected = feature_manifest_hash(default_feature_manifest())
    adapter = SklearnCompiledDispatcher.from_artifact(path, expected_manifest_hash=expected)
    svc = ShadowCompiledDispatcherService(adapter)

    agree = 0
    sample = rows[:60]
    for row in sample:
        src = source_prediction_from_targets(row["targets"])
        cmp = svc.compare(src, row["features"])
        assert cmp.compiled_available is True
        assert cmp.shadow_only is True
        assert cmp.fallback_reason is None
        assert cmp.action_agrees == (cmp.source_action == cmp.compiled_action)
        assert cmp.compiled_latency_ms is not None
        agree += int(bool(cmp.action_agrees))
    # The compiled model was distilled from these very traces — it should mostly agree on action.
    assert agree / len(sample) >= 0.7


# --------------------------------------------------------------------------- #
# Container wiring
# --------------------------------------------------------------------------- #
def test_default_container_shadow_is_disabled_noop():
    container = build_container(ROOT / "config")
    assert container.compiled_dispatcher_config.enabled is False
    avail = container.compiled_dispatcher_shadow.availability()
    assert avail.available is False
    assert avail.reason == REASON_DISABLED


def test_builder_enabled_with_artifact_is_available(trained_artifact):
    path, _rows = trained_artifact
    cfg = CompiledDispatcherConfig(enabled=True, artifact_path=str(path), shadow_mode=True)
    svc = _build_compiled_dispatcher_shadow(cfg)
    assert svc.availability().available is True


def test_builder_enabled_missing_artifact_is_no_artifact(tmp_path):
    cfg = CompiledDispatcherConfig(
        enabled=True, artifact_path=str(tmp_path / "nope.joblib"), shadow_mode=True
    )
    svc = _build_compiled_dispatcher_shadow(cfg)
    avail = svc.availability()
    assert avail.available is False
    assert avail.reason == REASON_NO_ARTIFACT


def test_builder_enabled_manifest_mismatch_fails_closed(trained_artifact, tmp_path):
    path, _rows = trained_artifact
    # Re-save the model with a tampered manifest hash so it mismatches the inference contract.
    model = CompiledDispatcherModel.load(path)
    model.feature_manifest_hash = "deadbeefdeadbeef"
    bad = model.save(tmp_path / "mismatch.joblib")
    cfg = CompiledDispatcherConfig(enabled=True, artifact_path=str(bad), shadow_mode=True)
    svc = _build_compiled_dispatcher_shadow(cfg)
    avail = svc.availability()
    assert avail.available is False
    assert avail.reason == REASON_MANIFEST_MISMATCH


# --------------------------------------------------------------------------- #
# Additive /rank surface
# --------------------------------------------------------------------------- #
def _ingested_client(container):
    client = TestClient(create_app(container))
    loads_payload = json.loads((ROOT / "sample_data" / "loads.json").read_text())
    client.post("/loads", json=loads_payload)
    return client


def _rank_payload():
    return json.loads((ROOT / "sample_data" / "rank_request.json").read_text())


def test_rank_compiled_shadow_null_when_off():
    container = build_container(ROOT / "config")
    client = _ingested_client(container)
    data = client.post("/rank", json=_rank_payload()).json()
    assert set(data.keys()) == {"truck_id", "ranked", "compiled_shadow"}
    assert data["compiled_shadow"] is None
    assert len(data["ranked"]) >= 1


def test_rank_banner_present_when_on_and_ranked_byte_identical(trained_artifact):
    path, _rows = trained_artifact
    # OFF baseline.
    off_ranked = _ingested_client(build_container(ROOT / "config")).post(
        "/rank", json=_rank_payload()
    ).json()["ranked"]

    # ON: same container, but enable shadow with the trained artifact.
    container = build_container(ROOT / "config")
    expected = feature_manifest_hash(default_feature_manifest())
    adapter = SklearnCompiledDispatcher.from_artifact(path, expected_manifest_hash=expected)
    container.compiled_dispatcher_config = replace(
        container.compiled_dispatcher_config, enabled=True
    )
    container.compiled_dispatcher_shadow = ShadowCompiledDispatcherService(adapter)

    on_data = _ingested_client(container).post("/rank", json=_rank_payload()).json()
    banner = on_data["compiled_shadow"]
    assert banner is not None
    assert banner["shadow_only"] is True
    assert banner["compiled_available"] is True
    # The compiled dispatcher is additive metadata only — ranked items are unchanged.
    assert on_data["ranked"] == off_ranked

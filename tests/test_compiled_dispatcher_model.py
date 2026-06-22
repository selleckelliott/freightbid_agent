"""Phase 6.3 — compiled multi-head dispatcher model tests.

Pins the contracts the user asked for: the model trains on ``inference_context`` features **only**
(a forbidden ``node_outputs`` / ``eval_labels`` field can never enter the feature matrix and is
rejected from the manifest); the artifact **refuses to serve** on a feature-manifest-hash mismatch
or a missing feature; training is deterministic; the artifact round-trips through joblib; per-head
predictions serialize to a stable DTO / 6-key runtime JSON; the bid head trains on biddable rows
only; rare classes appear in the metrics; and the model **beats the majority baseline** on action
macro-F1.

Fast cases run on tiny **crafted** traces. A single module-scoped seeded teacher batch (the Phase
6.1/6.2 tiny-cfg recipe) backs the learn-ability checks (determinism, beats-baseline, metrics).
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import load_conditions
from ml.calibration.recalibration_workflow import RecalibrationConfig
from ml.config import load_ml_config
from ml.data.build_compiled_dispatcher_dataset import build_dataset
from ml.data.compiled_agent_trace_schema import (
    APPROVAL_AUTO_ELIGIBLE,
    APPROVAL_HUMAN_REQUIRED,
    APPROVAL_NOT_APPLICABLE,
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
    AgentTrace,
    EvalLabels,
    InferenceContext,
    NodeOutputs,
    Recommendation,
    TraceMetadata,
    eval_label_field_names,
    inference_field_names,
    node_output_field_names,
)
from ml.data.compiled_dispatcher_formatters import to_structured_row
from ml.models.baseline_compiled_dispatcher import MajorityCompiledDispatcherBaseline
from ml.models.compiled_dispatcher_model import (
    TARGET_NAMES,
    CompiledDispatcherModel,
    CompiledDispatcherPrediction,
    FeatureManifestError,
    assert_manifest_inference_only,
    default_feature_manifest,
    derive_action,
    feature_manifest_hash,
    row_features,
)
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.training.compiled_dispatcher_dataset import split_rows
from ml.training.evaluate_compiled_dispatcher_model import evaluate_model
from ml.training.train_compiled_dispatcher_model import build_metadata, train
from ml.workflows.freightbid_workflow_graph import (
    BRANCH_CLEAN_BID,
    BRANCH_ESCALATED,
    BRANCH_INFEASIBLE,
    BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
    TERMINAL_APPROVAL_REQUIRED,
    TERMINAL_BID,
    TERMINAL_NO_BID,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_NO_FEASIBLE_BID,
    WARN_PAYMENT_RISK,
)


# --------------------------------------------------------------------------- #
# Crafted-trace factory (mirrors the 6.2 factory; instant, no engine)
# --------------------------------------------------------------------------- #
def _ctx(**over):
    base = dict(
        load_id="L-TEST-1", snapshot_time="2024-01-01T00:00:00", broker_id="BRK0001",
        equipment_type="HS", mode="TL", commodity="general",
        loaded_miles=200.0, weight=4000.0, length=30.0,
        origin_lat=35.0, origin_lon=-100.0, load_views="med", load_age_hours=3.0,
        has_posted_rate=True, posted_rate_per_mile=2.5, tarp_required=False,
        appointment_required=False, broker_credit_bucket="A", broker_days_to_pay=30,
        broker_bonded=True, broker_quick_pay_available=False, broker_age_days=500,
        market_rate=2.4, cost_per_loaded_mile=1.4, truck_equipment_type="hotshot",
    )
    base.update(over)
    return InferenceContext(**base)


def _node(**over):
    base = dict(
        estimated_cost=280.0, breakeven_rpm=1.4, market_to_breakeven_ratio=1.7,
        feasible=True, winnability_available=True, payment_risk_available=True,
        win_probability_at_target=0.9, expected_value_at_target=150.0,
        risk_adjusted_ev_at_target=120.0, p_default_at_target=0.05,
        p_collect_at_target=0.95, expected_pay_days_at_target=30.0,
        delay_penalty_at_target=2.0, risk_adjusted_ev_positive=True,
        risk_adjusted_warning=None, recommended_label="win",
        recommended_ask_engine=500.0, recommended_ask_rpm_engine=2.5,
        calibration_severity_before="OK", calibration_severity_after="OK",
        calibration_severity_operational="OK", recalibrator_promoted=False,
    )
    base.update(over)
    return NodeOutputs(**base)


def _rec(**over):
    base = dict(
        decision=DECISION_BID, recommended_load_id="L-TEST-1",
        recommended_bid_amount=500.0, recommended_bid_rpm=2.5, warnings=[],
        approval_decision=APPROVAL_AUTO_ELIGIBLE, explanation="Recommend $500 (2.5/mi).",
        terminal_state=TERMINAL_BID, hub_branch=BRANCH_CLEAN_BID,
    )
    base.update(over)
    return Recommendation(**base)


def _meta(**over):
    base = dict(
        source_policy_version="test", git_commit="deadbeef", config_hash="cfg",
        model_artifact_ids={"winnability": "w"}, random_seed=1, world_name="baseline",
        workflow_graph_version="1.0.0", teacher_trace_schema_version="1.0.0",
    )
    base.update(over)
    return TraceMetadata(**base)


def _trace(scenario_id="baseline::L-TEST-1", *, ctx=None, node=None, rec=None, meta=None):
    return AgentTrace(
        scenario_id=scenario_id,
        path=["start", "terminal"],
        inference_context=ctx or _ctx(),
        node_outputs=node or _node(),
        recommendation=rec or _rec(),
        eval_labels=EvalLabels(
            reservation_rpm=2.3, true_default_prob=0.05, true_pay_days=30.0,
            realized_win_prob_at_recommended=0.9, realized_collectible_profit_if_bid=120.0,
        ),
        metadata=meta or _meta(),
    )


def _clean_bid_row(scenario_id, *, rpm=2.5, market=2.4, load_id="L-1", miles=200.0):
    ctx = _ctx(load_id=load_id, market_rate=market, loaded_miles=miles)
    rec = _rec(recommended_load_id=load_id, recommended_bid_rpm=rpm,
               recommended_bid_amount=rpm * miles)
    return to_structured_row(_trace(scenario_id, ctx=ctx, rec=rec,
                                    node=_node(recommended_ask_rpm_engine=rpm)))


def _infeasible_row(scenario_id, *, load_id="L-INF"):
    ctx = _ctx(load_id=load_id, has_posted_rate=False, posted_rate_per_mile=None)
    node = _node(feasible=False, win_probability_at_target=None,
                 risk_adjusted_ev_at_target=None, p_default_at_target=None)
    rec = _rec(decision=DECISION_NO_BID, recommended_load_id=load_id,
               recommended_bid_amount=None, recommended_bid_rpm=None,
               warnings=[WARN_NO_FEASIBLE_BID], approval_decision=APPROVAL_NOT_APPLICABLE,
               hub_branch=BRANCH_INFEASIBLE, terminal_state=TERMINAL_NO_BID)
    return to_structured_row(_trace(scenario_id, ctx=ctx, node=node, rec=rec))


def _payment_escalation_row(scenario_id, *, load_id="L-PAY"):
    ctx = _ctx(load_id=load_id, broker_credit_bucket="D", broker_days_to_pay=75)
    node = _node(p_default_at_target=0.22)
    rec = _rec(decision=DECISION_APPROVAL_REQUIRED, recommended_load_id=load_id,
               warnings=[WARN_PAYMENT_RISK], approval_decision=APPROVAL_HUMAN_REQUIRED,
               hub_branch=BRANCH_ESCALATED, terminal_state=TERMINAL_APPROVAL_REQUIRED)
    return to_structured_row(_trace(scenario_id, ctx=ctx, node=node, rec=rec))


# --------------------------------------------------------------------------- #
# Feature manifest = inference_context only (the keystone)
# --------------------------------------------------------------------------- #
def test_feature_manifest_is_inference_only():
    manifest = default_feature_manifest()
    assert set(manifest).issubset(inference_field_names())
    assert set(manifest).isdisjoint(node_output_field_names())
    assert set(manifest).isdisjoint(eval_label_field_names())
    assert assert_manifest_inference_only(manifest) is True


def test_manifest_rejects_a_node_output_field():
    bad = default_feature_manifest() + ["risk_adjusted_ev_at_target"]
    with pytest.raises(ValueError):
        assert_manifest_inference_only(bad)


def test_manifest_rejects_an_eval_label_field():
    bad = default_feature_manifest() + [sorted(eval_label_field_names())[0]]
    with pytest.raises(ValueError):
        assert_manifest_inference_only(bad)


def test_constructing_with_a_leaky_manifest_raises():
    with pytest.raises(ValueError):
        CompiledDispatcherModel(feature_manifest=default_feature_manifest() + ["p_default_at_target"])


def test_forbidden_field_cannot_enter_the_feature_matrix():
    # Even if a caller smuggles a teacher-only field into the feature dict, the model builds its
    # frame from the manifest alone, so the prediction is byte-identical with/without it.
    rows = [_clean_bid_row(f"baseline::B{i}", rpm=2.3 + 0.1 * i, load_id=f"L{i}") for i in range(4)]
    model = CompiledDispatcherModel().fit(rows)
    clean = dict(row_features(rows[0]))
    smuggled = dict(clean)
    smuggled["risk_adjusted_ev_at_target"] = 9999.0  # node-output: must be ignored
    smuggled["realized_collectible_profit_if_bid"] = -1.0  # eval-label: must be ignored
    a = model.predict_raw([clean])
    b = model.predict_raw([smuggled])
    assert list(a["action"]) == list(b["action"])
    assert a["bid_ratio"][0] == b["bid_ratio"][0]
    assert a["risk_adjusted_ev"][0] == b["risk_adjusted_ev"][0]


# --------------------------------------------------------------------------- #
# Manifest-hash gating: refuse to serve on mismatch / missing feature
# --------------------------------------------------------------------------- #
def test_assert_compatible_refuses_on_hash_mismatch():
    model = CompiledDispatcherModel().fit([_clean_bid_row("baseline::B0")])
    assert model.assert_compatible(model.feature_manifest_hash) is True
    with pytest.raises(FeatureManifestError):
        model.assert_compatible("deadbeefdeadbeef")


def test_predict_refuses_when_a_manifest_feature_is_missing():
    model = CompiledDispatcherModel().fit([_clean_bid_row("baseline::B0")])
    feats = dict(row_features(_clean_bid_row("baseline::B1")))
    feats.pop("market_rate")
    with pytest.raises(FeatureManifestError):
        model.predict_dto(feats)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        CompiledDispatcherModel().predict_raw([row_features(_clean_bid_row("baseline::B0"))])


def test_manifest_hash_is_stable_and_matches_helper():
    model = CompiledDispatcherModel()
    assert model.feature_manifest_hash == feature_manifest_hash(default_feature_manifest())


# --------------------------------------------------------------------------- #
# Bid head trains on biddable rows only
# --------------------------------------------------------------------------- #
def test_bid_head_not_fit_without_biddable_rows():
    rows = [_infeasible_row(f"baseline::I{i}", load_id=f"LI{i}") for i in range(4)]
    model = CompiledDispatcherModel().fit(rows)
    assert model.bid_ratio_head.estimator is None  # no biddable rows entered the head
    assert model.ev_head.estimator is None  # no feasible EV either
    # action head learned the single 'no_bid' class
    dto = model.predict_dto(row_features(rows[0]))
    assert dto.decision == DECISION_NO_BID
    assert dto.recommended_bid is None
    assert dto.bid_ratio is None
    # infeasible rows have no risk-adjusted EV — the served EV must be null, not the head's fallback
    assert dto.risk_adjusted_ev is None


def test_bid_head_fit_when_biddable_rows_present():
    rows = [_clean_bid_row(f"baseline::B{i}", rpm=2.2 + 0.1 * i, load_id=f"LB{i}") for i in range(5)]
    model = CompiledDispatcherModel().fit(rows)
    assert model.bid_ratio_head.estimator is not None


def test_predict_batch_reconstructs_bid_from_ratio():
    rows = [_clean_bid_row(f"baseline::B{i}", rpm=2.2 + 0.1 * i, load_id=f"LB{i}") for i in range(5)]
    model = CompiledDispatcherModel().fit(rows)
    feats = row_features(rows[0])
    dto = model.predict_dto(feats)
    assert dto.decision == DECISION_BID
    assert dto.recommended_bid_rpm == pytest.approx(dto.bid_ratio * feats["market_rate"])
    assert dto.recommended_bid == pytest.approx(dto.recommended_bid_rpm * feats["loaded_miles"])


# --------------------------------------------------------------------------- #
# DTO / runtime-JSON contract
# --------------------------------------------------------------------------- #
def test_runtime_json_is_the_six_key_contract():
    dto = CompiledDispatcherPrediction(
        recommended_load_id="L1", decision=DECISION_BID, recommended_bid=1450.0,
        recommended_bid_rpm=2.5, bid_ratio=1.04, risk_adjusted_ev=122.456,
        approval_required=False, warnings=[WARN_PAYMENT_RISK],
    )
    rj = dto.to_runtime_json()
    assert list(rj.keys()) == [
        "recommended_load_id", "recommended_bid", "decision",
        "risk_adjusted_ev", "warnings", "explanation",
    ]
    assert rj["recommended_bid"] == 1450.0
    assert rj["risk_adjusted_ev"] == 122.46  # rounded to 2dp
    assert rj["warnings"] == [WARN_PAYMENT_RISK]


def test_no_bid_dto_suppresses_bid_amount():
    dto = CompiledDispatcherPrediction(
        recommended_load_id="L1", decision=DECISION_NO_BID, recommended_bid=None,
        recommended_bid_rpm=None, bid_ratio=None, risk_adjusted_ev=None,
        approval_required=False, warnings=[WARN_NO_FEASIBLE_BID],
    )
    rj = dto.to_runtime_json()
    assert rj["recommended_bid"] is None
    assert rj["decision"] == DECISION_NO_BID
    assert "No bid" in rj["explanation"]
    assert set(dto.to_dict()) == {
        "recommended_load_id", "decision", "recommended_bid", "recommended_bid_rpm",
        "bid_ratio", "risk_adjusted_ev", "approval_required", "warnings",
    }


# --------------------------------------------------------------------------- #
# Majority baseline behaves as the floor
# --------------------------------------------------------------------------- #
def test_majority_baseline_predicts_one_action():
    rows = (
        [_clean_bid_row(f"baseline::B{i}", load_id=f"LB{i}") for i in range(5)]
        + [_infeasible_row("baseline::I0", load_id="LI0")]
    )
    base = MajorityCompiledDispatcherBaseline().fit(rows)
    preds = base.predict_raw([row_features(r) for r in rows])
    assert set(preds["action"]) == {DECISION_BID}  # majority class only


# --------------------------------------------------------------------------- #
# Integration: a real seeded teacher batch (tiny-cfg recipe)
# --------------------------------------------------------------------------- #
SMOKE_WORLDS = {"baseline", "slow_pay", "degraded_corner"}


def _generate_tiny_rows():
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
def tiny_rows():
    return _generate_tiny_rows()


def test_training_is_deterministic(tiny_rows):
    train_rows, _val, test_rows = split_rows(tiny_rows, seed=63)
    feats = [row_features(r) for r in test_rows]
    m1 = CompiledDispatcherModel(random_state=63).fit(train_rows)
    m2 = CompiledDispatcherModel(random_state=63).fit(train_rows)
    r1, r2 = m1.predict_raw(feats), m2.predict_raw(feats)
    assert list(r1["action"]) == list(r2["action"])
    assert np.allclose(r1["bid_ratio"], r2["bid_ratio"])
    assert np.allclose(r1["risk_adjusted_ev"], r2["risk_adjusted_ev"])
    assert list(r1["approval_required"]) == list(r2["approval_required"])


def test_artifact_round_trips_through_joblib(tiny_rows, tmp_path):
    train_rows, _val, test_rows = split_rows(tiny_rows, seed=63)
    feats = [row_features(r) for r in test_rows]
    model = CompiledDispatcherModel(random_state=63).fit(train_rows)
    path = model.save(tmp_path / "compiled.joblib")
    loaded = CompiledDispatcherModel.load(path)
    assert loaded.feature_manifest_hash == model.feature_manifest_hash
    a, b = model.predict_raw(feats), loaded.predict_raw(feats)
    assert list(a["action"]) == list(b["action"])
    assert np.allclose(a["bid_ratio"], b["bid_ratio"])
    assert np.allclose(a["risk_adjusted_ev"], b["risk_adjusted_ev"])


def test_model_beats_majority_baseline_on_action_macro_f1(tiny_rows):
    report = train(tiny_rows, seed=63)
    cmp = report["comparison"]
    assert cmp["model_beats_baseline"] is True
    assert cmp["model_action_macro_f1"] > cmp["baseline_action_macro_f1"]


def test_rare_classes_appear_in_metrics(tiny_rows):
    train_rows, _val, test_rows = split_rows(tiny_rows, seed=63)
    model = CompiledDispatcherModel(random_state=63).fit(train_rows)
    ev = evaluate_model(model, test_rows)
    present = {derive_action(r) for r in test_rows}
    assert set(ev["action"]["per_class"]).issuperset(present)
    assert sum(m["support"] for m in ev["action"]["per_class"].values()) == len(test_rows)
    # more than one decision class is represented in the held-out slice
    assert len(present) >= 2


def test_metadata_carries_full_provenance(tiny_rows):
    report = train(tiny_rows, seed=63, dataset_provenance={
        "source_policy_version": "phase-5.5-full-risk-aware",
        "teacher_trace_schema_version": "1.0.0",
        "workflow_graph_version": "1.0.0",
        "dataset_version": "1.0.0",
        "determinism_hash": "abc123",
        "provenance": {"git_commit": "deadbeef"},
    })
    md = build_metadata(report)
    for key in (
        "feature_manifest_hash", "teacher_trace_schema_version", "workflow_graph_version",
        "source_policy_version", "random_seed", "rows", "target_names",
        "estimator_types", "test_metrics", "action_macro_f1_vs_baseline",
    ):
        assert key in md
    assert md["source_policy_version"] == "phase-5.5-full-risk-aware"
    assert md["dataset_git_commit"] == "deadbeef"
    assert md["feature_manifest_hash"] == feature_manifest_hash(default_feature_manifest())
    assert set(md["target_names"]) == set(TARGET_NAMES)
    assert md["rows"]["train"] > 0 and md["rows"]["validation"] > 0 and md["rows"]["test"] > 0

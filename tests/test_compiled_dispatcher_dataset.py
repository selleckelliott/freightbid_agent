"""Phase 6.2 — compiled-dispatcher dataset tests.

Pins the train-eligibility boundary (inputs from ``inference_context`` only; targets may draw
from ``node_outputs``), target/runtime-contract correctness, the procedure-free prompt, the
deterministic synthetic human-in-the-loop continuations, coverage, and end-to-end determinism.

Most cases run on tiny **crafted** traces (instant). A single module-scoped seeded teacher batch
(reusing the Phase 6.1 tiny-cfg recipe) backs the coverage + determinism integration checks.
"""
from dataclasses import replace
from pathlib import Path

import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import load_conditions
from domain.enums.bid_approval_status import BidApprovalStatus
from ml.calibration.recalibration_workflow import RecalibrationConfig
from ml.config import load_ml_config
from ml.data.build_compiled_dispatcher_dataset import build_dataset, build_summary
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
from ml.data.compiled_dispatcher_formatters import (
    CATEGORY_CALIBRATION_ESCALATION,
    CATEGORY_CLEAN_BID,
    CATEGORY_CLEAN_BID_WATCH,
    CATEGORY_INFEASIBLE_NO_BID,
    CATEGORY_NEGATIVE_EV_NO_BID,
    CATEGORY_PAYMENT_ESCALATION,
    FLAG_APPROVAL_REQUIRED,
    FLAG_HIGH_DEFAULT_RISK,
    FLAG_NO_SAFE_BID,
    FLAG_PROFITABLE_BID,
    FLAG_RECALIBRATION_APPLIED,
    HUMAN_ACTIONS,
    assert_features_inference_only,
    build_features,
    build_targets,
    coverage_flags,
    render_completion,
    render_conversation,
    render_prompt,
    runtime_json,
    scenario_category,
    synthetic_human_action,
    to_structured_row,
)
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.workflows.freightbid_workflow_graph import (
    BRANCH_CLEAN_BID,
    BRANCH_ESCALATED,
    BRANCH_INFEASIBLE,
    BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
    TERMINAL_APPROVAL_REQUIRED,
    TERMINAL_BID,
    TERMINAL_NO_BID,
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_NO_FEASIBLE_BID,
    WARN_PAYMENT_RISK,
)
import json


# --------------------------------------------------------------------------- #
# Crafted-trace factory (covers paths the canonical batch makes rare/absent)
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


def _clean_bid_trace():
    return _trace()


def _payment_escalation_trace(scenario_id="slow_pay::L-PAY-1"):
    return _trace(
        scenario_id,
        node=_node(p_default_at_target=0.22, calibration_severity_operational="OK"),
        rec=_rec(
            decision=DECISION_APPROVAL_REQUIRED, warnings=[WARN_PAYMENT_RISK],
            approval_decision=APPROVAL_HUMAN_REQUIRED, hub_branch=BRANCH_ESCALATED,
            terminal_state=TERMINAL_APPROVAL_REQUIRED,
        ),
        meta=_meta(world_name="slow_pay"),
    )


def _calibration_escalation_trace(scenario_id="degraded_corner::L-CAL-1"):
    return _trace(
        scenario_id,
        node=_node(calibration_severity_operational="ALERT"),
        rec=_rec(
            decision=DECISION_APPROVAL_REQUIRED, warnings=[WARN_CALIBRATION_ALERT],
            approval_decision=APPROVAL_HUMAN_REQUIRED, hub_branch=BRANCH_ESCALATED,
            terminal_state=TERMINAL_APPROVAL_REQUIRED,
        ),
    )


def _negative_ev_trace(scenario_id="risky_brokers::L-NEG-1"):
    return _trace(
        scenario_id,
        node=_node(risk_adjusted_ev_at_target=-45.0, risk_adjusted_ev_positive=False,
                   risk_adjusted_warning=WARN_NEGATIVE_RISK_ADJUSTED_EV),
        rec=_rec(
            decision=DECISION_NO_BID, recommended_bid_amount=None, recommended_bid_rpm=None,
            warnings=[WARN_NEGATIVE_RISK_ADJUSTED_EV], approval_decision=APPROVAL_NOT_APPLICABLE,
            hub_branch=BRANCH_NEGATIVE_RISK_ADJUSTED_EV, terminal_state=TERMINAL_NO_BID,
        ),
    )


def _infeasible_trace(scenario_id="baseline::L-INF-1"):
    return _trace(
        scenario_id,
        node=_node(feasible=False, win_probability_at_target=None,
                   risk_adjusted_ev_at_target=None, p_default_at_target=None),
        rec=_rec(
            decision=DECISION_NO_BID, recommended_bid_amount=None, recommended_bid_rpm=None,
            warnings=[WARN_NO_FEASIBLE_BID], approval_decision=APPROVAL_NOT_APPLICABLE,
            hub_branch=BRANCH_INFEASIBLE, terminal_state=TERMINAL_NO_BID,
        ),
    )


# --------------------------------------------------------------------------- #
# Train-eligibility boundary (the keystone)
# --------------------------------------------------------------------------- #
def test_features_are_exactly_the_inference_contract():
    feats = build_features(_clean_bid_trace())
    assert set(feats) == set(inference_field_names())
    assert assert_features_inference_only(feats) is True


def test_features_disjoint_from_teacher_only_fields():
    feats = build_features(_clean_bid_trace())
    assert set(feats).isdisjoint(node_output_field_names())
    assert set(feats).isdisjoint(eval_label_field_names())


def test_assert_rejects_a_node_output_key():
    feats = dict(build_features(_clean_bid_trace()))
    feats["risk_adjusted_ev_at_target"] = 1.0  # a teacher-only field must never be an input
    with pytest.raises(ValueError):
        assert_features_inference_only(feats)


def test_assert_rejects_an_eval_label_key():
    feats = dict(build_features(_clean_bid_trace()))
    feats["true_default_prob"] = 0.1
    with pytest.raises(ValueError):
        assert_features_inference_only(feats)


def test_assert_rejects_unknown_and_missing_keys():
    feats = dict(build_features(_clean_bid_trace()))
    feats["totally_made_up"] = 1
    with pytest.raises(ValueError):
        assert_features_inference_only(feats)
    short = dict(build_features(_clean_bid_trace()))
    short.pop("load_id")
    with pytest.raises(ValueError):
        assert_features_inference_only(short)


# --------------------------------------------------------------------------- #
# Targets + runtime contract (outputs may use node_outputs)
# --------------------------------------------------------------------------- #
def test_targets_mirror_recommendation_and_node_outputs():
    tr = _payment_escalation_trace()
    t = build_targets(tr)
    assert t["decision"] == tr.recommendation.decision
    assert t["recommended_load_id"] == tr.recommendation.recommended_load_id
    assert t["hub_branch"] == tr.recommendation.hub_branch
    assert t["approval_decision"] == tr.recommendation.approval_decision
    assert t["recommended_bid_amount"] == tr.recommendation.recommended_bid_amount
    assert t["warnings"] == list(tr.recommendation.warnings)
    assert t["explanation"] == tr.recommendation.explanation
    # output-side targets sourced from node_outputs (allowed — predicted, not leaked)
    assert t["risk_adjusted_ev"] == tr.node_outputs.risk_adjusted_ev_at_target
    assert t["win_probability"] == tr.node_outputs.win_probability_at_target
    assert t["p_default"] == tr.node_outputs.p_default_at_target


def test_runtime_json_is_the_six_key_contract():
    tr = _clean_bid_trace()
    rj = runtime_json(tr)
    assert list(rj) == [
        "recommended_load_id", "recommended_bid", "decision",
        "risk_adjusted_ev", "warnings", "explanation",
    ]
    assert rj["recommended_bid"] == tr.recommendation.recommended_bid_amount
    assert rj["decision"] == tr.recommendation.decision
    assert rj["warnings"] == list(tr.recommendation.warnings)


def test_runtime_json_consistent_with_targets_and_parses():
    tr = _payment_escalation_trace()
    rj = runtime_json(tr)
    t = build_targets(tr)
    assert rj["recommended_load_id"] == t["recommended_load_id"]
    assert rj["decision"] == t["decision"]
    assert rj["warnings"] == t["warnings"]
    parsed = json.loads(render_completion(tr))  # the assistant turn must be valid JSON
    assert parsed["decision"] == tr.recommendation.decision


def test_no_bid_runtime_json_has_null_bid():
    rj = runtime_json(_negative_ev_trace())
    assert rj["recommended_bid"] is None
    assert rj["decision"] == DECISION_NO_BID


# --------------------------------------------------------------------------- #
# Prompt: case facts only, no procedure, no model output
# --------------------------------------------------------------------------- #
BANNED_IN_PROMPT = [
    "risk_adjusted_ev", "risk-adjusted", "p_default", "p_collect", "win_prob",
    "win probability", "calibration", "recalibr", "reservation", "collectible",
    "breakeven", "expected value", "node", "terminal", "hub", "workflow", "true_default",
]


def test_prompt_has_no_procedure_or_model_output():
    prompt = render_prompt(build_features(_payment_escalation_trace())).lower()
    for banned in BANNED_IN_PROMPT:
        assert banned not in prompt, f"prompt leaks '{banned}'"


def test_prompt_contains_observable_case_facts():
    feats = build_features(_clean_bid_trace())
    prompt = render_prompt(feats)
    assert feats["load_id"] in prompt
    assert feats["broker_id"] in prompt


def test_prompt_requires_inference_only_features():
    feats = dict(build_features(_clean_bid_trace()))
    feats["p_default_at_target"] = 0.3
    with pytest.raises(ValueError):
        render_prompt(feats)


# --------------------------------------------------------------------------- #
# Coverage taxonomy (incl. the rare crafted paths)
# --------------------------------------------------------------------------- #
def test_scenario_categories_for_each_branch():
    assert scenario_category(_clean_bid_trace()) == CATEGORY_CLEAN_BID
    assert scenario_category(_payment_escalation_trace()) == CATEGORY_PAYMENT_ESCALATION
    assert scenario_category(_calibration_escalation_trace()) == CATEGORY_CALIBRATION_ESCALATION
    assert scenario_category(_negative_ev_trace()) == CATEGORY_NEGATIVE_EV_NO_BID
    assert scenario_category(_infeasible_trace()) == CATEGORY_INFEASIBLE_NO_BID


def test_clean_bid_watch_category():
    tr = _trace(rec=_rec(warnings=[WARN_CALIBRATION_WATCH]))
    assert scenario_category(tr) == CATEGORY_CLEAN_BID_WATCH


def test_coverage_flags():
    assert FLAG_PROFITABLE_BID in coverage_flags(_clean_bid_trace())
    pay = coverage_flags(_payment_escalation_trace())
    assert FLAG_APPROVAL_REQUIRED in pay and FLAG_HIGH_DEFAULT_RISK in pay
    assert FLAG_NO_SAFE_BID in coverage_flags(_negative_ev_trace())
    recal = coverage_flags(_trace(node=_node(recalibrator_promoted=True)))
    assert FLAG_RECALIBRATION_APPLIED in recal


# --------------------------------------------------------------------------- #
# Synthetic human-in-the-loop continuation
# --------------------------------------------------------------------------- #
def test_human_action_only_on_approval_required():
    assert synthetic_human_action(_clean_bid_trace()) is None
    assert synthetic_human_action(_negative_ev_trace()) is None
    action = synthetic_human_action(_payment_escalation_trace())
    assert action in HUMAN_ACTIONS


def test_human_action_is_deterministic_in_scenario_id_only():
    a = synthetic_human_action(_payment_escalation_trace("slow_pay::L-9"))
    b = synthetic_human_action(_payment_escalation_trace("slow_pay::L-9"))
    assert a == b
    # same id, different world -> same action (depends on scenario_id alone)
    other = _payment_escalation_trace("slow_pay::L-9")
    other = replace(other, metadata=_meta(world_name="baseline"))
    assert synthetic_human_action(other) == a


def test_all_four_human_actions_are_reachable():
    seen = set()
    for i in range(60):
        tr = _payment_escalation_trace(f"slow_pay::L-{i}")
        seen.add(synthetic_human_action(tr))
    assert seen == set(HUMAN_ACTIONS)


def test_conversation_shape_bid_vs_approval():
    bid = render_conversation(_clean_bid_trace())
    assert bid["human_action"] is None and bid["synthetic_continuation"] is False
    assert [m["role"] for m in bid["messages"]] == ["system", "user", "assistant"]

    appr = render_conversation(_payment_escalation_trace())
    assert appr["synthetic_continuation"] is True
    roles = [m["role"] for m in appr["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    last = appr["messages"][-1]["content"]
    assert last.startswith(f"[{appr['human_action']}]")
    assert appr["human_action"] in {a.value for a in HUMAN_ACTIONS}


def test_edited_action_uses_real_enum_value():
    # Find a scenario_id that hashes to EDITED and check the continuation text.
    tr = next(_payment_escalation_trace(f"slow_pay::E-{i}") for i in range(60)
              if synthetic_human_action(_payment_escalation_trace(f"slow_pay::E-{i}"))
              is BidApprovalStatus.EDITED)
    conv = render_conversation(tr)
    assert conv["human_action"] == BidApprovalStatus.EDITED.value
    assert "[edited]" in conv["messages"][-1]["content"]


def test_structured_row_excludes_human_action():
    row = to_structured_row(_payment_escalation_trace())
    assert set(row) == {
        "scenario_id", "world", "scenario_category", "coverage_flags", "features", "targets",
    }
    assert "human_action" not in row and "messages" not in row


# --------------------------------------------------------------------------- #
# Dataset assembly + determinism (crafted multi-trace)
# --------------------------------------------------------------------------- #
def _mixed_traces():
    return [
        _clean_bid_trace(),
        _payment_escalation_trace(),
        _calibration_escalation_trace(),
        _negative_ev_trace(),
        _infeasible_trace(),
    ]


def test_build_dataset_is_deterministic_and_order_independent():
    traces = _mixed_traces()
    one = build_dataset(traces)
    two = build_dataset(list(reversed(traces)))
    assert one.fingerprint() == two.fingerprint()
    assert one.rows == two.rows  # sorted internally by scenario_id
    assert one.conversations == two.conversations


def test_build_dataset_enforces_no_leakage_per_row():
    ds = build_dataset(_mixed_traces())
    for row in ds.rows:
        assert assert_features_inference_only(row["features"]) is True


def test_summary_shape_and_histograms():
    traces = _mixed_traces()
    ds = build_dataset(traces)
    s = build_summary(traces, ds)
    assert s["generated"]["n_examples"] == len(ds.rows)
    assert s["train_eligibility"]["leakage_check"] == "pass"
    assert s["train_eligibility"]["feature_fields"] == sorted(inference_field_names())
    assert sum(s["category_histogram"].values()) == len(ds.rows)
    assert s["determinism_hash"] == ds.fingerprint()
    # the five crafted categories should each appear once
    assert s["category_histogram"][CATEGORY_CLEAN_BID] == 1
    assert s["category_histogram"][CATEGORY_CALIBRATION_ESCALATION] == 1


# --------------------------------------------------------------------------- #
# Integration: a real seeded teacher batch -> dataset (Phase 6.1 tiny-cfg recipe)
# --------------------------------------------------------------------------- #
SMOKE_WORLDS = {"baseline", "slow_pay", "degraded_corner"}
SMOKE_DAYS = 6
SMOKE_MAX_LOADS = 120


def _generate_tiny_traces():
    from ml.workflows.teacher_trace_generator import generate_traces

    cfg = load_ml_config()
    cfg = replace(cfg, synthetic_data=replace(
        cfg.synthetic_data, loads_per_snapshot_mean=16.0, snapshots_per_day=4))
    bid_cfg = load_bid_recommender_config("config")
    thresholds = replace(CalibrationThresholds(), min_samples=40)
    recal = replace(RecalibrationConfig(), fit_days=2, eval_days=3, min_samples=40)
    conditions = [c for c in load_conditions(Path("config/broker_quality_stress.yaml"))
                  if c.name in SMOKE_WORLDS]
    return generate_traces(
        cfg, bid_cfg, thresholds, recal, conditions,
        days=SMOKE_DAYS, max_loads_per_world=SMOKE_MAX_LOADS,
    )


@pytest.fixture(scope="module")
def tiny_traces():
    return _generate_tiny_traces()


def test_integration_rows_are_inference_only(tiny_traces):
    ds = build_dataset(tiny_traces)
    assert len(ds.rows) == len(tiny_traces)
    for row in ds.rows:
        assert set(row["features"]) == set(inference_field_names())
        assert assert_features_inference_only(row["features"]) is True


def test_integration_covers_core_decision_paths(tiny_traces):
    ds = build_dataset(tiny_traces)
    cats = {r["scenario_category"] for r in ds.rows}
    # at tiny scale calibration is noisier, so the profitable-bid family may be clean_bid
    # and/or clean_bid_watch; either satisfies "a bid path is covered".
    assert cats & {CATEGORY_CLEAN_BID, CATEGORY_CLEAN_BID_WATCH}
    assert CATEGORY_PAYMENT_ESCALATION in cats
    assert CATEGORY_INFEASIBLE_NO_BID in cats
    flags = {f for r in ds.rows for f in r["coverage_flags"]}
    assert FLAG_PROFITABLE_BID in flags
    assert FLAG_APPROVAL_REQUIRED in flags


def test_integration_prompts_never_leak(tiny_traces):
    ds = build_dataset(tiny_traces)
    for conv in ds.conversations:
        prompt = conv["messages"][1]["content"].lower()
        for banned in BANNED_IN_PROMPT:
            assert banned not in prompt


def test_integration_dataset_deterministic(tiny_traces):
    assert build_dataset(tiny_traces).fingerprint() == build_dataset(tiny_traces).fingerprint()

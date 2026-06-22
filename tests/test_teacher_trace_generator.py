"""Tests for the Phase 6.1 workflow graph + teacher trace generator.

Three layers, fastest first:

* **schema** — the train-eligibility / no-leakage contract every later sub-phase relies on;
* **graph** — the declarative workflow graph validates, and its routing is well-formed;
* **hub** — the procedural decision hub routes on engine outputs alone, covering all four
  branches deterministically (this is where the rare ``negative_risk_adjusted_ev`` branch is
  pinned, since it almost never occurs in a real seeded batch);
* **integration** — two small seeded generations prove the teacher wraps (not reimplements)
  the engine, every trace is schema-complete and reaches a terminal, the hub is reproducible
  purely from ``node_outputs``, and generation is byte-for-byte deterministic.
"""
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from application.config_loader import load_bid_recommender_config
from benchmarks.run_broker_quality_stress import load_conditions
from ml.calibration.recalibration_workflow import RecalibrationConfig
from ml.config import load_ml_config
from ml.data.compiled_agent_trace_schema import (
    APPROVAL_AUTO_ELIGIBLE,
    APPROVAL_HUMAN_REQUIRED,
    APPROVAL_NOT_APPLICABLE,
    DECISION_NO_BID,
    DECISIONS,
    NON_TRAINABLE_SECTIONS,
    TRAINABLE_SECTIONS,
    AgentTrace,
    EvalLabels,
    InferenceContext,
    NodeOutputs,
    Recommendation,
    TraceMetadata,
    assert_no_leakage,
    eval_label_field_names,
    feature_eligible_fields,
    inference_field_names,
    node_output_field_names,
    trace_stream_fingerprint,
)
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.workflows.freightbid_workflow_graph import (
    BRANCH_CLEAN_BID,
    BRANCH_ESCALATED,
    BRANCH_INFEASIBLE,
    BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
    BRANCH_TERMINAL,
    CHOOSE_ACTION,
    EXPLAIN,
    SEV_ALERT,
    SEV_OK,
    SEV_WATCH,
    START,
    TERMINAL_APPROVAL_REQUIRED,
    TERMINAL_BID,
    TERMINAL_BY_DECISION,
    TERMINAL_NO_BID,
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_NO_FEASIBLE_BID,
    WARN_PAYMENT_RISK,
    DecisionHubPolicy,
    Edge,
    HubSignals,
    Node,
    WorkflowGraph,
    WorkflowGraphError,
    build_default_graph,
    decide,
)
from ml.workflows.teacher_trace_generator import build_summary, generate_traces


# --------------------------------------------------------------------------- #
# Schema: the train-eligibility / no-leakage contract (pure, instant)
# --------------------------------------------------------------------------- #
def test_no_leakage_invariant_holds():
    assert assert_no_leakage() is True
    inf = inference_field_names()
    assert inf.isdisjoint(node_output_field_names())
    assert inf.isdisjoint(eval_label_field_names())
    # The compiled model (6.3) may train on exactly the inference fields, nothing more.
    assert feature_eligible_fields() == inf
    assert TRAINABLE_SECTIONS == ("inference_context",)
    assert NON_TRAINABLE_SECTIONS == ("node_outputs", "eval_labels")


def test_realized_outcomes_are_never_feature_eligible():
    """Realized / latent ground truth lives only in eval_labels, never in the feature set."""
    for f in ("true_default_prob", "true_pay_days", "realized_collectible_profit_if_bid",
              "realized_win_prob_at_recommended", "reservation_rpm"):
        assert f in eval_label_field_names()
        assert f not in feature_eligible_fields()


def test_engine_internals_are_never_feature_eligible():
    """The teacher's own model outputs must not become compiled-model inputs."""
    for f in ("risk_adjusted_ev_at_target", "p_default_at_target", "win_probability_at_target",
              "calibration_severity_operational", "recommended_ask_engine"):
        assert f in node_output_field_names()
        assert f not in feature_eligible_fields()


def test_feature_eligible_field_count_is_pinned():
    # 25 observable decision-time fields — the exact contract 6.3's feature matrix enforces.
    assert len(feature_eligible_fields()) == 25


# --------------------------------------------------------------------------- #
# Graph: the declarative workflow validates and routes cleanly (pure, instant)
# --------------------------------------------------------------------------- #
def test_default_graph_validates_and_has_expected_shape():
    g = build_default_graph()
    assert g.validate() is True
    assert g.version == "1.0.0"
    assert len(g.nodes) == 16
    assert len(g.edges) == 19
    assert sum(n.kind == "start" for n in g.nodes) == 1
    assert sum(n.kind == "hub" for n in g.nodes) == 1
    assert sum(n.kind == "terminal" for n in g.nodes) == 3


def test_linear_prefix_runs_start_to_hub_without_repeats():
    g = build_default_graph()
    pre = g.linear_prefix()
    assert pre[0] == START
    assert pre[-1] == CHOOSE_ACTION
    assert len(pre) == 12
    assert len(set(pre)) == len(pre)


def test_each_branch_routes_through_explain_to_its_terminal():
    g = build_default_graph()
    for branch, terminal in BRANCH_TERMINAL.items():
        route = g.route(branch)
        assert route == [EXPLAIN, terminal]
        full = g.linear_prefix() + route
        assert full[0] == START
        assert full[-1] == terminal


def test_hub_reaches_all_three_terminals():
    g = build_default_graph()
    assert g.terminals_reachable_from(CHOOSE_ACTION) == {
        TERMINAL_BID, TERMINAL_NO_BID, TERMINAL_APPROVAL_REQUIRED,
    }


def test_validate_rejects_two_start_nodes():
    nodes = (Node("a", "start", "-", ""), Node("b", "start", "-", ""),
             Node("t", "terminal", "-", ""))
    edges = (Edge("a", "t"), Edge("b", "t"))
    with pytest.raises(WorkflowGraphError):
        WorkflowGraph(nodes, edges).validate()


def test_validate_rejects_terminal_with_outgoing_edge():
    nodes = (Node("s", "start", "-", ""), Node("h", "hub", "-", ""),
             Node("t", "terminal", "-", ""), Node("u", "terminal", "-", ""))
    edges = (Edge("s", "h"), Edge("h", "t", condition="x"),
             Edge("h", "u", condition="y"), Edge("t", "u"))
    with pytest.raises(WorkflowGraphError):
        WorkflowGraph(nodes, edges).validate()


def test_validate_rejects_unconditioned_hub_edge():
    nodes = (Node("s", "start", "-", ""), Node("h", "hub", "-", ""),
             Node("t", "terminal", "-", ""))
    edges = (Edge("s", "h"), Edge("h", "t"))  # hub out-edge carries no branch condition
    with pytest.raises(WorkflowGraphError):
        WorkflowGraph(nodes, edges).validate()


# --------------------------------------------------------------------------- #
# Hub: routes on engine outputs alone, all four branches (pure, instant)
# --------------------------------------------------------------------------- #
def _sig(*, feasible=True, payment_risk_available=True, ev_positive=True,
         p_default=0.05, sev=SEV_OK):
    return HubSignals(
        feasible=feasible,
        payment_risk_available=payment_risk_available,
        risk_adjusted_ev_positive=ev_positive,
        p_default_at_target=p_default,
        calibration_severity_operational=sev,
    )


def test_hub_infeasible_branch():
    d = decide(_sig(feasible=False))
    assert d.decision == DECISION_NO_BID
    assert d.branch == BRANCH_INFEASIBLE
    assert d.terminal_state == TERMINAL_NO_BID
    assert d.warnings == [WARN_NO_FEASIBLE_BID]
    assert d.approval_decision == APPROVAL_NOT_APPLICABLE


def test_hub_negative_risk_adjusted_ev_branch():
    d = decide(_sig(ev_positive=False, payment_risk_available=True))
    assert d.decision == DECISION_NO_BID
    assert d.branch == BRANCH_NEGATIVE_RISK_ADJUSTED_EV
    assert d.terminal_state == TERMINAL_NO_BID
    assert d.warnings == [WARN_NEGATIVE_RISK_ADJUSTED_EV]
    assert d.approval_decision == APPROVAL_NOT_APPLICABLE


def test_hub_negative_ev_only_when_payment_risk_available():
    # EV "not positive" without a payment model is not the negative-EV branch.
    d = decide(_sig(ev_positive=False, payment_risk_available=False, p_default=None))
    assert d.branch == BRANCH_CLEAN_BID


def test_hub_escalates_on_high_payment_default():
    d = decide(_sig(p_default=0.20, sev=SEV_OK))  # >= 0.15 default threshold
    assert d.decision == "approval_required"
    assert d.branch == BRANCH_ESCALATED
    assert d.terminal_state == TERMINAL_APPROVAL_REQUIRED
    assert WARN_PAYMENT_RISK in d.warnings
    assert d.approval_decision == APPROVAL_HUMAN_REQUIRED


def test_hub_escalates_on_calibration_alert():
    d = decide(_sig(p_default=0.01, sev=SEV_ALERT))
    assert d.branch == BRANCH_ESCALATED
    assert WARN_CALIBRATION_ALERT in d.warnings
    assert d.approval_decision == APPROVAL_HUMAN_REQUIRED


def test_hub_clean_bid_branch():
    d = decide(_sig(p_default=0.01, sev=SEV_OK))
    assert d.decision == "bid"
    assert d.branch == BRANCH_CLEAN_BID
    assert d.terminal_state == TERMINAL_BID
    assert d.warnings == []
    assert d.approval_decision == APPROVAL_AUTO_ELIGIBLE


def test_hub_watch_warns_without_escalating():
    d = decide(_sig(p_default=0.01, sev=SEV_WATCH))
    assert d.branch == BRANCH_CLEAN_BID
    assert d.warnings == [WARN_CALIBRATION_WATCH]
    assert d.approval_decision == APPROVAL_AUTO_ELIGIBLE


def test_hub_payment_threshold_is_config_driven():
    sig = _sig(p_default=0.20, sev=SEV_OK)
    assert decide(sig).branch == BRANCH_ESCALATED                       # default warns at 0.15
    lax = DecisionHubPolicy(payment_default_warn=0.30)
    assert decide(sig, lax).branch == BRANCH_CLEAN_BID                  # laxer policy does not


def test_hub_covers_all_four_branches():
    seen = {
        decide(_sig(feasible=False)).branch,
        decide(_sig(ev_positive=False)).branch,
        decide(_sig(p_default=0.20)).branch,
        decide(_sig(p_default=0.01, sev=SEV_OK)).branch,
    }
    assert seen == {
        BRANCH_INFEASIBLE, BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
        BRANCH_ESCALATED, BRANCH_CLEAN_BID,
    }


# --------------------------------------------------------------------------- #
# Integration: seeded teacher generation (a couple of small in-memory builds)
# --------------------------------------------------------------------------- #
SMOKE_WORLDS = {"baseline", "slow_pay", "degraded_corner"}
SMOKE_DAYS = 6
SMOKE_MAX_LOADS = 120


def _tiny_cfg():
    cfg = load_ml_config()
    return replace(
        cfg,
        synthetic_data=replace(cfg.synthetic_data, loads_per_snapshot_mean=16.0,
                               snapshots_per_day=4),
    )


def _generate():
    cfg = _tiny_cfg()
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
def traces():
    return _generate()


def test_traces_generated_and_cover_expected_worlds(traces):
    assert len(traces) > 0
    assert {t.metadata.world_name for t in traces} == SMOKE_WORLDS


def test_every_trace_is_schema_complete_with_provenance(traces):
    for t in traces:
        assert isinstance(t.inference_context, InferenceContext)
        assert isinstance(t.node_outputs, NodeOutputs)
        assert isinstance(t.recommendation, Recommendation)
        assert isinstance(t.eval_labels, EvalLabels)
        assert isinstance(t.metadata, TraceMetadata)
        m = t.metadata  # all eight provenance fields stamped
        assert m.source_policy_version and m.git_commit and m.config_hash
        assert m.model_artifact_ids and m.world_name
        assert m.workflow_graph_version and m.teacher_trace_schema_version
        assert isinstance(m.random_seed, int)


def test_every_trace_reaches_a_terminal(traces):
    terminals = {n.name for n in build_default_graph().terminals()}
    for t in traces:
        assert t.path[0] == START
        assert t.path[-1] in terminals
        assert t.recommendation.terminal_state == t.path[-1]
        assert t.recommendation.decision in DECISIONS
        assert TERMINAL_BY_DECISION[t.recommendation.decision] == t.recommendation.terminal_state


def test_procedural_hub_reproduces_every_recommendation(traces):
    """The graph is control flow, not a second engine: re-deciding from the recorded
    node_outputs reproduces the label exactly."""
    policy = DecisionHubPolicy()
    for t in traces:
        d = decide(HubSignals.from_node_outputs(t.node_outputs), policy)
        assert d.decision == t.recommendation.decision
        assert d.terminal_state == t.recommendation.terminal_state
        assert d.branch == t.recommendation.hub_branch
        assert d.warnings == t.recommendation.warnings
        assert d.approval_decision == t.recommendation.approval_decision


def test_teacher_labels_are_engine_outputs_not_recomputed(traces):
    """The recommended bid is the engine's chosen ask copied onto the label (the teacher
    wraps the recommender, it does not re-derive a price)."""
    for t in traces:
        if t.recommendation.decision == DECISION_NO_BID:
            assert t.recommendation.recommended_bid_amount is None
            assert t.recommendation.recommended_bid_rpm is None
        else:
            assert t.recommendation.recommended_bid_amount == t.node_outputs.recommended_ask_engine
            assert t.recommendation.recommended_bid_rpm == t.node_outputs.recommended_ask_rpm_engine


def test_inference_context_has_no_leakage_on_real_traces(traces):
    inf = inference_field_names()
    for t in traces[:25]:
        keys = set(asdict(t.inference_context).keys())
        assert keys == inf
        assert keys.isdisjoint(node_output_field_names())
        assert keys.isdisjoint(eval_label_field_names())


def test_seeded_batch_exercises_the_common_branches(traces):
    branches = {t.recommendation.hub_branch for t in traces}
    # negative_risk_adjusted_ev is rare in practice (the engine seldom recommends into a
    # guaranteed loss) and is pinned by the hub unit tests; the seeded batch reliably
    # exercises the other three.
    assert {BRANCH_CLEAN_BID, BRANCH_INFEASIBLE, BRANCH_ESCALATED} <= branches
    assert branches <= {
        BRANCH_CLEAN_BID, BRANCH_INFEASIBLE, BRANCH_ESCALATED,
        BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
    }


def test_trace_json_roundtrips(traces):
    t = traces[0]
    again = AgentTrace.from_json_dict(json.loads(json.dumps(t.to_json_dict())))
    assert again == t


def test_summary_exposes_the_train_eligibility_contract(traces):
    s = build_summary(traces, days=SMOKE_DAYS, max_loads_per_world=SMOKE_MAX_LOADS)
    assert s["generated"]["n_traces"] == len(traces)
    assert set(s["train_eligibility"]["trainable_sections"]) == {"inference_context"}
    assert set(s["train_eligibility"]["non_trainable_sections"]) == {"node_outputs", "eval_labels"}
    assert s["train_eligibility"]["feature_eligible_fields"] == sorted(feature_eligible_fields())
    assert s["determinism_hash"] == trace_stream_fingerprint(traces)
    assert sum(s["decision_histogram"].values()) == len(traces)
    assert sum(s["hub_branch_histogram"].values()) == len(traces)


def test_generation_is_deterministic(traces):
    """Two independent generations from the same seed/config are byte-for-byte identical."""
    other = _generate()
    assert trace_stream_fingerprint(other) == trace_stream_fingerprint(traces)
    assert [t.to_json_dict() for t in other] == [t.to_json_dict() for t in traces]

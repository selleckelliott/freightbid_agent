"""Compiled-agent teacher-trace schema (Phase 6.1).

One :class:`AgentTrace` is a single traversal of the FreightBid workflow graph
(:mod:`ml.workflows.freightbid_workflow_graph`) by the **real source-of-truth engine**
for one board-load scenario. It is the raw material every later Phase 6 sub-phase
consumes: 6.2 reshapes traces into a training dataset, 6.3 distills a compiled model
from them, 6.4 shadows that model, 6.5 benchmarks it.

The schema's defining feature is a **hard separation** of what a compiled model may learn
from versus what is teacher-only:

* :class:`InferenceContext` — **the only train-eligible inputs.** Decision-time observable
  facts a live caller would have *before* running any model (load-board columns, broker
  board metadata, the carrier's own truck/cost facts, the observable market anchor). No
  model outputs, no engine internals, no realized outcomes.
* :class:`NodeOutputs` — the teacher's intermediate tool/model outputs (haul cost, P(win),
  payment risk, risk-adjusted EV, calibration/recalibration status). **Teacher-only:** for
  auditing, debugging, explanations, **label generation**, and ablations — *never* a
  compiled-model input. If the compiled model needed these it would still depend on the
  source engine at runtime, defeating the whole point of compiling the procedure.
* :class:`Recommendation` — the chosen load, bid, decision, warnings, approval decision,
  explanation, and terminal state. **These are the labels** the compiled model predicts.
* :class:`EvalLabels` — realized win / default / pay-days and the latent ground truth.
  **Evaluation only — never an inference input.**
* :class:`TraceMetadata` — provenance stamped on every trace from day one
  (source policy, git commit, config hash, model-artifact ids, seed, world, graph + schema
  versions), so 6.3 / 6.5 are defensible and reproducible.

The no-leakage invariant — ``inference_context`` field names are disjoint from both
``node_outputs`` and ``eval_labels`` — is enforced by :func:`assert_no_leakage` and pinned
by the Phase 6.1 tests; :func:`feature_eligible_fields` is the exact contract 6.3's
feature-matrix test will enforce.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, FrozenSet, List, Optional

# Bumped when the trace shape changes; stamped onto every trace's metadata.
TEACHER_TRACE_SCHEMA_VERSION = "1.0.0"

# Decision actions the workflow's terminal hub may recommend (the user's
# "choose bid / no-bid / approval-needed").
DECISION_BID = "bid"
DECISION_NO_BID = "no_bid"
DECISION_APPROVAL_REQUIRED = "approval_required"
DECISIONS = (DECISION_BID, DECISION_NO_BID, DECISION_APPROVAL_REQUIRED)

# Recommended approval routing (the dispatcher recommends; it never auto-submits).
APPROVAL_AUTO_ELIGIBLE = "auto_eligible"
APPROVAL_HUMAN_REQUIRED = "human_required"
APPROVAL_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class InferenceContext:
    """Decision-time observable facts — **the only train-eligible inputs.**

    Mirrors what a live Truckstop adapter would expose for one board load plus the
    carrier's own truck/cost knowledge and the coarse observable market anchor. No model
    output, no engine internal, no realized outcome appears here. Observable broker board
    metadata (e.g. an advertised ``broker_days_to_pay`` bucket) is fair game; the broker's
    *latent* true pay behavior is not (it lives in :class:`EvalLabels`).
    """

    load_id: str
    snapshot_time: str
    broker_id: Optional[str]
    equipment_type: str
    mode: str
    commodity: Optional[str]
    loaded_miles: float
    weight: float
    length: float
    origin_lat: float
    origin_lon: float
    load_views: str
    load_age_hours: float
    has_posted_rate: bool
    posted_rate_per_mile: Optional[float]
    tarp_required: Optional[bool]
    appointment_required: Optional[bool]
    broker_credit_bucket: Optional[str]
    broker_days_to_pay: Optional[int]
    broker_bonded: Optional[bool]
    broker_quick_pay_available: Optional[bool]
    broker_age_days: Optional[int]
    market_rate: float
    cost_per_loaded_mile: float
    truck_equipment_type: str


@dataclass(frozen=True)
class NodeOutputs:
    """The teacher engine's intermediate node outputs — **teacher-only, never trained on.**

    Everything the source-of-truth engine *computes* on the way to a recommendation. The
    workflow's decision hub branches on a subset of these (feasibility, EV sign, payment
    risk, operational calibration severity); the rest are recorded for audit / explanation /
    label generation. ``market_to_breakeven_ratio`` is the informational market-context
    (a.k.a. desirability) signal — recorded, but **not** a hub predicate.
    """

    estimated_cost: float
    breakeven_rpm: float
    market_to_breakeven_ratio: float
    feasible: bool
    winnability_available: bool
    payment_risk_available: bool
    win_probability_at_target: Optional[float]
    expected_value_at_target: Optional[float]
    risk_adjusted_ev_at_target: Optional[float]
    p_default_at_target: Optional[float]
    p_collect_at_target: Optional[float]
    expected_pay_days_at_target: Optional[float]
    delay_penalty_at_target: Optional[float]
    risk_adjusted_ev_positive: Optional[bool]
    risk_adjusted_warning: Optional[str]
    recommended_label: Optional[str]
    recommended_ask_engine: Optional[float]
    recommended_ask_rpm_engine: Optional[float]
    calibration_severity_before: str
    calibration_severity_after: str
    calibration_severity_operational: str
    recalibrator_promoted: bool


@dataclass(frozen=True)
class Recommendation:
    """The dispatcher's chosen action for one load — **the labels.**

    Derived from :class:`NodeOutputs` by the procedural hub (that is the whole point of
    distillation: learn to produce these from observable facts alone). ``hub_branch`` names
    the branch the hub took, so the procedural-hub test can re-derive the route from
    ``node_outputs`` and confirm the graph is control flow, not a second engine.
    """

    decision: str
    recommended_load_id: str
    recommended_bid_amount: Optional[float]
    recommended_bid_rpm: Optional[float]
    warnings: List[str]
    approval_decision: str
    explanation: str
    terminal_state: str
    hub_branch: str


@dataclass(frozen=True)
class EvalLabels:
    """Realized / latent ground truth — **evaluation only, never an inference input.**

    The world's hidden reservation + broker payment latents and the oracle-realized
    collectible profit at the recommended ask. Used to *score* the compiled model later, not
    to build its features.
    """

    reservation_rpm: Optional[float]
    true_default_prob: float
    true_pay_days: float
    realized_win_prob_at_recommended: Optional[float]
    realized_collectible_profit_if_bid: Optional[float]


@dataclass(frozen=True)
class TraceMetadata:
    """Provenance stamped on every trace from day one (makes 6.3 / 6.5 defensible)."""

    source_policy_version: str
    git_commit: str
    config_hash: str
    model_artifact_ids: Dict[str, str]
    random_seed: int
    world_name: str
    workflow_graph_version: str
    teacher_trace_schema_version: str


@dataclass(frozen=True)
class AgentTrace:
    """One workflow traversal: observable context -> engine node outputs -> labels (+eval)."""

    scenario_id: str
    path: List[str]
    inference_context: InferenceContext
    node_outputs: NodeOutputs
    recommendation: Recommendation
    eval_labels: EvalLabels
    metadata: TraceMetadata

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "path": list(self.path),
            "inference_context": asdict(self.inference_context),
            "node_outputs": asdict(self.node_outputs),
            "recommendation": asdict(self.recommendation),
            "eval_labels": asdict(self.eval_labels),
            "metadata": asdict(self.metadata),
        }

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "AgentTrace":
        return cls(
            scenario_id=data["scenario_id"],
            path=list(data["path"]),
            inference_context=InferenceContext(**data["inference_context"]),
            node_outputs=NodeOutputs(**data["node_outputs"]),
            recommendation=Recommendation(**data["recommendation"]),
            eval_labels=EvalLabels(**data["eval_labels"]),
            metadata=TraceMetadata(**data["metadata"]),
        )


# --------------------------------------------------------------------------- #
# Train-eligibility contract
# --------------------------------------------------------------------------- #
# The compiled model (6.3) may learn from TRAINABLE_SECTIONS only. The other two are
# teacher-only / eval-only and must never enter a feature matrix.
TRAINABLE_SECTIONS = ("inference_context",)
NON_TRAINABLE_SECTIONS = ("node_outputs", "eval_labels")


def _field_names(dc_type) -> FrozenSet[str]:
    return frozenset(f.name for f in fields(dc_type))


def inference_field_names() -> FrozenSet[str]:
    return _field_names(InferenceContext)


def node_output_field_names() -> FrozenSet[str]:
    return _field_names(NodeOutputs)


def eval_label_field_names() -> FrozenSet[str]:
    return _field_names(EvalLabels)


def feature_eligible_fields() -> FrozenSet[str]:
    """The exact set of fields a compiled model may use as inputs (6.3's hard contract)."""
    return inference_field_names()


def assert_no_leakage() -> bool:
    """Assert ``inference_context`` is disjoint from ``node_outputs`` and ``eval_labels``.

    Raises ``ValueError`` on any overlap so a future field added to the wrong section fails
    loudly. Returns ``True`` on success so callers (and tests) can assert on it.
    """
    inf = inference_field_names()
    leaked_nodes = inf & node_output_field_names()
    leaked_labels = inf & eval_label_field_names()
    if leaked_nodes or leaked_labels:
        raise ValueError(
            "inference_context leaks teacher-only fields: "
            f"node_outputs={sorted(leaked_nodes)} eval_labels={sorted(leaked_labels)}"
        )
    return True


# Fail at import time if the schema ever drifts into a leak.
assert_no_leakage()


def trace_stream_fingerprint(traces: List[AgentTrace]) -> str:
    """Deterministic SHA-256 over the trace stream, excluding the volatile ``git_commit``.

    Two generations from the same seed/config produce the same fingerprint regardless of the
    commit they run on, so it is safe to record in the committed summary and to pin in tests.
    """
    h = hashlib.sha256()
    for tr in traces:
        payload = tr.to_json_dict()
        payload["metadata"] = {k: v for k, v in payload["metadata"].items() if k != "git_commit"}
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return h.hexdigest()

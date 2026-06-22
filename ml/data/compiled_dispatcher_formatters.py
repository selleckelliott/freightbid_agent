"""Deterministic formatters that turn one teacher :class:`AgentTrace` into training examples
for the compiled dispatcher (Phase 6.2).

Every example exists in **two forms** rendered from the same trace:

* a **structured feature/label row** for the committed sklearn path (6.3), and
* a **natural-language conversation** (prompt -> JSON completion, plus an optional
  human-in-the-loop continuation) for the optional LLM path.

The one inviolable rule is the **train-eligibility boundary**, and it is asymmetric:

* **Inputs (features / the prompt) are built from ``inference_context`` ONLY** — the 25
  decision-time observable fields a live caller would have. Never ``node_outputs`` (the
  teacher's own model outputs) and never ``eval_labels`` (realized/latent outcomes). If a
  compiled model needed a ``node_output`` as an *input* it would still depend on the source
  engine at runtime, defeating the point of compiling the procedure.
* **Outputs (targets / the JSON completion) may draw from ``node_outputs`` and the
  ``recommendation``** — e.g. the predicted ``risk_adjusted_ev`` the runtime contract asks
  for. Predicting a quantity the engine computed is a regression head, not leakage; the model
  estimates it *from observable facts alone* at inference time.

:func:`build_features` is the single chokepoint for inputs and :func:`assert_features_inference_only`
hard-asserts the boundary; both are pinned by the Phase 6.2 tests.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional

from domain.enums.bid_approval_status import BidApprovalStatus
from ml.data.compiled_agent_trace_schema import (
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
    AgentTrace,
    eval_label_field_names,
    inference_field_names,
    node_output_field_names,
)
from ml.workflows.freightbid_workflow_graph import (
    BRANCH_CLEAN_BID,
    BRANCH_ESCALATED,
    BRANCH_INFEASIBLE,
    BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
    WARN_CALIBRATION_ALERT,
    WARN_CALIBRATION_WATCH,
    WARN_PAYMENT_RISK,
)

# Bumped when the row/conversation shape changes; stamped onto the dataset summary.
DISPATCHER_DATASET_VERSION = "1.0.0"

# High-default flag threshold (mirrors the decision hub's payment_default_warn default).
HIGH_DEFAULT_FLAG = 0.15

# Primary scenario taxonomy (model-relevant: decision x branch x warning).
CATEGORY_CLEAN_BID = "clean_bid"
CATEGORY_CLEAN_BID_WATCH = "clean_bid_watch"
CATEGORY_PAYMENT_ESCALATION = "payment_escalation"
CATEGORY_CALIBRATION_ESCALATION = "calibration_escalation"
CATEGORY_NEGATIVE_EV_NO_BID = "negative_ev_no_bid"
CATEGORY_INFEASIBLE_NO_BID = "infeasible_no_bid"

# Secondary coverage flags (the user's path list, reported alongside the primary category).
FLAG_PROFITABLE_BID = "profitable_bid"
FLAG_NO_SAFE_BID = "no_safe_bid"
FLAG_APPROVAL_REQUIRED = "approval_required"
FLAG_HIGH_DEFAULT_RISK = "high_default_risk"
FLAG_SLOW_PAY_WORLD = "slow_pay_world"
FLAG_RECALIBRATION_APPLIED = "recalibration_applied"

# Synthetic human-in-the-loop actions for approval_required conversations (real domain enum).
HUMAN_ACTIONS = (
    BidApprovalStatus.APPROVED,
    BidApprovalStatus.EDITED,
    BidApprovalStatus.REJECTED,
    BidApprovalStatus.SUBMITTED_MOCK,
)


# --------------------------------------------------------------------------- #
# Inputs: inference_context ONLY
# --------------------------------------------------------------------------- #
def build_features(trace: AgentTrace) -> "OrderedDict[str, Any]":
    """The compiled model's inputs — a passthrough of ``inference_context``, nothing else.

    This is the *only* place inputs are constructed, so the train-eligibility boundary has a
    single chokepoint. The returned key set is exactly :func:`inference_field_names`.
    """
    feats = OrderedDict(sorted(asdict(trace.inference_context).items()))
    assert_features_inference_only(feats)
    return feats


def assert_features_inference_only(features: Mapping[str, Any]) -> bool:
    """Assert a feature mapping is drawn from ``inference_context`` alone.

    Raises ``ValueError`` if any key is unknown, or collides with a ``node_outputs`` /
    ``eval_labels`` field (teacher-only / eval-only). Returns ``True`` on success so tests can
    assert on it.
    """
    keys = set(features)
    inf = inference_field_names()
    leaked_nodes = keys & node_output_field_names()
    leaked_labels = keys & eval_label_field_names()
    if leaked_nodes or leaked_labels:
        raise ValueError(
            "compiled-dispatcher features leak teacher-only fields: "
            f"node_outputs={sorted(leaked_nodes)} eval_labels={sorted(leaked_labels)}"
        )
    unknown = keys - inf
    if unknown:
        raise ValueError(f"compiled-dispatcher features contain non-inference fields: {sorted(unknown)}")
    missing = inf - keys
    if missing:
        raise ValueError(f"compiled-dispatcher features missing inference fields: {sorted(missing)}")
    return True


# --------------------------------------------------------------------------- #
# Outputs: targets may draw from node_outputs + recommendation
# --------------------------------------------------------------------------- #
def build_targets(trace: AgentTrace) -> "OrderedDict[str, Any]":
    """The labels the compiled model predicts (output side — node_outputs allowed here)."""
    rec = trace.recommendation
    node = trace.node_outputs
    return OrderedDict([
        ("recommended_load_id", rec.recommended_load_id),
        ("decision", rec.decision),
        ("hub_branch", rec.hub_branch),
        ("approval_decision", rec.approval_decision),
        ("recommended_bid_amount", rec.recommended_bid_amount),
        ("recommended_bid_rpm", rec.recommended_bid_rpm),
        ("risk_adjusted_ev", node.risk_adjusted_ev_at_target),
        ("win_probability", node.win_probability_at_target),
        ("p_default", node.p_default_at_target),
        ("warnings", list(rec.warnings)),
        ("explanation", rec.explanation),
    ])


def runtime_json(trace: AgentTrace) -> "OrderedDict[str, Any]":
    """The inference-time output contract a served dispatcher returns (the 6.3 runtime shape)."""
    rec = trace.recommendation
    rae = trace.node_outputs.risk_adjusted_ev_at_target
    return OrderedDict([
        ("recommended_load_id", rec.recommended_load_id),
        ("recommended_bid", rec.recommended_bid_amount),
        ("decision", rec.decision),
        ("risk_adjusted_ev", round(rae, 2) if rae is not None else None),
        ("warnings", list(rec.warnings)),
        ("explanation", rec.explanation),
    ])


# --------------------------------------------------------------------------- #
# Coverage taxonomy (teacher-side: stratification + honest reporting, never a feature)
# --------------------------------------------------------------------------- #
def scenario_category(trace: AgentTrace) -> str:
    branch = trace.recommendation.hub_branch
    warnings = set(trace.recommendation.warnings)
    if branch == BRANCH_INFEASIBLE:
        return CATEGORY_INFEASIBLE_NO_BID
    if branch == BRANCH_NEGATIVE_RISK_ADJUSTED_EV:
        return CATEGORY_NEGATIVE_EV_NO_BID
    if branch == BRANCH_ESCALATED:
        return (CATEGORY_CALIBRATION_ESCALATION
                if WARN_CALIBRATION_ALERT in warnings else CATEGORY_PAYMENT_ESCALATION)
    if branch == BRANCH_CLEAN_BID:
        return CATEGORY_CLEAN_BID_WATCH if WARN_CALIBRATION_WATCH in warnings else CATEGORY_CLEAN_BID
    return branch  # pragma: no cover - graph guarantees one of the above


def coverage_flags(trace: AgentTrace) -> List[str]:
    rec = trace.recommendation
    node = trace.node_outputs
    flags: List[str] = []
    if rec.decision == DECISION_BID:
        flags.append(FLAG_PROFITABLE_BID)
    if rec.decision == DECISION_NO_BID:
        flags.append(FLAG_NO_SAFE_BID)
    if rec.decision == DECISION_APPROVAL_REQUIRED:
        flags.append(FLAG_APPROVAL_REQUIRED)
    if node.p_default_at_target is not None and node.p_default_at_target >= HIGH_DEFAULT_FLAG:
        flags.append(FLAG_HIGH_DEFAULT_RISK)
    if trace.metadata.world_name == "slow_pay":
        flags.append(FLAG_SLOW_PAY_WORLD)
    if node.recalibrator_promoted:
        flags.append(FLAG_RECALIBRATION_APPLIED)
    return flags


# --------------------------------------------------------------------------- #
# Natural-language rendering (prompt from features ONLY)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a FreightBid dispatcher assistant. Given one board load and the carrier's truck, "
    "return only a JSON recommendation with keys: recommended_load_id, recommended_bid, "
    "decision, risk_adjusted_ev, warnings, explanation."
)


def _fmt(v: Any, *, money: bool = False, none: str = "unknown") -> str:
    if v is None:
        return none
    if money:
        return f"${float(v):,.2f}"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def render_prompt(features: Mapping[str, Any]) -> str:
    """The dispatcher's case-fact prompt, built from ``inference_context`` features ONLY.

    Deliberately contains *no* workflow procedure (no node list, no routing rules) and *no*
    model output — just the observable truck / load / broker / market facts, mirroring what a
    live caller would type. The compiled model must internalize the procedure, not be handed it.
    """
    assert_features_inference_only(features)
    f = features
    posted = (f"{_fmt(f['posted_rate_per_mile'])}/mi" if f.get("has_posted_rate")
              else "no posted rate")
    lines = [
        "A dispatcher wants the best bid/no-bid plan for one load on the board. "
        "Use only the facts below.",
        "",
        "Truck:",
        f"- equipment: {_fmt(f['truck_equipment_type'])}",
        f"- cost basis: {_fmt(f['cost_per_loaded_mile'], money=True)} per loaded mile",
        "",
        f"Load {_fmt(f['load_id'])} (posted {_fmt(f['load_age_hours'])} h ago):",
        f"- {_fmt(f['loaded_miles'])} loaded mi, {_fmt(f['weight'])} lb, {_fmt(f['length'])} ft; "
        f"commodity {_fmt(f['commodity'])}; mode {_fmt(f['mode'])}; "
        f"equipment {_fmt(f['equipment_type'])}",
        f"- origin ({_fmt(f['origin_lat'])}, {_fmt(f['origin_lon'])}); posted rate {posted}",
        f"- board views: {_fmt(f['load_views'])}; tarp {_fmt(f['tarp_required'])}; "
        f"appointment {_fmt(f['appointment_required'])}",
        "",
        f"Broker {_fmt(f['broker_id'])}:",
        f"- credit bucket {_fmt(f['broker_credit_bucket'])}; advertised days-to-pay "
        f"{_fmt(f['broker_days_to_pay'])}",
        f"- bonded {_fmt(f['broker_bonded'])}; quick-pay {_fmt(f['broker_quick_pay_available'])}; "
        f"age {_fmt(f['broker_age_days'])} d",
        "",
        f"Market anchor: {_fmt(f['market_rate'])}/mi at this origin.",
        "",
        "Return the JSON recommendation.",
    ]
    return "\n".join(lines)


def render_completion(trace: AgentTrace) -> str:
    """The assistant turn: the runtime JSON recommendation (parseable, sorted keys)."""
    return json.dumps(runtime_json(trace), indent=2)


# --------------------------------------------------------------------------- #
# Synthetic human-in-the-loop continuation (conversations only; never a structured row)
# --------------------------------------------------------------------------- #
def synthetic_human_action(trace: AgentTrace) -> Optional[BidApprovalStatus]:
    """Deterministically assign a human action to an ``approval_required`` recommendation.

    Hashing ``scenario_id`` spreads the four outcomes (approve / edit / reject / submit-mock)
    reproducibly so the conversation dataset covers the full Phase 4.4 lifecycle without any
    dependence on hidden outcomes. ``None`` for non-escalated recommendations.
    """
    if trace.recommendation.decision != DECISION_APPROVAL_REQUIRED:
        return None
    digest = hashlib.sha256(trace.scenario_id.encode("utf-8")).hexdigest()
    return HUMAN_ACTIONS[int(digest, 16) % len(HUMAN_ACTIONS)]


def _human_turns(trace: AgentTrace, action: BidApprovalStatus) -> List[Dict[str, str]]:
    amount = trace.recommendation.recommended_bid_amount
    if action is BidApprovalStatus.APPROVED:
        user = "Approve this bid as recommended."
        ack = "Approved. The draft is marked approved; it is not submitted to any broker."
    elif action is BidApprovalStatus.EDITED:
        edited = round(float(amount) * 0.97, 2) if amount is not None else None
        user = f"Lower the bid to {_fmt(edited, money=True)} and keep it pending approval."
        ack = (f"Edited the draft to {_fmt(edited, money=True)} and left it pending approval "
               "for a human to confirm.")
    elif action is BidApprovalStatus.REJECTED:
        user = "Reject this bid; the broker risk is not worth it."
        ack = "Rejected. The draft is closed and no bid will be placed."
    else:  # SUBMITTED_MOCK
        user = "Approve and submit it (mock)."
        ack = ("Submitted in mock mode for workflow validation only — this is not a real "
               "broker or Truckstop submission.")
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": f"[{action.value}] {ack}"},
    ]


def render_conversation(trace: AgentTrace) -> "OrderedDict[str, Any]":
    """A full dispatcher conversation: system + user case facts -> JSON recommendation, plus a
    deterministic human-in-the-loop continuation when the recommendation is approval_required."""
    features = build_features(trace)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_prompt(features)},
        {"role": "assistant", "content": render_completion(trace)},
    ]
    action = synthetic_human_action(trace)
    if action is not None:
        messages.extend(_human_turns(trace, action))
    return OrderedDict([
        ("scenario_id", trace.scenario_id),
        ("world", trace.metadata.world_name),
        ("scenario_category", scenario_category(trace)),
        ("human_action", action.value if action is not None else None),
        ("synthetic_continuation", action is not None),
        ("messages", messages),
    ])


# --------------------------------------------------------------------------- #
# Structured row
# --------------------------------------------------------------------------- #
def to_structured_row(trace: AgentTrace) -> "OrderedDict[str, Any]":
    """One model-ready row: inference-only features + output-side targets + coverage labels."""
    return OrderedDict([
        ("scenario_id", trace.scenario_id),
        ("world", trace.metadata.world_name),
        ("scenario_category", scenario_category(trace)),
        ("coverage_flags", coverage_flags(trace)),
        ("features", build_features(trace)),
        ("targets", build_targets(trace)),
    ])

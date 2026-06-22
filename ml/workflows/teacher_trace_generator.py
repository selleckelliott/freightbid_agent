"""Teacher trace generator (Phase 6.1).

Walk the FreightBid workflow graph (:mod:`ml.workflows.freightbid_workflow_graph`) with the
**real source-of-truth engine** and record one
:class:`~ml.data.compiled_agent_trace_schema.AgentTrace` per board-load scenario. This is the
teacher dataset every later Phase 6 sub-phase consumes — the teacher *wraps* the engine, it
never re-implements it.

The engine is assembled **in memory**, reusing the exact machinery the Phase 5.5 risk-aware
stress harness already proved (``benchmarks/run_risk_aware_stress.py``): the calibrated
winnability model and the payment-risk model are trained once on the baseline world and
frozen; each scenario world is drawn from a later, disjoint operational seed; the Phase 5.4
recalibrator is fit + promoted per world on its own fit -> eval split; and the full
risk-aware recommender (risk-adjusted EV + the promoted recalibration wrapper) is the teacher.
No committed model artifacts are needed, so the teacher is fully self-contained and
deterministic.

Each node records its real input/output into ``node_outputs`` (teacher-only); the terminal
``choose_action`` hub branches on a strict subset of those outputs; the chosen action +
warnings + explanation become the ``recommendation`` labels; the world's hidden latents +
oracle-realized collectible profit become the eval-only ``eval_labels``; and provenance is
stamped on every trace.

Examples
--------
    # quick smoke (tiny seeded builds, 2 worlds, capped loads)
    python -m ml.workflows.teacher_trace_generator --fast

    # canonical generation -> committed summary + gitignored full traces
    python -m ml.workflows.teacher_trace_generator --days 14 \
        --out artifacts/teacher_trace_summary.json \
        --traces-out data/teacher_traces.jsonl
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import time
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from adapters.outbound.winnability.recalibrated_winnability_adapter import (
    RecalibratedWinnabilityAdapter,
)
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from benchmarks.run_bid_recommender_eval import _build_frame
from benchmarks.run_broker_quality_stress import Condition, load_conditions, world_cfg
from benchmarks.run_calibration_monitor import _calibrated_model
from benchmarks.run_risk_aware_stress import (
    SEED_OFFSET,
    _eval_window_snapshots,
    _score_ask,
    _train_payment_adapter,
)
from domain.models.bid_recommendation import MAX_EV, TARGET
from ml.brokers import BrokerPoolParams, broker_index, build_broker_pool
from ml.calibration.recalibration_workflow import (
    RecalibrationConfig,
    load_recalibration_config,
    recalibrate,
    time_split,
)
from ml.calibration.recalibrator import fit_recalibrator
from ml.config import MLConfig, load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.compiled_agent_trace_schema import (
    TEACHER_TRACE_SCHEMA_VERSION,
    AgentTrace,
    EvalLabels,
    InferenceContext,
    NodeOutputs,
    Recommendation,
    TraceMetadata,
    feature_eligible_fields,
    trace_stream_fingerprint,
)
from ml.data.load_history_schema import iso
from ml.data.outcome_schema import read_outcomes
from ml.features.winnability_features import BidQuery, market_rate_for
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.monitoring.calibration_report import load_calibration_config
from ml.training.winnability_dataset import LABEL as WIN_LABEL
from ml.training.winnability_dataset import resolve_path
from ml.workflows.freightbid_workflow_graph import (
    BRANCH_TERMINAL,
    DECISION_NO_BID,
    DecisionHubPolicy,
    HubSignals,
    WorkflowGraph,
    build_default_graph,
    decide,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDITIONS = ROOT / "config" / "broker_quality_stress.yaml"
DEFAULT_RECAL_CONFIG = ROOT / "config" / "recalibration.yaml"
DEFAULT_MONITOR_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_OUT = ROOT / "artifacts" / "teacher_trace_summary.json"
DEFAULT_TRACES_OUT = ROOT / "data" / "teacher_traces.jsonl"

# The teacher is the full Phase 5.5 risk-aware stack (risk-adjusted EV + promoted recalibration).
SOURCE_POLICY_VERSION = "phase-5.5-full-risk-aware"
# A single fixed hotshot asset for the offline teacher (the bid engine is carrier-relative; the
# truck contributes equipment + the per-loaded-mile cost basis, not a route).
TEACHER_TRUCK_EQUIPMENT = "hotshot"

# Worlds chosen for decision-path coverage (clean bids, payment escalation, EV-negative no-bids,
# calibration shifts). Drawn from config/broker_quality_stress.yaml.
DEFAULT_WORLDS = (
    "baseline", "risky_brokers", "unknown_credit", "slow_pay", "tight_brokers",
    "high_contention", "degraded_corner",
)
FAST_WORLDS = ("baseline", "risky_brokers")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:  # pragma: no cover - provenance best-effort
        return "unknown"


def _config_hash(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    recal_config: RecalibrationConfig,
    thresholds: CalibrationThresholds,
    overrides: Optional[Dict[str, Any]],
) -> str:
    payload = {
        "synthetic_data": {
            "seed": cfg.synthetic_data.seed,
            "loads_per_snapshot_mean": cfg.synthetic_data.loads_per_snapshot_mean,
            "snapshots_per_day": cfg.synthetic_data.snapshots_per_day,
        },
        "bid": dataclasses.asdict(bid_cfg),
        "recal": dataclasses.asdict(recal_config),
        "thresholds": dataclasses.asdict(thresholds),
        "overrides": overrides or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Per-scenario trace
# --------------------------------------------------------------------------- #
def _explain(decision: str, node_outputs: NodeOutputs, warnings: List[str]) -> str:
    sev = node_outputs.calibration_severity_operational
    if not node_outputs.feasible:
        return "No in-support, guardrail-clearing ask for this load; recommend no bid."
    rae = node_outputs.risk_adjusted_ev_at_target
    ask = node_outputs.recommended_ask_engine
    rpm = node_outputs.recommended_ask_rpm_engine
    pwin = node_outputs.win_probability_at_target
    pdef = node_outputs.p_default_at_target
    if decision == DECISION_NO_BID:
        return (
            f"Best in-support ask is collectible-EV-negative (risk-adjusted EV "
            f"{rae:.2f}); recommend no bid."
        )
    base = (
        f"Recommend ${ask:.0f} ({rpm:.2f}/mi); win {pwin:.0%}, "
        f"risk-adjusted EV {rae:.2f}, calibration {sev}"
    )
    if warnings:
        extra = []
        if pdef is not None:
            extra.append(f"p_default {pdef:.2f}")
        extra.append("escalate to human approval" if decision == "approval_required" else "")
        tail = "; ".join([w for w in extra if w])
        return f"{base}; warnings={','.join(warnings)}" + (f" ({tail})" if tail else "") + "."
    return base + "."


def _trace_one(
    graph: WorkflowGraph,
    policy: DecisionHubPolicy,
    recommender: EVBidRecommender,
    snapshot,
    world: MLConfig,
    bid_cfg: BidRecommenderConfig,
    pool_index: Dict[str, Any],
    reserve: float,
    sev_before: str,
    sev_after: str,
    sev_operational: str,
    promoted: bool,
    metadata: TraceMetadata,
) -> AgentTrace:
    miles = max(float(snapshot.loaded_miles), 1.0)
    cost = bid_cfg.cost_per_loaded_mile * miles
    market = market_rate_for(snapshot.origin_lat, snapshot.origin_lon)
    breakeven_rpm = cost / miles
    query = BidQuery.from_snapshot(snapshot)

    # FILTER node: feasible iff an in-support, guardrail-clearing candidate exists.
    scoring = recommender.score(query, estimated_total_cost=cost)
    winnability_available = scoring is not None
    feasible = scoring is not None and any(not c.extrapolated for c in scoring.candidates)

    tgt = None
    rec = None
    if feasible:
        rec = recommender.recommend(
            query, load_id=0, broker_id=snapshot.broker_id, estimated_total_cost=cost
        )
        tgt = rec.option(TARGET) or rec.option(MAX_EV)
        feasible = tgt is not None  # defensive: feasible only if a target rung exists

    if feasible:
        node_outputs = NodeOutputs(
            estimated_cost=round(cost, 2),
            breakeven_rpm=round(breakeven_rpm, 4),
            market_to_breakeven_ratio=round(market / breakeven_rpm, 4),
            feasible=True,
            winnability_available=rec.winnability_available,
            payment_risk_available=rec.payment_risk_available,
            win_probability_at_target=tgt.win_probability,
            expected_value_at_target=tgt.expected_value,
            risk_adjusted_ev_at_target=tgt.risk_adjusted_ev,
            p_default_at_target=tgt.p_default,
            p_collect_at_target=tgt.p_collect,
            expected_pay_days_at_target=tgt.expected_pay_days,
            delay_penalty_at_target=tgt.delay_penalty,
            risk_adjusted_ev_positive=rec.risk_adjusted_ev_positive,
            risk_adjusted_warning=rec.risk_adjusted_warning,
            recommended_label=rec.recommended_label,
            recommended_ask_engine=tgt.ask_amount,
            recommended_ask_rpm_engine=tgt.ask_rpm,
            calibration_severity_before=sev_before,
            calibration_severity_after=sev_after,
            calibration_severity_operational=sev_operational,
            recalibrator_promoted=promoted,
        )
    else:
        node_outputs = NodeOutputs(
            estimated_cost=round(cost, 2),
            breakeven_rpm=round(breakeven_rpm, 4),
            market_to_breakeven_ratio=round(market / breakeven_rpm, 4),
            feasible=False,
            winnability_available=winnability_available,
            payment_risk_available=False,
            win_probability_at_target=None,
            expected_value_at_target=None,
            risk_adjusted_ev_at_target=None,
            p_default_at_target=None,
            p_collect_at_target=None,
            expected_pay_days_at_target=None,
            delay_penalty_at_target=None,
            risk_adjusted_ev_positive=None,
            risk_adjusted_warning=None,
            recommended_label=None,
            recommended_ask_engine=None,
            recommended_ask_rpm_engine=None,
            calibration_severity_before=sev_before,
            calibration_severity_after=sev_after,
            calibration_severity_operational=sev_operational,
            recalibrator_promoted=promoted,
        )

    # Decision hub: route purely on the recorded node outputs.
    hub = decide(HubSignals.from_node_outputs(node_outputs), policy)
    post_hub = graph.route(hub.branch)  # [EXPLAIN, terminal]
    path = graph.linear_prefix() + post_hub

    bids = hub.decision != DECISION_NO_BID
    recommendation = Recommendation(
        decision=hub.decision,
        recommended_load_id=str(snapshot.load_id),
        recommended_bid_amount=(node_outputs.recommended_ask_engine if bids else None),
        recommended_bid_rpm=(node_outputs.recommended_ask_rpm_engine if bids else None),
        warnings=list(hub.warnings),
        approval_decision=hub.approval_decision,
        explanation=_explain(hub.decision, node_outputs, hub.warnings),
        terminal_state=hub.terminal_state,
        hub_branch=hub.branch,
    )

    broker = pool_index.get(snapshot.broker_id)
    p_default_true = broker.true_default_prob if broker is not None else 0.05
    pay_days_true = broker.true_pay_days if broker is not None else 35.0
    realized_pwin = None
    realized_collectible = None
    if feasible and reserve is not None:
        oracle = _score_ask(
            node_outputs.recommended_ask_rpm_engine, node_outputs.recommended_ask_engine,
            cost, reserve, world.outcomes.win_logistic_scale_rpm,
            p_default_true, pay_days_true, bid_cfg.annual_cash_cost_rate, bid_cfg.free_pay_days,
        )
        realized_pwin = round(float(oracle["p_win"]), 6)
        realized_collectible = round(float(oracle["collectible"]), 4)

    eval_labels = EvalLabels(
        reservation_rpm=(round(float(reserve), 4) if reserve is not None else None),
        true_default_prob=round(float(p_default_true), 4),
        true_pay_days=round(float(pay_days_true), 2),
        realized_win_prob_at_recommended=realized_pwin,
        realized_collectible_profit_if_bid=realized_collectible,
    )

    context = InferenceContext(
        load_id=str(snapshot.load_id),
        snapshot_time=iso(snapshot.snapshot_time),
        broker_id=snapshot.broker_id,
        equipment_type=query.equipment_type,
        mode=query.mode,
        commodity=query.commodity,
        loaded_miles=round(float(query.loaded_miles), 2),
        weight=round(float(query.weight), 2),
        length=round(float(query.length), 2),
        origin_lat=round(float(query.origin_lat), 5),
        origin_lon=round(float(query.origin_lon), 5),
        load_views=str(query.load_views),
        load_age_hours=round(float(query.load_age_hours), 3),
        has_posted_rate=query.rate_per_mile is not None,
        posted_rate_per_mile=(round(float(query.rate_per_mile), 4)
                              if query.rate_per_mile is not None else None),
        tarp_required=query.tarp_required,
        appointment_required=query.appointment_required,
        broker_credit_bucket=query.broker_credit_bucket,
        broker_days_to_pay=query.broker_days_to_pay,
        broker_bonded=query.broker_bonded,
        broker_quick_pay_available=query.broker_quick_pay_available,
        broker_age_days=query.broker_age_days,
        market_rate=round(float(market), 4),
        cost_per_loaded_mile=round(float(bid_cfg.cost_per_loaded_mile), 4),
        truck_equipment_type=TEACHER_TRUCK_EQUIPMENT,
    )

    scenario_id = f"{metadata.world_name}:{snapshot.load_id}:{iso(snapshot.snapshot_time)}"
    return AgentTrace(
        scenario_id=scenario_id,
        path=path,
        inference_context=context,
        node_outputs=node_outputs,
        recommendation=recommendation,
        eval_labels=eval_labels,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# World assembly + generation
# --------------------------------------------------------------------------- #
def generate_traces(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    thresholds: CalibrationThresholds,
    recal_config: RecalibrationConfig,
    conditions: List[Condition],
    *,
    days: int,
    max_loads_per_world: Optional[int],
    seed_offset: int = SEED_OFFSET,
    policy: Optional[DecisionHubPolicy] = None,
    graph: Optional[WorkflowGraph] = None,
    git_commit: Optional[str] = None,
) -> List[AgentTrace]:
    """Freeze the base models on baseline, then trace the workflow on every world's eval loads.

    Mirrors the Phase 5.5 assembly: calibrated winnability + payment-risk models trained once on
    the baseline training draw and frozen; each world drawn from a later operational seed; the
    Phase 5.4 recalibrator fit + promoted per world; the full risk-aware recommender is the
    teacher. Returns the traces in a deterministic order (worlds by (has-overrides, name); loads
    by (snapshot_time, load_id)).
    """
    graph = graph or build_default_graph()
    policy = policy or DecisionHubPolicy()
    git = git_commit if git_commit is not None else _git_commit()
    conditions = sorted(conditions, key=lambda c: (bool(c.overrides), c.name))
    op_seed = cfg.synthetic_data.seed + seed_offset
    traces: List[AgentTrace] = []

    with TemporaryDirectory() as base_tmp:
        base_world = world_cfg(cfg, Path(base_tmp))
        build_winnability_dataset(base_world, days=days)
        base_frame, base_snaps = _build_frame(base_world)
        base_model = _calibrated_model(base_frame, base_world.winnability.random_seed)
        base_adapter = ModelWinnabilityAdapter(base_model)
        base_outcomes = read_outcomes(resolve_path(base_world.outcomes.outcomes_path))
        payment_adapter = _train_payment_adapter(base_world, base_snaps, base_outcomes)
        win_id = f"winnability:isotonic:seed={base_world.winnability.random_seed}:days={days}"
        pay_id = f"payment:gbm:seed={base_world.payment_risk.random_seed}"

        bid_ra = replace(bid_cfg, risk_adjusted_ev_enabled=True)

        for cond in conditions:
            with TemporaryDirectory() as ctmp:
                world = world_cfg(cfg, Path(ctmp), cond.overrides or None)
                build_winnability_dataset(world, days=days, seed=op_seed)
                frame, snaps = _build_frame(world)
                outcomes = read_outcomes(resolve_path(world.outcomes.outcomes_path))
                pool_index = broker_index(
                    build_broker_pool(BrokerPoolParams.from_config(world.brokers))
                )

                # Phase 5.4 recalibrator: fit on the early window, promote only if safer.
                fit_df, eval_df = time_split(frame, recal_config.fit_days, recal_config.eval_days)
                raw_fit = base_model.predict_proba(fit_df)
                raw_eval = base_model.predict_proba(eval_df)
                result = recalibrate(
                    raw_fit, fit_df[WIN_LABEL].to_numpy(),
                    raw_eval, eval_df[WIN_LABEL].to_numpy(),
                    thresholds, recal_config, label=cond.name,
                )
                recalibrator = (
                    fit_recalibrator(raw_fit, fit_df[WIN_LABEL].to_numpy(), method=recal_config.method)
                    if result.promoted else None
                )
                sev_before = result.pre.severity
                sev_after = result.post.severity if result.post is not None else result.pre.severity
                sev_operational = sev_after if result.promoted else sev_before
                recal_id = f"recalibrator:{recal_config.method}" if result.promoted else "none"

                serve_adapter = (
                    RecalibratedWinnabilityAdapter(base_adapter, recalibrator)
                    if recalibrator is not None else base_adapter
                )
                recommender = EVBidRecommender(serve_adapter, bid_ra, payment=payment_adapter)

                metadata = TraceMetadata(
                    source_policy_version=SOURCE_POLICY_VERSION,
                    git_commit=git,
                    config_hash=_config_hash(cfg, bid_cfg, recal_config, thresholds, cond.overrides),
                    model_artifact_ids={
                        "winnability": win_id, "payment": pay_id, "recalibrator": recal_id,
                    },
                    random_seed=op_seed,
                    world_name=cond.name,
                    workflow_graph_version=graph.version,
                    teacher_trace_schema_version=TEACHER_TRACE_SCHEMA_VERSION,
                )

                reserve_by_key = {(o.load_id, o.snapshot_time): o.reservation_rpm for o in outcomes}
                eval_snaps = _eval_window_snapshots(
                    snaps, recal_config.fit_days, recal_config.eval_days
                )
                eval_snaps = sorted(eval_snaps, key=lambda s: (s.snapshot_time, s.load_id))
                if max_loads_per_world is not None:
                    eval_snaps = eval_snaps[:max_loads_per_world]

                for s in eval_snaps:
                    key = (s.load_id, s.snapshot_time)
                    if key not in reserve_by_key:
                        continue  # no realized ground truth for this load -> no eval labels
                    traces.append(_trace_one(
                        graph, policy, recommender, s, world, bid_cfg, pool_index,
                        reserve_by_key[key], sev_before, sev_after, sev_operational,
                        result.promoted, metadata,
                    ))
    return traces


def write_traces(traces: List[AgentTrace], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for tr in traces:
            fh.write(json.dumps(tr.to_json_dict()) + "\n")
    return len(traces)


def build_summary(
    traces: List[AgentTrace], *, days: int, max_loads_per_world: Optional[int],
) -> Dict[str, Any]:
    """A lean, committed summary: counts, coverage histograms, determinism hash, provenance."""
    decisions = Counter(t.recommendation.decision for t in traces)
    branches = Counter(t.recommendation.hub_branch for t in traces)
    terminals = Counter(t.recommendation.terminal_state for t in traces)
    warnings = Counter(w for t in traces for w in t.recommendation.warnings)

    per_world: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for t in traces:
        w = t.metadata.world_name
        if w not in per_world:
            per_world[w] = {
                "world": w,
                "n": 0,
                "decisions": Counter(),
                "branches": Counter(),
                "recalibrator_promoted": t.node_outputs.recalibrator_promoted,
                "calibration_severity_before": t.node_outputs.calibration_severity_before,
                "calibration_severity_after": t.node_outputs.calibration_severity_after,
                "calibration_severity_operational": t.node_outputs.calibration_severity_operational,
                "config_hash": t.metadata.config_hash,
            }
        per_world[w]["n"] += 1
        per_world[w]["decisions"][t.recommendation.decision] += 1
        per_world[w]["branches"][t.recommendation.hub_branch] += 1
    world_rows = []
    for w in per_world.values():
        w["decisions"] = dict(w["decisions"])
        w["branches"] = dict(w["branches"])
        world_rows.append(w)

    first = traces[0].metadata if traces else None
    return {
        "schema_version": TEACHER_TRACE_SCHEMA_VERSION,
        "workflow_graph_version": (traces[0].metadata.workflow_graph_version if traces else None),
        "source_policy_version": SOURCE_POLICY_VERSION,
        "generated": {
            "n_traces": len(traces),
            "n_worlds": len(world_rows),
            "days": days,
            "max_loads_per_world": max_loads_per_world,
            "random_seed": (first.random_seed if first else None),
        },
        "determinism_hash": trace_stream_fingerprint(traces),
        "decision_histogram": dict(decisions),
        "hub_branch_histogram": dict(branches),
        "terminal_histogram": dict(terminals),
        "warning_histogram": dict(warnings),
        "per_world": world_rows,
        "provenance": {
            "git_commit": (first.git_commit if first else None),
            "model_artifact_ids": (first.model_artifact_ids if first else None),
            "random_seed": (first.random_seed if first else None),
        },
        "train_eligibility": {
            "trainable_sections": ["inference_context"],
            "non_trainable_sections": ["node_outputs", "eval_labels"],
            "feature_eligible_fields": sorted(feature_eligible_fields()),
        },
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    g = summary["generated"]
    print(
        f"teacher traces: {g['n_traces']} over {g['n_worlds']} worlds "
        f"(days={g['days']}, seed={g['random_seed']})"
    )
    print(f"  decisions : {summary['decision_histogram']}")
    print(f"  branches  : {summary['hub_branch_histogram']}")
    print(f"  warnings  : {summary['warning_histogram']}")
    print(f"  hash      : {summary['determinism_hash'][:16]}...")
    print(f"{'world':<17} {'n':>5}  {'recal':<6} {'sev(before/op)':<16} decisions")
    print("-" * 72)
    for w in summary["per_world"]:
        recal = "yes" if w["recalibrator_promoted"] else "-"
        sev = f"{w['calibration_severity_before']}/{w['calibration_severity_operational']}"
        print(f"{w['world']:<17} {w['n']:>5}  {recal:<6} {sev:<16} {w['decisions']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conditions", default=str(DEFAULT_CONDITIONS),
                        help="Stress-conditions YAML providing the worlds to trace.")
    parser.add_argument("--recal-config", default=str(DEFAULT_RECAL_CONFIG))
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG))
    parser.add_argument("--days", type=int, default=14,
                        help="Synthetic horizon per world (must cover fit_days + eval_days).")
    parser.add_argument("--max-loads", type=int, default=80,
                        help="Cap eval-window loads traced per world.")
    parser.add_argument("--worlds", default=None,
                        help="Comma-separated world names (default: a coverage subset).")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + short windows + a 2-world subset (smoke).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--traces-out", default=str(DEFAULT_TRACES_OUT))
    args = parser.parse_args()

    thresholds, _ = load_calibration_config(args.monitor_config)
    recal_config = load_recalibration_config(args.recal_config)
    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")

    all_conditions = load_conditions(Path(args.conditions))
    days = args.days
    max_loads = args.max_loads
    wanted = (
        {s.strip() for s in args.worlds.split(",") if s.strip()}
        if args.worlds else set(DEFAULT_WORLDS)
    )
    if args.fast:
        days = 6
        recal_config = replace(recal_config, fit_days=2, eval_days=3, min_samples=40)
        thresholds = replace(thresholds, min_samples=40)
        max_loads = max_loads or 60
        if not args.worlds:
            wanted = set(FAST_WORLDS)
    conditions = [c for c in all_conditions if c.name in wanted]
    if not conditions:
        raise SystemExit(f"no matching worlds in {args.conditions} for {sorted(wanted)}")

    t0 = time.perf_counter()
    traces = generate_traces(
        cfg, bid_cfg, thresholds, recal_config, conditions,
        days=days, max_loads_per_world=max_loads,
    )
    n = write_traces(traces, args.traces_out)
    summary = build_summary(traces, days=days, max_loads_per_world=max_loads)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    _print_summary(summary)
    print(f"\nwrote {n} traces -> {args.traces_out}")
    print(f"wrote summary -> {args.out}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()

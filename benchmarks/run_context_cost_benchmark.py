"""Cost / context benchmark (Phase 6.5) — the other half of the compiled-vs-orchestrated capstone.

The decision-quality half lives in ``run_compiled_dispatcher_stress.py``. This module measures the
*price* of a decision under each system, on one shared baseline world::

    source   — the full Phase 5.5 risk-aware engine (the Phase 6.1 teacher: real services at every
               workflow node). One decision = a ``BidQuery`` scored + a route + destination risk +
               win-probability + payment-risk + risk-adjusted EV + calibration/approval routing.
    compiled — the frozen Phase 6.3 multi-head model behind the Phase 6.4 port: one sklearn predict
               from a 22-feature case-fact vector, **zero** source-engine port calls.

For the paper-inspired thesis (*compile the procedure into weights for ~2 orders of magnitude less
cost*), the non-LLM proxies for "context / cost" are concrete and measurable:

* **runtime** per decision (source orchestration vs compiled predict) + speedup,
* **source-engine calls avoided** — the winnability + payment ports the source hits per decision and
  the compiled model does not call at all,
* **decision payload size** — the source's full structured decision row vs the compiled model's
  compact 6-key runtime JSON,
* **feature/context width** — the compiled model's fixed 22-field manifest,
* **artifact load time** — cold-load the frozen joblib once, then serve from memory,
* a **recompile / flexibility test** — change one workflow rule, regenerate the teacher traces,
  recompile the model, and time how long the *whole* loop takes plus whether the model relearns the
  new behavior (the paper's "edit the procedure and recompile" flexibility argument).

Token/$ columns stay ``null`` here on purpose: the committed compiled path is sklearn, so there are no
tokens. They light up only when an LLM adapter is plugged in behind the same port.

Writes ``benchmarks/context_cost_summary.json`` (committed, lean).

Examples
--------
    python -m benchmarks.run_context_cost_benchmark --fast
    python -m benchmarks.run_context_cost_benchmark --days 21 --max-loads 200
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from adapters.outbound.compiled_dispatcher.sklearn_compiled_dispatcher import (
    SklearnCompiledDispatcher,
)
from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from benchmarks.run_bid_recommender_eval import _build_frame
from benchmarks.run_broker_quality_stress import load_conditions, world_cfg
from benchmarks.run_calibration_monitor import _calibrated_model
from benchmarks.run_risk_aware_stress import (
    SEED_OFFSET,
    _eval_window_snapshots,
    _train_payment_adapter,
)
from ml.brokers import BrokerPoolParams, broker_index, build_broker_pool
from ml.calibration.recalibration_workflow import load_recalibration_config
from ml.config import MLConfig, load_ml_config
from ml.data.build_compiled_dispatcher_dataset import build_dataset
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.compiled_agent_trace_schema import AgentTrace, TraceMetadata
from ml.data.compiled_dispatcher_formatters import build_features, build_targets
from ml.data.outcome_schema import read_outcomes
from ml.models.compiled_dispatcher_model import (
    CompiledDispatcherModel,
    default_feature_manifest,
    feature_manifest_hash,
)
from ml.monitoring.calibration_report import load_calibration_config
from ml.training.winnability_dataset import resolve_path
from ml.workflows.freightbid_workflow_graph import (
    WORKFLOW_GRAPH_VERSION,
    DecisionHubPolicy,
    build_default_graph,
)
from ml.workflows.teacher_trace_generator import (
    SOURCE_POLICY_VERSION,
    TEACHER_TRACE_SCHEMA_VERSION,
    _git_commit,
    _trace_one,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "compiled_dispatcher_stress.yaml"
DEFAULT_RECAL_CONFIG = ROOT / "config" / "recalibration.yaml"
DEFAULT_MONITOR_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_CONDITIONS = ROOT / "config" / "broker_quality_stress.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "context_cost_summary.json"

COMPILED_MODEL_SEED = 650  # matches run_compiled_dispatcher_stress
# The single rule we perturb in the recompile/flexibility test: tighten the payment-default
# escalation threshold so more loads route to human approval (the procedure literally changes).
RECOMPILE_PAYMENT_DEFAULT_WARN = 0.08


# --------------------------------------------------------------------------- #
# A transparent call-counting proxy around an engine port
# --------------------------------------------------------------------------- #
class _CountingPort:
    """Wrap a winnability/payment adapter and count invocations of one method (the engine call).

    Everything is delegated to the wrapped port; only ``method`` is tallied, so the count is exactly
    "how many times the source engine hit this port for a decision". The compiled model wraps nothing
    and calls neither port — its avoided-calls count is this tally.
    """

    def __init__(self, target: Any, method: str) -> None:
        self._target = target
        self._method = method
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if name == self._method and callable(attr):
            def _wrapped(*args: Any, **kwargs: Any) -> Any:
                self.calls += 1
                return attr(*args, **kwargs)

            return _wrapped
        return attr


# --------------------------------------------------------------------------- #
# Assemble one baseline world (mirrors the teacher's baseline block, no recalibration)
# --------------------------------------------------------------------------- #
class _BaselineWorld:
    def __init__(
        self,
        recommender: EVBidRecommender,
        world: MLConfig,
        bid_cfg: BidRecommenderConfig,
        pool_index: Dict[str, Any],
        snaps: Sequence[Any],
        reserve_by_key: Dict[Tuple[Any, Any], float],
        metadata: TraceMetadata,
        win_counter: _CountingPort,
        pay_counter: _CountingPort,
    ) -> None:
        self.recommender = recommender
        self.world = world
        self.bid_cfg = bid_cfg
        self.pool_index = pool_index
        self.snaps = list(snaps)
        self.reserve_by_key = reserve_by_key
        self.metadata = metadata
        self.win_counter = win_counter
        self.pay_counter = pay_counter


def _assemble_baseline_world(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    *,
    days: int,
    fit_days: int,
    eval_days: int,
    max_loads: Optional[int],
) -> _BaselineWorld:
    """Train base winnability + payment on the baseline draw, then build the operational eval world.

    The winnability + payment adapters are wrapped in counting proxies so the source engine's
    per-decision port calls can be measured. No recalibration (baseline does not drift), matching the
    cost benchmark's purpose: time the engine, not the monitor.
    """
    base_tmp = tempfile.mkdtemp(prefix="ctxcost_base_")
    base_world = world_cfg(cfg, Path(base_tmp))
    build_winnability_dataset(base_world, days=days)
    base_frame, base_snaps = _build_frame(base_world)
    base_model = _calibrated_model(base_frame, base_world.winnability.random_seed)
    base_adapter = ModelWinnabilityAdapter(base_model)
    base_outcomes = read_outcomes(resolve_path(base_world.outcomes.outcomes_path))
    payment_adapter = _train_payment_adapter(base_world, base_snaps, base_outcomes)

    op_seed = cfg.synthetic_data.seed + SEED_OFFSET
    op_tmp = tempfile.mkdtemp(prefix="ctxcost_op_")
    world = world_cfg(cfg, Path(op_tmp))
    build_winnability_dataset(world, days=days, seed=op_seed)
    _, snaps = _build_frame(world)
    outcomes = read_outcomes(resolve_path(world.outcomes.outcomes_path))
    pool_index = broker_index(build_broker_pool(BrokerPoolParams.from_config(world.brokers)))

    win_counter = _CountingPort(base_adapter, "win_probabilities")
    pay_counter = _CountingPort(payment_adapter, "estimate")
    bid_ra = replace(bid_cfg, risk_adjusted_ev_enabled=True)
    recommender = EVBidRecommender(win_counter, bid_ra, payment=pay_counter)

    metadata = TraceMetadata(
        source_policy_version=SOURCE_POLICY_VERSION,
        git_commit=_git_commit(),
        config_hash="context-cost-benchmark",
        model_artifact_ids={"winnability": "in-memory", "payment": "in-memory", "recalibrator": "none"},
        random_seed=op_seed,
        world_name="baseline",
        workflow_graph_version=WORKFLOW_GRAPH_VERSION,
        teacher_trace_schema_version=TEACHER_TRACE_SCHEMA_VERSION,
    )

    reserve_by_key = {(o.load_id, o.snapshot_time): o.reservation_rpm for o in outcomes}
    eval_snaps = _eval_window_snapshots(snaps, fit_days, eval_days)
    eval_snaps = sorted(eval_snaps, key=lambda s: (s.snapshot_time, s.load_id))
    eval_snaps = [s for s in eval_snaps if (s.load_id, s.snapshot_time) in reserve_by_key]
    if max_loads is not None:
        eval_snaps = eval_snaps[:max_loads]

    return _BaselineWorld(
        recommender, world, bid_cfg, pool_index, eval_snaps, reserve_by_key,
        metadata, win_counter, pay_counter,
    )


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def _pct(values: Sequence[float], q: float) -> Optional[float]:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
    return round(vals[idx], 4)


def _latency_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    vals = [v for v in values if v is not None]
    return {
        "mean_ms": round(sum(vals) / len(vals), 4) if vals else None,
        "p50_ms": _pct(vals, 0.50),
        "p95_ms": _pct(vals, 0.95),
        "n": len(vals),
    }


def _trace_world(
    world: _BaselineWorld,
    policy: DecisionHubPolicy,
    graph: Any,
) -> Tuple[List[AgentTrace], List[float], List[int], List[int]]:
    """Run the full source engine once per load; return traces + per-decision latency + port calls."""
    traces: List[AgentTrace] = []
    latencies: List[float] = []
    win_calls: List[int] = []
    pay_calls: List[int] = []
    for s in world.snaps:
        reserve = world.reserve_by_key[(s.load_id, s.snapshot_time)]
        world.win_counter.reset()
        world.pay_counter.reset()
        t0 = time.perf_counter()
        trace = _trace_one(
            graph, policy, world.recommender, s, world.world, world.bid_cfg,
            world.pool_index, reserve, "OK", "OK", "OK", False, world.metadata,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        win_calls.append(world.win_counter.calls)
        pay_calls.append(world.pay_counter.calls)
        traces.append(trace)
    return traces, latencies, win_calls, pay_calls


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def run_context_cost_benchmark(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    *,
    days: int,
    fit_days: int,
    eval_days: int,
    max_loads: Optional[int],
) -> Dict[str, Any]:
    graph = build_default_graph()
    base_policy = DecisionHubPolicy()

    print(f"Assembling one baseline world (days={days}, eval window={eval_days}d, "
          f"cap={max_loads})...")
    t0 = time.perf_counter()
    world = _assemble_baseline_world(
        cfg, bid_cfg, days=days, fit_days=fit_days, eval_days=eval_days, max_loads=max_loads,
    )
    assemble_s = time.perf_counter() - t0
    print(f"  {len(world.snaps)} priced eval loads in {assemble_s:.1f}s")

    # ---- SOURCE: full engine, one decision per load -----------------------
    print("Timing the SOURCE engine (full workflow per decision)...")
    traces, src_lat, win_calls, pay_calls = _trace_world(world, base_policy, graph)
    n = len(traces)
    if n == 0:
        raise SystemExit("no priced loads in the eval window; widen --days or --max-loads")

    # ---- COMPILE: train once on the source traces, freeze ------------------
    print("Compiling the dispatcher (train multi-head model on the source traces)...")
    rows = build_dataset(traces).rows
    t0 = time.perf_counter()
    model = CompiledDispatcherModel(random_state=COMPILED_MODEL_SEED).fit(rows)
    compile_s = time.perf_counter() - t0
    expected = feature_manifest_hash(default_feature_manifest())
    port = SklearnCompiledDispatcher(model, expected_manifest_hash=expected)

    # ---- COMPILED: one predict per load -----------------------------------
    print("Timing the COMPILED model (one predict per decision)...")
    cmp_lat: List[float] = []
    src_payload_bytes: List[int] = []
    cmp_payload_bytes: List[int] = []
    feature_widths: List[int] = []
    for trace in traces:
        features = build_features(trace)
        t0 = time.perf_counter()
        pred = port.predict(features)
        cmp_lat.append((time.perf_counter() - t0) * 1000.0)
        # Decision payload: source's full structured decision row vs the compiled runtime JSON.
        src_payload_bytes.append(len(json.dumps(build_targets(trace), default=str)))
        cmp_payload_bytes.append(len(json.dumps(pred.to_runtime_json(), default=str)))
        feature_widths.append(sum(1 for v in features.values() if v is not None))

    # ---- artifact cold-load timing ----------------------------------------
    art_tmp = Path(tempfile.mkdtemp(prefix="ctxcost_art_")) / "compiled_dispatcher_model.joblib"
    t0 = time.perf_counter()
    model.save(art_tmp)
    save_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    reloaded = CompiledDispatcherModel.load(art_tmp)
    load_s = time.perf_counter() - t0
    artifact_bytes = art_tmp.stat().st_size
    assert reloaded.feature_manifest_hash == model.feature_manifest_hash

    # ---- recompile / flexibility test -------------------------------------
    print(f"Recompile test: tighten payment_default_warn -> {RECOMPILE_PAYMENT_DEFAULT_WARN} "
          f"and recompile...")
    new_policy = DecisionHubPolicy(payment_default_warn=RECOMPILE_PAYMENT_DEFAULT_WARN)
    t0 = time.perf_counter()
    new_traces, _, _, _ = _trace_world(world, new_policy, graph)
    regen_s = time.perf_counter() - t0
    new_rows = build_dataset(new_traces).rows
    t0 = time.perf_counter()
    new_model = CompiledDispatcherModel(random_state=COMPILED_MODEL_SEED).fit(new_rows)
    recompile_fit_s = time.perf_counter() - t0
    new_port = SklearnCompiledDispatcher(new_model, expected_manifest_hash=expected)

    before_dist = Counter(build_targets(t)["decision"] for t in traces)
    after_dist = Counter(build_targets(t)["decision"] for t in new_traces)
    # Did the recompiled model relearn the changed rule? (in-sample agreement w/ the new teacher)
    relearn = sum(
        1 for t in new_traces
        if new_port.predict(build_features(t)).decision == build_targets(t)["decision"]
    )

    src_calls_total = [w + p for w, p in zip(win_calls, pay_calls)]
    mean_src_calls = sum(src_calls_total) / n
    src_stats = _latency_stats(src_lat)
    cmp_stats = _latency_stats(cmp_lat)
    speedup = (
        round(src_stats["mean_ms"] / cmp_stats["mean_ms"], 1)
        if src_stats["mean_ms"] and cmp_stats["mean_ms"] else None
    )
    mean_src_payload = sum(src_payload_bytes) / n
    mean_cmp_payload = sum(cmp_payload_bytes) / n

    return {
        "config": {
            "fast": days <= 8,
            "days": days,
            "eval_days": eval_days,
            "max_loads": max_loads,
            "sample_decisions": n,
            "world": "baseline",
            "eval_seed_offset": SEED_OFFSET,
            "compiled_model_seed": COMPILED_MODEL_SEED,
            "feature_manifest_hash": model.feature_manifest_hash,
            "feature_count": len(model.feature_manifest),
            "workflow_graph_version": WORKFLOW_GRAPH_VERSION,
            "compiled_used_for_decision": False,
            "shadow_only": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latency_per_decision": {
            "source": src_stats,
            "compiled": cmp_stats,
            "speedup_x": speedup,
            "note": (
                "Wall-clock is ~parity for the in-memory classical engine (both sides are "
                "dominated by Python/pandas per-call overhead). The runtime/$ win scales with how "
                "expensive each replaced node is - it materializes when the source nodes are "
                "frontier-model or remote-service calls (the paper's regime), which the "
                "calls-avoided metric captures directly."
            ),
        },
        "engine_calls_per_decision": {
            "winnability_calls_source": round(sum(win_calls) / n, 3),
            "payment_calls_source": round(sum(pay_calls) / n, 3),
            "total_source": round(mean_src_calls, 3),
            "total_compiled": 0,
            "calls_avoided_per_decision": round(mean_src_calls, 3),
        },
        "decision_payload_bytes": {
            "source_decision_row_mean": round(mean_src_payload, 1),
            "compiled_runtime_json_mean": round(mean_cmp_payload, 1),
            "reduction_x": (
                round(mean_src_payload / mean_cmp_payload, 1) if mean_cmp_payload else None
            ),
        },
        "context_width": {
            "compiled_feature_manifest_fields": len(model.feature_manifest),
            "compiled_features_populated_mean": round(sum(feature_widths) / n, 1),
            "source_workflow_nodes": len(graph.nodes),
        },
        "tokens_per_decision": {
            "source": None,
            "compiled": None,
            "note": "null for the sklearn path; populated only behind an LLM adapter",
        },
        "artifact": {
            "joblib_bytes": artifact_bytes,
            "save_seconds": round(save_s, 4),
            "cold_load_seconds": round(load_s, 4),
        },
        "recompile_test": {
            "rule_changed": f"DecisionHubPolicy.payment_default_warn 0.15 -> {RECOMPILE_PAYMENT_DEFAULT_WARN}",
            "regenerate_traces_seconds": round(regen_s, 2),
            "recompile_fit_seconds": round(recompile_fit_s, 2),
            "time_to_recompile_seconds": round(regen_s + recompile_fit_s, 2),
            "source_action_distribution_before": dict(before_dist),
            "source_action_distribution_after": dict(after_dist),
            "post_change_action_agreement": round(relearn / n, 4),
        },
        "timing": {
            "assemble_seconds": round(assemble_s, 1),
            "initial_compile_seconds": round(compile_s, 2),
        },
        "headline": (
            f"Compiling the dispatcher into weights replaces ~{mean_src_calls:.1f} source-engine "
            f"port calls per decision with one in-memory predict, shrinks the decision payload "
            f"~{round(mean_src_payload / mean_cmp_payload, 1)}x "
            f"({mean_src_payload:.0f}B -> {mean_cmp_payload:.0f}B), and needs only a "
            f"{len(model.feature_manifest)}-field context vs the {len(graph.nodes)}-node workflow; a "
            f"full rule-change recompile takes ~{regen_s + recompile_fit_s:.0f}s. Wall-clock latency "
            f"is ~parity for the in-memory engine - the runtime/$ win scales with how expensive each "
            f"replaced node is. Compiled stays shadow-only."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recal-config", default=str(DEFAULT_RECAL_CONFIG))
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG))
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--max-loads", type=int, default=200)
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded build + short window + capped loads (smoke).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    recal_config = load_recalibration_config(args.recal_config)
    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")

    days = args.days
    fit_days, eval_days = recal_config.fit_days, recal_config.eval_days
    max_loads = args.max_loads
    if args.fast:
        days = 6
        fit_days, eval_days = 2, 3
        max_loads = min(max_loads or 80, 80)

    print("=" * 92)
    print(f"Context/cost benchmark (Phase 6.5): source engine vs compiled dispatcher "
          f"on one baseline world ({days}d)")
    print("=" * 92)

    start = time.time()
    summary = run_context_cost_benchmark(
        cfg, bid_cfg, days=days, fit_days=fit_days, eval_days=eval_days, max_loads=max_loads,
    )
    summary["runtime_seconds"] = round(time.time() - start, 1)

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lat = summary["latency_per_decision"]
    calls = summary["engine_calls_per_decision"]
    rc = summary["recompile_test"]
    print("\n" + "-" * 92)
    print(f"  decisions sampled     : {summary['config']['sample_decisions']}")
    print(f"  source latency / dec  : {lat['source']['mean_ms']:.2f} ms "
          f"(p95 {lat['source']['p95_ms']:.2f})")
    print(f"  compiled latency / dec: {lat['compiled']['mean_ms']:.3f} ms "
          f"(p95 {lat['compiled']['p95_ms']:.3f})")
    print(f"  speedup               : {lat['speedup_x']}x")
    print(f"  source-engine calls   : {calls['total_source']} / decision avoided by compiled")
    print(f"  decision payload      : {summary['decision_payload_bytes']['source_decision_row_mean']:.0f}B "
          f"-> {summary['decision_payload_bytes']['compiled_runtime_json_mean']:.0f}B "
          f"({summary['decision_payload_bytes']['reduction_x']}x smaller)")
    print(f"  artifact cold-load    : {summary['artifact']['cold_load_seconds']*1000:.0f} ms "
          f"({summary['artifact']['joblib_bytes']/1024:.0f} KB)")
    print(f"  recompile (rule edit) : {rc['time_to_recompile_seconds']:.0f} s; "
          f"post-change agreement {rc['post_change_action_agreement']:.0%}")
    print("-" * 92)
    print("\n" + "=" * 92)
    print(summary["headline"])
    print("=" * 92)
    print(f"Wrote {out_path} ({summary['runtime_seconds'] / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

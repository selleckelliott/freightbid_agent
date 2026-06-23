"""Compiled-vs-orchestrated decision-quality stress test (Phase 6.5) — the Phase 6 capstone.

Answers the paper-inspired question for FreightBid:

    How much decision quality do we lose by serving the **compiled** dispatcher instead of
    the full **source** engine, across shifted broker-market worlds — and is the loss safe?

Three systems are scored on the **same** eval-window loads of each Phase 4.5/5.5 world:

1. ``source``   — the full Phase 5.5 risk-aware engine (the trusted authority). Its decision
   is produced by the Phase 6.1 teacher trace generator (the real engine at every node) and
   normalized into the shared prediction shape.
2. ``compiled`` — the frozen Phase 6.3 multi-head model, run **through the Phase 6.4 shadow
   service** (``shadow_only`` always True; fails closed; never decides).
3. ``baseline`` — the Phase 6.3 majority predictor, a trivial floor for context.

The compiled model is trained **once on the canonical multi-world teacher set** (the Phase
6.1/6.2/6.3 worlds — baseline plus the six broker-quality stress worlds), drawn from a *distinct*
operational seed, and then **frozen**; every world is distribution shift at inference, never
retraining (mirroring the Phase 5.5 base-model discipline). Even the in-distribution worlds are
scored out-of-sample because training and evaluation use different operational draws, and the three
eval-only worlds (``disappearing_loads`` / ``no_rate_heavy`` / ``sharp_win_curve``) are fully
out-of-distribution.

The headline metric is **collectible-profit regret** — the Phase 5.5 collectible oracle (true
broker latents + true win curve, evaluation-only) re-scored at each system's *own* chosen ask,
so following the compiled model's decisions instead of the source's has an honest dollar cost::

    arm collectible = 0                              if the arm does not bid
                    = score_ask(arm ask | oracle)    otherwise   (Phase 5.5 _score_ask)
    regret%         = (sum source - sum compiled) / |sum source| * 100

When the compiled model cannot serve (fail-closed: manifest mismatch / invalid output /
exception), the shadow service returns a fallback and the runtime defers to the source engine —
so a fallback row carries **zero regret** by construction (safe degradation) and is tallied
separately. The verdict per world (config-driven thresholds, defaults 2% / 5%)::

    PASS   regret <= regret_pass_pct  AND no safety-critical miss
    WATCH  regret <= regret_watch_pct OR a minor (non-critical) miss
    FAIL   regret >  regret_watch_pct OR a safety-critical miss

A **safety-critical** miss is the compiled model overriding caution where it matters most:
source ``no_bid`` but compiled bids; source ``approval_required`` but compiled ``clean_bid``;
or, in a risky world, compiled **suppressing** a source payment/calibration warning.

Writes ``benchmarks/compiled_dispatcher_stress_summary.json`` (committed, lean) for the chart
and the README.

Examples
--------
    # quick smoke (tiny seeded builds, 2 worlds, short windows, capped loads)
    python -m benchmarks.run_compiled_dispatcher_stress --fast

    # canonical sweep
    python -m benchmarks.run_compiled_dispatcher_stress --days 21 \
        --out benchmarks/compiled_dispatcher_stress_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from adapters.outbound.compiled_dispatcher.sklearn_compiled_dispatcher import (
    SklearnCompiledDispatcher,
)
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.services.shadow_compiled_dispatcher_service import (
    ShadowCompiledDispatcherService,
)
from benchmarks.run_broker_quality_stress import Condition, load_conditions, world_cfg
from benchmarks.run_risk_aware_stress import SEED_OFFSET, _score_ask
from ml.calibration.recalibration_workflow import load_recalibration_config
from ml.config import MLConfig, load_ml_config
from ml.data.build_compiled_dispatcher_dataset import build_dataset
from ml.data.compiled_agent_trace_schema import (
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
    AgentTrace,
)
from ml.data.compiled_dispatcher_formatters import build_features, build_targets
from ml.models.baseline_compiled_dispatcher import MajorityCompiledDispatcherBaseline
from ml.models.compiled_dispatcher_model import (
    CompiledDispatcherModel,
    default_feature_manifest,
    feature_manifest_hash,
)
from ml.monitoring.calibration_report import load_calibration_config
from ml.training.winnability_dataset import resolve_path
from ml.workflows.freightbid_workflow_graph import (
    WARN_CALIBRATION_ALERT,
    WARN_PAYMENT_RISK,
)
from ml.workflows.teacher_trace_generator import DEFAULT_WORLDS, FAST_WORLDS, generate_traces
from ports.compiled_dispatcher import (
    REASON_INVALID_OUTPUT,
    REASON_MANIFEST_MISMATCH,
    source_prediction_from_targets,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "compiled_dispatcher_stress.yaml"
DEFAULT_RECAL_CONFIG = ROOT / "config" / "recalibration.yaml"
DEFAULT_MONITOR_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "compiled_dispatcher_stress_summary.json"

# Systems compared (source is the authority; compiled/baseline are shadow-only).
SOURCE = "source"
COMPILED = "compiled"
BASELINE = "baseline"

# Verdicts (compiled vs source, on collectible-profit regret + safety).
VERDICT_PASS = "PASS"
VERDICT_WATCH = "WATCH"
VERDICT_FAIL = "FAIL"

# Train the compiled model on a baseline draw the eval never sees (out-of-sample even on
# baseline) — a distinct operational offset from the Phase 5.5 eval draw (SEED_OFFSET).
TRAIN_SEED_OFFSET = SEED_OFFSET + 1000
# Deterministic compiled-model seed (independent of the world seeds).
COMPILED_MODEL_SEED = 6_5_0
# Warnings whose suppression in a risky world is safety-critical. The soft `calibration_watch`
# tier is deliberately excluded: it rides the *clean-bid* branch (a minor signal), and the
# compiled model has no watch head (WARNING_HEADS models only payment / calibration_ALERT /
# no-feasible-bid). Counting an un-modelable watch warning as a safety-critical miss would make
# the metric a structural constant on watch-heavy worlds rather than a measure of real caution
# overrides; suppressing it is caught instead as a *minor* miss (warning_agreement < 1.0).
CRITICAL_WARNINGS = frozenset({WARN_PAYMENT_RISK, WARN_CALIBRATION_ALERT})
# Both of these place a bid on the board (approval_required still bids, pending a human).
_BIDDING = frozenset({DECISION_BID, DECISION_APPROVAL_REQUIRED})


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _scale_for(cond: Condition, cfg: MLConfig) -> float:
    """The world's win-curve logistic scale (override or baseline) — for the collectible oracle."""
    return float(cond.overrides.get("outcomes", {}).get(
        "win_logistic_scale_rpm", cfg.outcomes.win_logistic_scale_rpm
    ))


def _collectible(
    decision: Optional[str],
    ask_rpm: Optional[float],
    ask_amount: Optional[float],
    *,
    cost: float,
    reserve: float,
    scale: float,
    p_default: float,
    pay_days: float,
    cash_rate: float,
    free_days: float,
) -> float:
    """Realized collectible profit of one decision: 0 if it does not bid, else the oracle value."""
    if decision not in _BIDDING or ask_rpm is None or ask_amount is None:
        return 0.0
    return float(_score_ask(
        float(ask_rpm), float(ask_amount), cost, reserve, scale,
        p_default, pay_days, cash_rate, free_days,
    )["collectible"])


def _train_frozen_models(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    thresholds,
    recal_config,
    train_conditions: Sequence[Condition],
    *,
    days: int,
    max_loads: Optional[int],
) -> Tuple[CompiledDispatcherModel, MajorityCompiledDispatcherBaseline, Dict[str, Any]]:
    """Trace the canonical training worlds on the train draw, then fit + freeze the compiled model + baseline."""
    train_traces = generate_traces(
        cfg, bid_cfg, thresholds, recal_config, list(train_conditions),
        days=days, max_loads_per_world=max_loads, seed_offset=TRAIN_SEED_OFFSET,
    )
    rows = build_dataset(train_traces).rows
    model = CompiledDispatcherModel(random_state=COMPILED_MODEL_SEED).fit(rows)
    baseline = MajorityCompiledDispatcherBaseline().fit(rows)
    prov = {
        "train_worlds": [c.name for c in train_conditions],
        "train_seed_offset": TRAIN_SEED_OFFSET,
        "train_rows": len(rows),
        "feature_manifest_hash": model.feature_manifest_hash,
        "feature_count": len(model.feature_manifest),
        "compiled_model_seed": COMPILED_MODEL_SEED,
    }
    return model, baseline, prov


def _mean(values: Sequence[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _safety_critical(
    src_action: str,
    comp_action: Optional[str],
    src_warnings: set,
    comp_warnings: set,
    risky: bool,
) -> bool:
    """The compiled model overriding caution where it matters most (compiled must have served).

    Three cases: (a) the source rejected the load (``no_bid``) but compiled places a bid; (b) the
    source wanted human approval but compiled would auto-``bid`` it clean; (c) in a risky world,
    compiled suppresses a source payment/calibration warning. Any one ⇒ the world FAILs.
    """
    take_rejected = src_action == DECISION_NO_BID and comp_action in _BIDDING
    skip_approval = src_action == DECISION_APPROVAL_REQUIRED and comp_action == DECISION_BID
    drop_warning = risky and bool((src_warnings - comp_warnings) & CRITICAL_WARNINGS)
    return bool(take_rejected or skip_approval or drop_warning)


# --------------------------------------------------------------------------- #
# Per-load scoring
# --------------------------------------------------------------------------- #
def _score_trace(
    trace: AgentTrace,
    compiled_service: ShadowCompiledDispatcherService,
    baseline_service: ShadowCompiledDispatcherService,
    *,
    scale: float,
    bid_cfg: BidRecommenderConfig,
    risky: bool,
) -> Dict[str, Any]:
    """Compare source vs compiled vs baseline on one load and return a flat metrics row.

    Agreement + safety are computed for every load (a compiled bid on an *infeasible* load the
    source rejected is the most important miss to catch). Collectible profit is only priced when
    the world handed us a realized reserve for this load (``priceable``).
    """
    targets = build_targets(trace)
    features = build_features(trace)
    source = source_prediction_from_targets(targets)

    compiled_cmp = compiled_service.compare(source, features)
    baseline_cmp = baseline_service.compare(source, features)

    src_action = source.decision
    src_warnings = set(source.warnings or [])
    available = compiled_cmp.compiled_available
    comp_action = compiled_cmp.compiled_action
    comp_warnings = set(compiled_cmp.compiled_warnings or [])

    # --- safety classification (only meaningful when the compiled model served) ----------
    crit = minor = False
    no_bid_fn = approval_fn = False
    if available:
        crit = _safety_critical(src_action, comp_action, src_warnings, comp_warnings, risky)
        if not crit:
            disagree = (
                not compiled_cmp.action_agrees
                or not compiled_cmp.approval_agrees
                or (compiled_cmp.warning_agreement or 0.0) < 1.0
            )
            minor = bool(disagree)
        no_bid_fn = src_action == DECISION_NO_BID and comp_action != DECISION_NO_BID
        approval_fn = (
            src_action == DECISION_APPROVAL_REQUIRED and comp_action != DECISION_APPROVAL_REQUIRED
        )

    # --- collectible profit (priced only when a realized reserve exists) ------------------
    reserve = trace.eval_labels.reservation_rpm
    priceable = reserve is not None
    src_coll = comp_coll = base_coll = None
    if priceable:
        miles = max(float(features.get("loaded_miles") or 1.0), 1.0)
        ctx = dict(
            cost=float(trace.node_outputs.estimated_cost),
            reserve=float(reserve),
            scale=scale,
            p_default=float(trace.eval_labels.true_default_prob),
            pay_days=float(trace.eval_labels.true_pay_days),
            cash_rate=bid_cfg.annual_cash_cost_rate,
            free_days=bid_cfg.free_pay_days,
        )
        src_coll = _collectible(
            src_action, source.recommended_bid_rpm, source.recommended_bid, **ctx
        )
        if available:
            c_amount = compiled_cmp.compiled_bid
            c_rpm = (c_amount / miles) if c_amount is not None else None
            comp_coll = _collectible(comp_action, c_rpm, c_amount, **ctx)
        else:
            comp_coll = src_coll  # fail-closed: defer to the source engine -> zero regret
        b_amount = baseline_cmp.compiled_bid if baseline_cmp.compiled_available else None
        b_rpm = (b_amount / miles) if b_amount is not None else None
        base_coll = (
            _collectible(baseline_cmp.compiled_action, b_rpm, b_amount, **ctx)
            if baseline_cmp.compiled_available else src_coll
        )

    return {
        "world": trace.metadata.world_name,
        "source_action": src_action,
        "compiled_action": comp_action,
        "compiled_available": available,
        "shadow_only": compiled_cmp.shadow_only,
        "fallback_reason": compiled_cmp.fallback_reason,
        "invalid": compiled_cmp.fallback_reason == REASON_INVALID_OUTPUT,
        "action_agrees": compiled_cmp.action_agrees,
        "approval_agrees": compiled_cmp.approval_agrees,
        "warning_agreement": compiled_cmp.warning_agreement,
        "bid_delta": compiled_cmp.bid_delta,
        "ev_delta": compiled_cmp.ev_delta,
        "latency_ms": compiled_cmp.compiled_latency_ms,
        "crit": crit,
        "minor": minor,
        "src_no_bid": available and src_action == DECISION_NO_BID,
        "src_approval": available and src_action == DECISION_APPROVAL_REQUIRED,
        "no_bid_fn": no_bid_fn,
        "approval_fn": approval_fn,
        "priceable": priceable,
        "src_coll": src_coll,
        "comp_coll": comp_coll,
        "base_coll": base_coll,
    }


def _regret_pct(sum_source: float, sum_arm: float) -> float:
    """Signed regret of an arm vs source as a percent of |source| (positive ⇒ arm is worse)."""
    if abs(sum_source) < 1.0:  # collectible totals below $1 -> ratio is noise, call it flat
        return 0.0
    return round((sum_source - sum_arm) / abs(sum_source) * 100.0, 2)


def _verdict(regret_pct: float, crit: int, minor: int, pass_pct: float, watch_pct: float) -> str:
    if regret_pct > watch_pct or crit > 0:
        return VERDICT_FAIL
    if regret_pct > pass_pct or minor > 0:
        return VERDICT_WATCH
    return VERDICT_PASS


def _world_record(
    cond: Condition,
    rows: List[Dict[str, Any]],
    *,
    pass_pct: float,
    watch_pct: float,
) -> Dict[str, Any]:
    """Aggregate one world's per-load rows into agreement, regret, safety, and a verdict."""
    served = [r for r in rows if r["compiled_available"]]
    priced = [r for r in rows if r["priceable"]]
    n_total = len(rows)
    n_served = len(served)

    sum_src = sum(r["src_coll"] for r in priced)
    sum_comp = sum(r["comp_coll"] for r in priced)
    sum_base = sum(r["base_coll"] for r in priced)
    regret_pct = _regret_pct(sum_src, sum_comp)
    base_regret_pct = _regret_pct(sum_src, sum_base)

    crit = sum(1 for r in served if r["crit"])
    minor = sum(1 for r in served if r["minor"])
    n_src_no_bid = sum(1 for r in served if r["src_no_bid"])
    n_src_approval = sum(1 for r in served if r["src_approval"])

    def rate(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den else None

    return {
        "name": cond.name,
        "lens": cond.lens(),
        "n_loads": n_total,
        "n_served": n_served,
        "n_priced": len(priced),
        # decision quality (compiled vs source, over served loads)
        "action_agreement": round(_mean([r["action_agrees"] for r in served]) or 0.0, 4),
        "approval_agreement": round(_mean([r["approval_agrees"] for r in served]) or 0.0, 4),
        "warning_agreement": round(_mean([r["warning_agreement"] for r in served]) or 0.0, 4),
        "mean_bid_delta": round(_mean([r["bid_delta"] for r in served]) or 0.0, 2),
        "mean_ev_delta": round(_mean([r["ev_delta"] for r in served]) or 0.0, 2),
        # collectible profit + regret
        "collectible_source": round(sum_src, 2),
        "collectible_compiled": round(sum_comp, 2),
        "collectible_baseline": round(sum_base, 2),
        "regret_pct": regret_pct,
        "baseline_regret_pct": base_regret_pct,
        # safety
        "safety_critical_misses": crit,
        "safety_critical_rate": rate(crit, n_served),
        "minor_misses": minor,
        "no_bid_false_negative_rate": rate(sum(1 for r in served if r["no_bid_fn"]), n_src_no_bid),
        "approval_false_negative_rate": rate(
            sum(1 for r in served if r["approval_fn"]), n_src_approval
        ),
        "fallback_rate": rate(n_total - n_served, n_total),
        "invalid_output_rate": rate(sum(1 for r in rows if r["invalid"]), n_total),
        "mean_compiled_latency_ms": round(_mean([r["latency_ms"] for r in served]) or 0.0, 4),
        # the compiled model never owns the decision (Phase 6.4 invariant)
        "compiled_used_for_decision": False,
        "verdict": _verdict(regret_pct, crit, minor, pass_pct, watch_pct),
    }


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
def run_compiled_dispatcher_stress(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    thresholds,
    recal_config,
    conditions: List[Condition],
    *,
    train_conditions: Sequence[Condition],
    days: int,
    max_loads: Optional[int],
    risky_worlds: Sequence[str],
    pass_pct: float,
    watch_pct: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Train the compiled model once on the canonical worlds, then score it beside the source on every world."""
    conditions = sorted(conditions, key=lambda c: (bool(c.overrides), c.name))
    train_conditions = sorted(train_conditions, key=lambda c: (bool(c.overrides), c.name))
    train_names = ", ".join(c.name for c in train_conditions)
    risky = set(risky_worlds)

    print(f"Training compiled dispatcher once on {len(train_conditions)} canonical world(s) "
          f"[{train_names}] (train draw)...")
    t0 = time.perf_counter()
    model, baseline_model, prov = _train_frozen_models(
        cfg, bid_cfg, thresholds, recal_config, train_conditions, days=days, max_loads=max_loads,
    )
    prov["train_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"  trained on {prov['train_rows']} rows in {prov['train_seconds']}s "
          f"(manifest {prov['feature_manifest_hash'][:12]}, {prov['feature_count']} features)")

    expected = feature_manifest_hash(default_feature_manifest())
    compiled_service = ShadowCompiledDispatcherService(
        SklearnCompiledDispatcher(model, expected_manifest_hash=expected)
    )
    baseline_service = ShadowCompiledDispatcherService(
        SklearnCompiledDispatcher(baseline_model)
    )

    print(f"Scoring {len(conditions)} worlds beside the source engine (eval draw)...")
    t0 = time.perf_counter()
    eval_traces = generate_traces(
        cfg, bid_cfg, thresholds, recal_config, conditions,
        days=days, max_loads_per_world=max_loads, seed_offset=SEED_OFFSET,
    )
    prov["eval_source_seconds"] = round(time.perf_counter() - t0, 1)
    prov["eval_seed_offset"] = SEED_OFFSET

    by_world: Dict[str, List[Dict[str, Any]]] = {c.name: [] for c in conditions}
    scale_by_world = {c.name: _scale_for(c, cfg) for c in conditions}
    t0 = time.perf_counter()
    for trace in eval_traces:
        name = trace.metadata.world_name
        by_world.setdefault(name, []).append(_score_trace(
            trace, compiled_service, baseline_service,
            scale=scale_by_world.get(name, cfg.outcomes.win_logistic_scale_rpm),
            bid_cfg=bid_cfg, risky=name in risky,
        ))
    prov["compiled_score_seconds"] = round(time.perf_counter() - t0, 3)

    records: List[Dict[str, Any]] = []
    for i, cond in enumerate(conditions, 1):
        rec = _world_record(cond, by_world.get(cond.name, []), pass_pct=pass_pct, watch_pct=watch_pct)
        records.append(rec)
        _print_line(i, len(conditions), rec)
    return records, prov


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
_VERDICT_TAG = {VERDICT_PASS: "PASS ", VERDICT_WATCH: "WATCH", VERDICT_FAIL: "FAIL "}


def _print_line(i: int, n: int, r: Dict[str, Any]) -> None:
    crit = r["safety_critical_misses"]
    print(
        f"[{i:2d}/{n}] {_VERDICT_TAG.get(r['verdict'], r['verdict'])} {r['name']:<19} "
        f"regret {r['regret_pct']:+6.2f}%  act {r['action_agreement']:.2f}  "
        f"appr {r['approval_agreement']:.2f}  warn {r['warning_agreement']:.2f}  "
        f"crit {crit}  minor {r['minor_misses']}  fb {r['fallback_rate'] or 0.0:.2f}"
    )


def _tally(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tally = {VERDICT_PASS: 0, VERDICT_WATCH: 0, VERDICT_FAIL: 0}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return tally


def _format_table(records: List[Dict[str, Any]]) -> str:
    header = (
        f"{'world':<19} {'regret%':>8} {'base%':>8} {'act':>5} {'appr':>5} {'warn':>5} "
        f"{'crit':>4} {'minor':>5} {'fb':>5} {'verdict':<6}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        lines.append(
            f"{r['name']:<19} {r['regret_pct']:>+7.2f}% {r['baseline_regret_pct']:>+7.2f}% "
            f"{r['action_agreement']:>5.2f} {r['approval_agreement']:>5.2f} "
            f"{r['warning_agreement']:>5.2f} {r['safety_critical_misses']:>4d} "
            f"{r['minor_misses']:>5d} {(r['fallback_rate'] or 0.0):>5.2f} {r['verdict']:<6}"
        )
    return "\n".join(lines)


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    tally = _tally(records)
    crit_worlds = [r["name"] for r in records if r["safety_critical_misses"] > 0]
    worst = max(records, key=lambda r: r["regret_pct"]) if records else None
    mean_action = _mean([r["action_agreement"] for r in records]) or 0.0
    return {
        "tally": tally,
        "safety_critical_worlds": crit_worlds,
        "worst_regret_world": (worst["name"] if worst else None),
        "worst_regret_pct": (worst["regret_pct"] if worst else None),
        "mean_action_agreement": round(mean_action, 4),
        "headline": (
            f"Compiled dispatcher holds within budget on {tally[VERDICT_PASS]}/{len(records)} "
            f"worlds (watch {tally[VERDICT_WATCH]}, fail {tally[VERDICT_FAIL]}); "
            f"mean action agreement {mean_action:.0%}; "
            f"{len(crit_worlds)} world(s) with a safety-critical miss. "
            f"The source engine remains authoritative - compiled is shadow-only."
        ),
    }


def _load_sweep_config(path: Path) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return doc.get("compiled_dispatcher_stress", doc) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--recal-config", default=str(DEFAULT_RECAL_CONFIG))
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG))
    parser.add_argument("--conditions", default=None,
                        help="Override the stress-conditions YAML (default: the sweep config's).")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--max-loads", type=int, default=None)
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + short windows + a 2-world subset (smoke).")
    parser.add_argument("--only", default=None, help="Comma-separated condition names to run.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    sweep = _load_sweep_config(Path(args.config))
    thresholds, _ = load_calibration_config(args.monitor_config)
    recal_config = load_recalibration_config(args.recal_config)
    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")

    days = args.days if args.days is not None else int(sweep.get("days", 21))
    max_loads = args.max_loads if args.max_loads is not None else sweep.get("max_loads")
    pass_pct = float(sweep.get("regret_pass_pct", 2.0))
    watch_pct = float(sweep.get("regret_watch_pct", 5.0))
    risky_worlds = list(sweep.get("risky_worlds", []))

    conditions_path = Path(
        args.conditions or sweep.get("conditions", "config/broker_quality_stress.yaml")
    )
    if not conditions_path.is_absolute():
        conditions_path = ROOT / conditions_path
    all_conditions = load_conditions(conditions_path)
    conditions = list(all_conditions)

    # The compiled model is trained once on the canonical multi-world teacher set
    # (the Phase 6.1/6.2/6.3 worlds) regardless of which eval subset is requested.
    train_world_names = set(FAST_WORLDS if args.fast else DEFAULT_WORLDS)
    train_conditions = [c for c in all_conditions if c.name in train_world_names]
    if not any(c.name == "baseline" for c in train_conditions):
        train_conditions = (
            [c for c in all_conditions if c.name == "baseline"] + train_conditions
        )
    if not train_conditions:
        raise SystemExit("no canonical training worlds found in the conditions file")

    if args.fast:
        days = int(sweep.get("fast_days", 6))
        recal_config = replace(recal_config, fit_days=2, eval_days=3, min_samples=40)
        thresholds = replace(thresholds, min_samples=40)
        max_loads = max_loads or int(sweep.get("fast_max_loads", 120))
        if not args.only:
            fast = set(sweep.get("fast_conditions", ["baseline", "tight_brokers"]))
            conditions = [c for c in conditions if c.name in fast]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        conditions = [c for c in conditions if c.name in wanted]
        missing = wanted - {c.name for c in conditions}
        if missing:
            raise SystemExit(f"unknown condition(s): {sorted(missing)}")
    if not any(c.name == "baseline" for c in conditions):
        raise SystemExit("the 'baseline' condition is required (the compiled model trains on it)")
    if days < recal_config.fit_days + recal_config.eval_days:
        raise SystemExit(
            f"--days {days} is shorter than fit_days+eval_days "
            f"({recal_config.fit_days}+{recal_config.eval_days}); widen --days or the windows."
        )

    print("=" * 92)
    print(f"Compiled-vs-orchestrated stress (Phase 6.5): {len(conditions)} worlds x "
          f"3 systems on collectible-profit regret (compiled trained once on "
          f"{len(train_conditions)} canonical worlds, frozen; {days}d worlds)")
    print("=" * 92)

    start = time.time()
    records, prov = run_compiled_dispatcher_stress(
        cfg, bid_cfg, thresholds, recal_config, conditions,
        train_conditions=train_conditions,
        days=days, max_loads=max_loads, risky_worlds=risky_worlds,
        pass_pct=pass_pct, watch_pct=watch_pct,
    )
    elapsed = time.time() - start
    stats = _summarize(records)

    summary: Dict[str, Any] = {
        "config": {
            "fast": args.fast,
            "days": days,
            "max_loads": max_loads,
            "regret_pass_pct": pass_pct,
            "regret_watch_pct": watch_pct,
            "risky_worlds": risky_worlds,
            "condition_count": len(conditions),
            "trained_on": prov["train_worlds"],
            "train_seed_offset": prov["train_seed_offset"],
            "eval_seed_offset": prov["eval_seed_offset"],
            "train_rows": prov["train_rows"],
            "feature_manifest_hash": prov["feature_manifest_hash"],
            "feature_count": prov["feature_count"],
            "compiled_model_seed": prov["compiled_model_seed"],
            "shadow_only": True,
            "compiled_used_for_decision": False,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "timing": {
            "train_seconds": prov["train_seconds"],
            "eval_source_seconds": prov["eval_source_seconds"],
            "compiled_score_seconds": prov["compiled_score_seconds"],
        },
        "tally": stats["tally"],
        "safety_critical_worlds": stats["safety_critical_worlds"],
        "worst_regret_world": stats["worst_regret_world"],
        "worst_regret_pct": stats["worst_regret_pct"],
        "mean_action_agreement": stats["mean_action_agreement"],
        "headline": stats["headline"],
        "conditions": records,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + _format_table(records))
    print("\n" + "=" * 92)
    print(summary["headline"])
    print("=" * 92)
    print(f"Wrote {out_path} ({elapsed / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

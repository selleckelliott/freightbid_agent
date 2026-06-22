"""Risk-aware stress test (Phase 5.5) — the Phase 5 capstone.

Re-scores the bid policies under broker-quality stress on **realized collectible profit**,
combining the two Phase 5 tracks:

* **payment risk** — the Phase 5.1 risk-adjusted EV objective fed by the Phase 5.2 payment-risk
  model.
* **calibration** — the Phase 5.3 drift monitor + the Phase 5.4 post-hoc recalibrator, applied
  only when its promotion guardrail passes on a held-out window.

The research question::

    Does the full risk-aware bidding stack improve expected *collectible* profit under
    broker-quality stress, while recalibration repairs win-probability drift where needed?

**The metric change is the whole point of Phase 5.** Phase 4.5 measured ``realized = P(win) x
(ask - cost)`` — payment-blind, so broker payment quality was orthogonal to it. Phase 5.5
measures **realized collectible profit**, oracle-weighted over the win *and* the payment draw
using the world's **true** broker latents (``true_default_prob`` / ``true_pay_days``) and the
**true** win curve (``win_prob(reserve, ask)``)::

    if won & collected: ask - cost - realized_delay_penalty
    if won & defaulted: -cost
    if lost:            0
    =>  collectible = p_win_true * ( p_collect*ask - cost - p_collect*delay )
        delay       = ask * annual_cash_cost_rate * max(true_pay_days - free_pay_days, 0) / 365

This is exactly the Phase 5.1 ``risk_adjusted_profit`` evaluated with *oracle* inputs — the
payment analogue of the Phase 4.5 oracle ``realized = oracle_p * profit``. The **payment oracle**
is the broker pool rebuilt from ``cfg.brokers`` (the latents the simulator drew payment from),
exactly as ``reservation_rpm`` is the win oracle. It is **evaluation-only**: it never touches the
models, the recommender, or the features.

Four policy arms are compared on the **same** held-out eval-window loads of each world (same
realized outcomes — Common Random Numbers):

1. ``best_fixed``       — best of {conservative market x0.95, posted_rate, stretch x1.10}.
2. ``raw_ev``           — EV recommender, risk-blind ranking (Phase 4.3b).
3. ``risk_adjusted_ev`` — EV recommender ranking by collectible EV (Phase 5.1 + 5.2).
4. ``full_risk_aware``  — ``risk_adjusted_ev`` with the **promoted** Phase 5.4 recalibrator on P(win).

Three honest comparisons fall out: ``raw_ev`` vs ``best_fixed`` (does EV bidding still help?),
``risk_adjusted_ev`` vs ``raw_ev`` (does pricing payment risk improve collectible profit?), and
``full_risk_aware`` vs ``risk_adjusted_ev`` (does recalibration help in win-curve-shifted worlds?).
The **headline verdict** is ``full_risk_aware`` vs ``raw_ev`` (HOLDS > +1% / NEUTRAL +/-1% /
REGRESSION < -1%).

Both models are trained **once on the baseline world** and frozen; every world is distribution
shift at inference, never retraining. Each world is built at a later **operational draw**
(``synthetic_data.seed + SEED_OFFSET``, like Phase 5.4) so the scored loads are out-of-sample,
and a ``time_split`` carves the recalibration fit window from the (later, disjoint) eval window
the four arms are scored on. Writes ``benchmarks/risk_aware_stress_summary.json`` (committed,
lean) for the chart + README.

Examples
--------
    # quick smoke (tiny seeded builds, 2 worlds, short windows, capped loads)
    python -m benchmarks.run_risk_aware_stress --fast

    # canonical sweep
    python -m benchmarks.run_risk_aware_stress --days 21 \
        --out benchmarks/risk_aware_stress_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from adapters.outbound.payment_risk.model_adapter import ModelPaymentRiskAdapter
from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from adapters.outbound.winnability.recalibrated_winnability_adapter import (
    RecalibratedWinnabilityAdapter,
)
from application.config_loader import BidRecommenderConfig, load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from benchmarks.run_bid_recommender_eval import _build_frame
from benchmarks.run_broker_quality_stress import Condition, load_conditions, world_cfg
from benchmarks.run_calibration_monitor import _calibrated_model
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
from ml.data.outcome_schema import read_outcomes
from ml.data.outcome_simulator import win_prob
from ml.features.payment_features import PAYMENT_CATEGORICAL_COLUMNS, payment_feature_columns
from ml.features.winnability_features import BidQuery, market_rate_for
from ml.models.sklearn_payment_risk_model import SklearnPaymentRiskModel
from ml.monitoring.calibration_drift import CalibrationThresholds
from ml.monitoring.calibration_report import load_calibration_config
from ml.training.payment_risk_dataset import LABEL as PAY_LABEL
from ml.training.payment_risk_dataset import PAY_DAYS, build_payment_frame
from ml.training.train_payment_risk_model import _decide_calibration
from ml.training.winnability_dataset import LABEL as WIN_LABEL
from ml.training.winnability_dataset import resolve_path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "risk_aware_stress.yaml"
DEFAULT_RECAL_CONFIG = ROOT / "config" / "recalibration.yaml"
DEFAULT_MONITOR_CONFIG = ROOT / "config" / "calibration_monitor.yaml"
DEFAULT_OUT = ROOT / "benchmarks" / "risk_aware_stress_summary.json"

# Policy arms (ordered weakest -> fullest stack).
BEST_FIXED = "best_fixed"
RAW_EV = "raw_ev"
RISK_ADJUSTED_EV = "risk_adjusted_ev"
FULL_RISK_AWARE = "full_risk_aware"
ARMS = (BEST_FIXED, RAW_EV, RISK_ADJUSTED_EV, FULL_RISK_AWARE)
# Fixed sub-policies the best_fixed arm is the max over (per world).
_FIXED_SUBPOLICIES = ("conservative_fixed", "posted_rate", "stretch_fixed")

# Verdicts (full_risk_aware vs raw_ev), mirroring the +/-1% idiom of the 3.4 / 4.5 sweeps.
VERDICT_HOLDS = "HOLDS"
VERDICT_NEUTRAL = "NEUTRAL"
VERDICT_REGRESSION = "REGRESSION"

# Held-out operational draw (mirrors Phase 5.4): a fresh snapshot seed the frozen base models
# never trained on, so every world's eval-window loads are scored purely out-of-sample.
SEED_OFFSET = 1000

# Smoke subset: baseline reference + one decisive reserve/win-curve drifter.
_FAST_CONDITIONS = ("baseline", "tight_brokers")


# ---------------------------------------------------------------------------
# Collectible-profit oracle  (evaluation-only: true latents + true win curve)
# ---------------------------------------------------------------------------
def _score_ask(
    ask_rpm: float,
    ask_amount: float,
    cost: float,
    reserve: float,
    scale: float,
    p_default_true: float,
    pay_days_true: float,
    cash_rate: float,
    free_days: float,
) -> Dict[str, float]:
    """Realized collectible profit for one chosen ask, oracle-weighted over win + payment.

    Uses the world's **true** broker latents and win curve — never a model. ``p_win`` falls
    as the ask rises (``win_prob``); the collected branch pays the full ask less cost and a
    cash-cost penalty for slow-but-collected pay, the default branch still eats the operating
    cost, and a lost bid is zero.
    """
    p_win = win_prob(reserve, ask_rpm, scale)
    p_default = min(max(float(p_default_true), 0.0), 1.0)
    p_collect = 1.0 - p_default
    delay = ask_amount * cash_rate * max(float(pay_days_true) - free_days, 0.0) / 365.0
    collected_branch = ask_amount - cost - delay
    defaulted_branch = -cost
    won_value = p_collect * collected_branch + p_default * defaulted_branch
    return {
        "collectible": p_win * won_value,
        "p_win": p_win,
        "p_collect": p_collect,
        "p_default": p_default,
        "pay_days": float(pay_days_true),
        "delay_expected": p_win * p_collect * delay,
        "ask_rpm": float(ask_rpm),
    }


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, Any]:
    """Per-arm metrics over its scored eval loads (win-weighted where it matters)."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0, "collectible_profit": 0.0, "win_rate": 0.0, "avg_ask_rpm": 0.0,
            "default_rate_on_won_loads": None, "average_realized_pay_days": None,
            "delay_penalty_total": 0.0,
        }
    coll = np.array([r["collectible"] for r in rows], dtype=float)
    pwin = np.array([r["p_win"] for r in rows], dtype=float)
    pdef = np.array([r["p_default"] for r in rows], dtype=float)
    pcol = np.array([r["p_collect"] for r in rows], dtype=float)
    days = np.array([r["pay_days"] for r in rows], dtype=float)
    delay = np.array([r["delay_expected"] for r in rows], dtype=float)
    ask = np.array([r["ask_rpm"] for r in rows], dtype=float)
    sum_pwin = float(pwin.sum())
    sum_pwin_pcol = float((pwin * pcol).sum())
    return {
        "n": n,
        "collectible_profit": round(float(coll.mean()), 2),
        "win_rate": round(float(pwin.mean()), 4),
        "avg_ask_rpm": round(float(ask.mean()), 4),
        # Win-weighted expected default rate among the loads we actually win.
        "default_rate_on_won_loads": (
            round(float((pwin * pdef).sum() / sum_pwin), 4) if sum_pwin > 0 else None
        ),
        # Win-and-collect-weighted mean realized pay days.
        "average_realized_pay_days": (
            round(float((pwin * pcol * days).sum() / sum_pwin_pcol), 2)
            if sum_pwin_pcol > 0 else None
        ),
        "delay_penalty_total": round(float(delay.sum()), 2),
    }


def _uplift_pct(a: float, b: float) -> float:
    """Percentage uplift of ``a`` over ``b`` (``abs(b)`` denominator keeps the sign honest)."""
    if not b:
        return 0.0
    return round((a - b) / abs(b) * 100.0, 1)


def _verdict(uplift_pct: float, band: float) -> str:
    if uplift_pct >= band:
        return VERDICT_HOLDS
    if uplift_pct <= -band:
        return VERDICT_REGRESSION
    return VERDICT_NEUTRAL


# ---------------------------------------------------------------------------
# Frozen baseline models (trained once, never retrained)
# ---------------------------------------------------------------------------
def _train_payment_adapter(cfg: MLConfig, snapshots, outcomes) -> ModelPaymentRiskAdapter:
    """Train the Phase 5.2 payment-risk model in-memory on the baseline world and freeze it.

    Mirrors ``ml.training.train_payment_risk_model.train``: fit the GBM ``P(default)`` on the
    train slice, the optional ``E[pay_days]`` head on non-default train rows, and apply the same
    validation-only calibration decision the served artifact uses. No artifact is written.
    """
    df = build_payment_frame(snapshots, outcomes, cfg)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "validation"].reset_index(drop=True)
    cols = payment_feature_columns(df.columns)
    cats = [c for c in PAYMENT_CATEGORICAL_COLUMNS if c in cols]
    gbm = SklearnPaymentRiskModel(cols, cats, cfg.payment_risk.random_seed).fit(
        train_df, train_df[PAY_LABEL].to_numpy()
    )
    pay_train = train_df[(train_df["is_default"] == 0) & (train_df[PAY_DAYS].notna())]
    pay_train = pay_train.reset_index(drop=True)
    if len(pay_train) > 0:
        gbm.fit_pay_days(pay_train, pay_train[PAY_DAYS].to_numpy())
    served, _ = _decide_calibration(gbm, val_df, val_df[PAY_LABEL].to_numpy(), cfg)
    return ModelPaymentRiskAdapter(served)


# ---------------------------------------------------------------------------
# Per-world scoring
# ---------------------------------------------------------------------------
def _eval_window_snapshots(snapshots, fit_days: int, eval_days: int):
    """The later, disjoint eval-window snapshots (days ``[fit_days, fit_days+eval_days)``).

    Uses the same day-0 / calendar-day convention as
    :func:`ml.calibration.recalibration_workflow.time_split`, so the loads scored here are
    exactly the loads whose bid trials the recalibrator was judged on.
    """
    times = pd.to_datetime(pd.Series([s.snapshot_time for s in snapshots]))
    day0 = times.min().normalize()
    day_index = (times.dt.normalize() - day0).dt.days.to_numpy()
    mask = (day_index >= fit_days) & (day_index < fit_days + eval_days)
    return [s for s, keep in zip(snapshots, mask) if keep]


def _score_world(
    eval_snaps,
    outcomes,
    world: MLConfig,
    bid_cfg: BidRecommenderConfig,
    base_adapter: ModelWinnabilityAdapter,
    recalibrator,
    payment_adapter: ModelPaymentRiskAdapter,
    pool_index: Dict[str, Any],
    *,
    max_loads: Optional[int],
) -> Dict[str, List[Dict[str, float]]]:
    """Score the four policy arms on the eval-window loads against the collectible oracle.

    All recommender arms share the identical in-support candidate set (recalibration only
    rescales P(win); the extrapolation guard and profit floor are P-independent), so a load is
    scored for every arm or skipped for all — keeping the arms on the same loads.
    """
    scale = world.outcomes.win_logistic_scale_rpm
    cash_rate = bid_cfg.annual_cash_cost_rate
    free_days = bid_cfg.free_pay_days
    reserve_by_key = {
        (o.load_id, o.snapshot_time): o.reservation_rpm for o in outcomes
    }

    # Recommenders built once per world. raw_ev ranks risk-blind; risk_adjusted_ev / full add
    # the payment port (and full swaps in the recalibrated winnability adapter). When no
    # recalibrator was promoted, full is byte-identical to risk_adjusted_ev (the adapter's
    # ``recalibrator=None`` pass-through contract), so we reuse its target and skip a model call.
    bid_raw = replace(bid_cfg, risk_adjusted_ev_enabled=False)
    bid_ra = replace(bid_cfg, risk_adjusted_ev_enabled=True)
    raw_rec = EVBidRecommender(base_adapter, bid_raw, payment=None)
    ra_rec = EVBidRecommender(base_adapter, bid_ra, payment=payment_adapter)
    full_rec = (
        EVBidRecommender(
            RecalibratedWinnabilityAdapter(base_adapter, recalibrator), bid_ra,
            payment=payment_adapter,
        )
        if recalibrator is not None else None
    )

    eval_snaps = sorted(eval_snaps, key=lambda s: (s.snapshot_time, s.load_id))
    if max_loads is not None:
        eval_snaps = eval_snaps[:max_loads]

    rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in _FIXED_SUBPOLICIES}
    for arm in (RAW_EV, RISK_ADJUSTED_EV, FULL_RISK_AWARE):
        rows[arm] = []

    def _target(rec, query, cost) -> Optional[Tuple[float, float]]:
        r = rec.recommend(query, load_id=0, estimated_total_cost=cost)
        tgt = r.option(TARGET) or r.option(MAX_EV)
        return (tgt.ask_rpm, tgt.ask_amount) if tgt is not None else None

    for s in eval_snaps:
        key = (s.load_id, s.snapshot_time)
        if key not in reserve_by_key:
            continue
        miles = max(float(s.loaded_miles), 1.0)
        cost = bid_cfg.cost_per_loaded_mile * miles
        market = market_rate_for(s.origin_lat, s.origin_lon)
        reserve = reserve_by_key[key]
        query = BidQuery.from_snapshot(s)
        posted_rpm = query.rate_per_mile
        broker = pool_index.get(s.broker_id)
        p_default_true = broker.true_default_prob if broker is not None else 0.05
        pay_days_true = broker.true_pay_days if broker is not None else 35.0

        # Gate: skip loads with no in-support, guardrail-clearing candidate (rare) — matches
        # the Phase 4.5 sweep so the arms only score loads the recommender actually bids.
        gate = raw_rec.score(query, estimated_total_cost=cost)
        if gate is None or not any(not c.extrapolated for c in gate.candidates):
            continue

        # Recommender targets first, so a load missing any arm's target is skipped for all.
        raw_t = _target(raw_rec, query, cost)
        ra_t = _target(ra_rec, query, cost)
        full_t = _target(full_rec, query, cost) if full_rec is not None else ra_t
        if raw_t is None or ra_t is None or full_t is None:
            continue
        arm_targets = {RAW_EV: raw_t, RISK_ADJUSTED_EV: ra_t, FULL_RISK_AWARE: full_t}

        def score(ask_rpm: float, ask_amount: float) -> Dict[str, float]:
            return _score_ask(
                ask_rpm, ask_amount, cost, reserve, scale,
                p_default_true, pay_days_true, cash_rate, free_days,
            )

        # Fixed sub-policies (single ask each).
        rows["conservative_fixed"].append(score(market * 0.95, market * 0.95 * miles))
        anchor = posted_rpm if (posted_rpm and posted_rpm > 0) else market
        rows["posted_rate"].append(score(anchor, anchor * miles))
        rows["stretch_fixed"].append(score(market * 1.10, market * 1.10 * miles))
        for arm in (RAW_EV, RISK_ADJUSTED_EV, FULL_RISK_AWARE):
            ask_rpm, ask_amount = arm_targets[arm]
            rows[arm].append(score(ask_rpm, ask_amount))

    return rows


def _world_record(
    cond: Condition,
    rows: Dict[str, List[Dict[str, float]]],
    recal_result,
    promoted: bool,
    band: float,
) -> Dict[str, Any]:
    """Assemble one world's record: per-arm metrics, uplifts, verdict, recalibration severities."""
    # best_fixed = the strongest fixed sub-policy on collectible profit (per world).
    fixed_aggs = {name: _aggregate(rows[name]) for name in _FIXED_SUBPOLICIES}
    fixed_winner = max(fixed_aggs, key=lambda k: fixed_aggs[k]["collectible_profit"])
    best_fixed = dict(fixed_aggs[fixed_winner])
    best_fixed["fixed_winner"] = fixed_winner

    arms = {
        BEST_FIXED: best_fixed,
        RAW_EV: _aggregate(rows[RAW_EV]),
        RISK_ADJUSTED_EV: _aggregate(rows[RISK_ADJUSTED_EV]),
        FULL_RISK_AWARE: _aggregate(rows[FULL_RISK_AWARE]),
    }
    cp = {k: arms[k]["collectible_profit"] for k in ARMS}
    uplift_vs_raw = _uplift_pct(cp[FULL_RISK_AWARE], cp[RAW_EV])

    pre = recal_result.pre
    post = recal_result.post
    return {
        "name": cond.name,
        "lens": cond.lens(),
        "rationale": cond.rationale,
        "overrides": cond.overrides,
        "n_loads": arms[RAW_EV]["n"],
        "arms": arms,
        # The three honest comparisons (headline = full vs raw EV).
        "uplift_vs_raw_ev": uplift_vs_raw,
        "uplift_vs_fixed": _uplift_pct(cp[FULL_RISK_AWARE], cp[BEST_FIXED]),
        "risk_adj_uplift_vs_raw": _uplift_pct(cp[RISK_ADJUSTED_EV], cp[RAW_EV]),
        "full_uplift_vs_risk_adj": _uplift_pct(cp[FULL_RISK_AWARE], cp[RISK_ADJUSTED_EV]),
        "verdict": _verdict(uplift_vs_raw, band),
        # Recalibration story (from the Phase 5.4 guardrail on the held-out eval window).
        "recalibrator_promoted": promoted,
        "recalibrator_reason": recal_result.reason,
        "calibration_severity_before": pre.severity,
        "calibration_severity_after": post.severity if post is not None else pre.severity,
        "ece_before": pre.ece,
        "ece_after": post.ece if post is not None else None,
    }


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_risk_aware_stress(
    cfg: MLConfig,
    bid_cfg: BidRecommenderConfig,
    thresholds: CalibrationThresholds,
    recal_config: RecalibrationConfig,
    conditions: List[Condition],
    *,
    days: int,
    band: float,
    max_loads: Optional[int],
    seed_offset: int = SEED_OFFSET,
) -> List[Dict[str, Any]]:
    """Freeze the base models on baseline, then score the four arms on every shifted world.

    The calibrated winnability model and the payment-risk model are trained once on the
    baseline *training* draw and frozen. Each world's recalibration fit/eval windows and the
    four-arm scoring come from a later operational draw (``synthetic_data.seed + seed_offset``)
    the base models never saw, so the comparison is purely out-of-sample.
    """
    conditions = sorted(conditions, key=lambda c: (bool(c.overrides), c.name))
    records: List[Dict[str, Any]] = []
    op_seed = cfg.synthetic_data.seed + seed_offset

    with TemporaryDirectory() as base_tmp:
        base_world = world_cfg(cfg, Path(base_tmp))
        build_winnability_dataset(base_world, days=days)
        base_frame, base_snaps = _build_frame(base_world)
        base_model = _calibrated_model(base_frame, base_world.winnability.random_seed)
        base_adapter = ModelWinnabilityAdapter(base_model)
        base_outcomes = read_outcomes(resolve_path(base_world.outcomes.outcomes_path))
        payment_adapter = _train_payment_adapter(base_world, base_snaps, base_outcomes)

        for i, cond in enumerate(conditions, 1):
            t0 = time.perf_counter()
            with TemporaryDirectory() as ctmp:
                world = world_cfg(cfg, Path(ctmp), cond.overrides or None)
                build_winnability_dataset(world, days=days, seed=op_seed)
                frame, snaps = _build_frame(world)
                outcomes = read_outcomes(resolve_path(world.outcomes.outcomes_path))
                pool_index = broker_index(
                    build_broker_pool(BrokerPoolParams.from_config(world.brokers))
                )

                # Fit + promote the Phase 5.4 recalibrator on this world's fit -> eval windows.
                fit_df, eval_df = time_split(frame, recal_config.fit_days, recal_config.eval_days)
                raw_fit = base_model.predict_proba(fit_df)
                raw_eval = base_model.predict_proba(eval_df)
                result = recalibrate(
                    raw_fit, fit_df[WIN_LABEL].to_numpy(),
                    raw_eval, eval_df[WIN_LABEL].to_numpy(),
                    thresholds, recal_config, label=cond.name,
                )
                # Reconstruct the fitted map only when promoted (deterministic re-fit on the
                # same fit window); otherwise the full arm passes through the base model.
                recalibrator = (
                    fit_recalibrator(raw_fit, fit_df[WIN_LABEL].to_numpy(), method=recal_config.method)
                    if result.promoted else None
                )

                eval_snaps = _eval_window_snapshots(
                    snaps, recal_config.fit_days, recal_config.eval_days
                )
                rows = _score_world(
                    eval_snaps, outcomes, world, bid_cfg, base_adapter, recalibrator,
                    payment_adapter, pool_index, max_loads=max_loads,
                )

            record = _world_record(cond, rows, result, result.promoted, band)
            records.append(record)
            _print_line(i, len(conditions), record, time.perf_counter() - t0)

    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
_VERDICT_TAG = {VERDICT_HOLDS: "HOLDS  ", VERDICT_NEUTRAL: "neutral", VERDICT_REGRESSION: "REGRESS"}


def _print_line(i: int, n: int, record: Dict[str, Any], elapsed: float) -> None:
    tag = _VERDICT_TAG.get(record["verdict"], record["verdict"])
    promoted = "recal" if record["recalibrator_promoted"] else "  -  "
    print(
        f"[{i:2d}/{n}] {tag} {record['name']:<19} "
        f"full {record['arms'][FULL_RISK_AWARE]['collectible_profit']:>8.0f}  "
        f"vs raw {record['uplift_vs_raw_ev']:+6.1f}%  "
        f"vs fixed {record['uplift_vs_fixed']:+6.1f}%  "
        f"ra/raw {record['risk_adj_uplift_vs_raw']:+5.1f}%  {promoted}  ({elapsed:.0f}s)"
    )


def _tally(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tally = {VERDICT_HOLDS: 0, VERDICT_NEUTRAL: 0, VERDICT_REGRESSION: 0}
    for r in records:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    return tally


def _summarize(records: List[Dict[str, Any]], watch_worlds: Sequence[str]) -> Dict[str, Any]:
    tally = _tally(records)
    promoted = [r for r in records if r["recalibrator_promoted"]]
    watch = set(watch_worlds)
    payment_moved = [
        r for r in records if r["name"] in watch and r["risk_adj_uplift_vs_raw"] > 0
    ]
    return {
        "tally": tally,
        "promoted_count": len(promoted),
        "payment_quality_worlds_improved": [r["name"] for r in payment_moved],
        "headline": (
            f"Full risk-aware beats raw EV on collectible profit in "
            f"{tally[VERDICT_HOLDS]}/{len(records)} worlds "
            f"(neutral {tally[VERDICT_NEUTRAL]}, regression {tally[VERDICT_REGRESSION]}); "
            f"risk-adjusted EV improves {len(payment_moved)} payment-quality world(s); "
            f"recalibration promoted in {len(promoted)}."
        ),
    }


def _format_table(records: List[Dict[str, Any]]) -> str:
    header = (
        f"{'world':<19} {'fixed':>8} {'raw':>8} {'risk_adj':>9} {'full':>8} "
        f"{'full/raw':>9} {'full/fix':>9}  {'recal':<6} {'verdict':<8}"
    )
    lines = [header, "-" * len(header)]
    for r in records:
        a = r["arms"]
        recal = "yes" if r["recalibrator_promoted"] else "-"
        lines.append(
            f"{r['name']:<19} "
            f"{a[BEST_FIXED]['collectible_profit']:>8.0f} "
            f"{a[RAW_EV]['collectible_profit']:>8.0f} "
            f"{a[RISK_ADJUSTED_EV]['collectible_profit']:>9.0f} "
            f"{a[FULL_RISK_AWARE]['collectible_profit']:>8.0f} "
            f"{r['uplift_vs_raw_ev']:>+8.1f}% {r['uplift_vs_fixed']:>+8.1f}%  "
            f"{recal:<6} {r['verdict']:<8}"
        )
    return "\n".join(lines)


def _load_sweep_config(path: Path) -> Dict[str, Any]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return doc.get("risk_aware_stress", doc) or {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Risk-aware-stress sweep config (risk_aware_stress: block).")
    parser.add_argument("--recal-config", default=str(DEFAULT_RECAL_CONFIG),
                        help="Recalibration policy config (method/windows/guardrail).")
    parser.add_argument("--monitor-config", default=str(DEFAULT_MONITOR_CONFIG),
                        help="Calibration-monitor config for the shared severity thresholds.")
    parser.add_argument("--conditions", default=None,
                        help="Override the stress-conditions YAML (default: the sweep config's).")
    parser.add_argument("--days", type=int, default=None,
                        help="Synthetic horizon per world (must cover fit_days + eval_days).")
    parser.add_argument("--max-loads", type=int, default=None,
                        help="Cap eval-window loads scored per world (fast forces 150).")
    parser.add_argument("--fast", action="store_true",
                        help="Tiny seeded builds + short windows + a 2-world subset (smoke).")
    parser.add_argument("--only", default=None,
                        help="Comma-separated condition names to run (default: all).")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    sweep = _load_sweep_config(Path(args.config))
    thresholds, _ = load_calibration_config(args.monitor_config)
    recal_config = load_recalibration_config(args.recal_config)

    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")

    days = args.days if args.days is not None else int(sweep.get("days", 21))
    band = float(sweep.get("uplift_band_pct", 1.0))
    watch_worlds = list(sweep.get("watch_worlds", []))
    max_loads = args.max_loads

    conditions_path = Path(args.conditions or sweep.get("conditions", "config/broker_quality_stress.yaml"))
    if not conditions_path.is_absolute():
        conditions_path = ROOT / conditions_path
    conditions = load_conditions(conditions_path)

    if args.fast:
        days = 6
        recal_config = replace(recal_config, fit_days=2, eval_days=3, min_samples=40)
        thresholds = replace(thresholds, min_samples=40)
        max_loads = max_loads or 150
        if not args.only:
            conditions = [c for c in conditions if c.name in _FAST_CONDITIONS]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        conditions = [c for c in conditions if c.name in wanted]
        missing = wanted - {c.name for c in conditions}
        if missing:
            raise SystemExit(f"unknown condition(s): {sorted(missing)}")
    if not conditions:
        raise SystemExit("no conditions selected")
    if days < recal_config.fit_days + recal_config.eval_days:
        raise SystemExit(
            f"--days {days} is shorter than fit_days+eval_days "
            f"({recal_config.fit_days}+{recal_config.eval_days}); widen --days or the windows."
        )

    print("=" * 88)
    print(f"Risk-aware stress (Phase 5.5): {len(conditions)} worlds x 4 arms on collectible "
          f"profit (base models frozen on baseline; {days}d worlds, "
          f"fit {recal_config.fit_days}d / eval {recal_config.eval_days}d)")
    print("=" * 88)

    start = time.time()
    records = run_risk_aware_stress(
        cfg, bid_cfg, thresholds, recal_config, conditions,
        days=days, band=band, max_loads=max_loads,
    )
    elapsed = time.time() - start
    stats = _summarize(records, watch_worlds)

    summary: Dict[str, Any] = {
        "config": {
            "fast": args.fast,
            "days": days,
            "max_loads": max_loads,
            "uplift_band_pct": band,
            "condition_count": len(conditions),
            "trained_on": "baseline",
            "base_calibration": "isotonic(validation)",
            "recalibration": recal_config.as_dict(),
            "thresholds": thresholds.as_dict(),
            "annual_cash_cost_rate": bid_cfg.annual_cash_cost_rate,
            "free_pay_days": bid_cfg.free_pay_days,
            "cost_per_loaded_mile": bid_cfg.cost_per_loaded_mile,
            "win_logistic_scale_rpm_baseline": cfg.outcomes.win_logistic_scale_rpm,
            "winnability_seed": cfg.winnability.random_seed,
            "payment_seed": cfg.payment_risk.random_seed,
            "operational_seed": cfg.synthetic_data.seed + SEED_OFFSET,
            "operational_seed_offset": SEED_OFFSET,
            "watch_worlds": watch_worlds,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 1),
        "tally": stats["tally"],
        "promoted_count": stats["promoted_count"],
        "payment_quality_worlds_improved": stats["payment_quality_worlds_improved"],
        "headline": stats["headline"],
        "conditions": records,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + _format_table(records))
    print("\n" + "=" * 88)
    print(summary["headline"])
    print("=" * 88)
    print(f"Wrote {out_path} ({elapsed / 60.0:.1f} min).")


if __name__ == "__main__":
    main()

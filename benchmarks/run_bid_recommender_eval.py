"""Offline evaluation of the Phase 4.3 EV bid recommender.

Scores a held-out slice of synthetic loads and asks: does turning the calibrated
Phase 4.2 winnability model into an **expected-value bid ladder** actually pick better
bids than naive fixed-margin policies — and how close does it get to a clairvoyant
oracle?

The evaluation is **oracle-grounded**. Each load carries a hidden ``reservation_rpm``
(the minimum rpm the broker would accept) from the Phase 4.1 outcome world. The pure
simulator function ``win_prob(reservation, ask)`` gives the *true* probability an ask is
accepted, so for any chosen ask we can score an **oracle-weighted realized profit**
``win_prob(reserve, ask) x profit_if_won`` — honest even for off-grid asks (recommended
bids rarely land on the 6-point training grid), and low-variance versus a single
Bernoulli draw. The oracle is **evaluation-only**: it never touches the recommender,
the model, or the features (a unit test enforces this leakage discipline).

Policies compared (same held-out loads, same per-loaded-mile cost proxy):

* ``conservative_fixed`` — ask = market_rate x 0.95.
* ``posted_rate``        — ask = posted rpm (or market_rate when "call for rate").
* ``stretch_fixed``      — ask = market_rate x 1.10.
* ``recommender_max_ev`` — the EV recommender's max-EV rung.
* ``recommender_target`` — the EV recommender's recommended (target) rung.

Reported per policy: avg ask ($/rpm), avg model & oracle P(win), avg model-EV, avg
**oracle-weighted realized profit**, and avg **EV-regret vs the oracle** (the oracle's
best achievable EV over the same in-support candidate set minus what the policy
realized; >= 0 by construction). Plus a calibration-by-selected-bid reliability table
for the target policy. Writes ``benchmarks/bid_recommender_summary.json`` (committed).

Examples
--------
    # quick smoke (tiny synthetic build, capped loads)
    python -m benchmarks.run_bid_recommender_eval --fast

    # canonical run (uses the configured dataset; builds it if missing)
    python -m benchmarks.run_bid_recommender_eval \
        --out benchmarks/bid_recommender_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional

import numpy as np

from adapters.outbound.winnability.model_adapter import ModelWinnabilityAdapter
from application.config_loader import load_bid_recommender_config
from application.ev_bid_recommender import EVBidRecommender
from domain.models.bid_recommendation import MAX_EV, TARGET
from ml.config import load_ml_config
from ml.data.build_winnability_dataset import build_winnability_dataset
from ml.data.outcome_schema import read_outcomes
from ml.data.outcome_simulator import win_prob
from ml.features.winnability_features import (
    CATEGORICAL_COLUMNS,
    BidQuery,
    feature_columns,
    market_rate_for,
)
from ml.models.sklearn_winnability_model import SklearnWinnabilityModel
from ml.training.winnability_dataset import (
    LABEL,
    _split_boundaries,
    build_winnability_frame,
    load_snapshots_and_trials,
    resolve_path,
)

ROOT = Path(__file__).resolve().parents[1]

POLICY_LABELS = {
    "conservative_fixed": "Conservative fixed (market x0.95)",
    "posted_rate": "Posted-rate",
    "stretch_fixed": "Stretch fixed (market x1.10)",
    "recommender_max_ev": "EV recommender (max-EV)",
    "recommender_target": "EV recommender (target)",
}


def _train_model(frame, seed: int) -> SklearnWinnabilityModel:
    """Quick-train the winnability model on the train split (seeded, seconds)."""
    train_df = frame[frame["split"] == "train"].reset_index(drop=True)
    cols = feature_columns(frame.columns)
    cats = [c for c in CATEGORICAL_COLUMNS if c in cols]
    return SklearnWinnabilityModel(cols, cats, seed).fit(train_df, train_df[LABEL].to_numpy())


def _eval_ask(adapter, query: BidQuery, ask_rpm: float, cost: float, miles: float,
              reserve: float, scale: float) -> Dict[str, float]:
    ask = ask_rpm * miles
    profit = ask - cost
    model_p = float(adapter.win_probabilities(query, [ask_rpm])[0])
    oracle_p = win_prob(reserve, ask_rpm, scale)
    return {
        "ask": ask,
        "ask_rpm": ask_rpm,
        "profit": profit,
        "model_p": model_p,
        "oracle_p": oracle_p,
        "model_ev": model_p * profit,
        "realized": oracle_p * profit,
    }


def evaluate(cfg, bid_cfg, frame, snapshots, outcomes_path: Path, *,
             max_loads: Optional[int] = None) -> Dict:
    scale = cfg.outcomes.win_logistic_scale_rpm
    seed = cfg.winnability.random_seed
    model = _train_model(frame, seed)
    adapter = ModelWinnabilityAdapter(model)
    recommender = EVBidRecommender(adapter, bid_cfg)

    reserve_by_key = {
        (o.load_id, o.snapshot_time): o.reservation_rpm for o in read_outcomes(outcomes_path)
    }
    _, boundary_val = _split_boundaries(
        [s.snapshot_time for s in snapshots],
        cfg.winnability.train_fraction,
        cfg.winnability.validation_fraction,
    )
    test_snaps = [
        s
        for s in snapshots
        if s.snapshot_time >= boundary_val and (s.load_id, s.snapshot_time) in reserve_by_key
    ]
    test_snaps.sort(key=lambda s: (s.snapshot_time, s.load_id))
    if max_loads is not None:
        test_snaps = test_snaps[:max_loads]

    rows: Dict[str, List[Dict[str, float]]] = {k: [] for k in POLICY_LABELS}
    oracle_best: List[float] = []
    target_calibration: List[Dict[str, float]] = []
    best_example = None  # (sort_key, load_id, query, cost, miles, market, reserve, posted_rpm, rec)

    for s in test_snaps:
        miles = max(float(s.loaded_miles), 1.0)
        cost = bid_cfg.cost_per_loaded_mile * miles
        market = market_rate_for(s.origin_lat, s.origin_lon)
        reserve = reserve_by_key[(s.load_id, s.snapshot_time)]
        query = BidQuery.from_snapshot(s)
        posted_rpm = query.rate_per_mile

        scoring = recommender.score(query, estimated_total_cost=cost)
        in_support = [c for c in scoring.candidates if not c.extrapolated] if scoring else []
        if not in_support:
            continue  # no in-support, guardrail-clearing candidate -> skip (rare)

        # Oracle's best achievable EV over the same in-support candidate set.
        best = max(win_prob(reserve, c.ask_rpm, scale) * c.profit_if_won for c in in_support)
        oracle_best.append(best)

        # Fixed policies (single ask each).
        rows["conservative_fixed"].append(
            _eval_ask(adapter, query, market * 0.95, cost, miles, reserve, scale)
        )
        rows["posted_rate"].append(
            _eval_ask(
                adapter, query, posted_rpm if posted_rpm else market, cost, miles, reserve, scale
            )
        )
        rows["stretch_fixed"].append(
            _eval_ask(adapter, query, market * 1.10, cost, miles, reserve, scale)
        )

        # Recommender policies (reuse the already-scored candidates; no extra model calls).
        max_ev_c = max(in_support, key=lambda c: c.expected_value)
        rows["recommender_max_ev"].append({
            "ask": max_ev_c.ask_amount, "ask_rpm": max_ev_c.ask_rpm,
            "profit": max_ev_c.profit_if_won, "model_p": max_ev_c.win_probability,
            "oracle_p": win_prob(reserve, max_ev_c.ask_rpm, scale),
            "model_ev": max_ev_c.expected_value,
            "realized": win_prob(reserve, max_ev_c.ask_rpm, scale) * max_ev_c.profit_if_won,
        })

        rec = recommender.recommend(query, load_id=0, estimated_total_cost=cost)
        tgt = rec.option(TARGET) or rec.option(MAX_EV)
        tgt_oracle_p = win_prob(reserve, tgt.ask_rpm, scale)
        rows["recommender_target"].append({
            "ask": tgt.ask_amount, "ask_rpm": tgt.ask_rpm, "profit": tgt.profit_if_won,
            "model_p": tgt.win_probability, "oracle_p": tgt_oracle_p,
            "model_ev": tgt.expected_value, "realized": tgt_oracle_p * tgt.profit_if_won,
        })
        target_calibration.append({"model_p": tgt.win_probability, "oracle_p": tgt_oracle_p})

        # Track the most illustrative load for the example EV-curve panel: prefer a
        # ladder whose rungs sit at *distinct* asks (so the spread is visible), then a
        # fuller ladder, then a longer load (clearer $ spread), then a stable tie-break
        # on load_id. Deterministic given the seeded model.
        n_distinct = len({round(o.ask_rpm, 3) for o in rec.options})
        ex_key = (n_distinct, len(rec.options), -abs(miles - 600.0))
        if (
            best_example is None
            or ex_key > best_example[0]
            or (ex_key == best_example[0] and str(s.load_id) < str(best_example[1]))
        ):
            best_example = (ex_key, s.load_id, query, cost, miles, market, reserve, posted_rpm, rec)

    summary = _summarize(rows, oracle_best, target_calibration, len(test_snaps))
    summary["example_load"] = _example_payload(adapter, best_example, scale, bid_cfg)
    return summary


def _example_payload(adapter, best_example, scale: float, bid_cfg) -> Optional[Dict]:
    """Render one held-out load's full ask-vs-{P(win), profit, EV} curve plus its
    ladder rungs, so the chart can show *why* the recommender picks the bid it does.

    The curve is sampled across the model's **trained support**
    ``[trained_ask_ratio_min, trained_ask_ratio_max] x market_rate`` (where the EV
    estimate is trustworthy); the oracle curve is overlaid for honesty. One extra
    batched model call — the oracle stays evaluation-only."""
    if best_example is None:
        return None
    _, load_id, query, cost, miles, market, reserve, posted_rpm, rec = best_example
    lo = bid_cfg.trained_ask_ratio_min * market
    hi = bid_cfg.trained_ask_ratio_max * market
    grid = [float(r) for r in np.linspace(lo, hi, 49)]
    model_p = adapter.win_probabilities(query, grid) or []
    ask_amount = [r * miles for r in grid]
    profit = [a - cost for a in ask_amount]
    oracle_p = [win_prob(reserve, r, scale) for r in grid]
    model_ev = [p * pr for p, pr in zip(model_p, profit)]
    oracle_ev = [op * pr for op, pr in zip(oracle_p, profit)]
    rungs = [
        {
            "label": o.label,
            "ask_rpm": round(o.ask_rpm, 4),
            "ask_amount": round(o.ask_amount, 2),
            "win_probability": round(o.win_probability, 4),
            "expected_value": round(o.expected_value, 2),
            "profit_if_won": round(o.profit_if_won, 2),
        }
        for o in rec.options
    ]
    return {
        "load_id": str(load_id),
        "loaded_miles": round(miles, 1),
        "market_rate": round(market, 4),
        "estimated_cost": round(cost, 2),
        "breakeven_rpm": round(cost / miles, 4),
        "posted_rpm": round(posted_rpm, 4) if posted_rpm else None,
        "trained_band_ratio": [bid_cfg.trained_ask_ratio_min, bid_cfg.trained_ask_ratio_max],
        "recommended_label": rec.recommended_label,
        "rungs": rungs,
        "curve": {
            "ask_rpm": [round(r, 4) for r in grid],
            "ask_amount": [round(a, 2) for a in ask_amount],
            "model_win_prob": [round(float(p), 4) for p in model_p],
            "oracle_win_prob": [round(float(p), 4) for p in oracle_p],
            "profit_if_won": [round(p, 2) for p in profit],
            "model_ev": [round(float(e), 2) for e in model_ev],
            "oracle_ev": [round(float(e), 2) for e in oracle_ev],
        },
    }


def _summarize(rows, oracle_best, target_calibration, n_test) -> Dict:
    oracle_best_avg = float(np.mean(oracle_best)) if oracle_best else 0.0
    policies = {}
    for key, recs in rows.items():
        if not recs:
            continue
        realized = float(np.mean([r["realized"] for r in recs]))
        policies[key] = {
            "label": POLICY_LABELS[key],
            "n": len(recs),
            "avg_ask": round(float(np.mean([r["ask"] for r in recs])), 2),
            "avg_ask_rpm": round(float(np.mean([r["ask_rpm"] for r in recs])), 4),
            "avg_profit_if_won": round(float(np.mean([r["profit"] for r in recs])), 2),
            "avg_model_win_prob": round(float(np.mean([r["model_p"] for r in recs])), 4),
            "avg_oracle_win_prob": round(float(np.mean([r["oracle_p"] for r in recs])), 4),
            "avg_model_ev": round(float(np.mean([r["model_ev"] for r in recs])), 2),
            "avg_realized_profit": round(realized, 2),
            "avg_ev_regret_vs_oracle": round(oracle_best_avg - realized, 2),
        }

    best_fixed = max(
        (policies[k]["avg_realized_profit"] for k in ("conservative_fixed", "posted_rate", "stretch_fixed") if k in policies),
        default=0.0,
    )
    target_realized = policies.get("recommender_target", {}).get("avg_realized_profit", 0.0)
    uplift = ((target_realized - best_fixed) / best_fixed * 100.0) if best_fixed else 0.0

    return {
        "oracle_best_avg_ev": round(oracle_best_avg, 2),
        "policies": policies,
        "calibration_by_selected_bid": _calibration_table(target_calibration),
        "headline": {
            "target_realized_profit": round(target_realized, 2),
            "best_fixed_realized_profit": round(best_fixed, 2),
            "target_uplift_pct_vs_best_fixed": round(uplift, 1),
            "target_ev_regret_vs_oracle": policies.get("recommender_target", {}).get(
                "avg_ev_regret_vs_oracle", 0.0
            ),
        },
        "n_test_loads": n_test,
    }


def _calibration_table(pairs: List[Dict[str, float]], n_bins: int = 10) -> List[Dict]:
    if not pairs:
        return []
    table = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        bucket = [p for p in pairs if (lo <= p["model_p"] < hi) or (i == n_bins - 1 and p["model_p"] == 1.0)]
        if not bucket:
            continue
        table.append({
            "bin": f"[{lo:.1f},{hi:.1f})",
            "count": len(bucket),
            "mean_predicted_win_prob": round(float(np.mean([p["model_p"] for p in bucket])), 4),
            "mean_oracle_win_prob": round(float(np.mean([p["oracle_p"] for p in bucket])), 4),
        })
    return table


def _build_frame(cfg, snapshot_path=None, trials_path=None, outcomes_path=None):
    snapshots, trials = load_snapshots_and_trials(cfg)
    frame = build_winnability_frame(snapshots, trials, cfg)
    return frame, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.3 EV bid recommender offline eval")
    parser.add_argument("--out", default="benchmarks/bid_recommender_summary.json")
    parser.add_argument("--fast", action="store_true",
                        help="tiny seeded build + capped loads for a quick smoke")
    parser.add_argument("--days", type=int, default=None,
                        help="override synthetic horizon (fast mode defaults to 6)")
    parser.add_argument("--max-loads", type=int, default=None,
                        help="cap the number of held-out loads scored")
    args = parser.parse_args()

    cfg = load_ml_config()
    bid_cfg = load_bid_recommender_config("config")
    start = time.time()

    if args.fast:
        days = args.days or 6
        max_loads = args.max_loads or 200
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = replace(cfg, outcomes=replace(
                cfg.outcomes,
                snapshot_path=str(tmp / "snap.jsonl"),
                outcomes_path=str(tmp / "out.jsonl"),
                trials_path=str(tmp / "trials.jsonl"),
            ))
            build_winnability_dataset(cfg, days=days)
            frame, snapshots = _build_frame(cfg)
            summary = evaluate(cfg, bid_cfg, frame, snapshots,
                               resolve_path(cfg.outcomes.outcomes_path), max_loads=max_loads)
    else:
        # Canonical: use the configured dataset (built on demand if missing).
        outcomes_path = resolve_path(cfg.outcomes.outcomes_path)
        if not outcomes_path.exists():
            build_winnability_dataset(cfg, days=args.days)
        frame, snapshots = _build_frame(cfg)
        summary = evaluate(cfg, bid_cfg, frame, snapshots, outcomes_path,
                           max_loads=args.max_loads)

    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["runtime_seconds"] = round(time.time() - start, 1)
    summary["config"] = {
        "fast": args.fast,
        "win_logistic_scale_rpm": cfg.outcomes.win_logistic_scale_rpm,
        "cost_per_loaded_mile": bid_cfg.cost_per_loaded_mile,
        "anchor_multipliers": list(bid_cfg.anchor_multipliers),
        "trained_ask_ratio_band": [bid_cfg.trained_ask_ratio_min, bid_cfg.trained_ask_ratio_max],
        "winnability_seed": cfg.winnability.random_seed,
    }

    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print_report(summary, out_path)


def _print_report(summary: Dict, out_path: Path) -> None:
    print(f"\nEV bid recommender eval — {summary['n_test_loads']} held-out loads "
          f"({summary['runtime_seconds']}s)")
    print(f"Oracle best avg EV: ${summary['oracle_best_avg_ev']:,.2f}\n")
    header = f"{'Policy':<34}{'avg ask':>9}{'P(win) m/o':>13}{'realized':>10}{'regret':>9}"
    print(header)
    print("-" * len(header))
    for key in POLICY_LABELS:
        p = summary["policies"].get(key)
        if not p:
            continue
        print(f"{p['label']:<34}{p['avg_ask']:>9.0f}"
              f"{p['avg_model_win_prob']:>6.2f}/{p['avg_oracle_win_prob']:<6.2f}"
              f"{p['avg_realized_profit']:>10.0f}{p['avg_ev_regret_vs_oracle']:>9.0f}")
    h = summary["headline"]
    print(f"\nTarget vs best fixed: ${h['target_realized_profit']:,.0f} vs "
          f"${h['best_fixed_realized_profit']:,.0f} "
          f"({h['target_uplift_pct_vs_best_fixed']:+.1f}%); "
          f"target EV-regret ${h['target_ev_regret_vs_oracle']:,.0f}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

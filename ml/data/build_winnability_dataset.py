"""Build the Phase 4.1 *labeled winnability dataset*.

Orchestrates the whole outcome world and emits three seeded, byte-reproducible
JSONL artifacts under ``data/`` (gitignored, like the Phase 3.1 history):

1. ``winnability_snapshots.jsonl`` — decision-time ``LoadSnapshotRecord``s with the
   observable broker/quality columns attached (no latent fields).
2. ``winnability_outcomes.jsonl`` — one ``LoadOutcomeRecord`` per load: coverage,
   payment, negotiation, plus the hidden ground-truth ``reservation_rpm`` /
   ``contention_intensity`` (labels only — never on the snapshot).
3. ``winnability_trials.jsonl`` — materialized ``(bid_rpm, won)`` rows over a neutral
   rpm grid: the directly-trainable winnability table for Phase 4.2.

No model is trained here (that is 4.2). Run::

    python -m ml.data.build_winnability_dataset --config config/ml_config.yaml
    python -m ml.data.build_winnability_dataset --days 5   # quick smoke build
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

from ml.brokers import BrokerPoolParams, build_broker_pool
from ml.config import MLConfig, load_ml_config
from ml.data.load_history_schema import write_jsonl
from ml.data.outcome_schema import write_bid_trials, write_outcomes
from ml.data.outcome_simulator import (
    OutcomeConfig,
    sample_bid_trials,
    simulate_outcomes,
)
from ml.data.synthetic_history_generator import GeneratorParams, generate_history

ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_winnability_dataset(
    cfg: MLConfig,
    *,
    days: Optional[int] = None,
    seed: Optional[int] = None,
    write: bool = True,
) -> Dict[str, Any]:
    """Generate snapshots, simulate outcomes, sample bid trials, write artifacts.

    Returns a summary dict (counts + label balance). Deterministic: the same config
    yields byte-identical artifacts.
    """
    params = GeneratorParams.from_config(cfg.synthetic_data)
    if days is not None:
        params = replace(params, days=days)
    if seed is not None:
        params = replace(params, seed=seed)

    pool = build_broker_pool(BrokerPoolParams.from_config(cfg.brokers))
    ocfg = OutcomeConfig.from_config(cfg.outcomes)

    records = generate_history(params, pool)
    outcomes = simulate_outcomes(records, pool, ocfg)
    trials = sample_bid_trials(records, outcomes, ocfg)

    snapshot_path = _resolve(cfg.outcomes.snapshot_path)
    outcomes_path = _resolve(cfg.outcomes.outcomes_path)
    trials_path = _resolve(cfg.outcomes.trials_path)
    if write:
        write_jsonl(records, snapshot_path)
        write_outcomes(outcomes, outcomes_path)
        write_bid_trials(trials, trials_path)

    n = max(len(outcomes), 1)
    payment = Counter(o.payment_outcome for o in outcomes)
    trial_wins = sum(1 for t in trials if t.won)
    return {
        "brokers": len(pool),
        "snapshots": len(records),
        "outcomes": len(outcomes),
        "trials": len(trials),
        "covered_rate": round(sum(o.covered for o in outcomes) / n, 4),
        "censored": sum(o.cover_censored for o in outcomes),
        "negotiation_required": sum(o.negotiation_required for o in outcomes),
        "payment_outcomes": dict(payment),
        "trial_win_rate": round(trial_wins / max(len(trials), 1), 4),
        "paths": {
            "snapshots": str(snapshot_path),
            "outcomes": str(outcomes_path),
            "trials": str(trials_path),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the Phase 4.1 labeled winnability dataset."
    )
    parser.add_argument("--config", default=None, help="Path to ml_config.yaml")
    parser.add_argument(
        "--days", type=int, default=None, help="Override generation horizon (smoke)."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the snapshot generation seed."
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_arg_parser().parse_args(argv)
    cfg = load_ml_config(args.config) if args.config else load_ml_config()
    summary = build_winnability_dataset(cfg, days=args.days, seed=args.seed)
    print("Phase 4.1 winnability dataset built:")
    for key in ("brokers", "snapshots", "outcomes", "trials"):
        print(f"  {key:20s} {summary[key]:>10,}")
    print(f"  covered_rate         {summary['covered_rate']:>10}")
    print(f"  censored             {summary['censored']:>10,}")
    print(f"  negotiation_required {summary['negotiation_required']:>10,}")
    print(f"  payment_outcomes     {summary['payment_outcomes']}")
    print(f"  trial_win_rate       {summary['trial_win_rate']:>10}")
    for label, path in summary["paths"].items():
        print(f"  -> {label:9s} {path}")


if __name__ == "__main__":
    main()

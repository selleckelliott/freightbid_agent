"""One-command reproduction entry point (Phase 3.5).

Non-destructive by default: regenerates the legible demo plus a *reduced* rolling
A/B into the gitignored ``benchmarks/reproduced/`` directory, prints a run-metadata
header and the consolidated benchmark table, and leaves ``git status`` clean.

Modes
-----
    python -m benchmarks.reproduce                     # fast smoke (~2-3 min) -> benchmarks/reproduced/
    python -m benchmarks.reproduce --update-artifacts  # refresh committed demo SVGs, then exit
    python -m benchmarks.reproduce --full              # canonical long benchmark (~70 min)

The fast path works from a clean checkout: the gitignored model artifact is absent
on a fresh clone, so it quick-trains a small seeded destination model into
``benchmarks/reproduced/`` (training is seconds) purely to drive the reduced
rolling chart. The committed canonical numbers in the table below are the source
of truth; the fast run is a smoke, not a canonical result.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPRODUCED = ROOT / "benchmarks" / "reproduced"
COMMITTED = ROOT / "benchmarks"
CANONICAL_MODEL = ROOT / "ml" / "artifacts" / "destination_desirability_model.joblib"

# Fast-path knobs (smoke, NOT canonical).
FAST_EPISODES = 8
FAST_HORIZON_DAYS = 3
FAST_TIME_LIMIT = 0.15
FAST_TRAIN_DAYS = 14
FAST_SEED = 1000  # matches config/rolling_replay.yaml base_seed

# Canonical knobs (--full).
FULL_REPLAY_EPISODES = 150
FULL_STRESS_EPISODES = 30

PROFIT_PLANNER = "ORToolsProfitAwarePlanner"
DEST_PLANNER = "ORToolsDestinationAwarePlanner"

# Consolidated "results at a glance" — sourced from the committed per-phase
# sections / summary JSONs (see README).
HEADLINE_ROWS = [
    ("Heuristic baseline", "rule-based scoring", "$396.38 profit / 11.3 mi DH / 88.1% feasible"),
    ("OR-Tools profit-aware", "CP-SAT, profit objective", "$396.79 profit / 12.0 mi DH"),
    ("OR-Tools deadhead-control", "tuned objective weights", "$392.97 profit / 7.4 mi DH (-34.3%)"),
    ("ML destination model", "Hurdle GBM", "MAE 49.3 vs 61.2 zone / <=50 mi 76%"),
    ("Destination-aware (one-shot)", "model-in-planner", "-12.9% deadhead at ~free profit"),
    ("Rolling replay (sequential)", "multi-day MPC A/B", "+3.9% profit / -4.7% deadhead (150 eps)"),
    ("Stress test (robustness)", "18 shifted markets", "0 regressions; HOLDS 7/18, neutral 11/18"),
]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _module(mod: str, *args: str) -> list[str]:
    return [sys.executable, "-m", mod, *args]


def _print_headline_table() -> None:
    w0 = max(len(r[0]) for r in HEADLINE_ROWS)
    w1 = max(len(r[1]) for r in HEADLINE_ROWS)
    rule = "-" * (w0 + w1 + 50)
    print("\nResults at a glance (committed canonical numbers)")
    print(rule)
    print(f"{'Layer':<{w0}}  {'Approach':<{w1}}  Headline result")
    print(rule)
    for layer, approach, result in HEADLINE_ROWS:
        print(f"{layer:<{w0}}  {approach:<{w1}}  {result}")
    print(rule)


def _print_metadata(mode: str, *, episodes: int, horizon_days: int,
                    time_limit: float, model_path: Path) -> None:
    have_model = model_path.exists()
    print("=" * 78)
    print("FreightBid Agent - reproduce")
    print("=" * 78)
    print(f"  mode:              {mode}")
    print(f"  seed (base):       {FAST_SEED}")
    print(f"  episodes:          {episodes}")
    print(f"  scenarios:         1 (baseline rolling condition)")
    print(f"  horizon (days):    {horizon_days}")
    print(f"  solver time limit: {time_limit}s/decision")
    print(f"  planners:          {PROFIT_PLANNER} vs {DEST_PLANNER}")
    print(f"  model artifact:    {model_path}  ({'present' if have_model else 'will quick-train'})")
    print("=" * 78)


def _quick_train(out_dir: Path) -> Path | None:
    """Quick-train a small seeded destination model into ``out_dir``."""
    model_path = out_dir / "destination_model.joblib"
    if model_path.exists():
        print(f"(reusing quick model at {model_path.relative_to(ROOT)})")
        return model_path
    try:
        from ml.config import load_ml_config
        from ml.training.train_destination_model import train
    except Exception as exc:  # pragma: no cover - defensive
        print(f"(skipping quick-train: {exc})")
        return None
    base = load_ml_config()
    cfg = replace(
        base,
        synthetic_data=replace(
            base.synthetic_data,
            days=FAST_TRAIN_DAYS,
            output_path=str(out_dir / "synthetic_history.jsonl"),
        ),
        artifacts=replace(
            base.artifacts,
            model_path=str(model_path),
            metadata_path=str(out_dir / "destination_model_metadata.json"),
        ),
    )
    print(f"\nQuick-training destination model ({FAST_TRAIN_DAYS}-day synthetic) -> "
          f"{model_path.relative_to(ROOT)}")
    train(cfg)
    return model_path


def _fast() -> None:
    REPRODUCED.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if CANONICAL_MODEL.exists():
        model_path = CANONICAL_MODEL
    else:
        model_path = REPRODUCED / "destination_model.joblib"

    _print_metadata(
        "fast (non-destructive -> benchmarks/reproduced/)",
        episodes=FAST_EPISODES, horizon_days=FAST_HORIZON_DAYS,
        time_limit=FAST_TIME_LIMIT, model_path=model_path,
    )

    # 1) Faithful CLI demo SVGs (no model needed).
    from benchmarks.render_demo import render_demo
    for p in render_demo(REPRODUCED):
        print(f"wrote {p.relative_to(ROOT)}")

    # 2) Ensure a model for the reduced rolling A/B (quick-train on a clean checkout).
    if model_path == CANONICAL_MODEL:
        print(f"(using canonical model at {model_path.relative_to(ROOT)})")
    else:
        _quick_train(REPRODUCED)

    # 3) Reduced rolling A/B + chart (best-effort; degrades to profit-aware only).
    summary = REPRODUCED / "rolling_replay_summary.json"
    chart = REPRODUCED / "rolling_replay_comparison.png"
    try:
        _run(_module(
            "benchmarks.run_rolling_replay",
            "--episodes", str(FAST_EPISODES),
            "--horizon-days", str(FAST_HORIZON_DAYS),
            "--time-limit", str(FAST_TIME_LIMIT),
            "--model-path", str(model_path),
            "--out", str(summary),
        ))
        _run(_module(
            "benchmarks.chart_rolling_replay",
            "--results", str(summary), "--out-png", str(chart),
        ))
    except subprocess.CalledProcessError as exc:
        print(f"(reduced rolling chart step failed, continuing: {exc})")

    _print_headline_table()
    print(f"\nFast reproduce complete in {time.perf_counter() - t0:.1f}s. "
          f"Outputs in {REPRODUCED.relative_to(ROOT)}/ (gitignored); git status stays clean.")


def _update_artifacts() -> None:
    from benchmarks.render_demo import render_demo
    print("Refreshing committed demo SVGs (benchmarks/demo_*.svg)...")
    for p in render_demo(COMMITTED):
        print(f"wrote {p.relative_to(ROOT)}")


def _ensure_canonical_model() -> None:
    if CANONICAL_MODEL.exists():
        return
    print("Training canonical destination model (full ml_config)...")
    _run(_module("ml.training.train_destination_model"))


def _full() -> None:
    t0 = time.perf_counter()
    _ensure_canonical_model()
    _run(_module("benchmarks.run_rolling_replay", "--episodes", str(FULL_REPLAY_EPISODES)))
    _run(_module("benchmarks.chart_rolling_replay"))
    _run(_module("benchmarks.run_stress_test", "--episodes", str(FULL_STRESS_EPISODES)))
    _run(_module("benchmarks.chart_stress_test"))
    _print_headline_table()
    print(f"\nFull reproduce complete in {(time.perf_counter() - t0) / 60:.1f} min. "
          "Committed canonical artifacts regenerated.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--update-artifacts", action="store_true",
                       help="regenerate the committed demo SVGs, then exit")
    group.add_argument("--full", action="store_true",
                       help="run the canonical long benchmark (~70 min)")
    args = parser.parse_args(argv)

    if args.update_artifacts:
        _update_artifacts()
    elif args.full:
        _full()
    else:
        _fast()


if __name__ == "__main__":
    main()

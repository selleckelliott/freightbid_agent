"""Objective tuning harness for the profit-aware planner (Phase 2.3).

Sweeps the profit-aware objective's two *independent* preference knobs and
maps the profit-vs-deadhead Pareto frontier against the heuristic baseline:

* ``deadhead_cost_multiplier`` — scales the cost-model-derived rate for empty
  miles (1.0 = true cost; higher = extra deadhead aversion).
* ``skip_profit_floor_dollars`` — solver pickiness: the static profit a load
  must clear before skipping it costs the solver anything. Swept only at or
  above the business ``min_expected_profit`` — a lower objective floor would
  reward serving loads the replay pipeline rejects.

The objective is ``sum(miles x D) + sum(margin x P)``: scaling ``D`` and
``P`` jointly cannot change the argmin, so only their *ratio* is worth
sweeping — a "skip penalty multiplier" axis would just duplicate this grid
(verify empirically with ``--invariance-check``). Solver time limit is held
constant across the sweep; runtime-vs-quality is a separate one-dimensional
question answered by ``--time-study``.

Named profiles from ``config/objective_profiles.yaml`` are annotated onto
their grid points. Pareto efficiency is computed over (avg profit, avg
deadhead) among configs with feasible rate >= 85%; the *recommended* config
is the knee: largest deadhead reduction costing < 2% of the best config's
profit.

Examples
--------
    # Smoke run over the first 100 scenarios:
    python -m benchmarks.tune_objective --limit 100

    # Full sweep, write JSON:
    python -m benchmarks.tune_objective --out benchmarks/tuning_results.json

    # Scaling-invariance demonstration (3 configs, 50 scenarios):
    python -m benchmarks.tune_objective --invariance-check

    # Append a time-limit study of the recommended config to existing JSON:
    python -m benchmarks.tune_objective --time-study --out benchmarks/tuning_results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from adapters.inbound.api.container import build_container
from application.config_loader import load_objective_profiles
from application.ortools_distance_planner import ORToolsDistancePlanner
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from domain.policies.ortools_objective_weights import ORToolsObjectiveWeights

from .compare_planners import DEFAULT_SCENARIO_DIR, ROOT, _load_scenarios, _run_planner
from .pareto import pareto_flags

GRID_MULTIPLIERS = [0.75, 1.0, 1.25, 1.6, 2.0, 2.5]
GRID_FLOORS = [50.0, 75.0, 100.0, 150.0]
MIN_FEASIBLE_RATE = 0.85
KNEE_PROFIT_TOLERANCE = 0.02  # max profit sacrifice (vs best config) at the knee
TIME_STUDY_LIMITS = [0.1, 0.2, 0.5, 1.0]


def _build_profit_planner(container, weights: ORToolsObjectiveWeights,
                          time_limit: float) -> ORToolsProfitAwarePlanner:
    return ORToolsProfitAwarePlanner(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        objective_weights=weights,
        solver_time_limit_seconds=time_limit,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )


def _weights_for(container, multiplier: float, floor: float,
                 profit_multiplier: float = 1.0) -> ORToolsObjectiveWeights:
    return ORToolsObjectiveWeights.from_cost_model(
        container.config.cost_model,
        container.config.average_speed_mph,
        deadhead_cost_multiplier=multiplier,
        profit_multiplier=profit_multiplier,
        skip_profit_floor_dollars=floor,
    )


def _deltas(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, float]:
    def pct(key: str) -> float:
        base = baseline[key]
        return 0.0 if base == 0 else (metrics[key] - base) / abs(base) * 100.0

    return {
        "profit_delta_pct": round(pct("avg_profit"), 2),
        "deadhead_delta_pct": round(pct("avg_deadhead_miles"), 2),
        "feasible_delta_pct": round(pct("feasible_rate"), 2),
        "loads_delta_pct": round(pct("avg_loads_selected"), 2),
    }


def _mark_recommended(configs: List[Dict[str, Any]]) -> None:
    """Knee rule: among Pareto-efficient configs, pick the one with the least
    deadhead whose profit is within KNEE_PROFIT_TOLERANCE of the best
    config's profit."""
    front = [c for c in configs if c["pareto_efficient"]]
    for c in configs:
        c["recommended"] = False
    if not front:
        return
    best_profit = max(c["metrics"]["avg_profit"] for c in front)
    knee_pool = [
        c for c in front
        if c["metrics"]["avg_profit"] >= best_profit * (1 - KNEE_PROFIT_TOLERANCE)
    ]
    knee = min(knee_pool, key=lambda c: c["metrics"]["avg_deadhead_miles"])
    knee["recommended"] = True


def _print_table(configs: List[Dict[str, Any]], baseline: Dict[str, Any]) -> None:
    print(f"\n{'mult':>5} {'floor':>6} {'profit':>9} {'d-prof':>8} {'deadhd':>7} "
          f"{'d-dead':>8} {'feas':>6} {'loads':>6}  flags / profile")
    print("-" * 78)
    for c in configs:
        m, d = c["metrics"], c["deltas"]
        flags = ("*" if c["pareto_efficient"] else " ") + \
                (">" if c.get("recommended") else " ")
        name = c["profile_name"] or ""
        print(f"{c['params']['deadhead_cost_multiplier']:>5.2f} "
              f"{c['params']['skip_profit_floor_dollars']:>6.0f} "
              f"{m['avg_profit']:>9.2f} {d['profit_delta_pct']:>+7.1f}% "
              f"{m['avg_deadhead_miles']:>7.1f} {d['deadhead_delta_pct']:>+7.1f}% "
              f"{m['feasible_rate']*100:>5.1f}% {m['avg_loads_selected']:>6.2f}  "
              f"{flags} {name}")
    print("-" * 78)
    print(f"baseline heuristic: profit ${baseline['avg_profit']:.2f}, "
          f"deadhead {baseline['avg_deadhead_miles']:.1f} mi, "
          f"feasible {baseline['feasible_rate']*100:.1f}%")
    print("flags: * = Pareto-efficient (feasible >= "
          f"{MIN_FEASIBLE_RATE:.0%}), > = recommended (knee)")


def run_sweep(args) -> None:
    scenarios = _load_scenarios(Path(args.scenario_dir), args.limit)
    container = build_container(ROOT / "config")
    profiles = load_objective_profiles(ROOT / "config")
    profile_by_point = {
        (p.deadhead_cost_multiplier, p.skip_profit_floor_dollars): p.name
        for p in profiles.values()
    }

    print(f"Heuristic baseline over {len(scenarios)} scenarios...")
    heuristic = _run_planner(container.planner.build_plan, scenarios)

    print("OR-Tools distance baseline...")
    distance_planner = ORToolsDistancePlanner(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        solver_time_limit_seconds=args.time_limit,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )
    distance = _run_planner(distance_planner.build_plan, scenarios)

    if args.profiles_only:
        points = sorted(profile_by_point)
    else:
        points = [(m, f) for m in GRID_MULTIPLIERS for f in GRID_FLOORS]

    configs: List[Dict[str, Any]] = []
    for i, (multiplier, floor) in enumerate(points, 1):
        name = profile_by_point.get((multiplier, floor))
        tag = f" [{name}]" if name else ""
        print(f"[{i:>2}/{len(points)}] multiplier={multiplier:g} "
              f"floor=${floor:g}{tag}...", flush=True)
        weights = _weights_for(container, multiplier, floor)
        planner = _build_profit_planner(container, weights, args.time_limit)
        metrics = _run_planner(planner.build_plan, scenarios)
        configs.append({
            "profile_name": name,
            "params": {
                "deadhead_cost_multiplier": multiplier,
                "skip_profit_floor_dollars": floor,
                "deadhead_cost_cents_per_mile": weights.deadhead_cost_cents_per_mile,
                "profit_cents_multiplier": weights.profit_cents_multiplier,
            },
            "metrics": metrics,
            "deltas": _deltas(metrics, heuristic),
        })

    flags = pareto_flags(
        [c["metrics"] for c in configs], min_feasible_rate=MIN_FEASIBLE_RATE
    )
    for config, flag in zip(configs, flags):
        config["pareto_efficient"] = flag
    _mark_recommended(configs)

    _print_table(configs, heuristic)

    if args.out:
        payload = {
            "scenario_count": len(scenarios),
            "ortools_time_limit_s": args.time_limit,
            "min_feasible_rate": MIN_FEASIBLE_RATE,
            "knee_profit_tolerance": KNEE_PROFIT_TOLERANCE,
            "baselines": [
                {"key": "heuristic", "label": "Heuristic", "metrics": heuristic},
                {"key": "ortools_distance", "label": "OR-Tools Distance",
                 "metrics": distance},
            ],
            "configs": configs,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote tuning results to {args.out}")


def run_invariance_check(args) -> None:
    """Demonstrate that the objective is scale-invariant: doubling *both*
    rates yields identical plans, doubling only one changes them."""
    limit = args.limit or 50
    scenarios = _load_scenarios(Path(args.scenario_dir), limit)
    container = build_container(ROOT / "config")

    cases = [
        ("base        (D=1.0x, P=1.0x)", 1.0, 1.0),
        ("joint scale (D=3.0x, P=3.0x)", 3.0, 3.0),
        ("ratio change (D=3.0x, P=1.0x)", 3.0, 1.0),
    ]
    results = []
    for label, dead_mult, profit_mult in cases:
        weights = _weights_for(container, dead_mult, 50.0,
                               profit_multiplier=profit_mult)
        planner = _build_profit_planner(container, weights, args.time_limit)
        metrics = _run_planner(planner.build_plan, scenarios)
        results.append((label, metrics))
        print(f"{label}: profit ${metrics['avg_profit']:.2f}, "
              f"deadhead {metrics['avg_deadhead_miles']:.2f} mi, "
              f"loads {metrics['avg_loads_selected']:.3f}")

    keys = ("avg_profit", "avg_deadhead_miles", "avg_loads_selected")
    base, joint, ratio = (m for _label, m in results)
    same = all(abs(base[k] - joint[k]) < 1e-9 for k in keys)
    different = any(abs(base[k] - ratio[k]) > 1e-9 for k in keys)
    print()
    print(f"joint scaling preserved every plan:  {'YES' if same else 'NO'}")
    print(f"ratio change altered plans:          {'YES' if different else 'NO'}")
    if same and different:
        print("=> only the D/P *ratio* matters; a separate skip-penalty "
              "multiplier axis would be redundant.")
    else:
        raise SystemExit("Invariance check FAILED - investigate before sweeping.")


def run_time_study(args) -> None:
    """Re-run the recommended config at several solver time limits."""
    out_path = Path(args.out or ROOT / "benchmarks" / "tuning_results.json")
    if not out_path.exists():
        raise SystemExit(f"{out_path} not found - run the sweep first.")
    payload = json.loads(out_path.read_text())
    recommended = next(
        (c for c in payload["configs"] if c.get("recommended")), None
    )
    if recommended is None:
        raise SystemExit("No recommended config in results - rerun the sweep.")
    params = recommended["params"]

    limit = args.limit or 200
    scenarios = _load_scenarios(Path(args.scenario_dir), limit)
    container = build_container(ROOT / "config")
    weights = _weights_for(
        container,
        params["deadhead_cost_multiplier"],
        params["skip_profit_floor_dollars"],
    )

    name = recommended["profile_name"] or "recommended"
    print(f"Time study for {name} "
          f"(mult={params['deadhead_cost_multiplier']:g}, "
          f"floor=${params['skip_profit_floor_dollars']:g}) "
          f"over {len(scenarios)} scenarios")
    points = []
    for t in TIME_STUDY_LIMITS:
        planner = _build_profit_planner(container, weights, t)
        metrics = _run_planner(planner.build_plan, scenarios)
        points.append({"time_limit_s": t, "metrics": metrics})
        print(f"  {t:>4.1f}s: profit ${metrics['avg_profit']:.2f}, "
              f"deadhead {metrics['avg_deadhead_miles']:.1f} mi, "
              f"runtime {metrics['avg_runtime_ms']:.0f} ms")

    payload["time_study"] = {
        "profile_name": recommended["profile_name"],
        "params": params,
        "scenario_count": len(scenarios),
        "points": points,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Appended time study to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N scenarios.")
    parser.add_argument("--time-limit", type=float, default=0.2,
                        help="OR-Tools solver time limit per scenario (seconds).")
    parser.add_argument("--out", type=str, default=None,
                        help="Write tuning results to this JSON file.")
    parser.add_argument("--profiles-only", action="store_true",
                        help="Run only the named profiles, not the full grid.")
    parser.add_argument("--invariance-check", action="store_true",
                        help="Run the 3-config scaling-invariance demonstration.")
    parser.add_argument("--time-study", action="store_true",
                        help="Append a solver-time-limit study of the recommended "
                             "config to the existing results JSON.")
    args = parser.parse_args()

    if args.invariance_check:
        run_invariance_check(args)
    elif args.time_study:
        run_time_study(args)
    else:
        run_sweep(args)


if __name__ == "__main__":
    main()

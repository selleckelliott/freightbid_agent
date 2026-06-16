"""Head-to-head comparison: heuristic vs OR-Tools distance / profit / destination.

Runs every planner over the generated scenario suite and reports aggregate
plan-quality and runtime metrics side by side. Same scenarios, same load inputs,
same cost accounting (every planner's financials come from
``EvaluateLoadsService``), so the only difference is load selection/sequencing.

Note on objectives: ``ORToolsDistancePlanner`` minimises travel *distance*
(and therefore deadhead) subject to time windows, pickup-before-delivery and
the driver's HOS budget. ``ORToolsProfitAwarePlanner`` (Phase 2.2) instead
minimises *negative expected profit* — cost-model-priced deadhead plus
profit-proportional penalties for skipping loads — so it may decline freight
that is not worth its empty miles. ``ORToolsDestinationAwarePlanner`` (Phase
3.2) adds one more term: each load is additionally discounted by the Phase 3.1
model's *expected onward-deadhead cost*, so it declines freight that would
strand the truck in a weak market even when the immediate trip is profitable.
The four-way result is a clean ablation: heuristic baseline, distance objective,
business objective, business objective + learned destination desirability. The
destination-aware column appears only when the trained model artifact exists
locally (it is gitignored).

Examples
--------
    # Compare over the first 200 scenarios (fast), OR-Tools 0.3s/solve:
    python -m benchmarks.compare_planners --limit 200 --time-limit 0.3

    # Full 1000-scenario comparison, write JSON:
    python -m benchmarks.compare_planners --time-limit 0.3 --out benchmarks/compare_results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import load_from_dto, truck_from_dto
from adapters.inbound.api.schemas import LoadDTO, TruckStateDTO
from application.ortools_distance_planner import ORToolsDistancePlanner
from application.ortools_profit_aware_planner import ORToolsProfitAwarePlanner
from application.ortools_destination_aware_planner import (
    ORToolsDestinationAwarePlanner,
)
from application.destination_desirability_service import (
    DestinationDesirabilityService,
)
from domain.models.load import Load
from domain.models.plan import Plan
from domain.models.truck_state import TruckState

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = ROOT / "benchmarks" / "scenarios" / "gen"


def _load_scenarios(scenario_dir: Path, limit: int | None) -> List[Tuple[str, TruckState, List[Load]]]:
    files = sorted(scenario_dir.glob("scenario_*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(
            f"No scenarios in {scenario_dir}. Generate with:\n"
            "  python -m benchmarks.scenario_generator --count 1000 --seed 42 "
            "--out-dir benchmarks/scenarios/gen"
        )
    scenarios: List[Tuple[str, TruckState, List[Load]]] = []
    for path in files:
        doc = json.loads(path.read_text())
        truck = truck_from_dto(TruckStateDTO(**doc["truck"]))
        loads = [load_from_dto(LoadDTO(**l)) for l in doc["loads"]]
        scenarios.append((path.stem, truck, loads))
    return scenarios


def _run_planner(
    build_plan: Callable[[List[Load], TruckState], Plan],
    scenarios: List[Tuple[str, TruckState, List[Load]]],
) -> Dict[str, Any]:
    profits: List[float] = []
    deadheads: List[float] = []
    loads_selected: List[int] = []
    runtimes_ms: List[float] = []
    n_feasible = 0

    for _name, truck, loads in scenarios:
        t0 = time.perf_counter()
        plan = build_plan(loads, truck)
        runtimes_ms.append((time.perf_counter() - t0) * 1000.0)

        profits.append(plan.expected_profit)
        deadheads.append(plan.expected_deadhead_miles)
        loads_selected.append(len(plan.stops))
        if plan.feasible:
            n_feasible += 1

    n = len(scenarios)
    return {
        "scenarios": n,
        "feasible_rate": n_feasible / n if n else 0.0,
        "avg_profit": statistics.mean(profits) if profits else 0.0,
        "total_profit": sum(profits),
        "avg_deadhead_miles": statistics.mean(deadheads) if deadheads else 0.0,
        "avg_loads_selected": statistics.mean(loads_selected) if loads_selected else 0.0,
        "total_loads_selected": sum(loads_selected),
        "avg_runtime_ms": statistics.mean(runtimes_ms) if runtimes_ms else 0.0,
        "median_runtime_ms": statistics.median(runtimes_ms) if runtimes_ms else 0.0,
    }


def _print_block(title: str, m: Dict[str, Any]) -> None:
    print(f"{title}:")
    print(f"  Feasible rate:      {m['feasible_rate'] * 100:5.1f}%")
    print(f"  Avg profit:         ${m['avg_profit']:,.2f}")
    print(f"  Avg deadhead:       {m['avg_deadhead_miles']:,.1f} mi")
    print(f"  Avg loads selected: {m['avg_loads_selected']:.2f}")
    print(f"  Total loads:        {m['total_loads_selected']}")
    print(f"  Avg runtime:        {m['avg_runtime_ms']:.2f} ms "
          f"(median {m['median_runtime_ms']:.2f} ms)")


def _pct_delta(new: float, old: float) -> str:
    if old == 0:
        return "  n/a"
    return f"{(new - old) / abs(old) * 100:+6.1f}%"


def _print_delta(title: str, new: Dict[str, Any], old: Dict[str, Any]) -> None:
    print(f"Delta ({title}):")
    print(f"  Profit:    {_pct_delta(new['avg_profit'], old['avg_profit'])}")
    print(f"  Deadhead:  {_pct_delta(new['avg_deadhead_miles'], old['avg_deadhead_miles'])} "
          f"(lower is better)")
    print(f"  Loads:     {_pct_delta(new['avg_loads_selected'], old['avg_loads_selected'])}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N scenarios.")
    parser.add_argument("--time-limit", type=float, default=0.3,
                        help="OR-Tools solver time limit per scenario (seconds).")
    parser.add_argument("--out", type=str, default=None, help="Write metrics to this JSON file.")
    args = parser.parse_args()

    scenarios = _load_scenarios(Path(args.scenario_dir), args.limit)
    container = build_container(ROOT / "config")

    ortools_kwargs = dict(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        solver_time_limit_seconds=args.time_limit,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )
    planners: List[Tuple[str, str, Callable[[List[Load], TruckState], Plan]]] = [
        ("heuristic", "Heuristic", container.planner.build_plan),
        (
            "ortools_distance",
            "OR-Tools Distance",
            ORToolsDistancePlanner(**ortools_kwargs).build_plan,
        ),
        (
            "ortools_profit_aware",
            "OR-Tools Profit-Aware",
            ORToolsProfitAwarePlanner(
                objective_weights=container.config.ortools_objective_weights,
                **ortools_kwargs,
            ).build_plan,
        ),
    ]

    # Phase 3.2: a destination-aware planner that additionally discounts each
    # load by the Phase 3.1 model's expected onward-deadhead cost. Artifact-
    # gated — the trained ``.joblib`` is gitignored, so the comparison still
    # runs (three-way) wherever the model has not been built locally.
    model_path = ROOT / "ml" / "artifacts" / "destination_desirability_model.joblib"
    if model_path.exists():
        service = DestinationDesirabilityService.from_artifact(model_path)
        planners.append(
            (
                "ortools_destination_aware",
                "OR-Tools Destination-Aware",
                ORToolsDestinationAwarePlanner(
                    objective_weights=container.config.ortools_objective_weights,
                    destination_service=service,
                    **ortools_kwargs,
                ).build_plan,
            )
        )
    else:
        print(
            f"(destination-aware planner skipped: no model artifact at {model_path}; "
            "train it with `python -m ml.training.train_destination_model`)"
        )

    print("=" * 78)
    print(f"Planner comparison over {len(scenarios)} scenarios "
          f"(OR-Tools {args.time_limit:.2f}s/solve)")
    print("=" * 78)

    results: Dict[str, Dict[str, Any]] = {}
    labels: Dict[str, str] = {}
    for key, label, build_plan in planners:
        results[key] = _run_planner(build_plan, scenarios)
        labels[key] = label
        _print_block(label, results[key])
        print("-" * 78)

    _print_delta("OR-Tools Distance vs Heuristic",
                 results["ortools_distance"], results["heuristic"])
    _print_delta("OR-Tools Profit-Aware vs Heuristic",
                 results["ortools_profit_aware"], results["heuristic"])
    _print_delta("OR-Tools Profit-Aware vs OR-Tools Distance",
                 results["ortools_profit_aware"], results["ortools_distance"])
    if "ortools_destination_aware" in results:
        _print_delta("OR-Tools Destination-Aware vs OR-Tools Profit-Aware",
                     results["ortools_destination_aware"], results["ortools_profit_aware"])
    print("=" * 78)

    if args.out:
        payload = {
            "scenarios": len(scenarios),
            "ortools_time_limit_s": args.time_limit,
            "planners": [
                {"key": key, "label": labels[key], "metrics": results[key]}
                for key, _label, _fn in planners
            ],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()

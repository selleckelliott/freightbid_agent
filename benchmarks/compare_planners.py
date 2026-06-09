"""Head-to-head comparison: heuristic planner vs OR-Tools planner.

Runs both planners over the generated scenario suite and reports aggregate
plan-quality and runtime metrics side by side. Same scenarios, same load
inputs, same cost accounting (both planners' financials come from
``EvaluateLoadsService``), so the only difference is load selection/sequencing.

Note on objective: the Phase 2 OR-Tools planner minimises travel *distance*
(and therefore deadhead) subject to time windows, pickup-before-delivery and
the driver's HOS budget. It does **not** optimise profit yet, so the headline
expected win is reduced deadhead miles; profit parity-or-better is a bonus.

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
from application.ortools_planner import ORToolsPlanner
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

    ortools_planner = ORToolsPlanner(
        distance_provider=container.evaluator.distance_provider,
        evaluate_loads_service=container.evaluator,
        constraints=container.config.planning_constraints,
        solver_time_limit_seconds=args.time_limit,
        average_speed_mph=container.config.average_speed_mph,
        load_unload_hours=container.config.planning_constraints.average_load_unload_hours,
    )

    print("=" * 78)
    print(f"Planner comparison over {len(scenarios)} scenarios "
          f"(OR-Tools {args.time_limit:.2f}s/solve)")
    print("=" * 78)

    heuristic = _run_planner(container.planner.build_plan, scenarios)
    ortools = _run_planner(ortools_planner.build_plan, scenarios)

    _print_block("HeuristicPlanner", heuristic)
    print("-" * 78)
    _print_block("ORToolsPlanner", ortools)
    print("-" * 78)
    print("Delta (OR-Tools vs Heuristic):")
    print(f"  Profit:    {_pct_delta(ortools['avg_profit'], heuristic['avg_profit'])}")
    print(f"  Deadhead:  {_pct_delta(ortools['avg_deadhead_miles'], heuristic['avg_deadhead_miles'])} "
          f"(lower is better)")
    print(f"  Loads:     {_pct_delta(ortools['avg_loads_selected'], heuristic['avg_loads_selected'])}")
    print("=" * 78)

    if args.out:
        payload = {
            "scenarios": len(scenarios),
            "ortools_time_limit_s": args.time_limit,
            "heuristic": heuristic,
            "ortools": ortools,
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"Wrote metrics to {args.out}")


if __name__ == "__main__":
    main()

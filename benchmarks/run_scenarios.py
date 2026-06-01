"""Run generated scenarios through the recommend pipeline and report results.

Unlike ``bench_scenarios.py`` (which only measures timing), this script
actually executes each scenario and surfaces the ranked output so you can
inspect correctness, feasibility rate, profit distribution, etc.

Examples
--------

    # Process all generated scenarios, print a summary table:
    python -m benchmarks.run_scenarios

    # Show the top-N ranked loads for the first 3 scenarios:
    python -m benchmarks.run_scenarios --limit 3 --show-ranked

    # Dump full per-scenario results to JSON:
    python -m benchmarks.run_scenarios --out benchmarks/results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import load_from_dto, truck_from_dto
from adapters.inbound.api.schemas import LoadDTO, TruckStateDTO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = ROOT / "benchmarks" / "scenarios" / "gen"


def _load_scenario(path: Path) -> Dict[str, Any]:
    doc = json.loads(path.read_text())
    return {
        "name": doc.get("name", path.stem),
        "truck": truck_from_dto(TruckStateDTO(**doc["truck"])),
        "loads": [load_from_dto(LoadDTO(**l)) for l in doc["loads"]],
        "top_n": int(doc.get("top_n", 10)),
        "n_loads": len(doc["loads"]),
    }


def _run(scenario_dir: Path, limit: int | None) -> List[Dict[str, Any]]:
    container = build_container(ROOT / "config")
    files = sorted(scenario_dir.glob("scenario_*.json"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(
            f"No scenarios in {scenario_dir}. Generate with:\n"
            "  python -m benchmarks.scenario_generator --count 1000 --seed 42 "
            "--out-dir benchmarks/scenarios/gen"
        )

    results: List[Dict[str, Any]] = []
    for path in files:
        sc = _load_scenario(path)
        t0 = time.perf_counter()
        ranked = container.recommender.recommend_loads(
            sc["loads"], sc["truck"], sc["top_n"]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        ranked_payload = []
        for evaluation, score in ranked:
            ranked_payload.append(
                {
                    "load_id": score.load_id,
                    "score": round(score.score, 2),
                    "expected_profit": round(score.expected_profit, 2),
                    "expected_revenue": round(score.expected_revenue, 2),
                    "rate_per_mile": round(score.rate_per_mile, 2),
                    "deadhead_miles": round(score.deadhead_miles, 1),
                    "driver_hours": round(score.driver_hours, 2),
                    "origin": f"{evaluation.load.origin_city}, {evaluation.load.origin_state}",
                    "destination": (
                        f"{evaluation.load.destination_city}, "
                        f"{evaluation.load.destination_state}"
                    ),
                }
            )

        results.append(
            {
                "scenario": sc["name"],
                "truck_origin": f"{sc['truck'].current_city}, {sc['truck'].current_state}",
                "n_loads_in": sc["n_loads"],
                "n_loads_ranked": len(ranked),
                "top_n": sc["top_n"],
                "feasibility_rate": round(
                    len(ranked) / sc["n_loads"] if sc["n_loads"] else 0.0, 3
                ),
                "best_score": ranked_payload[0]["score"] if ranked_payload else None,
                "best_profit": ranked_payload[0]["expected_profit"] if ranked_payload else None,
                "elapsed_ms": round(elapsed_ms, 3),
                "ranked": ranked_payload,
            }
        )

    return results


def _print_summary(results: List[Dict[str, Any]]) -> None:
    n = len(results)
    times = [r["elapsed_ms"] for r in results]
    feas = [r["feasibility_rate"] for r in results]
    best_scores = [r["best_score"] for r in results if r["best_score"] is not None]
    best_profits = [r["best_profit"] for r in results if r["best_profit"] is not None]
    no_feasible = sum(1 for r in results if r["n_loads_ranked"] == 0)

    print("=" * 78)
    print(f"Scenarios run:           {n}")
    print(f"Scenarios w/ 0 feasible: {no_feasible}")
    print(f"Total loads ingested:    {sum(r['n_loads_in'] for r in results)}")
    print(f"Total loads ranked:      {sum(r['n_loads_ranked'] for r in results)}")
    print("-" * 78)
    print(f"Per-scenario elapsed ms: min={min(times):.2f}  "
          f"median={statistics.median(times):.2f}  "
          f"mean={statistics.mean(times):.2f}  max={max(times):.2f}")
    print(f"Feasibility rate:        min={min(feas):.2f}  "
          f"median={statistics.median(feas):.2f}  "
          f"mean={statistics.mean(feas):.2f}  max={max(feas):.2f}")
    if best_scores:
        print(f"Best score:              min={min(best_scores):.2f}  "
              f"median={statistics.median(best_scores):.2f}  "
              f"mean={statistics.mean(best_scores):.2f}  max={max(best_scores):.2f}")
        print(f"Best expected profit:    min=${min(best_profits):.2f}  "
              f"median=${statistics.median(best_profits):.2f}  "
              f"mean=${statistics.mean(best_profits):.2f}  "
              f"max=${max(best_profits):.2f}")
    print("=" * 78)


def _print_ranked(results: List[Dict[str, Any]]) -> None:
    for r in results:
        print(f"\n[{r['scenario']}] truck @ {r['truck_origin']}  "
              f"({r['n_loads_in']} loads in -> {r['n_loads_ranked']} feasible, "
              f"top_n={r['top_n']}, {r['elapsed_ms']:.2f} ms)")
        if not r["ranked"]:
            print("  (no feasible loads)")
            continue
        print(f"  {'rank':>4}  {'load':>5}  {'score':>8}  {'profit':>10}  "
              f"{'rpm':>6}  {'dh_mi':>6}  lane")
        for i, row in enumerate(r["ranked"], 1):
            print(f"  {i:>4}  {row['load_id']:>5}  {row['score']:>8.2f}  "
                  f"${row['expected_profit']:>9.2f}  ${row['rate_per_mile']:>5.2f}  "
                  f"{row['deadhead_miles']:>6.1f}  "
                  f"{row['origin']} -> {row['destination']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario-dir", default=str(DEFAULT_SCENARIO_DIR))
    p.add_argument("--limit", type=int, default=None, help="Only run the first N scenarios.")
    p.add_argument("--out", type=str, default=None, help="Write full results to this JSON file.")
    p.add_argument("--show-ranked", action="store_true",
                   help="Print the per-scenario ranked tables in addition to the summary.")
    args = p.parse_args()

    results = _run(Path(args.scenario_dir), args.limit)

    if args.show_ranked:
        _print_ranked(results)
    _print_summary(results)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Wrote full results to {args.out}")


if __name__ == "__main__":
    main()

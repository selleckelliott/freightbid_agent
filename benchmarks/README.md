# Benchmarks

Performance harness for the FreightBid Agent. Kept separate from `tests/`
so perf runs don't gate correctness CI.

---

## Planner Comparison — Heuristic vs OR-Tools (Phase 2)

1,000 scenarios, OR-Tools capped at 0.2 s/solve:

| Metric | Heuristic | OR-Tools Distance | OR-Tools Profit-Aware |
|---|---|---|---|
| Feasible rate | 88.1% | 82.4% | 88.1% |
| Avg profit / scenario | $396.38 | $240.14 (−39.4%) | **$396.79 (+0.1%)** |
| Avg deadhead miles | 11.3 | **8.9 (−20.9%)** | 12.0 (+6.6%) |
| Avg loads selected | 0.91 | 0.88 | 0.91 |
| Avg runtime / scenario | 0.17 ms | 199 ms | 197 ms |

Chart: [`compare_chart.png`](compare_chart.png)  
Raw data: [`compare_results.json`](compare_results.json)

> Minimizing distance saves deadhead but craters profit; switching the solver
> objective to cost-model-derived expected profit recovers it. See the main
> README's Phase 2 section for the objective formulation.

---

## Current Baseline — `heuristic_baseline_1000`

| Metric | Value |
|---|---|
| Scenarios | 1 000 |
| Scenarios w/ 0 feasible loads | 119 |
| Mean Feasibility Rate | 22% |
| Mean Expected Profit (best load) | $445 |
| Median Expected Profit (best load) | $439 |
| Mean Latency per Scenario | 0.13 ms |

Chart: [`results/heuristic_baseline_1000.png`](results/heuristic_baseline_1000.png)  
Raw data: [`results/heuristic_baseline_1000.json`](results/heuristic_baseline_1000.json)

> Origins: Intermountain West + West Coast cities.  
> Loads biased within 200 mi of truck, 85% trailer-type match, max lane 450 mi.

---

## Setup

```bash
pip install pytest-benchmark matplotlib numpy
```

## Running

From the repo root (`FreightBid_Agent/`):

```bash
# micro-benchmarks (timing only)
python -m pytest benchmarks/ --benchmark-only -v

# run 1 000 scenarios and print summary
python -m benchmarks.run_scenarios

# dump results to JSON
python -m benchmarks.run_scenarios --out benchmarks/results.json

# generate charts from results JSON
python -m benchmarks.chart_results --results benchmarks/results.json --show

# three-way planner comparison (heuristic vs OR-Tools distance vs profit-aware)
python -m benchmarks.compare_planners --time-limit 0.2 --out benchmarks/compare_results.json

# chart the comparison
python -m benchmarks.chart_comparison --results benchmarks/compare_results.json

# regenerate 1 000 scenarios (reproducible with --seed)
python -m benchmarks.scenario_generator --count 1000 --seed 42 \
    --out-dir benchmarks/scenarios/gen
```

## Layout

| File | Purpose |
|---|---|
| `conftest.py` | Shared pytest fixtures (synthetic data) |
| `bench_scoring.py` | Micro-benchmarks for `HeuristicScoringStrategy` |
| `bench_scenarios.py` | Scenario-driven pytest-benchmark tests |
| `run_scenarios.py` | Run scenarios, print ranked results + summary |
| `chart_results.py` | Visualize results JSON → 6-panel PNG chart |
| `compare_planners.py` | Heuristic vs OR-Tools planners over all scenarios |
| `chart_comparison.py` | Visualize comparison JSON → grouped-bar PNG |
| `scenario_generator.py` | Generate random realistic scenarios |
| `scenarios/scenario_001.json` | Hand-crafted canonical example |
| `scenarios/gen/` | Generated scenarios (gitignored — reproducible) |
| `results/` | Saved baselines (JSON + PNG) |

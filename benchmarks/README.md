# Benchmarks

Performance harness for the FreightBid Agent. Kept separate from `tests/`
so perf runs don't gate correctness CI.

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
| `scenario_generator.py` | Generate random realistic scenarios |
| `scenarios/scenario_001.json` | Hand-crafted canonical example |
| `scenarios/gen/` | Generated scenarios (gitignored — reproducible) |
| `results/` | Saved baselines (JSON + PNG) |

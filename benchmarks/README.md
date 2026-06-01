# Benchmarks

Performance harness for the FreightBid Agent. Kept separate from `tests/`
so perf runs don't gate correctness CI.

## Setup

```bash
pip install pytest-benchmark
```

## Running

From the repo root (`FreightBid_Agent/`):

```bash
# run all benchmarks
pytest benchmarks/ --benchmark-only

# compare against a saved baseline
pytest benchmarks/ --benchmark-only --benchmark-autosave
pytest benchmarks/ --benchmark-only --benchmark-compare

# scope to one module
pytest benchmarks/bench_scoring.py --benchmark-only -v
```

## Layout

- `conftest.py` — shared fixtures: loaded `AppConfig`, scoring strategy,
  synthetic `LoadEvaluation` generators parameterized by dataset size.
- `bench_scoring.py` — micro-benchmarks for `HeuristicScoringStrategy`.

Add new `bench_*.py` modules for additional surfaces
(`bench_evaluate.py`, `bench_recommend.py`, `bench_api.py`, ...).

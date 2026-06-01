"""Benchmarks that drive the full recommend pipeline over generated scenarios.

Run with::

    # Generate scenarios first (one-time, deterministic):
    python -m benchmarks.scenario_generator --count 1000 --seed 42 \\
        --out-dir benchmarks/scenarios/gen

    # Then benchmark them:
    python -m pytest benchmarks/bench_scenarios.py --benchmark-only

Override the directory or sample size with env vars:

    $env:FREIGHTBID_SCENARIO_DIR = "benchmarks/scenarios/gen"
    $env:FREIGHTBID_SCENARIO_SAMPLE = "100"   # bench a subset
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

import pytest

from adapters.inbound.api.container import build_container
from adapters.inbound.api.mappers import load_from_dto, truck_from_dto
from adapters.inbound.api.schemas import LoadDTO, TruckStateDTO
from domain.models.load import Load
from domain.models.truck_state import TruckState

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_DIR = ROOT / "benchmarks" / "scenarios" / "gen"


def _scenario_dir() -> Path:
    return Path(os.environ.get("FREIGHTBID_SCENARIO_DIR", DEFAULT_SCENARIO_DIR))


def _scenario_files() -> List[Path]:
    d = _scenario_dir()
    if not d.exists():
        return []
    files = sorted(d.glob("scenario_*.json"))
    sample = os.environ.get("FREIGHTBID_SCENARIO_SAMPLE")
    if sample:
        files = files[: int(sample)]
    return files


@pytest.fixture(scope="session")
def container():
    return build_container(ROOT / "config")


@pytest.fixture(scope="session")
def scenarios() -> List[Tuple[str, TruckState, List[Load], int]]:
    """Load and pre-parse every scenario file into domain objects.

    Parsing happens once so benchmarks measure pipeline work, not JSON I/O.
    """
    parsed: List[Tuple[str, TruckState, List[Load], int]] = []
    for path in _scenario_files():
        doc = json.loads(path.read_text())
        truck = truck_from_dto(TruckStateDTO(**doc["truck"]))
        loads = [load_from_dto(LoadDTO(**l)) for l in doc["loads"]]
        top_n = int(doc.get("top_n", 10))
        parsed.append((path.stem, truck, loads, top_n))
    return parsed


def _require_scenarios(scenarios):
    if not scenarios:
        pytest.skip(
            f"No scenarios in {_scenario_dir()}. Generate them with: "
            "python -m benchmarks.scenario_generator --count 1000 --seed 42 "
            "--out-dir benchmarks/scenarios/gen"
        )


def test_recommend_single_scenario(benchmark, container, scenarios):
    """Time one recommend_loads call on the first scenario (warm path)."""
    _require_scenarios(scenarios)
    _name, truck, loads, top_n = scenarios[0]

    ranked = benchmark(container.recommender.recommend_loads, loads, truck, top_n)
    assert len(ranked) <= top_n


def test_recommend_all_scenarios(benchmark, container, scenarios):
    """End-to-end: rank every scenario in the batch.

    This is the headline number — total time to process the full scenario
    suite through the production recommend pipeline.
    """
    _require_scenarios(scenarios)

    def _run():
        out = []
        for _name, truck, loads, top_n in scenarios:
            out.append(container.recommender.recommend_loads(loads, truck, top_n))
        return out

    results = benchmark.pedantic(_run, rounds=3, iterations=1, warmup_rounds=1)
    assert len(results) == len(scenarios)


def test_plan_all_scenarios(benchmark, container, scenarios):
    """End-to-end planning over every scenario (48h plan builder)."""
    _require_scenarios(scenarios)

    def _run():
        out = []
        for _name, truck, loads, _top_n in scenarios:
            out.append(container.planner.build_plan(loads, truck))
        return out

    results = benchmark.pedantic(_run, rounds=3, iterations=1, warmup_rounds=1)
    assert len(results) == len(scenarios)

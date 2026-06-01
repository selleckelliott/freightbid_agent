"""Shared fixtures and synthetic-data generators for benchmarks.

Kept independent from tests/conftest.py so benchmark runs don't depend on
test-only fixtures and vice versa.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, List

import pytest

from application.config_loader import AppConfig, load_config
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.scoring.heuristic_scoring import HeuristicScoringStrategy
from domain.scoring.scoring_strategy import ScoringStrategy

ROOT = Path(__file__).resolve().parents[1]

# Standard dataset sizes exercised across benchmarks. Override per-test with
# @pytest.mark.parametrize if a benchmark needs a different sweep.
DATASET_SIZES = (10, 100, 1_000, 10_000)


@pytest.fixture(scope="session")
def app_config() -> AppConfig:
    return load_config(ROOT / "config")


@pytest.fixture(scope="session")
def scoring_strategy(app_config: AppConfig) -> ScoringStrategy:
    return HeuristicScoringStrategy(
        scoring_weights=app_config.scoring_weights,
        cost_model=app_config.cost_model,
    )


def _make_load(load_id: int, rng: random.Random) -> Load:
    miles = rng.uniform(50.0, 1_200.0)
    rate_per_mile = rng.uniform(1.50, 4.50)
    pickup = datetime(2026, 5, 27, tzinfo=timezone.utc) + timedelta(
        hours=rng.uniform(0, 48)
    )
    delivery = pickup + timedelta(hours=miles / 50.0 + 1.0)
    return Load(
        load_id=load_id,
        weight=rng.uniform(5_000, 44_000),
        created_at=pickup - timedelta(hours=12),
        origin_city="OriginCity",
        origin_state="TX",
        origin_latitude=rng.uniform(25.0, 49.0),
        origin_longitude=rng.uniform(-124.0, -67.0),
        destination_city="DestCity",
        destination_state="CA",
        destination_latitude=rng.uniform(25.0, 49.0),
        destination_longitude=rng.uniform(-124.0, -67.0),
        pickup_window_start=pickup,
        pickup_window_end=pickup + timedelta(hours=2),
        delivery_window_start=delivery,
        delivery_window_end=delivery + timedelta(hours=2),
        miles=miles,
        total_rate=miles * rate_per_mile,
        equipment_type="Dry Van",
    )


def _make_evaluation(load: Load, rng: random.Random) -> LoadEvaluation:
    deadhead = rng.uniform(0.0, 200.0)
    total_miles = load.miles + deadhead
    driver_hours = total_miles / 50.0 + 1.5
    revenue = load.total_rate
    # Realistic-ish cost so expected_profit is meaningful in the score.
    fuel = total_miles * 0.55
    time_cost = driver_hours * 30.0
    total_cost = fuel + time_cost
    return LoadEvaluation(
        load=load,
        deadhead_miles=deadhead,
        total_miles=total_miles,
        driver_hours=driver_hours,
        expected_revenue=revenue,
        deadhead_cost=deadhead * 0.55,
        load_cost=load.miles * 0.55,
        toll_cost=0.0,
        time_cost=time_cost,
        total_cost=total_cost,
        expected_profit=revenue - total_cost,
    )


@pytest.fixture(scope="session")
def make_evaluations() -> Callable[[int, int], List[LoadEvaluation]]:
    """Factory: ``make_evaluations(n, seed=0)`` -> deterministic list."""

    def _factory(n: int, seed: int = 0) -> List[LoadEvaluation]:
        rng = random.Random(seed)
        return [_make_evaluation(_make_load(i + 1, rng), rng) for i in range(n)]

    return _factory

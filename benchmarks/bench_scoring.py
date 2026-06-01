"""Micro-benchmarks for HeuristicScoringStrategy.

Run with: ``pytest benchmarks/bench_scoring.py --benchmark-only``
"""
from __future__ import annotations

import pytest

from benchmarks.conftest import DATASET_SIZES


def test_score_single_load(benchmark, scoring_strategy, make_evaluations):
    evaluation = make_evaluations(1)[0]
    result = benchmark(scoring_strategy.score_load, evaluation)
    assert result.load_id == evaluation.load.load_id


@pytest.mark.parametrize("n", DATASET_SIZES)
def test_score_batch(benchmark, scoring_strategy, make_evaluations, n):
    evaluations = make_evaluations(n)

    def _run():
        return [scoring_strategy.score_load(e) for e in evaluations]

    results = benchmark(_run)
    assert len(results) == n


@pytest.mark.parametrize("n", DATASET_SIZES)
def test_rank_top_n(benchmark, scoring_strategy, make_evaluations, n):
    """Score + sort: closer to what /rank does on the hot path."""
    evaluations = make_evaluations(n)
    top_n = 10

    def _run():
        scored = [scoring_strategy.score_load(e) for e in evaluations]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_n]

    results = benchmark(_run)
    assert len(results) == min(top_n, n)
    assert all(
        results[i].score >= results[i + 1].score for i in range(len(results) - 1)
    )

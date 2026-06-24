"""Phase 8.3 - fleet dispatch benchmark: committed-summary reconciliation tests.

Guards that the shipped ``benchmarks/fleet_dispatch_summary.json`` is the genuine
output of the current code (not a stale artifact) by re-running one fast condition
through the same ``load_config`` -> ``run_fleet_condition`` path the runner uses
and matching the committed numbers exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.run_fleet_dispatch import DEFAULT_CONFIG, DEFAULT_OUT, load_config
from simulation.fleet_experiment import run_fleet_condition

ROOT = Path(__file__).resolve().parents[1]

_REQUIRED_CONDITION_KEYS = {
    "name", "rationale", "fleet_size", "homogeneous", "episode_count",
    "verdict", "greedy", "fleet_aware", "headline_delta", "paired_profit",
    "paired_deadhead", "per_episode",
}


@pytest.fixture(scope="module")
def committed():
    return json.loads(DEFAULT_OUT.read_text(encoding="utf-8"))


def test_committed_summary_shape(committed):
    assert committed["headline"] == "fleet_aware vs greedy (coordination value)"
    cfg = committed["config"]
    assert cfg["fleet_size"] >= 1 and cfg["episode_count"] >= 1
    conditions = committed["conditions"]
    assert conditions, "summary must contain at least one condition"
    names = [c["name"] for c in conditions]
    assert "homogeneous_contention" in names
    for c in conditions:
        assert _REQUIRED_CONDITION_KEYS <= set(c)
        assert c["verdict"] in {"HOLDS", "NEUTRAL", "REGRESSION"}
        assert len(c["per_episode"]["greedy"]) == c["episode_count"]
        assert len(c["per_episode"]["fleet_aware"]) == c["episode_count"]


def test_no_regression_verdicts(committed):
    """The shipped benchmark must not advertise a coordination regression."""
    for c in committed["conditions"]:
        assert c["verdict"] != "REGRESSION", c["name"]


def test_committed_summary_reconciles_with_code(committed, container):
    """Re-run the fastest committed condition end-to-end and match its numbers.

    Uses ``homogeneous_contention`` (a deliberately thin board -> fast) so the
    full-episode reconciliation stays cheap while still proving the artifact is
    reproducible from the current engine + config.
    """
    settings = load_config(DEFAULT_CONFIG)
    cfg = committed["config"]
    assert settings.episode_count == cfg["episode_count"]
    assert settings.fleet_size == cfg["fleet_size"]
    assert settings.base_seed == cfg["base_seed"]

    target = "homogeneous_contention"
    spec = next(s for s in settings.specs if s.name == target)
    committed_cond = next(c for c in committed["conditions"] if c["name"] == target)

    result = run_fleet_condition(
        spec,
        container=container,
        base=settings.base,
        fleet_size=settings.fleet_size,
        episode_count=settings.episode_count,
        base_seed=settings.base_seed,
        solver_time_limit=settings.time_limit,
        max_candidates_per_truck=settings.max_candidates,
    ).to_dict()

    assert result["verdict"] == committed_cond["verdict"]
    for arm in ("greedy", "fleet_aware"):
        for metric in ("total_profit", "total_deadhead_miles", "loads_completed"):
            assert result[arm]["metrics"][metric]["mean"] == pytest.approx(
                committed_cond[arm]["metrics"][metric]["mean"], rel=1e-9, abs=1e-6
            ), f"{arm}.{metric}"
    for key in ("mean", "ci_low", "ci_high"):
        assert result["paired_profit"][key] == pytest.approx(
            committed_cond["paired_profit"][key], rel=1e-9, abs=1e-6
        )

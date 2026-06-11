"""Unit tests for the Pareto-dominance utilities (Phase 2.3 tuning)."""
import pytest

from benchmarks.pareto import dominates, is_dominated, pareto_flags


def _m(profit, deadhead, feasible=1.0):
    return {
        "avg_profit": profit,
        "avg_deadhead_miles": deadhead,
        "feasible_rate": feasible,
    }


def test_dominated_config_detected():
    a = _m(profit=100, deadhead=10)
    b = _m(profit=120, deadhead=9)

    assert dominates(b, a)
    assert is_dominated(a, [a, b])
    assert not is_dominated(b, [a, b])
    assert pareto_flags([a, b]) == [False, True]


def test_non_dominated_tradeoff_preserved():
    a = _m(profit=130, deadhead=15)
    b = _m(profit=110, deadhead=8)

    assert not dominates(a, b)
    assert not dominates(b, a)
    assert pareto_flags([a, b]) == [True, True]


def test_identical_points_do_not_dominate_each_other():
    a = _m(profit=100, deadhead=10)
    b = _m(profit=100, deadhead=10)

    assert not dominates(a, b)
    assert pareto_flags([a, b]) == [True, True]


def test_equal_on_one_axis_better_on_other_dominates():
    a = _m(profit=100, deadhead=10)
    b = _m(profit=100, deadhead=8)

    assert dominates(b, a)
    assert pareto_flags([a, b]) == [False, True]


def test_feasibility_filter_excludes_impractical_configs():
    # Best on both axes but rarely produces a plan at all.
    flashy = _m(profit=500, deadhead=2, feasible=0.40)
    solid = _m(profit=300, deadhead=12, feasible=0.90)

    flags = pareto_flags([flashy, solid], min_feasible_rate=0.85)

    assert flags == [False, True]
    # And the impractical config doesn't knock practical ones off the front.
    assert pareto_flags([flashy, solid], min_feasible_rate=0.0) == [True, False]


@pytest.mark.parametrize("min_rate", [0.0, 0.85])
def test_pareto_flags_align_with_input_order(min_rate):
    metrics = [
        _m(profit=200, deadhead=20, feasible=0.9),
        _m(profit=250, deadhead=18, feasible=0.9),  # dominates the first
        _m(profit=180, deadhead=5, feasible=0.9),
    ]

    flags = pareto_flags(metrics, min_feasible_rate=min_rate)

    assert flags == [False, True, True]
    assert len(flags) == len(metrics)

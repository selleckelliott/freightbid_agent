"""Pareto-dominance utilities for the objective tuning harness (Phase 2.3).

Pure functions over metric mappings — no solver, no I/O — so the dominance
logic is unit-testable in microseconds.

Convention: configs trade off **average profit (maximize)** against
**average deadhead miles (minimize)**. A config is *dominated* when some
other config is at least as good on both dimensions and strictly better on
at least one. The Pareto front is the set of non-dominated configs,
optionally restricted to configs whose feasible rate clears a practical
threshold first (an impractical planner that rarely produces a plan can
look artificially great on the other two axes).
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

PROFIT_KEY = "avg_profit"
DEADHEAD_KEY = "avg_deadhead_miles"
FEASIBLE_KEY = "feasible_rate"


def dominates(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    profit_key: str = PROFIT_KEY,
    deadhead_key: str = DEADHEAD_KEY,
) -> bool:
    """True if metrics ``a`` Pareto-dominate metrics ``b``."""
    better_or_equal = (
        a[profit_key] >= b[profit_key] and a[deadhead_key] <= b[deadhead_key]
    )
    strictly_better = (
        a[profit_key] > b[profit_key] or a[deadhead_key] < b[deadhead_key]
    )
    return better_or_equal and strictly_better


def is_dominated(
    candidate: Mapping[str, Any],
    others: Iterable[Mapping[str, Any]],
    *,
    profit_key: str = PROFIT_KEY,
    deadhead_key: str = DEADHEAD_KEY,
) -> bool:
    """True if any of ``others`` dominates ``candidate``.

    ``candidate`` may appear in ``others``; an identical metric point never
    dominates itself (the strictness clause fails).
    """
    return any(
        dominates(other, candidate, profit_key=profit_key, deadhead_key=deadhead_key)
        for other in others
    )


def pareto_flags(
    metrics: Sequence[Mapping[str, Any]],
    *,
    min_feasible_rate: float = 0.0,
    profit_key: str = PROFIT_KEY,
    deadhead_key: str = DEADHEAD_KEY,
    feasible_key: str = FEASIBLE_KEY,
) -> list[bool]:
    """Per-config Pareto-efficiency flags, aligned with ``metrics``.

    Configs below ``min_feasible_rate`` are flagged ``False`` outright and
    excluded from the comparison pool, so an impractical config can neither
    join the front nor knock practical configs off it.
    """
    eligible = [
        m for m in metrics if float(m.get(feasible_key, 1.0)) >= min_feasible_rate
    ]
    return [
        float(m.get(feasible_key, 1.0)) >= min_feasible_rate
        and not is_dominated(
            m, eligible, profit_key=profit_key, deadhead_key=deadhead_key
        )
        for m in metrics
    ]

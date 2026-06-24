"""Fleet pair scoring (Phase 8.1).

Both fleet policies — the uncoordinated greedy baseline and the globally optimised
assignment policy — differ *only* in how they resolve contention. What a single
``(truck, load)`` pair is *worth*, and whether it is even allowed, is shared
machinery, factored out here so the two policies stay honest about scoring the
same thing.

A :class:`FleetPairScorer` turns a ``(load, truck)`` pair into a
:class:`ScoredPair` or ``None``:

* ``None`` — the pair is infeasible (wrong equipment, over capacity, past the
  deadhead cap, not enough driver hours, window missed, or expected profit below
  the business floor). Infeasible pairs never enter an assignment, exactly as the
  single-truck engine refuses them.
* a :class:`ScoredPair` — feasible, carrying the same realized ``PlanStop``
  financials the single-truck planner would emit, plus a ``score`` used to rank /
  match.

**Positivity invariant.** Every feasible pair scores ``>= 0``. The default
:class:`ProfitPairScorer` scores expected profit, which feasibility already gates
at ``min_expected_profit`` (> 0). A destination-aware scorer (Phase 8.3) adds a
*non-negative* onward-desirability bonus, so the invariant holds for it too. The
assignment policy relies on it: serving a feasible load can only ever increase the
objective, so coordination never idles a truck that has an available load merely
because another scorer could rate it lower.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from application.evaluate_loads import EvaluateLoadsService
from domain.models.fleet import Assignment
from domain.models.load import Load
from domain.models.load_evaluation import LoadEvaluation
from domain.models.plan import PlanStop
from domain.models.truck_state import TruckState
from domain.policies.constraints import PlanningConstraints
from domain.policies.feasibility import feasibility_checker


def stop_from_evaluation(evaluation: LoadEvaluation, rationale: str = "") -> PlanStop:
    """Build the executed ``PlanStop`` for a load, mirroring ``PlanBuilderService``.

    Identical field-for-field to the stop the single-truck greedy planner appends,
    so a fleet assignment and a single-truck decision over the same pair carry
    byte-identical financials.
    """
    load = evaluation.load
    return PlanStop(
        load_id=load.load_id,
        pickup_eta=evaluation.pickup_eta,
        delivery_eta=evaluation.delivery_eta,
        deadhead_miles=evaluation.deadhead_miles,
        load_miles=load.miles,
        revenue=evaluation.expected_revenue,
        cost=evaluation.total_cost,
        profit=evaluation.expected_profit,
        rationale=rationale,
    )


@dataclass
class ScoredPair:
    """A feasible ``(truck, load)`` pair, its financials, and its selection score."""

    truck_id: int
    load: Load
    evaluation: LoadEvaluation
    stop: PlanStop
    score: float
    rationale: str = ""

    def to_assignment(self) -> Assignment:
        return Assignment(
            truck_id=self.truck_id,
            load_id=self.load.load_id,
            stop=self.stop,
            score=self.score,
            rationale=self.rationale,
        )


@runtime_checkable
class FleetPairScorer(Protocol):
    def score(self, load: Load, truck: TruckState) -> Optional[ScoredPair]: ...


class ProfitPairScorer:
    """Feasibility-gated expected-profit scorer (the default, ML-free).

    The pair is evaluated through the shared ``EvaluateLoadsService`` and gated by
    the same ``feasibility_checker`` (with the cost model, so the profit floor
    applies) the single-truck engine uses. The score is the pair's expected
    profit. Because feasibility requires ``expected_profit >= min_expected_profit``
    (> 0), every returned score is strictly positive — the positivity invariant.
    """

    def __init__(
        self,
        evaluator: EvaluateLoadsService,
        constraints: PlanningConstraints,
    ):
        self._evaluator = evaluator
        self._constraints = constraints

    def score(self, load: Load, truck: TruckState) -> Optional[ScoredPair]:
        evaluation = self._evaluator.evaluate_one(load, truck)
        feasible, reason = feasibility_checker(
            evaluation,
            truck,
            self._constraints,
            self._evaluator.cost_model,
        )
        if not feasible:
            return None
        rationale = (
            f"profit ${evaluation.expected_profit:,.0f} "
            f"(deadhead {evaluation.deadhead_miles:.0f}mi)"
        )
        return ScoredPair(
            truck_id=truck.truck_id,
            load=load,
            evaluation=evaluation,
            stop=stop_from_evaluation(evaluation, rationale),
            score=evaluation.expected_profit,
            rationale=rationale,
        )

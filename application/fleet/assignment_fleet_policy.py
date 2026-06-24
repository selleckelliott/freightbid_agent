"""Globally-optimised fleet assignment policy (Phase 8.1) — the treatment arm.

Where :class:`GreedyFleetPolicy` lets trucks grab loads one at a time, this policy
looks at the **whole** ``truck x load`` field at once and picks the set of
assignments that maximises total score, subject to: each truck takes at most one
load, each load goes to at most one truck. That is a maximum-weight bipartite
matching, solved exactly with OR-Tools **CP-SAT**.

Only feasible pairs (the scorer returned a :class:`ScoredPair`, not ``None``)
become decision variables, so infeasible pairs can never be chosen. Thanks to the
scorer's positivity invariant (every feasible pair scores ``>= 0``), the optimum
never leaves a truck idle when it has an available feasible load — coordination
can only *re-route* loads to their best truck, never destroy utilisation.

Determinism is enforced explicitly (single worker, fixed seed, integer-cent
weights, stable output order) so the benchmark's paired comparison is reproducible.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ortools.sat.python import cp_model

from application.fleet.pair_scorer import FleetPairScorer, ScoredPair
from domain.models.fleet import Assignment
from domain.models.load import Load
from domain.models.truck_state import TruckState

# Scores (dollars) are scaled to integer cents for the CP-SAT integer objective.
_SCORE_SCALE = 100


class AssignmentFleetPolicy:
    def __init__(
        self,
        scorer: FleetPairScorer,
        *,
        max_candidates_per_truck: Optional[int] = None,
        solver_time_limit_seconds: float = 1.0,
    ):
        self._scorer = scorer
        self._max_candidates_per_truck = max_candidates_per_truck
        self._time_limit = solver_time_limit_seconds

    def assign(
        self,
        trucks: List[TruckState],
        candidates_by_truck_id: Dict[int, List[Load]],
        now: datetime,
    ) -> List[Assignment]:
        # 1. Score every (truck, load) pair; keep only feasible ones.
        scored_pairs = self._score_pairs(trucks, candidates_by_truck_id)
        if not scored_pairs:
            return []

        # 2. Build and solve the max-weight matching with CP-SAT.
        model = cp_model.CpModel()
        x: Dict[int, cp_model.IntVar] = {}
        by_truck: Dict[int, List[int]] = {}
        by_load: Dict[int, List[int]] = {}

        for idx, pair in enumerate(scored_pairs):
            x[idx] = model.NewBoolVar(f"x_{pair.truck_id}_{pair.load.load_id}")
            by_truck.setdefault(pair.truck_id, []).append(idx)
            by_load.setdefault(pair.load.load_id, []).append(idx)

        for idxs in by_truck.values():
            model.AddAtMostOne(x[i] for i in idxs)
        for idxs in by_load.values():
            model.AddAtMostOne(x[i] for i in idxs)

        model.Maximize(
            sum(
                int(round(pair.score * _SCORE_SCALE)) * x[idx]
                for idx, pair in enumerate(scored_pairs)
            )
        )

        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = self._time_limit
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return []

        # 3. Collect chosen assignments, in a stable (truck_id, load_id) order.
        chosen = [
            pair
            for idx, pair in enumerate(scored_pairs)
            if solver.Value(x[idx]) == 1
        ]
        chosen.sort(key=lambda p: (p.truck_id, p.load.load_id))
        return [p.to_assignment() for p in chosen]

    def _score_pairs(
        self,
        trucks: List[TruckState],
        candidates_by_truck_id: Dict[int, List[Load]],
    ) -> List[ScoredPair]:
        pairs: List[ScoredPair] = []
        for truck in sorted(trucks, key=lambda t: t.truck_id):
            scored_for_truck: List[ScoredPair] = []
            for load in candidates_by_truck_id.get(truck.truck_id, []):
                scored = self._scorer.score(load, truck)
                if scored is not None:
                    scored_for_truck.append(scored)
            if self._max_candidates_per_truck is not None:
                # Keep this truck's top-N by score; ties broken by lower load_id so
                # the candidate cap is deterministic.
                scored_for_truck.sort(
                    key=lambda p: (-p.score, p.load.load_id)
                )
                scored_for_truck = scored_for_truck[: self._max_candidates_per_truck]
            pairs.extend(scored_for_truck)
        return pairs

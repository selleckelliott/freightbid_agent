"""Greedy (uncoordinated) fleet policy (Phase 8.1) — the baseline arm.

Each free truck, taken in ascending ``truck_id`` order, grabs its own best
feasible load; once a truck claims a load, that load is removed so a later truck
cannot take it. The result is **conflict-free** (no double-booking) but
**uncoordinated**: an early, low-``truck_id`` truck can snatch a load that would
have been far more profitable for a later truck, and nobody re-optimises.

This is deliberately the same myopic rule the single-truck planner follows, run
truck-by-truck. It is the honest baseline for Phase 8: the gap between this and
the globally-optimised :class:`AssignmentFleetPolicy` is the *value of
coordination*, not merely the value of avoiding double-booking (this baseline
already avoids that).
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from application.fleet.pair_scorer import FleetPairScorer
from domain.models.fleet import Assignment
from domain.models.load import Load
from domain.models.truck_state import TruckState


class GreedyFleetPolicy:
    def __init__(self, scorer: FleetPairScorer):
        self._scorer = scorer

    def assign(
        self,
        trucks: List[TruckState],
        candidates_by_truck_id: Dict[int, List[Load]],
        now: datetime,
    ) -> List[Assignment]:
        claimed: set[int] = set()
        assignments: List[Assignment] = []

        for truck in sorted(trucks, key=lambda t: t.truck_id):
            best = None
            for load in candidates_by_truck_id.get(truck.truck_id, []):
                if load.load_id in claimed:
                    continue
                scored = self._scorer.score(load, truck)
                if scored is None:
                    continue
                # Higher score wins; ties broken by lower load_id so the result is
                # deterministic regardless of candidate-list ordering.
                if (
                    best is None
                    or scored.score > best.score
                    or (
                        scored.score == best.score
                        and scored.load.load_id < best.load.load_id
                    )
                ):
                    best = scored
            if best is not None:
                claimed.add(best.load.load_id)
                assignments.append(best.to_assignment())

        return assignments

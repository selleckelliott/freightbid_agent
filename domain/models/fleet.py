"""Fleet domain models (Phase 8.1).

The single-truck engine plans one ``TruckState`` over a board of ``Load``s and
returns a ``Plan`` of sequenced ``PlanStop``s. Fleet dispatch adds one concept on
top: an :class:`Assignment` — *this* truck takes *that* load, now — together with
the realized financials of doing so (lifted straight off the same
``EvaluateLoadsService`` the rest of the engine uses, so a fleet assignment
reconciles with a single-truck stop exactly).

:class:`Fleet` is a thin convenience over ``List[TruckState]`` used when building
and querying a set of trucks; it owns no mutable simulation state (each truck's
evolving position/HOS lives in its own ``TruckSimulator`` during a replay).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List, Optional

from domain.models.plan import PlanStop
from domain.models.truck_state import TruckState


@dataclass
class Assignment:
    """One truck claiming one load, with the realized financials of doing so.

    ``stop`` is the same ``PlanStop`` shape the single-truck planner emits for the
    load it executes, so the fleet simulator advances a truck through an
    assignment with the identical ``TruckSimulator.execute_load`` path the rolling
    replay already uses. ``score`` is the policy's selection value (>= expected
    profit for the profit scorer; a destination-aware scorer may add a
    non-negative onward-desirability bonus), kept distinct from ``stop.profit`` so
    business profit and selection score never get conflated.
    """

    truck_id: int
    load_id: int
    stop: PlanStop
    score: float
    rationale: str = ""

    @property
    def profit(self) -> float:
        return self.stop.profit

    @property
    def revenue(self) -> float:
        return self.stop.revenue

    @property
    def cost(self) -> float:
        return self.stop.cost

    @property
    def deadhead_miles(self) -> float:
        return self.stop.deadhead_miles

    @property
    def loaded_miles(self) -> float:
        return self.stop.load_miles


@dataclass
class Fleet:
    """A set of trucks. Thin helper — querying only, no mutable replay state."""

    trucks: List[TruckState]

    def __len__(self) -> int:
        return len(self.trucks)

    def __iter__(self) -> Iterator[TruckState]:
        return iter(self.trucks)

    @property
    def truck_ids(self) -> List[int]:
        return [t.truck_id for t in self.trucks]

    def by_id(self, truck_id: int) -> Optional[TruckState]:
        return next((t for t in self.trucks if t.truck_id == truck_id), None)

    def free_at(self, now: datetime) -> List[TruckState]:
        """Trucks available to be dispatched at ``now`` (``available_at <= now``).

        Returned in ascending ``truck_id`` order so any downstream tie-breaking is
        deterministic regardless of how the fleet was assembled.
        """
        return sorted(
            (t for t in self.trucks if t.available_at <= now),
            key=lambda t: t.truck_id,
        )

    def equipment_mix(self) -> Dict[str, int]:
        """Count of trucks per trailer type (homogeneous vs heterogeneous view)."""
        mix: Dict[str, int] = {}
        for t in self.trucks:
            mix[t.trailer_type] = mix.get(t.trailer_type, 0) + 1
        return mix

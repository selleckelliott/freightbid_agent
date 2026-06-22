"""Recalibrated winnability adapter (Phase 5.4).

Wraps a base :class:`~ports.winnability.WinnabilityPort` and a fitted
:class:`~ml.calibration.recalibrator.Recalibrator`, applying the post-hoc map to every
probability the base adapter returns. The base model is untouched — this is the *repair layer*
the recalibration workflow promotes when the Phase 5.3 monitor flags drift.

Behavior-preserving by construction: with ``recalibrator=None`` (no map promoted) or a base
adapter that returns ``None`` (no model available), the adapter returns exactly what the base
returns — so wiring it in with no promoted recalibrator changes nothing (the same "optional by
default" contract as the Phase 4.3 winnability adapters).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ml.calibration.recalibrator import Recalibrator
from ml.features.winnability_features import BidQuery
from ports.winnability import WinnabilityPort


class RecalibratedWinnabilityAdapter(WinnabilityPort):
    def __init__(
        self, base: WinnabilityPort, recalibrator: Optional[Recalibrator] = None
    ) -> None:
        self._base = base
        self._recalibrator = recalibrator

    @property
    def is_active(self) -> bool:
        """True when a recalibrator is attached and will modify base probabilities."""
        return self._recalibrator is not None

    def win_probabilities(
        self, query: BidQuery, bid_rpms: Sequence[float]
    ) -> Optional[List[float]]:
        base_probs = self._base.win_probabilities(query, bid_rpms)
        if base_probs is None or not base_probs or self._recalibrator is None:
            return base_probs
        repaired = self._recalibrator.transform(base_probs)
        return [float(p) for p in repaired]

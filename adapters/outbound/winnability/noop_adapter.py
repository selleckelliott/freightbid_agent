"""No-op winnability adapter (Phase 4.3).

Returns ``None`` for every query, signaling "no winnability model available" so the EV
recommender falls back to the existing cost-plus-margin behavior. This is the adapter
wired when winnability is disabled or no artifact is present — guaranteeing zero
behavior change versus the pre-4.3 recommender.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from ml.features.winnability_features import BidQuery
from ports.winnability import WinnabilityPort


class NoopWinnabilityAdapter(WinnabilityPort):
    def win_probabilities(
        self, query: BidQuery, bid_rpms: Sequence[float]
    ) -> Optional[List[float]]:
        return None

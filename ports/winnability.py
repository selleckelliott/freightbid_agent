"""Outbound port for bid-winnability estimation (Phase 4.3).

The EV bid recommender depends on this port, never on the model: a
``ModelWinnabilityAdapter`` wraps the trained Phase 4.2 artifact, while a
``NoopWinnabilityAdapter`` returns ``None`` so the recommender degrades to today's
cost-plus-margin behavior. One port, two adapters — the model stays optional and
swappable.

The query is a :class:`~ml.features.winnability_features.BidQuery`, the serving-time
DTO designed for exactly this reuse, so adapters build features with the *same*
builder used at training (no train/serve skew).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ml.features.winnability_features import BidQuery


class WinnabilityPort(ABC):
    @abstractmethod
    def win_probabilities(
        self, query: BidQuery, bid_rpms: Sequence[float]
    ) -> Optional[List[float]]:
        """``P(win)`` for each ask in ``bid_rpms`` (same order), or ``None``.

        ``None`` signals "no winnability model available" — the caller should fall
        back to cost-plus-margin. A non-``None`` result must have exactly one
        probability per requested ask.
        """
        ...

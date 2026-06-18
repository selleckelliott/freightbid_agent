"""Model-backed winnability adapter (Phase 4.3).

Wraps the trained Phase 4.2 ``SklearnWinnabilityModel`` and scores a grid of candidate
asks for one board load. Features are built with ``build_winnability_features`` — the
exact training-time builder — so there is no train/serve skew.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

from ml.features.winnability_features import BidQuery, build_winnability_features
from ports.winnability import WinnabilityPort


class ModelWinnabilityAdapter(WinnabilityPort):
    def __init__(self, model) -> None:
        self._model = model

    @classmethod
    def from_artifact(cls, model_path: str | Path) -> "ModelWinnabilityAdapter":
        from ml.models.sklearn_winnability_model import SklearnWinnabilityModel

        return cls(SklearnWinnabilityModel.load(model_path))

    def win_probabilities(
        self, query: BidQuery, bid_rpms: Sequence[float]
    ) -> Optional[List[float]]:
        asks = list(bid_rpms)
        if not asks:
            return []
        rows = [build_winnability_features(query, float(ask)) for ask in asks]
        frame = pd.DataFrame(rows)
        return [float(p) for p in self._model.predict_proba(frame)]

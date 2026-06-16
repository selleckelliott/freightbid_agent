"""Destination desirability service (Phase 3.1 boundary).

A thin facade over the trained model so future planner work (Phase 3.2) can ask
a single question — *how much next-deadhead should I expect if I deliver here?* —
without knowing anything about feature engineering or model internals.

This file intentionally does **not** touch the OR-Tools planner. It only defines
the contract Phase 3.2 will call:

    future_deadhead_penalty = predict_next_deadhead(...) * deadhead_cost_per_mile
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ml.config import FeatureConfig
from ml.features.destination_features import DestinationQuery, build_features


class DestinationDesirabilityService:
    def __init__(self, model: Any, feature_config: FeatureConfig | None = None) -> None:
        self._model = model
        self._feature_config = feature_config or FeatureConfig()

    @classmethod
    def from_artifact(
        cls,
        model_path: str | Path,
        feature_config: FeatureConfig | None = None,
    ) -> "DestinationDesirabilityService":
        from ml.models.sklearn_destination_model import SklearnDestinationModel

        return cls(SklearnDestinationModel.load(model_path), feature_config)

    def predict_next_deadhead(
        self,
        *,
        destination_lat: float,
        destination_lon: float,
        destination_state: str,
        arrival_time: datetime,
        equipment_type: str,
        visible_loads: Sequence[Any] = (),
        load_age_hours: float = 0.0,
        mode: str = "TL",
    ) -> float:
        """Predicted ``expected_next_deadhead_miles`` for a candidate delivery.

        ``visible_loads`` is the decision-time board (loads currently posted);
        their proximity to the destination drives the market-density features.
        """
        query = DestinationQuery(
            destination_lat=destination_lat,
            destination_lon=destination_lon,
            destination_state=destination_state,
            equipment_type=equipment_type,
            arrival_dt=arrival_time,
            mode=mode,
            load_age_hours_value=load_age_hours,
            load_id=None,
        )
        feats = build_features(query, list(visible_loads), self._feature_config)
        frame = pd.DataFrame([feats])
        return float(self._model.predict(frame)[0])

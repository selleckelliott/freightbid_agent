"""Tests for the destination desirability service facade (Phase 3.1)."""
from datetime import datetime, timezone

import numpy as np

from application.destination_desirability_service import DestinationDesirabilityService


class _StubModel:
    """Records the frame it was asked to predict on and returns a fixed value."""

    def __init__(self):
        self.last_frame = None

    def predict(self, frame):
        self.last_frame = frame
        return np.array([42.0])


def test_service_returns_model_prediction_and_builds_one_row():
    stub = _StubModel()
    service = DestinationDesirabilityService(stub)

    value = service.predict_next_deadhead(
        destination_lat=39.7392,
        destination_lon=-104.9903,
        destination_state="CO",
        arrival_time=datetime(2026, 1, 2, 14, 0, tzinfo=timezone.utc),
        equipment_type="Flatbed",
        visible_loads=[],
    )

    assert value == 42.0
    assert stub.last_frame.shape[0] == 1
    assert "destination_zone" in stub.last_frame.columns
    assert stub.last_frame["destination_zone"].iloc[0] == "Denver"

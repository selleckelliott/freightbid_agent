"""The sklearn compiled dispatcher: wraps the frozen Phase 6.3 ``CompiledDispatcherModel``.

Loads the committed-shape joblib artifact and serves predictions for **shadow comparison only**.
Fail-closed by construction:

* if the artifact's ``feature_manifest_hash`` does not match the expected inference contract, the
  adapter reports unavailable (``manifest_mismatch``) and ``predict`` refuses to serve;
* a manifest error raised by the model at predict time is translated into
  :class:`CompiledDispatcherUnavailable` so the shadow service can record the reason rather than
  propagating an exception into the source recommendation path.

Mirrors ``adapters/outbound/winnability/model_adapter.py`` (``from_artifact`` classmethod that
lazy-loads the ml model). The adapter never drafts, approves, or submits a bid.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from ports.compiled_dispatcher import (
    REASON_MANIFEST_MISMATCH,
    CompiledDispatcherAvailability,
    CompiledDispatcherPort,
    CompiledDispatcherPrediction,
    CompiledDispatcherUnavailable,
)


class SklearnCompiledDispatcher(CompiledDispatcherPort):
    """Wraps a loaded ``CompiledDispatcherModel`` (shadow-only)."""

    def __init__(
        self,
        model: Any,
        *,
        artifact_path: Optional[str | Path] = None,
        expected_manifest_hash: Optional[str] = None,
    ):
        self._model = model
        self._artifact_path = str(artifact_path) if artifact_path is not None else None
        self._manifest_ok = (
            expected_manifest_hash is None
            or getattr(model, "feature_manifest_hash", None) == expected_manifest_hash
        )

    @classmethod
    def from_artifact(
        cls,
        path: str | Path,
        *,
        expected_manifest_hash: Optional[str] = None,
    ) -> "SklearnCompiledDispatcher":
        from ml.models.compiled_dispatcher_model import CompiledDispatcherModel

        model = CompiledDispatcherModel.load(path)
        return cls(model, artifact_path=path, expected_manifest_hash=expected_manifest_hash)

    @property
    def feature_manifest_hash(self) -> Optional[str]:
        return getattr(self._model, "feature_manifest_hash", None)

    def availability(self) -> CompiledDispatcherAvailability:
        if not self._manifest_ok:
            return CompiledDispatcherAvailability(
                available=False,
                reason=REASON_MANIFEST_MISMATCH,
                artifact_path=self._artifact_path,
                feature_manifest_hash=self.feature_manifest_hash,
            )
        return CompiledDispatcherAvailability(
            available=True,
            reason=None,
            artifact_path=self._artifact_path,
            feature_manifest_hash=self.feature_manifest_hash,
        )

    def predict(self, features: Mapping[str, Any]) -> CompiledDispatcherPrediction:
        if not self._manifest_ok:
            raise CompiledDispatcherUnavailable(REASON_MANIFEST_MISMATCH)
        from ml.models.compiled_dispatcher_model import FeatureManifestError

        try:
            return self._model.predict_dto(features)
        except FeatureManifestError as exc:
            raise CompiledDispatcherUnavailable(REASON_MANIFEST_MISMATCH) from exc

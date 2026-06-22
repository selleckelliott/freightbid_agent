"""The compiled multi-head dispatcher model (Phase 6.3).

One **artifact, five heads.** Rather than force FreightBid's mixed targets through a single
multi-output estimator, the compiled dispatcher packages a purpose-built head per target into one
:class:`CompiledDispatcherModel`, each with the right estimator, masking, and class handling:

* ``action_head`` — multiclass ``decision`` ∈ {bid, no_bid, approval_required} (class-balanced).
  In the canonical batch these correspond exactly to the user's named scenario categories
  (clean_bid → bid, payment_escalation → approval_required, infeasible_no_bid → no_bid).
* ``bid_ratio_head`` — regresses the **dimensionless** ``bid_rpm / market_rate`` on *biddable* rows
  only, so the model learns a market-relative multiplier (~1.0) that generalizes across worlds far
  better than raw dollars. Served bid is reconstructed: ``rpm = ratio × market_rate``,
  ``amount = rpm × loaded_miles``.
* ``risk_adjusted_ev_head`` — regresses the teacher ``risk_adjusted_ev`` on *feasible* rows.
* ``warning_heads`` — independent binary heads for ``payment_risk`` / ``calibration_alert`` /
  ``no_feasible_bid`` (a head whose label is constant in training degrades to a constant predictor
  rather than failing — ``calibration_alert`` is absent in the canonical batch).
* ``approval_required_head`` — binary "human approval required".

**The boundary is enforced in the weights, not just the docs.** The feature manifest is
``inference_context`` minus pure identifiers (load id, snapshot time, broker id); :func:`assert_manifest_inference_only`
rejects any ``node_outputs``/``eval_labels`` field at construction, and the manifest is hashed onto
the artifact. At inference the model **refuses to serve** if the caller's manifest hash does not
match (:class:`FeatureManifestError`) — a compiled model that silently consumed a teacher-only field
would defeat the whole point of compiling the procedure.
"""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from ml.data.compiled_agent_trace_schema import (
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
    eval_label_field_names,
    inference_field_names,
    node_output_field_names,
)
from ml.workflows.freightbid_workflow_graph import (
    WARN_CALIBRATION_ALERT,
    WARN_NO_FEASIBLE_BID,
    WARN_PAYMENT_RISK,
)

MODEL_NAME = "CompiledDispatcherMultiHead"
COMPILED_DISPATCHER_MODEL_VERSION = "1.0.0"
DEFAULT_RANDOM_STATE = 63

# inference_context fields that are pure identifiers / bookkeeping — never model inputs.
IDENTIFIER_FIELDS = ("load_id", "snapshot_time", "broker_id")

# Categorical model features (the rest of the manifest is numeric; bools coerce to 0/1).
CATEGORICAL_FEATURES = (
    "equipment_type",
    "mode",
    "commodity",
    "load_views",
    "broker_credit_bucket",
    "truck_equipment_type",
)

# The three binary warning heads, in stable order.
WARNING_HEADS = (WARN_PAYMENT_RISK, WARN_CALIBRATION_ALERT, WARN_NO_FEASIBLE_BID)

TARGET_NAMES = (
    "action",
    "bid_ratio",
    "risk_adjusted_ev",
    *(f"warning::{w}" for w in WARNING_HEADS),
    "approval_required",
)

_HGB_CLF_KW = dict(
    learning_rate=0.1,
    max_iter=200,
    max_leaf_nodes=15,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=False,  # fully deterministic: no internal RNG validation split
    categorical_features="from_dtype",
)
_HGB_REG_KW = dict(
    learning_rate=0.1,
    max_iter=300,
    max_leaf_nodes=15,
    min_samples_leaf=10,
    l2_regularization=0.1,
    early_stopping=False,
    categorical_features="from_dtype",
)


class FeatureManifestError(RuntimeError):
    """Raised when an inference caller's feature manifest does not match the artifact's."""


# --------------------------------------------------------------------------- #
# Feature manifest (inference_context only)
# --------------------------------------------------------------------------- #
def default_feature_manifest() -> List[str]:
    """The ordered model feature manifest: ``inference_context`` minus pure identifiers."""
    return [f for f in sorted(inference_field_names()) if f not in IDENTIFIER_FIELDS]


def assert_manifest_inference_only(manifest: Sequence[str]) -> bool:
    """Assert every manifest field is an ``inference_context`` field and none is teacher-only."""
    keys = set(manifest)
    leaked_nodes = keys & node_output_field_names()
    leaked_labels = keys & eval_label_field_names()
    if leaked_nodes or leaked_labels:
        raise ValueError(
            "compiled-dispatcher feature manifest leaks teacher-only fields: "
            f"node_outputs={sorted(leaked_nodes)} eval_labels={sorted(leaked_labels)}"
        )
    unknown = keys - inference_field_names()
    if unknown:
        raise ValueError(
            f"compiled-dispatcher feature manifest has non-inference fields: {sorted(unknown)}"
        )
    return True


def feature_manifest_hash(manifest: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(manifest), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# Target derivation (output side — may read recommendation + node-output targets)
# --------------------------------------------------------------------------- #
def row_features(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row["features"]


def derive_action(row: Mapping[str, Any]) -> str:
    return row["targets"]["decision"]


def derive_bid_ratio(row: Mapping[str, Any]) -> Optional[float]:
    """``bid_rpm / market_rate`` — a market-relative multiplier — or ``None`` for no-bid rows."""
    rpm = row["targets"]["recommended_bid_rpm"]
    market = row["features"]["market_rate"]
    if rpm is None or not market:
        return None
    return float(rpm) / float(market)


def derive_ev(row: Mapping[str, Any]) -> Optional[float]:
    return row["targets"]["risk_adjusted_ev"]


def derive_warnings(row: Mapping[str, Any]) -> Dict[str, int]:
    present = set(row["targets"]["warnings"])
    return {w: int(w in present) for w in WARNING_HEADS}


def derive_approval_required(row: Mapping[str, Any]) -> int:
    return int(row["targets"]["decision"] == DECISION_APPROVAL_REQUIRED)


# --------------------------------------------------------------------------- #
# Prediction DTO
# --------------------------------------------------------------------------- #
@dataclass
class CompiledDispatcherPrediction:
    recommended_load_id: Optional[str]
    decision: str
    recommended_bid: Optional[float]
    recommended_bid_rpm: Optional[float]
    bid_ratio: Optional[float]
    risk_adjusted_ev: Optional[float]
    approval_required: bool
    warnings: List[str]

    def explanation(self) -> str:
        if self.decision == DECISION_NO_BID:
            warns = ", ".join(self.warnings) if self.warnings else "no feasible/collectible bid"
            return f"No bid ({warns})."
        bid = (
            f"${self.recommended_bid:,.0f} ({self.recommended_bid_rpm:.2f}/mi)"
            if self.recommended_bid is not None
            else "n/a"
        )
        ev = f"{self.risk_adjusted_ev:.2f}" if self.risk_adjusted_ev is not None else "n/a"
        route = "human approval required" if self.approval_required else "auto-eligible"
        warns = ", ".join(self.warnings) if self.warnings else "none"
        return f"Recommend {bid}; risk-adjusted EV {ev}; {route}; warnings={warns}."

    def to_runtime_json(self) -> "OrderedDict[str, Any]":
        """The served 6-key contract (matches the Phase 6.2 ``runtime_json`` shape)."""
        return OrderedDict(
            [
                ("recommended_load_id", self.recommended_load_id),
                (
                    "recommended_bid",
                    round(self.recommended_bid, 2) if self.recommended_bid is not None else None,
                ),
                ("decision", self.decision),
                (
                    "risk_adjusted_ev",
                    round(self.risk_adjusted_ev, 2) if self.risk_adjusted_ev is not None else None,
                ),
                ("warnings", list(self.warnings)),
                ("explanation", self.explanation()),
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #
class _ClassifierHead:
    """A class-balanced HGB classifier that degrades to a constant when training is single-class."""

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state
        self.estimator: Optional[HistGradientBoostingClassifier] = None
        self.constant_label: Optional[Any] = None
        self.classes_: List[Any] = []

    def fit(self, X: pd.DataFrame, y: Sequence[Any]) -> "_ClassifierHead":
        y = np.asarray(y)
        self.classes_ = list(np.unique(y))
        if len(self.classes_) < 2:
            self.constant_label = self.classes_[0] if self.classes_ else 0
            return self
        self.estimator = HistGradientBoostingClassifier(
            **_HGB_CLF_KW, class_weight="balanced", random_state=self.random_state
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            return np.array([self.constant_label] * len(X))
        return self.estimator.predict(X)


class _RegressorHead:
    """An HGB regressor fit on a masked subset; degrades to a constant mean when empty."""

    def __init__(self, random_state: int) -> None:
        self.random_state = random_state
        self.estimator: Optional[HistGradientBoostingRegressor] = None
        self.constant_value: float = 0.0

    def fit(self, X: pd.DataFrame, y: Sequence[float]) -> "_RegressorHead":
        y = np.asarray(y, dtype=float)
        if len(y) == 0:
            self.constant_value = 0.0
            return self
        if len(np.unique(y)) < 2:
            self.constant_value = float(y[0])
            return self
        self.estimator = HistGradientBoostingRegressor(
            **_HGB_REG_KW, random_state=self.random_state
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.estimator is None:
            return np.full(len(X), self.constant_value, dtype=float)
        return self.estimator.predict(X)


# --------------------------------------------------------------------------- #
# The compiled model
# --------------------------------------------------------------------------- #
class CompiledDispatcherModel:
    def __init__(
        self,
        feature_manifest: Optional[Sequence[str]] = None,
        categorical_features: Sequence[str] = CATEGORICAL_FEATURES,
        random_state: int = DEFAULT_RANDOM_STATE,
    ) -> None:
        self.feature_manifest: List[str] = list(feature_manifest or default_feature_manifest())
        assert_manifest_inference_only(self.feature_manifest)
        self.categorical_features: List[str] = [
            c for c in categorical_features if c in self.feature_manifest
        ]
        self.random_state = random_state
        self.feature_manifest_hash = feature_manifest_hash(self.feature_manifest)
        self.category_levels_: Dict[str, List[Any]] = {}

        self.action_head = _ClassifierHead(random_state)
        self.bid_ratio_head = _RegressorHead(random_state + 1)
        self.ev_head = _RegressorHead(random_state + 2)
        self.warning_heads: Dict[str, _ClassifierHead] = {
            w: _ClassifierHead(random_state + 3 + i) for i, w in enumerate(WARNING_HEADS)
        }
        self.approval_head = _ClassifierHead(random_state + 10)
        self.provenance: Dict[str, Any] = {}
        self._fitted = False

    # ----- feature frame ----------------------------------------------------- #
    def _prepare(self, feature_dicts: Sequence[Mapping[str, Any]], *, fit: bool) -> pd.DataFrame:
        df = pd.DataFrame(
            [{k: fd.get(k) for k in self.feature_manifest} for fd in feature_dicts],
            columns=self.feature_manifest,
        )
        for col in self.feature_manifest:
            if col in self.categorical_features:
                if fit:
                    cat = df[col].astype("category")
                    self.category_levels_[col] = list(cat.cat.categories)
                    df[col] = cat
                else:
                    df[col] = pd.Categorical(
                        df[col], categories=self.category_levels_.get(col, [])
                    )
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        return df

    # ----- fit --------------------------------------------------------------- #
    def fit(self, rows: Sequence[Mapping[str, Any]]) -> "CompiledDispatcherModel":
        ordered = sorted(rows, key=lambda r: r["scenario_id"])
        feats = [row_features(r) for r in ordered]
        X = self._prepare(feats, fit=True)

        self.action_head.fit(X, [derive_action(r) for r in ordered])
        self.approval_head.fit(X, [derive_approval_required(r) for r in ordered])
        for w, head in self.warning_heads.items():
            head.fit(X, [derive_warnings(r)[w] for r in ordered])

        bid_ratios = [derive_bid_ratio(r) for r in ordered]
        bid_mask = [r is not None for r in bid_ratios]
        self.bid_ratio_head.fit(X[bid_mask], [v for v in bid_ratios if v is not None])

        evs = [derive_ev(r) for r in ordered]
        ev_mask = [v is not None for v in evs]
        self.ev_head.fit(X[ev_mask], [v for v in evs if v is not None])

        self._fitted = True
        return self

    # ----- manifest gating --------------------------------------------------- #
    def assert_compatible(self, incoming_manifest_hash: str) -> bool:
        if incoming_manifest_hash != self.feature_manifest_hash:
            raise FeatureManifestError(
                "feature manifest hash mismatch — refusing to serve: "
                f"artifact={self.feature_manifest_hash[:12]} incoming={str(incoming_manifest_hash)[:12]}"
            )
        return True

    def _validate_features(self, features: Mapping[str, Any]) -> None:
        missing = [f for f in self.feature_manifest if f not in features]
        if missing:
            raise FeatureManifestError(
                f"feature mapping is missing manifest fields — refusing to serve: {missing}"
            )

    # ----- predict ----------------------------------------------------------- #
    def predict_raw(self, feature_dicts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Raw per-head outputs (no reconciliation) — the evaluation surface."""
        if not self._fitted:
            raise RuntimeError("CompiledDispatcherModel.predict called before fit")
        for fd in feature_dicts:
            self._validate_features(fd)
        X = self._prepare(feature_dicts, fit=False)
        return {
            "action": self.action_head.predict(X),
            "bid_ratio": self.bid_ratio_head.predict(X),
            "risk_adjusted_ev": self.ev_head.predict(X),
            "warnings": {w: head.predict(X) for w, head in self.warning_heads.items()},
            "approval_required": self.approval_head.predict(X),
        }

    def predict_batch(
        self, feature_dicts: Sequence[Mapping[str, Any]]
    ) -> List[CompiledDispatcherPrediction]:
        raw = self.predict_raw(feature_dicts)
        actions = raw["action"]
        ratios = raw["bid_ratio"]
        evs = raw["risk_adjusted_ev"]
        approvals = raw["approval_required"]
        warn_preds = raw["warnings"]

        out: List[CompiledDispatcherPrediction] = []
        for i, fd in enumerate(feature_dicts):
            decision = str(actions[i])
            ratio = float(ratios[i])
            market = fd.get("market_rate")
            miles = fd.get("loaded_miles")
            # The EV head is trained on feasible rows only; the teacher emits a null risk-adjusted EV
            # exactly when the row is infeasible. Reproduce that contract by suppressing EV when the
            # model's own infeasibility signal (the no-feasible-bid warning) fires — this still emits
            # a (negative) EV for a feasible negative-EV no-bid, matching the teacher.
            infeasible = int(warn_preds[WARN_NO_FEASIBLE_BID][i]) == 1
            if decision == DECISION_NO_BID or market is None or miles is None:
                rpm = None
                amount = None
                ratio_out: Optional[float] = None
            else:
                rpm = ratio * float(market)
                amount = rpm * float(miles)
                ratio_out = ratio
            ev_out = None if infeasible else float(evs[i])
            warnings = [w for w in WARNING_HEADS if int(warn_preds[w][i]) == 1]
            out.append(
                CompiledDispatcherPrediction(
                    recommended_load_id=fd.get("load_id"),
                    decision=decision,
                    recommended_bid=amount,
                    recommended_bid_rpm=rpm,
                    bid_ratio=ratio_out,
                    risk_adjusted_ev=ev_out,
                    approval_required=bool(int(approvals[i]) == 1),
                    warnings=warnings,
                )
            )
        return out

    def predict_dto(self, features: Mapping[str, Any]) -> CompiledDispatcherPrediction:
        return self.predict_batch([features])[0]

    # ----- persistence ------------------------------------------------------- #
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_name": MODEL_NAME,
                "model_version": COMPILED_DISPATCHER_MODEL_VERSION,
                "feature_manifest": self.feature_manifest,
                "feature_manifest_hash": self.feature_manifest_hash,
                "categorical_features": self.categorical_features,
                "category_levels": self.category_levels_,
                "random_state": self.random_state,
                "action_head": self.action_head,
                "bid_ratio_head": self.bid_ratio_head,
                "ev_head": self.ev_head,
                "warning_heads": self.warning_heads,
                "approval_head": self.approval_head,
                "provenance": self.provenance,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CompiledDispatcherModel":
        payload = joblib.load(Path(path))
        obj = cls(
            feature_manifest=payload["feature_manifest"],
            categorical_features=payload["categorical_features"],
            random_state=payload.get("random_state", DEFAULT_RANDOM_STATE),
        )
        obj.feature_manifest_hash = payload["feature_manifest_hash"]
        obj.category_levels_ = payload.get("category_levels", {})
        obj.action_head = payload["action_head"]
        obj.bid_ratio_head = payload["bid_ratio_head"]
        obj.ev_head = payload["ev_head"]
        obj.warning_heads = payload["warning_heads"]
        obj.approval_head = payload["approval_head"]
        obj.provenance = payload.get("provenance", {})
        obj._fitted = True
        return obj

    def set_provenance(self, **kwargs: Any) -> None:
        self.provenance.update(kwargs)
        self.provenance.setdefault("trained_at", datetime.now(timezone.utc).isoformat())
        self.provenance["feature_manifest_hash"] = self.feature_manifest_hash
        self.provenance["model_name"] = MODEL_NAME
        self.provenance["model_version"] = COMPILED_DISPATCHER_MODEL_VERSION

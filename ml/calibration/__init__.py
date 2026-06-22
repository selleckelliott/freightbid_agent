"""Post-hoc probability recalibration (Phase 5.4).

Repairs flagged win-probability drift *without retraining the base model*: a lightweight,
monotonic recalibrator is fit on a recent window of labeled outcomes and wrapped around the
frozen winnability model, then promoted only if it provably improves calibration on a later,
disjoint holdout window. See :mod:`ml.calibration.recalibrator` (the map) and
:mod:`ml.calibration.recalibration_workflow` (fit → evaluate → promote).
"""

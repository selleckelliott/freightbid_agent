"""Model-monitoring utilities (Phase 5.3+).

Currently hosts the label-based **calibration drift monitor**: given predicted
probabilities and the realized binary outcomes they were meant to predict, is the model
still calibrated? Detection only — repair is Phase 5.4.
"""

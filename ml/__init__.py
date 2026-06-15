"""FreightBid ML layer (Phase 3.1).

The first learned component: a *destination desirability* model that predicts
``expected_next_deadhead_miles`` for a load's destination given decision-time
signals (market density, arrival time, equipment). Lower predicted next-deadhead
means a better place to end up.

Subpackages:
    data/      synthetic history schema, generator, label construction
    features/  decision-time feature builder + market zones
    models/    baselines + the scikit-learn regressor
    training/  dataset assembly, metrics, train/evaluate entry points
    artifacts/ saved model binary (gitignored) + committed metadata JSON

Nothing here wires into the OR-Tools planner yet; Phase 3.2 consumes the saved
model through ``application.destination_desirability_service``.
"""

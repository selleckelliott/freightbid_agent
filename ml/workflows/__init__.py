"""FreightBid training-time workflow graph + teacher trace generation (Phase 6).

Phase 6 turns the orchestrated FreightBid engine into an explicit, declarative workflow graph
(:mod:`ml.workflows.freightbid_workflow_graph`) and traces it with the real source-of-truth
engine (:mod:`ml.workflows.teacher_trace_generator`) to produce the teacher dataset a later
sub-phase compiles into a small dispatcher model.
"""

"""FreightBid workflow as an explicit, declarative graph (Phase 6.1).

This is the "procedure" an orchestration framework would otherwise inline into a system
prompt every turn. Phase 6 instead makes it an explicit graph so a teacher can trace it and
a later phase can *compile* it into a small model's weights.

The graph is **procedural control flow, not a second source of truth.** Each step node is
bound to a real source-of-truth engine capability; the terminal :data:`CHOOSE_ACTION` hub
**branches on the engine's recorded outputs** (feasibility, risk-adjusted EV sign, payment
risk, operational calibration severity) and **never recomputes the engine's formulas**. The
hub decision is therefore reproducible purely from
:class:`~ml.data.compiled_agent_trace_schema.NodeOutputs` (pinned by the Phase 6.1
procedural-hub test).

Data-path note
--------------
The validated Phase 4/5 bid engine operates on origin-market + load + broker + ask features;
the bid snapshot carries no destination coordinate. The graph is therefore faithful to the
**bid-decision engine** the project actually validated, not a destination planner:

* the paper's *"plan route"* node becomes :data:`ESTIMATE_HAUL_COST` — the real cost basis
  (``cost_per_loaded_mile * loaded_miles``) the recommender consumes;
* the paper's *"score destination risk"* node becomes :data:`SCORE_MARKET_CONTEXT` — an
  honest market-desirability read (market rate vs. breakeven), recorded as an informational
  node output and **never** used as a hub predicate.

Everything else maps one-to-one onto the paper's flowchart.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ml.data.compiled_agent_trace_schema import (
    APPROVAL_AUTO_ELIGIBLE,
    APPROVAL_HUMAN_REQUIRED,
    APPROVAL_NOT_APPLICABLE,
    DECISION_APPROVAL_REQUIRED,
    DECISION_BID,
    DECISION_NO_BID,
)

# Bumped when the node set / edges / hub predicates change; stamped onto every trace.
WORKFLOW_GRAPH_VERSION = "1.0.0"

# -- Node names (the paper's flowchart, adapted to the bid-decision engine) -----------
START = "start"
INTAKE_GOAL = "intake_dispatcher_goal"
READ_TRUCK = "read_truck_state"
INSPECT_BOARD = "inspect_load_board"
FILTER_INFEASIBLE = "filter_infeasible_loads"
ESTIMATE_HAUL_COST = "estimate_haul_cost"          # adapts "plan route"
SCORE_MARKET_CONTEXT = "score_market_context"      # adapts "score destination risk"
ESTIMATE_WIN_PROBABILITY = "estimate_win_probability"
ESTIMATE_PAYMENT_RISK = "estimate_payment_risk"
COMPUTE_RISK_ADJUSTED_EV = "compute_risk_adjusted_ev"
CHECK_CALIBRATION = "check_calibration"
CHOOSE_ACTION = "choose_action"                    # the decision hub
EXPLAIN = "explain_recommendation"
# Terminal states (one per recommendable action).
TERMINAL_BID = "terminal_bid"
TERMINAL_NO_BID = "terminal_no_bid"
TERMINAL_APPROVAL_REQUIRED = "terminal_approval_required"

TERMINAL_BY_DECISION = {
    DECISION_BID: TERMINAL_BID,
    DECISION_NO_BID: TERMINAL_NO_BID,
    DECISION_APPROVAL_REQUIRED: TERMINAL_APPROVAL_REQUIRED,
}

# Hub branch labels (each must be reachable across a seeded batch / the hub unit test).
BRANCH_INFEASIBLE = "infeasible"
BRANCH_NEGATIVE_RISK_ADJUSTED_EV = "negative_risk_adjusted_ev"
BRANCH_ESCALATED = "escalated"
BRANCH_CLEAN_BID = "clean_bid"
BRANCH_TERMINAL = {
    BRANCH_INFEASIBLE: TERMINAL_NO_BID,
    BRANCH_NEGATIVE_RISK_ADJUSTED_EV: TERMINAL_NO_BID,
    BRANCH_ESCALATED: TERMINAL_APPROVAL_REQUIRED,
    BRANCH_CLEAN_BID: TERMINAL_BID,
}

# Warning codes the hub may attach.
WARN_PAYMENT_RISK = "payment_risk"
WARN_CALIBRATION_ALERT = "calibration_alert"
WARN_CALIBRATION_WATCH = "calibration_watch"
WARN_NEGATIVE_RISK_ADJUSTED_EV = "negative_risk_adjusted_ev"
WARN_NO_FEASIBLE_BID = "no_feasible_bid"

# Operational calibration severities (mirrors ml.monitoring.calibration_drift).
SEV_OK = "OK"
SEV_WATCH = "WATCH"
SEV_ALERT = "ALERT"


@dataclass(frozen=True)
class Node:
    """One workflow node bound to a real engine capability."""

    name: str
    kind: str  # "start" | "step" | "hub" | "terminal"
    capability: str
    description: str


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    condition: Optional[str] = None  # branch label on routed out-edges; None on linear edges


@dataclass(frozen=True)
class DecisionHubPolicy:
    """Config-overridable thresholds for the ``choose_action`` hub.

    Defaults are deliberately simple and explainable. They are *routing* thresholds applied
    to engine outputs — they do not recompute any engine quantity.
    """

    payment_default_warn: float = 0.15      # p_default at target >= this -> payment warning + escalate
    alert_escalates: bool = True            # operational calibration ALERT -> warning + escalate
    watch_warns: bool = True                # operational calibration WATCH -> warning (no forced escalate)


@dataclass(frozen=True)
class HubSignals:
    """The *only* inputs the hub reads — a strict subset of recorded ``node_outputs``.

    Constructing this from a :class:`NodeOutputs` (see :meth:`from_node_outputs`) and feeding
    it to :func:`decide` reproduces the routing exactly, which is how the procedural-hub test
    proves the graph never introduces independent calculations.
    """

    feasible: bool
    payment_risk_available: bool
    risk_adjusted_ev_positive: Optional[bool]
    p_default_at_target: Optional[float]
    calibration_severity_operational: str

    @classmethod
    def from_node_outputs(cls, node_outputs) -> "HubSignals":
        return cls(
            feasible=node_outputs.feasible,
            payment_risk_available=node_outputs.payment_risk_available,
            risk_adjusted_ev_positive=node_outputs.risk_adjusted_ev_positive,
            p_default_at_target=node_outputs.p_default_at_target,
            calibration_severity_operational=node_outputs.calibration_severity_operational,
        )


@dataclass(frozen=True)
class HubDecision:
    decision: str
    terminal_state: str
    warnings: List[str]
    approval_decision: str
    branch: str


def decide(signals: HubSignals, policy: Optional[DecisionHubPolicy] = None) -> HubDecision:
    """Route one scenario to a terminal action from engine outputs alone.

    Branch order (first match wins):

    1. **infeasible** — no in-support, guardrail-clearing candidate -> ``no_bid``.
    2. **negative_risk_adjusted_ev** — payment risk available and the best in-support ask is
       collectible-EV-negative -> ``no_bid`` (the engine's honest "every ask loses money").
    3. **escalated** — feasible and EV-positive but a risk warning fires (high payment-default
       probability or an operational calibration ALERT) -> ``approval_required``.
    4. **clean_bid** — feasible, EV-positive, no escalating warning -> ``bid`` (a non-escalating
       calibration ``WATCH`` is still surfaced as a warning).
    """
    policy = policy or DecisionHubPolicy()
    warnings: List[str] = []

    if not signals.feasible:
        return HubDecision(
            decision=DECISION_NO_BID,
            terminal_state=TERMINAL_NO_BID,
            warnings=[WARN_NO_FEASIBLE_BID],
            approval_decision=APPROVAL_NOT_APPLICABLE,
            branch=BRANCH_INFEASIBLE,
        )

    if signals.payment_risk_available and signals.risk_adjusted_ev_positive is False:
        return HubDecision(
            decision=DECISION_NO_BID,
            terminal_state=TERMINAL_NO_BID,
            warnings=[WARN_NEGATIVE_RISK_ADJUSTED_EV],
            approval_decision=APPROVAL_NOT_APPLICABLE,
            branch=BRANCH_NEGATIVE_RISK_ADJUSTED_EV,
        )

    escalate = False
    if (
        signals.payment_risk_available
        and signals.p_default_at_target is not None
        and signals.p_default_at_target >= policy.payment_default_warn
    ):
        warnings.append(WARN_PAYMENT_RISK)
        escalate = True

    sev = signals.calibration_severity_operational
    if sev == SEV_ALERT and policy.alert_escalates:
        warnings.append(WARN_CALIBRATION_ALERT)
        escalate = True
    elif sev == SEV_WATCH and policy.watch_warns:
        warnings.append(WARN_CALIBRATION_WATCH)

    if escalate:
        return HubDecision(
            decision=DECISION_APPROVAL_REQUIRED,
            terminal_state=TERMINAL_APPROVAL_REQUIRED,
            warnings=warnings,
            approval_decision=APPROVAL_HUMAN_REQUIRED,
            branch=BRANCH_ESCALATED,
        )
    return HubDecision(
        decision=DECISION_BID,
        terminal_state=TERMINAL_BID,
        warnings=warnings,
        approval_decision=APPROVAL_AUTO_ELIGIBLE,
        branch=BRANCH_CLEAN_BID,
    )


class WorkflowGraphError(ValueError):
    """Raised when the declared graph violates a structural invariant."""


@dataclass(frozen=True)
class WorkflowGraph:
    """An ordered, validated description of the FreightBid decision procedure."""

    nodes: Tuple[Node, ...]
    edges: Tuple[Edge, ...]
    version: str = WORKFLOW_GRAPH_VERSION

    # -- lookups ----------------------------------------------------------------
    def node(self, name: str) -> Node:
        for n in self.nodes:
            if n.name == name:
                return n
        raise KeyError(name)

    def successors(self, name: str) -> List[Edge]:
        return [e for e in self.edges if e.src == name]

    def predecessors(self, name: str) -> List[Edge]:
        return [e for e in self.edges if e.dst == name]

    def start(self) -> Node:
        starts = [n for n in self.nodes if n.kind == "start"]
        if len(starts) != 1:
            raise WorkflowGraphError(f"expected exactly one START node, found {len(starts)}")
        return starts[0]

    def hub(self) -> Node:
        hubs = [n for n in self.nodes if n.kind == "hub"]
        if len(hubs) != 1:
            raise WorkflowGraphError(f"expected exactly one hub node, found {len(hubs)}")
        return hubs[0]

    def terminals(self) -> List[Node]:
        return [n for n in self.nodes if n.kind == "terminal"]

    def hub_branch_labels(self) -> List[str]:
        return [e.condition for e in self.successors(self.hub().name) if e.condition is not None]

    def terminals_reachable_from(self, name: str) -> set:
        seen, out, stack = set(), set(), [name]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            if self.node(u).kind == "terminal":
                out.add(u)
            stack.extend(e.dst for e in self.successors(u))
        return out

    def linear_prefix(self) -> List[str]:
        """The deterministic step sequence START..hub (the nodes every trace walks in order)."""
        order: List[str] = []
        cur = self.start().name
        hub = self.hub().name
        seen = set()
        while True:
            order.append(cur)
            if cur in seen:  # safety; validate() rules this out
                raise WorkflowGraphError("cycle in linear prefix")
            seen.add(cur)
            if cur == hub:
                return order
            outs = self.successors(cur)
            if len(outs) != 1:
                raise WorkflowGraphError(
                    f"node {cur!r} on the linear prefix must have exactly one successor"
                )
            cur = outs[0].dst

    def route(self, branch: str) -> List[str]:
        """The nodes traversed after the hub for ``branch`` (follows matching conditioned edges).

        Returns ``[hub_successor, ..., terminal]`` — for the default graph this is
        ``[EXPLAIN, terminal]``. Used by the teacher generator and the procedural-hub test so
        the recorded path is exactly what the declared graph dictates.
        """
        hub = self.hub().name
        cur = hub
        out: List[str] = []
        seen = set()
        while self.node(cur).kind != "terminal":
            edges = self.successors(cur)
            match = [e for e in edges if e.condition == branch]
            chosen = match[0] if match else [e for e in edges if e.condition is None][0]
            cur = chosen.dst
            if cur in seen:
                raise WorkflowGraphError(f"cycle routing branch {branch!r}")
            seen.add(cur)
            out.append(cur)
        return out

    # -- validation -------------------------------------------------------------
    def validate(self) -> bool:
        """Single START, acyclic, fully reachable, every path ends at a terminal, hub branches
        cover all decisions. Raises :class:`WorkflowGraphError` on any violation."""
        names = [n.name for n in self.nodes]
        if len(names) != len(set(names)):
            raise WorkflowGraphError("duplicate node names")
        name_set = set(names)
        for e in self.edges:
            if e.src not in name_set or e.dst not in name_set:
                raise WorkflowGraphError(f"edge references unknown node: {e}")

        start = self.start()
        hub = self.hub()  # exactly one
        if not self.terminals():
            raise WorkflowGraphError("graph has no terminal nodes")

        # START has no predecessors.
        if self.predecessors(start.name):
            raise WorkflowGraphError("START node must have no incoming edges")

        # Terminals are sinks; every non-terminal has >=1 successor.
        for n in self.nodes:
            outs = self.successors(n.name)
            if n.kind == "terminal":
                if outs:
                    raise WorkflowGraphError(f"terminal {n.name!r} must have no outgoing edges")
            elif not outs:
                raise WorkflowGraphError(f"non-terminal {n.name!r} must have >=1 successor")

        self._assert_acyclic()
        self._assert_reachable(start.name)
        self._assert_all_reach_terminal()

        # Hub branch coverage: every out-edge is labelled + unique, and every decision's
        # terminal is reachable from the hub.
        hub_edges = self.successors(hub.name)
        labels = [e.condition for e in hub_edges]
        if any(c is None for c in labels):
            raise WorkflowGraphError("every hub out-edge must carry a branch condition")
        if len(labels) != len(set(labels)):
            raise WorkflowGraphError("duplicate hub branch labels")
        reach = self.terminals_reachable_from(hub.name)
        for term in TERMINAL_BY_DECISION.values():
            if term not in reach:
                raise WorkflowGraphError(f"hub cannot reach terminal {term!r}")
        return True

    def _assert_acyclic(self) -> None:
        WHITE, GREY, BLACK = 0, 1, 2
        color = {n.name: WHITE for n in self.nodes}
        adj: Dict[str, List[str]] = {
            n.name: [e.dst for e in self.successors(n.name)] for n in self.nodes
        }

        def visit(u: str) -> None:
            color[u] = GREY
            for v in adj[u]:
                if color[v] == GREY:
                    raise WorkflowGraphError(f"cycle detected via edge {u} -> {v}")
                if color[v] == WHITE:
                    visit(v)
            color[u] = BLACK

        for n in self.nodes:
            if color[n.name] == WHITE:
                visit(n.name)

    def _assert_reachable(self, start_name: str) -> None:
        seen = set()
        stack = [start_name]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            stack.extend(e.dst for e in self.successors(u))
        missing = {n.name for n in self.nodes} - seen
        if missing:
            raise WorkflowGraphError(f"unreachable nodes from START: {sorted(missing)}")

    def _assert_all_reach_terminal(self) -> None:
        terminals = {n.name for n in self.terminals()}
        adj: Dict[str, List[str]] = {
            n.name: [e.dst for e in self.successors(n.name)] for n in self.nodes
        }
        memo: Dict[str, bool] = {}

        def reaches(u: str, stack: set) -> bool:
            if u in terminals:
                return True
            if u in memo:
                return memo[u]
            stack.add(u)
            ok = any(v not in stack and reaches(v, stack) for v in adj[u])
            stack.discard(u)
            memo[u] = ok
            return ok

        for n in self.nodes:
            if not reaches(n.name, set()):
                raise WorkflowGraphError(f"node {n.name!r} cannot reach any terminal")


def build_default_graph() -> WorkflowGraph:
    """The canonical Phase 6.1 FreightBid workflow graph (validated before return)."""
    nodes: Tuple[Node, ...] = (
        Node(START, "start", "-", "Entry point."),
        Node(INTAKE_GOAL, "step", "dispatcher goal",
             "Intake the dispatcher goal: best bid/no-bid plan for this truck and board."),
        Node(READ_TRUCK, "step", "truck state",
             "Read the carrier's truck state: equipment + per-loaded-mile cost basis."),
        Node(INSPECT_BOARD, "step", "load board",
             "Inspect the board load: observable load + broker columns (BidQuery.from_snapshot)."),
        Node(FILTER_INFEASIBLE, "step", "EVBidRecommender.score",
             "Filter infeasible loads: require an in-support, guardrail-clearing candidate ask."),
        Node(ESTIMATE_HAUL_COST, "step", "cost basis",
             "Estimate haul cost (cost_per_loaded_mile * miles) and breakeven rpm "
             "(adapts 'plan route' to the bid engine's real cost basis)."),
        Node(SCORE_MARKET_CONTEXT, "step", "market_rate_for",
             "Score market context: market rate vs. breakeven desirability "
             "(adapts 'score destination risk'; informational, not a hub predicate)."),
        Node(ESTIMATE_WIN_PROBABILITY, "step", "WinnabilityPort",
             "Estimate win probability at the recommended ask (calibrated winnability model)."),
        Node(ESTIMATE_PAYMENT_RISK, "step", "PaymentRiskPort",
             "Estimate payment risk: p_default, p_collect, expected pay-days."),
        Node(COMPUTE_RISK_ADJUSTED_EV, "step", "EVBidRecommender.recommend",
             "Compute the risk-adjusted EV ladder and the recommended target ask."),
        Node(CHECK_CALIBRATION, "step", "calibration monitor + recalibration",
             "Check calibration drift / recalibration status (operational severity)."),
        Node(CHOOSE_ACTION, "hub", "decision hub",
             "Choose bid / no-bid / approval-required by branching on engine outputs."),
        Node(EXPLAIN, "step", "explanation",
             "Explain the recommendation in terms of the signals that drove it."),
        Node(TERMINAL_BID, "terminal", "-", "Recommend submitting the bid (auto-eligible)."),
        Node(TERMINAL_NO_BID, "terminal", "-",
             "Recommend not bidding (infeasible or collectible-EV-negative)."),
        Node(TERMINAL_APPROVAL_REQUIRED, "terminal", "-",
             "Recommend escalating to a human approver before any bid."),
    )

    linear = [
        START, INTAKE_GOAL, READ_TRUCK, INSPECT_BOARD, FILTER_INFEASIBLE,
        ESTIMATE_HAUL_COST, SCORE_MARKET_CONTEXT, ESTIMATE_WIN_PROBABILITY,
        ESTIMATE_PAYMENT_RISK, COMPUTE_RISK_ADJUSTED_EV, CHECK_CALIBRATION, CHOOSE_ACTION,
    ]
    edges: List[Edge] = [Edge(linear[i], linear[i + 1]) for i in range(len(linear) - 1)]
    # Each hub branch passes through EXPLAIN to its terminal; the branch label is carried on
    # both legs so a trace's post-hub path is fully determined by the hub's branch.
    for branch, terminal in BRANCH_TERMINAL.items():
        edges.append(Edge(CHOOSE_ACTION, EXPLAIN, condition=branch))
        edges.append(Edge(EXPLAIN, terminal, condition=branch))

    graph = WorkflowGraph(nodes=nodes, edges=tuple(edges))
    graph.validate()
    return graph

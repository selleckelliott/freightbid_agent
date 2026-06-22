"""Bid recommendation domain models (Phase 4.3).

The output of the expected-value bid recommender: a small **ladder** of candidate
asks, each annotated with its win probability, profit-if-won, and expected value, so a
human dispatcher sees the margin-vs-win-probability tradeoff rather than a single
opaque number. Pure value objects — no model, IO, or framework concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

# Ladder rung labels (ordered conservative -> stretch).
CONSERVATIVE = "conservative"
TARGET = "target"
MAX_EV = "max_ev"
STRETCH = "stretch"
LADDER_LABELS = (CONSERVATIVE, TARGET, MAX_EV, STRETCH)


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate ask scored for expected value, before ladder labeling.

    ``extrapolated`` is ``True`` when the ask's ``ask_to_market_ratio`` falls outside
    the model's trained support, so it is excluded from ladder selection (but still
    returned for charting/diagnostics).
    """

    ask_rpm: float
    ask_amount: float
    profit_if_won: float
    win_probability: float
    expected_value: float
    extrapolated: bool
    # -- Phase 5.1: optional risk-adjusted EV (None unless payment risk is wired) --
    # The objective shifts from "expected profit if won" to "expected *collectible*
    # profit after default + payment-delay risk". All None when payment risk is off or
    # unavailable, so ``ranking_ev`` collapses to ``expected_value`` and behavior is
    # identical to Phase 4.3.
    risk_adjusted_ev: Optional[float] = None
    p_default: Optional[float] = None
    p_collect: Optional[float] = None
    expected_pay_days: Optional[float] = None
    delay_penalty: Optional[float] = None
    expected_collected_revenue: Optional[float] = None
    risk_adjusted_profit_if_won: Optional[float] = None

    @property
    def ranking_ev(self) -> float:
        """The objective the ladder ranks by: risk-adjusted EV when payment risk is
        available, else the raw expected value (so risk-off ranking is unchanged)."""
        return self.risk_adjusted_ev if self.risk_adjusted_ev is not None else self.expected_value


@dataclass(frozen=True)
class CandidateScoring:
    """The full scored candidate curve for one load — the raw material a ladder is
    selected from, and what the offline benchmark / chart consume directly."""

    estimated_cost: float
    market_rate: float
    breakeven_rpm: float
    candidates: List[ScoredCandidate]


@dataclass(frozen=True)
class BidOption:
    """One candidate ask scored for expected value."""

    label: str
    ask_amount: float
    ask_rpm: float
    estimated_cost: float
    profit_if_won: float
    win_probability: float
    expected_value: float
    extrapolated: bool
    rationale: str
    # -- Phase 5.1: optional risk-adjusted EV (None unless payment risk is wired) --
    risk_adjusted_ev: Optional[float] = None
    p_default: Optional[float] = None
    p_collect: Optional[float] = None
    expected_pay_days: Optional[float] = None
    delay_penalty: Optional[float] = None
    expected_collected_revenue: Optional[float] = None
    risk_adjusted_profit_if_won: Optional[float] = None

    @property
    def ranking_ev(self) -> float:
        """Risk-adjusted EV when available, else raw expected value."""
        return self.risk_adjusted_ev if self.risk_adjusted_ev is not None else self.expected_value


@dataclass(frozen=True)
class BidRecommendation:
    """A scored bid ladder for one load, with a recommended default (``target``).

    ``winnability_available`` is ``False`` when no winnability model is wired (the
    no-op adapter): the recommender then falls back to the existing cost-plus-margin
    target and the ladder collapses to that single, EV-free option.
    """

    load_id: int
    broker_id: Optional[str]
    estimated_cost: float
    breakeven_ask: float
    market_rate: float
    options: List[BidOption]
    recommended_label: str
    recommended_ask: float
    winnability_available: bool
    rationale: str
    # -- Phase 5.1: risk-adjusted EV (defaults preserve pre-5.1 construction) -----
    # ``payment_risk_available`` is True only when the risk-adjusted objective was
    # actually applied (flag on AND a payment estimate was produced). When every
    # in-support candidate has a negative risk-adjusted EV the recommender still
    # returns its best (least-negative) option but flags it via
    # ``risk_adjusted_ev_positive=False`` + ``risk_adjusted_warning`` rather than
    # suppressing the recommendation.
    payment_risk_available: bool = False
    risk_adjusted_ev_positive: Optional[bool] = None
    risk_adjusted_warning: Optional[str] = None

    def option(self, label: str) -> Optional[BidOption]:
        """Return the ladder rung with ``label``, or ``None`` if it was omitted."""
        return next((o for o in self.options if o.label == label), None)

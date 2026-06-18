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

    def option(self, label: str) -> Optional[BidOption]:
        """Return the ladder rung with ``label``, or ``None`` if it was omitted."""
        return next((o for o in self.options if o.label == label), None)

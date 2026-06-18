"""Expected-value bid recommender (Phase 4.3).

Turns the calibrated Phase 4.2 winnability model — accessed through a
:class:`~ports.winnability.WinnabilityPort` — into a human-reviewable **bid ladder**.
For a load and a candidate ask,

    profit_if_won = ask_amount - estimated_total_cost
    EV(ask)       = P(win | ask) x profit_if_won

A higher ask lifts ``profit_if_won`` but lowers ``P(win)``, so EV peaks at an interior
ask. Rather than emit one number we return four rungs — conservative / target /
max-EV / stretch — and recommend **target** (near-max-EV with a sane win floor), which
keeps a human in the loop.

Two design rules keep the EV curve trustworthy:

* **Candidates are market-anchored.** The 4.2 model was trained against
  ``market_rate``; candidates are generated as ``market_rate x multiplier`` so they
  live in the model's trained support. Posted-rate and breakeven anchors are added,
  but any ask whose ``ask_to_market_ratio`` falls outside the trained envelope is
  flagged ``extrapolated`` and excluded from ladder selection.
* **No model ⇒ today's behavior.** When the port returns ``None`` (no-op adapter) the
  recommender falls back to a cost-plus-margin target — zero behavior change.
"""
from __future__ import annotations

from math import isnan, nan
from typing import Callable, List, Optional, Sequence

from application.config_loader import BidRecommenderConfig
from domain.models.bid_recommendation import (
    CONSERVATIVE,
    MAX_EV,
    STRETCH,
    TARGET,
    BidOption,
    BidRecommendation,
    CandidateScoring,
    ScoredCandidate,
)
from ml.features.winnability_features import BidQuery, market_rate_for
from ports.winnability import WinnabilityPort

# A fallback strategy maps (query, estimated_cost, market_rate, loaded_miles) -> ask $.
MarginFallback = Callable[[BidQuery, float, float, float], float]


class EVBidRecommender:
    def __init__(
        self,
        winnability: WinnabilityPort,
        config: BidRecommenderConfig,
        margin_fallback: Optional[MarginFallback] = None,
    ) -> None:
        self._win = winnability
        self._cfg = config
        self._margin_fallback = margin_fallback or self._default_margin_fallback

    # -- public API --------------------------------------------------------
    def score(
        self,
        query: BidQuery,
        *,
        estimated_total_cost: Optional[float] = None,
    ) -> Optional[CandidateScoring]:
        """Score the full candidate curve for one load, or ``None`` if no model.

        Public so the offline benchmark and chart consume the *exact* candidate set
        (asks, model ``P(win)``, profit, EV, extrapolation flag) the ladder is selected
        from — no re-derivation, no drift. ``None`` mirrors the port: no winnability
        signal available.
        """
        cfg = self._cfg
        miles = max(float(query.loaded_miles), 1.0)
        cost = (
            float(estimated_total_cost)
            if estimated_total_cost is not None
            else cfg.cost_per_loaded_mile * miles
        )
        market_rate = market_rate_for(query.origin_lat, query.origin_lon)
        breakeven_rpm = cost / miles
        posted_rpm = query.rate_per_mile  # None for "call for rate"

        candidate_rpms = self._candidate_rpms(market_rate, breakeven_rpm, posted_rpm)
        probs = self._win.win_probabilities(query, candidate_rpms)
        if probs is None:
            return None

        candidates = self._score(candidate_rpms, probs, cost, miles, market_rate)
        return CandidateScoring(
            estimated_cost=round(cost, 2),
            market_rate=round(market_rate, 4),
            breakeven_rpm=round(breakeven_rpm, 4),
            candidates=candidates,
        )

    def recommend(
        self,
        query: BidQuery,
        *,
        load_id: int,
        broker_id: Optional[str] = None,
        estimated_total_cost: Optional[float] = None,
    ) -> BidRecommendation:
        """Recommend a bid ladder for one board load.

        ``estimated_total_cost`` overrides the per-loaded-mile proxy — a live caller
        passes ``LoadEvaluation.total_cost``; the synthetic benchmark passes ``None``.
        """
        cfg = self._cfg
        miles = max(float(query.loaded_miles), 1.0)
        cost = (
            float(estimated_total_cost)
            if estimated_total_cost is not None
            else cfg.cost_per_loaded_mile * miles
        )
        market_rate = market_rate_for(query.origin_lat, query.origin_lon)
        breakeven_rpm = cost / miles

        scoring = self.score(query, estimated_total_cost=cost)
        if scoring is None:
            return self._fallback_recommendation(
                query, load_id, broker_id, cost, market_rate, breakeven_rpm, miles
            )

        eligible = [s for s in scoring.candidates if not s.extrapolated]
        if not eligible:
            return self._fallback_recommendation(
                query, load_id, broker_id, cost, market_rate, breakeven_rpm, miles,
                winnability_available=True,
                note="no in-support candidate cleared the guardrails; "
                "served cost-plus-margin",
            )

        options = self._build_ladder(eligible, market_rate, cost)
        recommended = self._pick_recommended(options)
        return BidRecommendation(
            load_id=load_id,
            broker_id=broker_id,
            estimated_cost=round(cost, 2),
            breakeven_ask=round(cost, 2),
            market_rate=round(market_rate, 4),
            options=options,
            recommended_label=recommended.label,
            recommended_ask=recommended.ask_amount,
            winnability_available=True,
            rationale=self._summary_rationale(options, recommended, cost, market_rate),
        )

    # -- candidate generation ---------------------------------------------
    def _candidate_rpms(
        self, market_rate: float, breakeven_rpm: float, posted_rpm: Optional[float]
    ) -> List[float]:
        cfg = self._cfg
        raw: List[float] = [market_rate * m for m in cfg.anchor_multipliers]
        raw.append(breakeven_rpm + cfg.min_margin_rpm)
        if posted_rpm is not None and posted_rpm > 0:
            raw.append(posted_rpm)

        anchor = posted_rpm if (posted_rpm and posted_rpm > 0) else market_rate
        max_rpm = min(cfg.max_rate_per_mile, anchor * cfg.max_anchor_multiplier)
        min_rpm = max(cfg.min_rate_per_mile, breakeven_rpm + cfg.min_margin_rpm)

        kept = set()
        for rpm in raw:
            rpm = round(rpm, 4)
            if rpm < min_rpm or rpm > max_rpm:
                continue  # guardrail: below-margin or absurd-high asks are dropped
            kept.add(rpm)
        ordered = sorted(kept)
        if len(ordered) > cfg.max_candidate_count:
            ordered = self._thin(ordered, cfg.max_candidate_count)
        return ordered

    @staticmethod
    def _thin(values: List[float], cap: int) -> List[float]:
        """Keep an evenly-spaced subset (always including the endpoints)."""
        if cap <= 1 or len(values) <= cap:
            return values
        step = (len(values) - 1) / (cap - 1)
        idx = sorted({round(i * step) for i in range(cap)})
        return [values[i] for i in idx]

    def _score(
        self,
        rpms: Sequence[float],
        probs: Sequence[float],
        cost: float,
        miles: float,
        market_rate: float,
    ) -> List[ScoredCandidate]:
        cfg = self._cfg
        out: List[ScoredCandidate] = []
        for rpm, p in zip(rpms, probs):
            ask = rpm * miles
            profit = ask - cost
            if profit < cfg.min_profit_dollars:
                continue  # guardrail: profit floor
            ratio = rpm / market_rate if market_rate > 0 else nan
            extrapolated = isnan(ratio) or not (
                cfg.trained_ask_ratio_min <= ratio <= cfg.trained_ask_ratio_max
            )
            out.append(
                ScoredCandidate(
                    ask_rpm=round(rpm, 4),
                    ask_amount=round(ask, 2),
                    profit_if_won=round(profit, 2),
                    win_probability=float(p),
                    expected_value=round(p * profit, 2),
                    extrapolated=extrapolated,
                )
            )
        return out

    # -- ladder ------------------------------------------------------------
    def _build_ladder(
        self, eligible: List[ScoredCandidate], market_rate: float, cost: float
    ) -> List[BidOption]:
        cfg = self._cfg
        max_ev_c = max(eligible, key=lambda s: s.expected_value)
        max_ev = max_ev_c.expected_value

        cons_pool = [s for s in eligible if s.win_probability >= cfg.conservative_min_win_prob]
        conservative = max(cons_pool, key=lambda s: s.expected_value) if cons_pool else None

        tgt_pool = [
            s
            for s in eligible
            if s.expected_value >= cfg.target_ev_tolerance * max_ev
            and s.win_probability >= cfg.target_min_win_prob
        ]
        target = max(tgt_pool, key=lambda s: s.expected_value) if tgt_pool else None

        str_pool = [s for s in eligible if s.win_probability >= cfg.stretch_min_win_prob]
        stretch = max(str_pool, key=lambda s: s.ask_rpm) if str_pool else None

        rungs = [
            (CONSERVATIVE, conservative),
            (TARGET, target),
            (MAX_EV, max_ev_c),
            (STRETCH, stretch),
        ]
        options: List[BidOption] = []
        for label, s in rungs:
            if s is None:
                continue
            options.append(
                BidOption(
                    label=label,
                    ask_amount=s.ask_amount,
                    ask_rpm=s.ask_rpm,
                    estimated_cost=round(cost, 2),
                    profit_if_won=s.profit_if_won,
                    win_probability=round(s.win_probability, 4),
                    expected_value=s.expected_value,
                    extrapolated=s.extrapolated,
                    rationale=self._rung_rationale(label, s, max_ev),
                )
            )
        return options

    def _pick_recommended(self, options: List[BidOption]) -> BidOption:
        by_label = {o.label: o for o in options}
        return by_label.get(TARGET) or by_label[MAX_EV]

    # -- rationale ---------------------------------------------------------
    def _rung_rationale(self, label: str, s: ScoredCandidate, max_ev: float) -> str:
        ev_pct = (s.expected_value / max_ev * 100.0) if max_ev > 0 else 0.0
        if label == CONSERVATIVE:
            return (
                f"High win probability {s.win_probability:.0%} at ${s.ask_amount:,.0f} "
                f"(${s.ask_rpm:.2f}/mi); EV ${s.expected_value:,.0f} ({ev_pct:.0f}% of max)."
            )
        if label == TARGET:
            return (
                f"Balanced default: captures {ev_pct:.0f}% of max EV "
                f"(${s.expected_value:,.0f}) while keeping win probability at "
                f"{s.win_probability:.0%}."
            )
        if label == MAX_EV:
            return (
                f"Mathematically strongest EV ${s.expected_value:,.0f} at "
                f"${s.ask_amount:,.0f} (win probability {s.win_probability:.0%})."
            )
        return (
            f"Higher margin ${s.profit_if_won:,.0f} at ${s.ask_amount:,.0f} but win "
            f"probability drops to {s.win_probability:.0%} (EV ${s.expected_value:,.0f})."
        )

    def _summary_rationale(
        self,
        options: List[BidOption],
        recommended: BidOption,
        cost: float,
        market_rate: float,
    ) -> str:
        by_label = {o.label: o for o in options}
        parts = [
            f"Estimated cost ${cost:,.0f} (market ${market_rate:.2f}/mi). "
            f"Recommended {recommended.label} bid ${recommended.ask_amount:,.0f} "
            f"(win {recommended.win_probability:.0%}, EV ${recommended.expected_value:,.0f})."
        ]
        cons = by_label.get(CONSERVATIVE)
        if cons and cons.ask_amount < recommended.ask_amount:
            parts.append(
                f"A lower conservative bid ${cons.ask_amount:,.0f} raises win "
                f"probability to {cons.win_probability:.0%} but cuts EV."
            )
        stretch = by_label.get(STRETCH)
        if stretch and stretch.ask_amount > recommended.ask_amount:
            parts.append(
                f"A stretch bid ${stretch.ask_amount:,.0f} lifts margin but drops "
                f"win probability to {stretch.win_probability:.0%}."
            )
        return " ".join(parts)

    # -- no-model fallback -------------------------------------------------
    def _default_margin_fallback(
        self, query: BidQuery, cost: float, market_rate: float, miles: float
    ) -> float:
        return cost * (1.0 + self._cfg.fallback_target_margin)

    def _fallback_recommendation(
        self,
        query: BidQuery,
        load_id: int,
        broker_id: Optional[str],
        cost: float,
        market_rate: float,
        breakeven_rpm: float,
        miles: float,
        *,
        winnability_available: bool = False,
        note: Optional[str] = None,
    ) -> BidRecommendation:
        cfg = self._cfg
        target_ask = self._margin_fallback(query, cost, market_rate, miles)
        target_ask = max(target_ask, cost + cfg.min_margin_rpm * miles)
        target_ask = max(target_ask, cfg.min_rate_per_mile * miles)
        target_ask = min(target_ask, cfg.max_rate_per_mile * miles)
        rpm = target_ask / miles
        profit = target_ask - cost
        reason = note or "no winnability model wired"
        option = BidOption(
            label=TARGET,
            ask_amount=round(target_ask, 2),
            ask_rpm=round(rpm, 4),
            estimated_cost=round(cost, 2),
            profit_if_won=round(profit, 2),
            win_probability=nan,
            expected_value=nan,
            extrapolated=False,
            rationale=(
                f"Cost-plus-margin fallback ({reason}): "
                f"${target_ask:,.0f} at margin {cfg.fallback_target_margin:.0%}."
            ),
        )
        return BidRecommendation(
            load_id=load_id,
            broker_id=broker_id,
            estimated_cost=round(cost, 2),
            breakeven_ask=round(cost, 2),
            market_rate=round(market_rate, 4),
            options=[option],
            recommended_label=TARGET,
            recommended_ask=option.ask_amount,
            winnability_available=winnability_available,
            rationale=(
                f"No expected-value signal ({reason}); recommending cost-plus-margin "
                f"target ${target_ask:,.0f} (margin {cfg.fallback_target_margin:.0%} "
                f"over ${cost:,.0f} cost)."
            ),
        )

"""No-op payment-risk adapter (Phase 5.2).

Returns ``None`` for every query, signaling "no payment-risk model available" so the
Phase 5.1 risk-adjusted EV recommender falls back to its risk-blind behavior. This is
the adapter wired when payment risk is disabled or no artifact is present — guaranteeing
zero behavior change versus the pre-5.2 recommender.
"""
from __future__ import annotations

from typing import Optional

from ml.features.winnability_features import BidQuery
from ports.payment_risk import PaymentEstimate, PaymentRiskPort


class NoopPaymentRiskAdapter(PaymentRiskPort):
    def estimate(self, query: BidQuery) -> Optional[PaymentEstimate]:
        return None

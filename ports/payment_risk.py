"""Outbound port for broker payment-risk estimation (Phase 5.2).

The Phase 5.1 risk-adjusted EV recommender will depend on this port, never on the
model: a ``ModelPaymentRiskAdapter`` wraps the trained Phase 5.2 artifact, while a
``NoopPaymentRiskAdapter`` returns ``None`` so the recommender degrades to its
risk-blind behavior. One port, two adapters — the model stays optional and swappable,
exactly like the Phase 4.3 winnability boundary.

The query is a :class:`~ml.features.winnability_features.BidQuery`, reused as-is: it
already carries every observable broker + load field the payment feature builder reads,
and payment simply ignores the candidate ask it also holds. So adapters build features
with the *same* builder used at training (no train/serve skew).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ml.features.winnability_features import BidQuery


@dataclass(frozen=True)
class PaymentEstimate:
    """A broker's decision-time payment risk.

    ``p_default`` is ``P(broker never pays)``; ``p_collect`` is its complement
    (``1 - p_default``), the factor 5.1 multiplies expected margin by.
    ``expected_pay_days`` is the optional slow-pay estimate (``None`` when the model
    carries no pay-days head), a discount input for the same risk-adjusted EV.
    """

    p_default: float
    p_collect: float
    expected_pay_days: Optional[float]


class PaymentRiskPort(ABC):
    @abstractmethod
    def estimate(self, query: BidQuery) -> Optional[PaymentEstimate]:
        """Payment risk for the load/broker in ``query``, or ``None``.

        ``None`` signals "no payment-risk model available" — the caller should fall
        back to its risk-blind behavior.
        """
        ...

from enum import Enum


class BidApprovalStatus(str, Enum):
    """Lifecycle states for a human-in-the-loop bid draft (Phase 4.4).

    Non-terminal states (``DRAFTED``, ``EDITED``, ``APPROVED``) can still transition;
    terminal states (``REJECTED``, ``SUBMITTED_MOCK``, ``EXPIRED``) never move. An
    ``APPROVED`` draft may still **expire** before it is submitted.

    ``SUBMITTED_MOCK`` is a *simulated* terminal state for workflow validation only — it
    never represents a real broker/Truckstop submission.
    """

    DRAFTED = "drafted"
    EDITED = "edited"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUBMITTED_MOCK = "submitted_mock"
    EXPIRED = "expired"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL


_TERMINAL = frozenset(
    {
        BidApprovalStatus.REJECTED,
        BidApprovalStatus.SUBMITTED_MOCK,
        BidApprovalStatus.EXPIRED,
    }
)

"""The no-op compiled dispatcher: always unavailable (the default wiring).

This is what the container injects when the ``compiled_dispatcher`` flag is off, the artifact is
missing, or it failed to load — so the shadow service degrades to a clean "compiled model
unavailable (<reason>)" comparison instead of raising. Mirrors
``adapters/outbound/winnability/noop_adapter.py``.
"""
from __future__ import annotations

from typing import Any, Mapping

from ports.compiled_dispatcher import (
    REASON_DISABLED,
    CompiledDispatcherAvailability,
    CompiledDispatcherPort,
    CompiledDispatcherPrediction,
    CompiledDispatcherUnavailable,
)


class NoopCompiledDispatcher(CompiledDispatcherPort):
    """Reports the compiled dispatcher unavailable; ``predict`` always fails closed."""

    def __init__(self, reason: str = REASON_DISABLED):
        self._reason = reason

    def availability(self) -> CompiledDispatcherAvailability:
        return CompiledDispatcherAvailability(available=False, reason=self._reason)

    def predict(self, features: Mapping[str, Any]) -> CompiledDispatcherPrediction:
        raise CompiledDispatcherUnavailable(self._reason)

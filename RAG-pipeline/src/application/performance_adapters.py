from __future__ import annotations

from typing import Any

from src.core.performance import CircuitBreaker
from src.observability import get_observability


class ObservableCircuitBreaker(CircuitBreaker):
    """Application-layer metrics adapter; provider-neutral breaker remains in the core."""
    def call(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        before = self.state.value
        try:
            return super().call(operation, *args, **kwargs)
        finally:
            after = self.state.value
            metrics = get_observability().metrics
            for state in ("closed", "open", "half_open"):
                metrics.labels(metrics.circuit_state, dependency=self.name, state=state).set(int(after == state))
            if after != before:
                metrics.labels(metrics.circuit_transitions, dependency=self.name, state=after).inc()

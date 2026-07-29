from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import Request

from src.api.errors import ApiError
from src.application.config import ApiSettings


class SlidingWindowRateLimiter:
    def __init__(self, requests: int, window_seconds: int):
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, identity: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            events = self._events[identity]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                raise ApiError(429, "rate_limit_exceeded", "Request rate limit exceeded", {
                    "Retry-After": str(min(60, max(1, int(self.window_seconds))))
                })
            events.append(now)


class ApiSecurity:
    def __init__(self, settings: ApiSettings):
        self.settings = settings
        self.rate_limiter = SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        )

    def authorize(self, request: Request) -> None:
        supplied = request.headers.get("X-API-Key", "")
        if self.settings.auth_enabled:
            if not self.settings.api_key:
                raise ApiError(503, "authentication_not_configured", "API authentication is not configured")
            if not supplied or not hmac.compare_digest(supplied, self.settings.api_key):
                raise ApiError(401, "unauthorized", "A valid API key is required")
        client_host = request.client.host if request.client else "unknown"
        raw_identity = supplied if supplied else client_host
        identity = hashlib.sha256(raw_identity.encode("utf-8")).hexdigest()
        request.state.authorization_scope = identity
        self.rate_limiter.check(identity)

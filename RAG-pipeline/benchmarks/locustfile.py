from __future__ import annotations

import itertools
import os

from locust import HttpUser, between, task


QUERIES = (
    "What deployment technology is described?",
    "Which database stores control-plane records?",
    "How are ingestion messages delivered?",
    "What prevents stale document versions from being queried?",
)


class DeterministicInfrastructureUser(HttpUser):
    wait_time = between(0.01, 0.05)

    def on_start(self) -> None:
        self._queries = itertools.cycle(QUERIES)
        api_key = os.getenv("RAG_API_KEY", "")
        self.headers = {"X-API-Key": api_key} if api_key else {}

    @task
    def query(self) -> None:
        with self.client.post(
            "/api/v1/query", headers=self.headers,
            json={"query": next(self._queries), "top_k": 5, "retrieval_mode": "hybrid"},
            name="POST /api/v1/query", catch_response=True,
        ) as response:
            if response.status_code not in (200, 429, 503):
                response.failure(f"unexpected status class {response.status_code // 100}xx")


class OptionalGroqSmokeUser(DeterministicInfrastructureUser):
    wait_time = between(2.0, 3.0)

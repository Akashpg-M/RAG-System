from __future__ import annotations

import hashlib
from typing import Any

from src.core.cache_keys import graph_extraction_key, representation_key


class CachingQueryProcessor:
    def __init__(self, wrapped: Any, cache: Any, provider: str, model: str,
                 prompt_version: str = "query-rewrite-v1", retrieval_mode: str = "configured"):
        self.wrapped, self.cache = wrapped, cache
        self.provider, self.model = provider, model
        self.prompt_version, self.retrieval_mode = prompt_version, retrieval_mode

    def process_query(self, query: str) -> dict[str, Any]:
        key = representation_key(query, self.provider, self.model, self.prompt_version,
                                 {"temperature": 0.0}, self.retrieval_mode)
        value = self.cache.get_or_compute(key, lambda: dict(self.wrapped.process_query(query)))
        return dict(value)


class CachingGraphExtractor:
    def __init__(self, wrapped: Any, cache: Any, model: str, prompt_version: str = "graph-v1",
                 ontology_version: str = "software-v1", schema_version: str = "graph-v1",
                 parser_chunker_version: str = "configured"):
        self.wrapped, self.cache, self.model = wrapped, cache, model
        self.prompt_version, self.ontology_version = prompt_version, ontology_version
        self.schema_version, self.parser_chunker_version = schema_version, parser_chunker_version

    def extract_triples(self, text: str) -> list[dict[str, Any]]:
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        key = graph_extraction_key(content_hash, self.model, self.prompt_version, self.ontology_version,
                                   self.schema_version, self.parser_chunker_version)
        value = self.cache.get_or_compute(key, lambda: list(self.wrapped.extract_triples(text)))
        # Return fresh dictionaries; ingestion attaches the new version's chunk identity.
        return [dict(item) for item in value]

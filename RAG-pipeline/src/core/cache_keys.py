from __future__ import annotations

import hashlib
import json
import unicodedata
from typing import Any, Mapping


NORMALIZATION_VERSION = "nfkc-ws-v1"
CACHE_SCHEMA_VERSION = "stage6-v1"


def normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def stable_hash(value: Any) -> str:
    if isinstance(value, str):
        serialized = value
    else:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def query_embedding_key(query: str, model_version: str) -> str:
    return ":".join((CACHE_SCHEMA_VERSION, "query-embedding", NORMALIZATION_VERSION,
                     stable_hash(normalize_query(query)), stable_hash(model_version)))


def content_embedding_key(content_hash: str, model_version: str, pipeline_version: str) -> str:
    return ":".join((CACHE_SCHEMA_VERSION, "content-embedding", content_hash,
                     stable_hash(model_version), stable_hash(pipeline_version)))


def representation_key(query: str, provider: str, model: str, prompt_version: str,
                       generation_parameters: Mapping[str, Any], retrieval_mode: str) -> str:
    dimensions = {"query": stable_hash(normalize_query(query)), "provider": provider, "model": model,
                  "prompt": prompt_version, "parameters": generation_parameters, "mode": retrieval_mode,
                  "normalization": NORMALIZATION_VERSION}
    return f"{CACHE_SCHEMA_VERSION}:representations:{stable_hash(dimensions)}"


def retrieval_key(query: str, revision: int, retrieval_config: Mapping[str, Any], filters: Mapping[str, Any],
                  namespace: str, authorization_scope: str, top_k: int, index_schema: str) -> str:
    dimensions = {"query": stable_hash(normalize_query(query)), "revision": revision,
                  "retrieval": retrieval_config, "filters": filters, "namespace": namespace,
                  "scope": stable_hash(authorization_scope), "top_k": top_k, "index_schema": index_schema}
    return f"{CACHE_SCHEMA_VERSION}:retrieval:{stable_hash(dimensions)}"


def graph_extraction_key(content_hash: str, model: str, prompt_version: str, ontology_version: str,
                         schema_version: str, parser_chunker_version: str) -> str:
    dimensions = {"content": content_hash, "model": model, "prompt": prompt_version,
                  "ontology": ontology_version, "schema": schema_version,
                  "parser_chunker": parser_chunker_version}
    return f"{CACHE_SCHEMA_VERSION}:graph:{stable_hash(dimensions)}"

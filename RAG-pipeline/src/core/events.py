from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from urllib.parse import unquote_plus


EVENT_VERSION = "1.0"


def idempotency_key(namespace: str, object_key: str, object_version: str, pipeline_version: str) -> str:
    """Stable key for effectively-once indexing across at-least-once transports."""
    canonical = "\0".join((namespace.strip(), object_key.strip(), object_version.strip(), pipeline_version.strip()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IngestionEvent:
    event_id: str
    task_id: str
    document_id: str
    version_id: str
    namespace: str
    object_key: str
    object_version: str
    pipeline_version: str
    source_uri: str
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = EVENT_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_context: Dict[str, str] = field(default_factory=dict)

    @property
    def idempotency_key(self) -> str:
        return idempotency_key(self.namespace, self.object_key, self.object_version, self.pipeline_version)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str | bytes) -> "IngestionEvent":
        payload = json.loads(value)
        if payload.get("schema_version") not in ("1", EVENT_VERSION):
            raise ValueError("unsupported ingestion event version")
        required = {"event_id", "task_id", "document_id", "version_id", "namespace", "object_key",
                    "object_version", "pipeline_version", "source_uri"}
        if not required <= payload.keys():
            raise ValueError("invalid ingestion event envelope")
        return cls(**{key: payload[key] for key in cls.__dataclass_fields__ if key in payload})


def parse_s3_notification(value: str | bytes, pipeline_version: str = "stage-3") -> list[dict[str, Optional[str]]]:
    """Parse native S3 ObjectCreated notifications, including SNS-style body wrapping."""
    payload = json.loads(value)
    if "Message" in payload and isinstance(payload["Message"], str):
        payload = json.loads(payload["Message"])
    parsed = []
    for record in payload.get("Records", []):
        if not str(record.get("eventName", "")).startswith("ObjectCreated:"):
            continue
        s3 = record["s3"]
        bucket = s3["bucket"]["name"]
        obj = s3["object"]
        key = unquote_plus(obj["key"])
        version = obj.get("versionId") or obj.get("eTag") or str(obj.get("sequencer", ""))
        parsed.append({
            "namespace": bucket, "object_key": key, "object_version": version,
            "pipeline_version": pipeline_version, "source_uri": f"s3://{bucket}/{key}",
        })
    return parsed

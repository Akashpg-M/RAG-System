import os
import uuid

import pytest

from src.core.contracts import ChildChunk
from src.vector_store import ProductionVectorStore


QDRANT_URL = os.getenv("RAG_QDRANT_TEST_URL", "http://127.0.0.1:6333")


@pytest.mark.integration
def test_real_qdrant_keeps_versions_separate_and_cleans_one_version():
    collection = f"test_stage4_{uuid.uuid4().hex}"
    try:
        store = ProductionVectorStore(collection, 2, url=QDRANT_URL)
    except Exception as error:
        pytest.skip(f"real Qdrant unavailable: {type(error).__name__}")
    first = ChildChunk("document#v1#c0", "document", "document#v1#p0", "first", 1, "hash-v1",
                       {"document_id": "document", "version_id": "v1", "namespace": "default"})
    second = ChildChunk("document#v2#c0", "document", "document#v2#p0", "second", 1, "hash-v2",
                        {"document_id": "document", "version_id": "v2", "namespace": "default"})
    try:
        store.upsert_chunks_bulk([first, second], [[1.0, 0.0], [0.0, 1.0]])
        assert store.version_chunks("document", "v1") == {"document#v1#c0": "hash-v1"}
        assert {hit["version_id"] for hit in store.search_similar([1.0, 1.0], 10)} == {"v1", "v2"}
        store.delete_version("document", "v1")
        assert {hit["version_id"] for hit in store.search_similar([1.0, 1.0], 10)} == {"v2"}
    finally:
        store.client.delete_collection(collection)

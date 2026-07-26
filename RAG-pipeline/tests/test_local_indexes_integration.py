from src.config import Config
from src.graph_store import KnowledgeGraphStore
from src.main import ProductionIngestionPipeline
from src.models import EmbeddingCache
from src.sparse_store import SparseStore
from src.vector_store import ProductionVectorStore

from test_ingestion_pipeline import FakeChunker, FakeEmbedder, FakeExtractor


def test_pipeline_updates_and_deletes_real_local_indexes(tmp_path, monkeypatch):
    source = tmp_path / "document.md"
    source.write_text("content", encoding="utf-8")
    monkeypatch.setattr(Config, "QDRANT_STORAGE_PATH", str(tmp_path / "qdrant"))

    sparse = SparseStore(str(tmp_path / "sparse.db"))
    graph = KnowledgeGraphStore(str(tmp_path / "graph.db"))
    vector = ProductionVectorStore("test", vector_dim=2)
    pipeline = ProductionIngestionPipeline(
        sparse, graph, vector, FakeEmbedder(), EmbeddingCache(str(tmp_path / "cache.db")), FakeExtractor()
    )
    chunker = FakeChunker()

    pipeline.ingest_document(str(source), chunker)
    assert sparse.search("child")
    assert vector.search_similar([1.0, 0.0], limit=2)
    assert graph.traverse_graph_hops(["api"], max_hops=1)

    chunker.count = 1
    pipeline.ingest_document(str(source), chunker)
    assert len(vector.search_similar([1.0, 0.0], limit=10)) == 1

    assert pipeline.delete_document(str(source)) == 1
    assert sparse.search("child") == []
    assert vector.search_similar([1.0, 0.0], limit=10) == []
    assert graph.traverse_graph_hops(["api"], max_hops=1) == []


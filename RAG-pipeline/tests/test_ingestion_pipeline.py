from src.main import ProductionIngestionPipeline
from src.models import ChildChunk, ParentChunk


class FakeCache:
    def __init__(self):
        self.values = {"cached": [1.0, 0.0]}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value):
        self.values[key] = value


class FakeEmbedder:
    def get_embeddings_batched(self, texts):
        return [[0.0, float(index + 2)] for index, _ in enumerate(texts)]


class FakeSparseStore:
    def __init__(self):
        self.documents = {}

    def delete_document(self, document_id):
        stale = [cid for cid, chunk in self.documents.items() if chunk.document_id == document_id]
        for cid in stale:
            del self.documents[cid]
        return stale

    def add_documents(self, chunks):
        self.documents.update((chunk.chunk_id, chunk) for chunk in chunks)


class FakeGraphStore:
    def __init__(self):
        self.triples = []

    def delete_by_chunk_ids(self, chunk_ids):
        self.triples = [t for t in self.triples if t["chunk_id"] not in chunk_ids]

    def delete_document(self, document_id):
        self.triples = [t for t in self.triples if not t["chunk_id"].startswith(f"{document_id}#")]

    def add_triples_bulk(self, triples):
        self.triples.extend(triples)


class FakeVectorStore:
    def __init__(self):
        self.points = {}

    def delete_document(self, document_id):
        self.points = {
            cid: value for cid, value in self.points.items()
            if value[0].document_id != document_id
        }

    def upsert_chunks_bulk(self, chunks, embeddings):
        assert len(chunks) == len(embeddings)
        self.points.update((chunk.chunk_id, (chunk, vector)) for chunk, vector in zip(chunks, embeddings))


class FakeExtractor:
    def extract_triples(self, text):
        return [{"source": "API", "relation": "USES", "target": "Database"}]


class FakeChunker:
    def __init__(self):
        self.count = 2

    def process_file(self, file_path, document_id):
        parent = ParentChunk(f"{document_id}#p0", document_id, "parent", {"source": file_path})
        hashes = ["cached", "new"][: self.count]
        children = [
            ChildChunk(
                f"{document_id}#c{index}", document_id, parent.parent_id,
                f"child {index}", 2, content_hash, {"source": file_path},
            )
            for index, content_hash in enumerate(hashes)
        ]
        return [parent], children


def test_ingest_preserves_embedding_order_and_replaces_document(tmp_path):
    source = tmp_path / "doc.md"
    source.write_text("content", encoding="utf-8")
    sparse, graph, vector = FakeSparseStore(), FakeGraphStore(), FakeVectorStore()
    pipeline = ProductionIngestionPipeline(sparse, graph, vector, FakeEmbedder(), FakeCache(), FakeExtractor())
    chunker = FakeChunker()

    pipeline.ingest_document(str(source), chunker)
    document_id = pipeline.document_id_for(str(source))
    assert vector.points[f"{document_id}#c0"][1] == [1.0, 0.0]
    assert vector.points[f"{document_id}#c1"][1] == [0.0, 2.0]
    assert graph.triples[0]["chunk_id"] == f"{document_id}#p0"

    chunker.count = 1
    pipeline.ingest_document(str(source), chunker)
    assert list(sparse.documents) == [f"{document_id}#c0"]
    assert list(vector.points) == [f"{document_id}#c0"]
    assert len(graph.triples) == 1

    assert pipeline.delete_document(str(source)) == 1
    assert not sparse.documents
    assert not vector.points
    assert not graph.triples

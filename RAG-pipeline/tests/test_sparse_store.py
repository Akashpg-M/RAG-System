import sqlite3

from src.models import ChildChunk
from src.sparse_store import SparseStore


def make_chunk(document_id, text, content_hash="hash"):
    return ChildChunk("chunk", document_id, "parent", text, 1, content_hash, {})


def test_replacing_chunk_removes_stale_postings_and_document_delete_is_complete(tmp_path):
    store = SparseStore(str(tmp_path / "sparse.db"))
    store.add_documents([make_chunk("doc", "alpha beta")])
    store.add_documents([make_chunk("doc", "gamma", "new-hash")])

    assert store.search("alpha") == []
    assert store.search("gamma")[0]["chunk_id"] == "chunk"
    assert store.delete_document("doc") == ["chunk"]
    assert store.search("gamma") == []
    with sqlite3.connect(store.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM postings").fetchone()[0] == 0


from src.fusion import ReciprocalRankFusion
from src.retrieval import AgenticRetrievalEngine, RetrieverManager


class Retriever:
    def __init__(self, result=None, error=None):
        self.result, self.error = result or [], error

    def retrieve(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.result


class Processor:
    def process_query(self, query):
        return {"original_query": query, "rewritten_query": query, "hyde_document": query}


class BrokenReranker:
    def predict(self, inputs):
        raise RuntimeError("offline")


def hit(chunk_id):
    return {"chunk_id": chunk_id, "text": "relevant", "rrf_score": 0.0}


def test_one_retriever_and_reranker_failure_do_not_discard_other_results():
    manager = RetrieverManager(
        dense=Retriever([hit("dense")]),
        sparse=Retriever(error=RuntimeError("sparse unavailable")),
        graph=Retriever([hit("graph")]),
    )
    engine = AgenticRetrievalEngine(Processor(), manager, ReciprocalRankFusion(), reranker=BrokenReranker())
    results = engine.retrieve_context("query", top_k=2)
    assert {result["chunk_id"] for result in results} == {"dense", "graph"}


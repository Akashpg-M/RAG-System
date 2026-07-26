import json
from types import SimpleNamespace

from src.graph.graph_extractor import GraphExtractor
from src.graph.ontology import DomainOntology
from src.graph_store import KnowledgeGraphStore


class Client:
    def __init__(self, payload):
        message = SimpleNamespace(content=json.dumps(payload))
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=message)]))
        )


def test_domain_concept_allowed_and_provider_failure_is_safe():
    ontology = DomainOntology(domain="software", entities=["SERVICE"], relations=["USES"])
    payload = {"triples": [{
        "source": "Worker", "source_type": "DOMAIN_CONCEPT", "relation": "USES",
        "target": "API", "target_type": "SERVICE",
    }]}
    assert len(GraphExtractor(Client(payload), "model", ontology).extract_triples("text")) == 1
    assert GraphExtractor(None, "model", ontology).extract_triples("text") == []


def test_graph_hop_limit_and_document_deletion(tmp_path):
    store = KnowledgeGraphStore(str(tmp_path / "graph.db"))
    store.add_triples_bulk([
        {"source": "a", "relation": "uses", "target": "b", "chunk_id": "doc#p0"},
        {"source": "b", "relation": "uses", "target": "c", "chunk_id": "doc#p1"},
        {"source": "c", "relation": "uses", "target": "d", "chunk_id": "doc#p2"},
    ])
    assert [edge["hop_level"] for edge in store.traverse_graph_hops(["a"], max_hops=2)] == [1, 2]
    store.delete_document("doc")
    assert store.traverse_graph_hops(["a"], max_hops=2) == []


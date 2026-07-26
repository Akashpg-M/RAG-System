from types import SimpleNamespace

from src.generation import ProductionResponseGenerator
from src.semantic_processor import SemanticQueryProcessor


class FailingClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")))
        )


def test_semantic_provider_failure_falls_back_to_raw_query():
    processor = SemanticQueryProcessor(llm_client=FailingClient())
    assert processor.process_query("raw") == {
        "original_query": "raw",
        "rewritten_query": "raw",
        "hyde_document": "raw",
    }


def test_generation_failure_is_generic_and_context_is_xml_escaped():
    generator = ProductionResponseGenerator(llm_client=FailingClient())
    context = [{"chunk_id": "a&b", "text": "<unsafe>", "metadata": {"source": "x&y"}, "rerank_score": 1.0}]
    messages = generator._build_xml_context_prompt("query", context)
    assert "a&amp;b" in messages[1]["content"]
    assert "&lt;unsafe&gt;" in messages[1]["content"]
    assert list(generator.generate_stream("query", context)) == ["\n[Generation is temporarily unavailable.]"]


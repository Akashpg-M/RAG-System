from sentence_transformers import CrossEncoder

from src.embedder import ProductionEmbedder
from src.generation import ProductionResponseGenerator
from src.semantic_processor import SemanticQueryProcessor


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def predict(self, pairs):
        return self.model.predict(pairs)


__all__ = [
    "CrossEncoderReranker", "ProductionEmbedder", "ProductionResponseGenerator", "SemanticQueryProcessor",
]


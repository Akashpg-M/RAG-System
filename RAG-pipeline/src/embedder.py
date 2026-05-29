# src/embedder.py
import logging
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from src.config import Config

logger = logging.getLogger("Embedder")

class ProductionEmbedder:
    def __init__(self):
        # Local execution using SentenceTransformers for predictable, zero-cost vectors
        logger.info(f"Initializing local embedding engine: {Config.EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
        # dimension size for all-MiniLM-L6-v2 is 384
        self.vector_dim = self.model.get_sentence_embedding_dimension()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generates embeddings in efficient batches with automatic fault-tolerant retries.
        """
        logger.info(f"Generating embeddings for text payload cluster of size: {len(texts)}")
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()
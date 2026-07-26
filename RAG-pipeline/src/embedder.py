# src/embedder.py
import logging
import threading
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from src.config import Config

logger = logging.getLogger("Embedder")

class ProductionEmbedder:
    def __init__(self, model_name: str = None):
        # Local execution using SentenceTransformers for predictable, zero-cost vectors
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        logger.info(f"Initializing local embedding engine: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        # dimension size for all-MiniLM-L6-v2 is 384
        self.vector_dim = self.model.get_sentence_embedding_dimension()
        self._encode_lock = threading.Lock()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=10))
    def get_embeddings_batched(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Generates embeddings in efficient batches with automatic fault-tolerant retries.
        """
        logger.info(f"Generating embeddings for text payload cluster of size: {len(texts)}")
        if not texts:
            return []
        with self._encode_lock:
            embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()

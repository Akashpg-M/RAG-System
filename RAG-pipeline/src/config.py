# src/config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup structured logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

class Config:
    QDRANT_STORAGE_PATH = os.getenv("QDRANT_STORAGE_PATH", "./qdrant_local_data")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 30))
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

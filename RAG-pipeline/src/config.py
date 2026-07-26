"""Legacy configuration facade; new code receives ``AppConfig`` explicitly."""

import logging

from src.application.config import AppConfig, Profile, load_config, profile_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)

_legacy = load_config(Profile.LOCAL)


class Config:
    QDRANT_STORAGE_PATH = _legacy.storage.qdrant_path
    EMBEDDING_MODEL_NAME = _legacy.models.embedding_model
    CHUNK_SIZE = _legacy.pipeline.chunk_size
    CHUNK_OVERLAP = _legacy.pipeline.chunk_overlap
    GROQ_API_KEY = _legacy.models.groq_api_key
    GROQ_MODEL_NAME = _legacy.models.groq_model


__all__ = ["AppConfig", "Config", "Profile", "load_config", "profile_config"]

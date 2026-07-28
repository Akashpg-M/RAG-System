import json
import logging
from typing import Dict, Any
from groq import Groq
from src.config import Config

logger = logging.getLogger("SemanticQueryProcessor")

class SemanticQueryProcessor:
    """
    Decoupled Linguistic Synthesis Layer.
    Can be dynamically bypassed for local deterministic testing.
    """
    def __init__(self, semantic_enabled: bool = True, llm_client=None, model_name: str = None, api_key: str = None):
        resolved_key = Config.GROQ_API_KEY if api_key is None else api_key
        self.model_name = model_name or Config.GROQ_MODEL_NAME
        self.enabled = semantic_enabled and bool(resolved_key or llm_client)
        self.llm_client = None
        
        if self.enabled:
            logger.info("Initializing Groq Semantic Processor Pipeline Link...")
            self.llm_client = llm_client or Groq(api_key=resolved_key)

    def process_query(self, raw_query: str) -> Dict[str, Any]:
        """Generates rewritten and HyDE representations if semantic operations are enabled."""
        if not self.enabled or not self.llm_client:
            logger.info("Semantic processing disabled. Emitting raw fallback query mapping.")
            return {
                "original_query": raw_query,
                "rewritten_query": raw_query,
                "hyde_document": raw_query
            }
            
        sys_prompt = """
        You are a backend search optimization engine. Analyze the user query.
        Output a valid JSON object with exactly two keys:
        1. "rewritten_query": A clean, highly technical version of the query optimized for vector space.
        2. "hyde_document": A 2-to-3 sentence hypothetical technical answer to the query.
        """
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": raw_query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            data = json.loads(response.choices[0].message.content)
            return {
                "original_query": raw_query,
                "rewritten_query": data.get("rewritten_query", raw_query),
                "hyde_document": data.get("hyde_document", raw_query)
            }
        except Exception:
            logger.error("query_representation_failed", extra={
                "component": "semantic_processor", "error_code": "generation", "outcome": "degraded",
            })
            return {
                "original_query": raw_query,
                "rewritten_query": raw_query,
                "hyde_document": raw_query
            }

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
    def __init__(self, semantic_enabled: bool = True):
        self.enabled = semantic_enabled
        self.llm_client = None
        
        if self.enabled:
            logger.info("Initializing Groq Semantic Processor Pipeline Link...")
            self.llm_client = Groq(api_key=Config.GROQ_API_KEY)

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
                model="llama-3.3-70b-versatile",
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
        except Exception as e:
            logger.error(f"Semantic API failure: {str(e)}. Falling back to raw representations.")
            return {
                "original_query": raw_query,
                "rewritten_query": raw_query,
                "hyde_document": raw_query
            }
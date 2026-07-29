import logging
import time
from html import escape
from typing import List, Dict, Any, Iterator
from groq import Groq
from src.config import Config
from src.observability import get_observability
from src.core.performance import CircuitBreaker

logger = logging.getLogger("StreamingGenerator")

class ProductionResponseGenerator:
    """
    Context-Grounded Response Generation Layer.
    Injects retrieved multi-modal evidence contexts into XML blocks 
    and handles token streaming via Groq.
    """
    def __init__(self, llm_client=None, model_name: str = None, api_key: str = None,
                 circuit_breaker: CircuitBreaker | None = None):
        # Establish the runtime link to the ultra-fast execution backend
        resolved_key = Config.GROQ_API_KEY if api_key is None else api_key
        self.llm_client = llm_client or (Groq(api_key=resolved_key) if resolved_key else None)
        self.model_name = model_name or Config.GROQ_MODEL_NAME
        self.circuit_breaker = circuit_breaker

    def _build_xml_context_prompt(self, query: str, context_pool: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Constructs a deterministic system and user prompt utilizing strict XML encapsulation."""
        
        system_instruction = (
            "You are an expert, elite backend software and distributed systems engineer.\n"
            "Your task is to answer the user's technical question using ONLY the verified evidence "
            "provided inside the <retrieved_context> XML blocks.\n\n"
            "CRITICAL EXECUTION CONSTRAINTS:\n"
            "1. Rely strictly on the provided context. If the text does not contain the answer, "
            "state clearly that the information is missing from the system index.\n"
            "2. Do not hallucinate, speculate, or extrapolate beyond the explicit documentation.\n"
            "3. Cite your sources implicitly by referencing the 'chunk_id' or 'source' metadata when stating facts.\n"
            "4. Keep your formatting highly structured, leveraging markdown, code blocks, and bullet points where applicable."
        )

        # Build raw XML blocks tracking text, provenance, and graph ties
        context_str_accumulator = []
        for idx, doc in enumerate(context_pool, start=1):
            metadata = doc.get("metadata", {})
            relations = metadata.get("graph_context_relations", [])
            
            block = f'<document index="{idx}">\n'
            block += f'  <chunk_id>{escape(str(doc["chunk_id"]))}</chunk_id>\n'
            block += f'  <source_file>{escape(str(metadata.get("source", "unknown")))}</source_file>\n'
            
            if relations:
                block += '  <discovered_knowledge_graph_paths>\n'
                for rel in relations:
                    block += f'    <path>{escape(str(rel))}</path>\n'
                block += '  </discovered_knowledge_graph_paths>\n'
                
            block += f'  <content>\n{escape(str(doc["text"]))}\n  </content>\n'
            block += '</document>\n'
            context_str_accumulator.append(block)

        compiled_context = "<retrieved_context>\n" + "".join(context_str_accumulator) + "</retrieved_context>"

        user_content = (
            f"Here is the verified documentation context:\n{compiled_context}\n\n"
            f"User Question: {query}\n\n"
            f"Provide a highly detailed, technically rigorous response grounded strictly in the context above:"
        )

        return [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

    def generate_stream(self, query: str, context_pool: List[Dict[str, Any]], score_threshold: float = -2.0) -> Iterator[str]:
        """
        Filters the candidate pool, constructs the prompt, and yields text tokens as they stream from the API.
        """
        # Filter down candidate pool using a Cross-Encoder threshold to discard noise
        viable_context = [doc for doc in context_pool if doc.get("rerank_score", 0.0) >= score_threshold]
        
        # If everything falls below the threshold, log a warning but fall back to the top 2 elements to prevent total failures
        if not viable_context and context_pool:
            logger.warning("All contexts fell below the rerank score threshold. Falling back to top candidates.")
            viable_context = context_pool[:2]

        telemetry = get_observability()
        with telemetry.span("query.prompt_construct", {"rag.context_count": len(viable_context)}):
            messages = self._build_xml_context_prompt(query, viable_context)

        try:
            if self.llm_client is None:
                raise RuntimeError("GROQ_API_KEY is not configured")
            started = time.perf_counter()
            first_token = True
            with telemetry.span("query.generation.provider", {"gen_ai.system": "groq"}):
                def operation():
                    return self.llm_client.chat.completions.create(
                        model=self.model_name, messages=messages, temperature=0.1, stream=True
                    )

                stream = self.circuit_breaker.call(operation) if self.circuit_breaker else operation()
                for chunk in stream:
                    token = chunk.choices[0].delta.content
                    if token:
                        if first_token:
                            telemetry.metrics.labels(telemetry.metrics.ttft).observe(time.perf_counter() - started)
                            first_token = False
                        yield token
                    
        except Exception:
            logger.exception("generation_provider_failed", extra={
                "component": "generation_provider", "error_code": "generation",
            })
            yield "\n[Generation is temporarily unavailable.]"

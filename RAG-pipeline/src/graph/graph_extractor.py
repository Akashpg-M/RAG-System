# src/graph/graph_extractor.py
import json
import logging
from typing import Dict, Any, List, Optional
from src.graph.prompt_builder import PromptBuilder
from src.graph.ontology import DomainOntology

logger = logging.getLogger("GraphExtractor")

class GraphExtractor:
    def __init__(self, llm_client: Any, model_name: str, ontology: DomainOntology):
        """
        Dependency injection container for the graph construction layer.
        """
        self.client = llm_client
        self.model_name = model_name
        self.ontology = ontology
        self.system_prompt = PromptBuilder.build(ontology.model_dump())        
        self.allowed_entities = set(ontology.entities)
        self.allowed_relations = set(ontology.relations)

    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """
        Executes synchronous extraction, schema validation, and ontology filtering.
        """
        try:
            # Polymorphic wrapper to support multiple providers (Groq/OpenAI/Ollama)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Extract triples from the following text:\n\n{text}"}
                ],
                temperature=0.0,  # Zero out creativity for deterministic extraction
                response_format={"type": "json_object"}
            )
            
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            
            validated_triples = []
            seen_triples = set()
            
            for triple in data.get("triples", []):
                # 1. Structural Schema Validation
                if not all(k in triple for k in ("source", "source_type", "relation", "target", "target_type")):
                    continue
                
                # 2. Strict Ontology Restriction Filtering
                if triple["source_type"] not in self.allowed_entities:
                    continue
                if triple["target_type"] not in self.allowed_entities:
                    continue
                if triple["relation"] not in self.allowed_relations:
                    continue
                
                # 3. Entity Level Normalization
                triple["source"] = self._normalize_entity(triple["source"])
                triple["target"] = self._normalize_entity(triple["target"])
                
                # 4. Deduplication Verification
                triple_key = (triple["source"], triple["relation"], triple["target"])
                if triple_key in seen_triples:
                    continue
                    
                seen_triples.add(triple_key)
                validated_triples.append(triple)
                
            return validated_triples
            
        except Exception as e:
            logger.error(f"Failed graph extraction execution trace: {str(e)}")
            return []

    def _normalize_entity(self, entity_name: str) -> str:
        """Applies dynamic, domain-specific sanitization from the injected ontology."""
        cleaned = entity_name.strip().lower()
        
        # Check if the messy name exists in our JSON configuration keys
        if cleaned in self.ontology.normalization_map:
            # Return the clean, canonical version
            return self.ontology.normalization_map[cleaned]
            
        return entity_name.strip()
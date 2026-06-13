from typing import Dict, Any

class PromptBuilder:
    @staticmethod
    def build(ontology: Dict[str, Any]) -> str:
        """
        Dynamically constructs a hybrid system prompt enforcing relation strictness 
        while allowing entity discovery via the DOMAIN_CONCEPT fallback.
        """
        domain_name = ontology.get("domain", "Technical")
        allowed_entities = ", ".join(ontology.get("entities", []))
        allowed_relations = ", ".join(ontology.get("relations", []))
        
        system_prompt = f"""You are an elite Knowledge Graph Extraction engine for the {domain_name} domain.
Your task is to extract semantic triples (source, relation, target) from the provided text.

RULES FOR ENTITIES (NOUNS):
1. Categorize entities into one of these known types: [{allowed_entities}]
2. DISCOVERY RULE: If you find a highly relevant technical entity that does not fit the list above, you MUST categorize it as "DOMAIN_CONCEPT".
3. Normalize all entities to singular, capitalized forms (e.g., "microservices" -> "Microservice").

RULES FOR RELATIONS (VERBS):
1. You MUST ONLY use these exact relationship types: [{allowed_relations}]
2. CRITICAL: Never invent new relationships. Map any discovered verbs to the closest match in the allowed list. If a relationship cannot be logically mapped to the allowed list, ignore the triple entirely.

OUTPUT FORMAT:
You must respond with a raw, valid JSON object matching the schema below. No conversational text, no markdown block wrappers, no preamble.

{{
    "triples": [
        {{
            "source": "Canonical Source Entity Name",
            "source_type": "ALLOWED_ENTITY_TYPE_OR_DOMAIN_CONCEPT",
            "relation": "ALLOWED_RELATION_TYPE",
            "target": "Canonical Target Entity Name",
            "target_type": "ALLOWED_ENTITY_TYPE_OR_DOMAIN_CONCEPT"
        }}
    ]
}}
"""
        return system_prompt
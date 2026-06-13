# src/graph/ontology.py
import json
import os
from typing import List, Dict
from pydantic import BaseModel, ValidationError

class DomainOntology(BaseModel):
    domain: str
    entities: List[str]
    relations: List[str]
    normalization_map: Dict[str, str] = {}

    @classmethod
    def load_pipeline_configs(cls, ontology_path: str, aliases_path: str) -> "DomainOntology":
        """Loads both the structural schema and the normalization mapping files safely."""
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file missing at: {ontology_path}")
            
        with open(ontology_path, "r") as f:
            data = json.load(f)
            
        # Safely bind the aliases if the file exists
        if os.path.exists(aliases_path):
            with open(aliases_path, "r") as f:
                alias_data = json.load(f)
                data["normalization_map"] = alias_data.get("normalization_map", {})
        else:
            data["normalization_map"] = {}
            
        try:
            return cls(**data)
        except ValidationError as e:
            raise RuntimeError(f"Ontology configuration mapping is invalid: {str(e)}")
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class DocumentChunk:
    chunk_id: str
    document_id: str
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RawDocument:
    document_id: str
    filename: str
    raw_text: str
    
    @classmethod
    def from_text(cls, filename: str, raw_text: str):
        hash_input = f"{filename}{raw_text}".encode('utf-8')
        doc_id = hashlib.sha256(hash_input).hexdigest()[:16]
        return cls(document_id=doc_id, filename=filename, raw_text=raw_text)
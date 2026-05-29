import tiktoken
from src.models import RawDocument, DocumentChunk
from typing import List
import re

class IngestionEngine:
    def __init__(self, model_name: str = "gpt-4", chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
          raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.encoder = tiktoken.encoding_for_model(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_chunk_{chunk_index}"

    def process_document(self, document: RawDocument) -> List[DocumentChunk]:
        processed_chunks = []
        # sections = document.raw_text.split("\n## ")
        sections = re.split(r'\n\s*##\s+', document.raw_text)
        chunk_index = 0
        for section in sections:
            if not section.strip():
                continue
                
            lines = section.split("\n", 1)
            header = lines[0].strip() if len(lines) > 1 else "Introduction"
            content = lines[1].strip() if len(lines) > 1 else lines[0]
            
            tokens = self.encoder.encode(content)
            start_idx = 0
            
            while start_idx < len(tokens):
                end_idx = min(start_idx + self.chunk_size, len(tokens))
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = self.encoder.decode(chunk_tokens)
                
                chunk = DocumentChunk(
                    chunk_id=self._generate_chunk_id(document.document_id, chunk_index),
                    document_id=document.document_id,
                    text=chunk_text,
                    token_count=len(chunk_tokens),
                    metadata={
                        "source_file": document.filename,
                        "section_header": header
                    }
                )
                processed_chunks.append(chunk)
                chunk_index += 1
                
                if end_idx == len(tokens):
                    break
                start_idx += (self.chunk_size - self.chunk_overlap)
                
        return processed_chunks
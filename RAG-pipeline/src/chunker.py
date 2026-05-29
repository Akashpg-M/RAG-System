# src/chunker.py
import re
import hashlib
import tiktoken
from typing import List
from src.config import Config
from src.models import RawDocument, DocumentChunk

class SemanticChunker:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.chunk_size = Config.CHUNK_SIZE
        self.chunk_overlap = Config.CHUNK_OVERLAP

    def _split_into_sentences(self, text: str) -> List[str]:
        # Simple sentence splitter avoiding breaking mid-abbreviation
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        return [s.strip() for s in sentence_end.split(text) if s.strip()]

    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        chunks = []
        # Paragraph splitting preserves structural blocks
        paragraphs = doc.raw_text.split("\n\n")
        chunk_idx = 0
        
        current_chunk_tokens = []
        current_chunk_text = []

        for para in paragraphs:
            sentences = self.split_into_sentences(para) if len(self.encoder.encode(para)) > self.chunk_size else [para]
            
            for sentence in sentences:
                sentence_tokens = self.encoder.encode(sentence)
                
                if len(current_chunk_tokens) + len(sentence_tokens) <= self.chunk_size:
                    current_chunk_tokens.extend(sentence_tokens)
                    current_chunk_text.append(sentence)
                else:
                    # Flush completed chunk
                    if current_chunk_text:
                        chunks.append(self._build_chunk(doc, chunk_idx, current_chunk_text, current_chunk_tokens))
                        chunk_idx += 1
                    
                    # Handle sliding window overlap tracking
                    # Retain last few tokens/sentences for context matching
                    current_chunk_tokens = sentence_tokens
                    current_chunk_text = [sentence]
                    
        if current_chunk_text:
            chunks.append(self._build_chunk(doc, chunk_idx, current_chunk_text, current_chunk_tokens))
            
        return chunks

    def _build_chunk(self, doc: RawDocument, idx: int, text_list: List[str], token_list: List[int]) -> DocumentChunk:
        combined_text = " ".join(text_list).strip()
        content_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
        chunk_id = f"{doc.document_id}#c{idx}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=doc.document_id,
            text=combined_text,
            token_count=len(token_list),
            content_hash=content_hash,
            metadata={"source": doc.filename, "index": idx}
        )

    def split_into_sentences(self, text: str) -> List[str]:
        return self._split_into_sentences(text)

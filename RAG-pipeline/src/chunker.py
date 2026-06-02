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

    def split_into_sentences(self, text: str) -> List[str]:
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        return [s.strip() for s in sentence_end.split(text) if s.strip()]

    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        chunks = []
        sentences = self.split_into_sentences(doc.raw_text)
        
        current_chunk_sentences = []
        current_chunk_tokens = 0
        chunk_idx = 0
        
        idx = 0
        while idx < len(sentences):
            sentence = sentences[idx]
            sentence_tokens = len(self.encoder.encode(sentence))
            
            # Guard against massive outlier sentences
            if sentence_tokens > self.chunk_size:
                # Force fallback split for extreme edge cases
                sub_tokens = self.encoder.encode(sentence)
                for start in range(0, len(sub_tokens), self.chunk_size - self.chunk_overlap):
                    end = start + self.chunk_size
                    slice_text = self.encoder.decode(sub_tokens[start:end])
                    chunks.append(self._build_chunk(doc, chunk_idx, [slice_text], sub_tokens[start:end]))
                    chunk_idx += 1
                idx += 1
                continue

            if current_chunk_tokens + sentence_tokens <= self.chunk_size:
                current_chunk_sentences.append(sentence)
                current_chunk_tokens += sentence_tokens
                idx += 1
            else:
                # Flush completed chunk
                raw_tokens = self.encoder.encode(" ".join(current_chunk_sentences))
                chunks.append(self._build_chunk(doc, chunk_idx, current_chunk_sentences, raw_tokens))
                chunk_idx += 1
                
                # SLIDING WINDOW BACKPRESSURE OVERLAP LOGIC
                # Rollback index to implement a true semantic overlap based on target token volume
                overlap_tokens_collected = 0
                overlap_sentences = []
                
                # Step backwards through recently sent sentences to fill overlap quota
                for back_sentence in reversed(current_chunk_sentences):
                    back_tokens = len(self.encoder.encode(back_sentence))
                    if overlap_tokens_collected + back_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, back_sentence)
                        overlap_tokens_collected += back_tokens
                    else:
                        break
                        
                current_chunk_sentences = overlap_sentences
                current_chunk_tokens = overlap_tokens_collected
                
        # Final flush for trailing blocks
        if current_chunk_sentences:
            raw_tokens = self.encoder.encode(" ".join(current_chunk_sentences))
            chunks.append(self._build_chunk(doc, chunk_idx, current_chunk_sentences, raw_tokens))
            
        return chunks

    def _build_chunk(self, doc: RawDocument, idx: int, text_list: List[str], tokens: List[int]) -> DocumentChunk:
        combined_text = " ".join(text_list).strip()
        content_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
        return DocumentChunk(
            chunk_id=f"{doc.document_id}#c{idx}",
            document_id=doc.document_id,
            text=combined_text,
            token_count=len(tokens),
            content_hash=content_hash,
            metadata={"source": doc.filename, "index": idx}
        )
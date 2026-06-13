# # src/chunker.py
# import re
# from typing import List, Dict, Any, Tuple
# from functools import lru_cache
# from src.models import ParentChunk, ChildChunk

# class SemanticDoclingChunker:
#     def __init__(self, encoder_instance: Any):
#         self.encoder = encoder_instance
#         # Phase 4 Optimization: Compile regex rules exactly once on initialization
#         self.clean_regex = re.compile(r'\s+')
        
#     @lru_cache(maxsize=4096)
#     def cached_token_count(self, text: str) -> int:
#         """Phase 4 Optimization: Avoid redundant tokenizer invocations for repeated text fragments."""
#         return len(self.encoder.encode(text))

#     def process_file(self, file_path: str, doc_hash: str) -> Tuple[List[ParentChunk], List[ChildChunk]]:
#         """
#         Pure computational function. Parses and returns structural objects.
#         Zero external side-effects, zero outbound database or network model writes.
#         """
#         # (Document loading and structural traversal logic executes here)
#         parent_chunks: List[ParentChunk] = []
#         child_chunks: List[ChildChunk] = []
        
#         # Example generation highlighting Phase 4 context deduplication:
#         # parent_metadata structural change: omit storing massive parent text blocks inside children.
#         # Instead, map via Parent ID references cleanly.
        
#         return parent_chunks, child_chunks

# src/chunker.py
import re
import hashlib
import logging
from typing import List, Tuple, Any
from functools import lru_cache

from docling.document_converter import DocumentConverter
from docling_core.types.doc.document import SectionHeaderItem, TextItem, TableItem, CodeItem

from src.models import ParentChunk, ChildChunk
from src.config import Config

logger = logging.getLogger("DoclingChunker")

class SemanticDoclingChunker:
    def __init__(self, encoder_instance: Any):
        self.encoder = encoder_instance
        self.clean_regex = re.compile(r'\s+')
        self.converter = DocumentConverter()
        self.child_size = Config.CHUNK_SIZE
        self.child_overlap = Config.CHUNK_OVERLAP

    @lru_cache(maxsize=4096)
    def cached_token_count(self, text: str) -> int:
        """Phase 4 Optimization: Avoid redundant tokenizer invocations."""
        return len(self.encoder.encode(text))

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        return [s.strip() for s in sentence_end.split(text) if s.strip()]

    def process_file(self, file_path: str, doc_id: str) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        logger.info(f"Docling parsing AST for file: {file_path}")
        conversion_result = self.converter.convert(file_path)
        docling_doc = conversion_result.document

        parent_chunks = []
        child_chunks = []

        current_section_title = "Document Introduction"
        current_section_blocks = []
        parent_idx = 0
        child_global_idx = 0

        for item, level in docling_doc.iterate_items():
            if isinstance(item, SectionHeaderItem):
                if current_section_blocks or current_section_title != "Document Introduction":
                    p_chunk, c_chunks, child_global_idx = self._finalize_section(
                        file_path, doc_id, parent_idx, current_section_title,
                        current_section_blocks, child_global_idx
                    )
                    if p_chunk:
                        parent_chunks.append(p_chunk)
                        child_chunks.extend(c_chunks)
                        parent_idx += 1

                current_section_title = item.text.strip()
                current_section_blocks = []
                continue

            block_text = ""
            content_type = "text"

            if isinstance(item, TextItem):
                block_text = item.text.strip()
            elif isinstance(item, TableItem):
                block_text = item.export_to_markdown()
                content_type = "table"
            elif isinstance(item, CodeItem):
                block_text = f"```{getattr(item, 'language', '')}\n{item.text}\n```"
                content_type = "code"

            if block_text:
                current_section_blocks.append({"text": block_text, "type": content_type})

        if current_section_blocks or current_section_title:
            p_chunk, c_chunks, child_global_idx = self._finalize_section(
                file_path, doc_id, parent_idx, current_section_title,
                current_section_blocks, child_global_idx
            )
            if p_chunk:
                parent_chunks.append(p_chunk)
                child_chunks.extend(c_chunks)

        return parent_chunks, child_chunks

    def _finalize_section(self, file_path, doc_id, parent_idx, title, blocks, start_child_idx):
        if not title and not blocks:
            return None, [], start_child_idx

        parent_id = f"{doc_id}#p{parent_idx}"
        full_text_parts = [f"## {title}"] if title else []
        full_text_parts.extend([b["text"] for b in blocks])
        parent_text = "\n\n".join(full_text_parts)

        parent_chunk = ParentChunk(
            parent_id=parent_id,
            text=parent_text,
            metadata={"source": file_path, "title": title}
        )

        child_chunks = []
        current_child_texts = []
        current_tokens = 0
        child_idx = start_child_idx

        for block in blocks:
            b_text = block["text"]
            b_type = block["type"]
            b_tokens = self.cached_token_count(b_text)

            if b_tokens > self.child_size:
                if current_child_texts:
                    child_chunks.append(self._build_child_node(file_path, doc_id, parent_chunk, child_idx, current_child_texts))
                    child_idx += 1
                    current_child_texts = []
                    current_tokens = 0

                if b_type == "text":
                    sentences = self._split_into_sentences(b_text)
                    temp_sentences = []
                    temp_tokens = 0
                    for s in sentences:
                        s_tok = self.cached_token_count(s)
                        if temp_tokens + s_tok > self.child_size and temp_sentences:
                            child_chunks.append(self._build_child_node(file_path, doc_id, parent_chunk, child_idx, temp_sentences))
                            child_idx += 1
                            
                            overlap_tokens = 0
                            overlap_s = []
                            for back_s in reversed(temp_sentences):
                                back_tok = self.cached_token_count(back_s)
                                if overlap_tokens + back_tok <= self.child_overlap:
                                    overlap_s.insert(0, back_s)
                                    overlap_tokens += back_tok
                                else:
                                    break
                            temp_sentences = overlap_s
                            temp_tokens = overlap_tokens

                        temp_sentences.append(s)
                        temp_tokens += s_tok

                    if temp_sentences:
                        current_child_texts = [" ".join(temp_sentences)]
                        current_tokens = self.cached_token_count(current_child_texts[0])
                else:
                    child_chunks.append(self._build_child_node(file_path, doc_id, parent_chunk, child_idx, [b_text]))
                    child_idx += 1

            elif current_tokens + b_tokens <= self.child_size:
                current_child_texts.append(b_text)
                current_tokens += b_tokens
            else:
                child_chunks.append(self._build_child_node(file_path, doc_id, parent_chunk, child_idx, current_child_texts))
                child_idx += 1
                
                last_block = current_child_texts[-1]
                last_tok = self.cached_token_count(last_block)
                if last_tok <= self.child_overlap:
                    current_child_texts = [last_block, b_text]
                    current_tokens = last_tok + b_tokens
                else:
                    current_child_texts = [b_text]
                    current_tokens = b_tokens

        if current_child_texts:
            child_chunks.append(self._build_child_node(file_path, doc_id, parent_chunk, child_idx, current_child_texts))
            child_idx += 1

        return parent_chunk, child_chunks, child_idx

    def _build_child_node(self, file_path: str, doc_id: str, parent: ParentChunk, idx: int, text_blocks: List[str]) -> ChildChunk:
        combined_text = "\n\n".join(text_blocks).strip()
        content_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
        c_id = f"{doc_id}#c{idx}"

        return ChildChunk(
            chunk_id=c_id,
            parent_id=parent.parent_id,
            text=combined_text,
            token_count=self.cached_token_count(combined_text),
            content_hash=content_hash,
            # Phase 4 Opt: Only store the parent ID reference, not the entire string clone
            metadata={"source": file_path, "parent_id": parent.parent_id} 
        )
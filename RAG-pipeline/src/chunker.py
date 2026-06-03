# src/chunker.py
import re
import hashlib
import tiktoken
from typing import List, Tuple
from src.config import Config
from src.models import RawDocument, ParentChunk, ChildChunk, KnowledgeGraphIndex

class HierarchicalGraphChunker:
    """
    Advanced Data Ingestion Engine: Performs document pre-processing, 
    nested Parent-Child structural generation, and Graph RAG entity-triple parsing.
    """
    def __init__(self, graph_index: KnowledgeGraphIndex):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.graph_index = graph_index
        # Hardcoded constraints matching enterprise specifications
        self.parent_size = 512
        self.child_size = 128
        self.child_overlap = 30

    def pre_process_text(self, text: str) -> str:
        """
        Document Pre-processing Layer: Normalizes whitespaces, sanitizes anomalies, 
        and extracts unified structures to ensure formatting changes do not pollute vectors.
        """
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def _split_into_sentences(self, text: str) -> List[str]:
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        return [s.strip() for s in sentence_end.split(text) if s.strip()]

    def extract_graph_triples(self, text: str, chunk_id: str):
        """
        Graph RAG Extraction Engine: Parses semantic entities and operational verbs 
        natively out of text nodes to populate the multi-hop Knowledge Graph database.
        """
        # Look for explicit system design relationships: (Concept) -> [verb/relation] -> (Concept)
        patterns = [
            (r'(system|infrastructure|engine)\s+is\s+built\s+on\s+([\w\s]+?)(?=\.|\,)', "built_on"),
            (r'(deployment|runtime)\s+uses\s+([\w\s]+?)(?=\.|\,)', "uses_infrastructure"),
            (r'([\w\s]+?)\s+process\s+([\w\s]+?)(?=\.|\,)', "processes_tasks"),
            (r'([\w\s]+?)\s+requires\s+([\w\s]+?)(?=\.|\,)', "depends_on")
        ]
        
        for pattern, relation in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    self.graph_index.add_triple(match[0], relation, match[1], chunk_id)

    def process_document(self, doc: RawDocument) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Hierarchical Ingestion Execution Strategy: Builds unified structural Parent Chunks 
        and maps smaller, overlapping Child vectors to them.
        """
        sanitized_text = self.pre_process_text(doc.raw_text)
        
        # Split document by structural markdown headers to maintain block boundaries
        sections = [s.strip() for s in sanitized_text.split("\n## ") if s.strip()]
        
        parent_chunks = []
        child_chunks = []
        parent_idx = 0
        child_global_idx = 0

        for section in sections:
            # Reconstruct markdown header formatting if stripped
            full_section_text = f"## {section}" if not section.startswith("##") else section
            
            # Generate Parent Chunk Envelope
            p_id = f"{doc.document_id}#p{parent_idx}"
            parent_node = ParentChunk(parent_id=p_id, text=full_section_text, metadata={"source": doc.filename})
            parent_chunks.append(parent_node)
            
            # Sub-segment the Parent into targeted child text blocks using a true sliding window
            sentences = self._split_into_sentences(full_section_text)
            current_child_sentences = []
            current_child_tokens = 0
            
            s_idx = 0
            while s_idx < len(sentences):
                sentence = sentences[s_idx]
                sentence_tokens = len(self.encoder.encode(sentence))
                
                if current_child_tokens + sentence_tokens <= self.child_size:
                    current_child_sentences.append(sentence)
                    current_child_tokens += sentence_tokens
                    s_idx += 1
                else:
                    # Flush Child block
                    child_node = self._build_child_node(doc, parent_node, child_global_idx, current_child_sentences)
                    child_chunks.append(child_node)
                    child_global_idx += 1
                    
                    # Compute sliding context overlap window bounds
                    overlap_tokens = 0
                    overlap_sentences = []
                    for back_s in reversed(current_child_sentences):
                        back_tk = len(self.encoder.encode(back_s))
                        if overlap_tokens + back_tk <= self.child_overlap:
                            overlap_sentences.insert(0, back_s)
                            overlap_tokens += back_tk
                        else:
                            break
                    current_child_sentences = overlap_sentences
                    current_child_tokens = overlap_tokens
                    
            if current_child_sentences:
                child_node = self._build_child_node(doc, parent_node, child_global_idx, current_child_sentences)
                child_chunks.append(child_node)
                child_global_idx += 1
                
            parent_idx += 1
            
        return parent_chunks, child_chunks

    def _build_child_node(self, doc: RawDocument, parent: ParentChunk, idx: int, sentences: List[str]) -> ChildChunk:
        combined_text = " ".join(sentences).strip()
        content_hash = hashlib.sha256(combined_text.encode('utf-8')).hexdigest()[:16]
        c_id = f"{doc.document_id}#c{idx}"
        
        # Build relational Graph connections out of the raw text window
        self.extract_graph_triples(combined_text, c_id)
        
        return ChildChunk(
            chunk_id=c_id,
            parent_id=parent.parent_id,
            text=combined_text,
            token_count=len(self.encoder.encode(combined_text)),
            content_hash=content_hash,
            metadata={"source": doc.filename, "parent_context": parent.text}
        )
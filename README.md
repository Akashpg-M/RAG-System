# RAG - Hybrid Knowledge Retrieval System

> Retrieval-Augmented Generation system combining dense vector search, BM25 sparse retrieval, and LLM-extracted knowledge graphs into a unified, concurrently-executed hybrid search engine.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [End-to-End System Flow](#end-to-end-system-flow)
- [High-Level Retrieval Pipeline](#high-level-retrieval-pipeline)
- [Repository Structure](#repository-structure)
- [Component Architecture](#component-architecture)
- [Knowledge Graph Extraction Pipeline](#knowledge-graph-extraction-pipeline)
- [Hybrid Retrieval Pipeline](#hybrid-retrieval-pipeline)
- [Technology Stack](#technology-stack)
- [Production Features](#production-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Example Workflow](#example-workflow)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview

RAG is a **multi-index, hybrid retrieval system** designed for high-precision question answering over dense technical corpora. The system simultaneously operates three fully independent retrieval engines - **dense vector search** (Qdrant), **BM25 sparse search** (SQLite), and **Knowledge Graph traversal** (SQLite adjacency matrix) - running them in parallel and fusing their results using **Reciprocal Rank Fusion (RRF)** before a final neural cross-encoder reranking pass.

The pipeline is built to address a fundamental limitation of naive RAG systems: semantic embeddings compress meaning but lose precision on exact identifiers (model numbers, version strings, product codes), while keyword search finds exact matches but has no semantic understanding. The knowledge graph layer adds a third retrieval mode that captures *relational context* the entities involved in a query and how they connect - which neither vector nor keyword search can surface.

---

## Key Features

| Feature | Implementation |
|---|---|
| Layout-aware document parsing | Docling AST parser preserving headers, tables, and code blocks |
| Hierarchical chunking | Parent (structural) + Child (token-bounded) chunk strategy |
| Local embedding inference | `all-MiniLM-L6-v2` via SentenceTransformers on CPU |
| Exact-match keyword search | BM25 Inverted Index implemented natively on SQLite |
| Dense approximate nearest-neighbour | Qdrant with batched content-hash deduplication |
| LLM knowledge graph extraction | Groq `llama3-70b-8192` at `temperature=0.0` with JSON schema enforcement |
| Entity alias resolution | External `aliases.json` mapping raw entity surface forms to canonical names |
| Concurrent retrieval | Python `ThreadPoolExecutor` with GIL-released I/O across all three engines |
| Score-space-agnostic fusion | Reciprocal Rank Fusion (RRF) across dense, sparse, and graph results |
| Neural reranking | MiniLM Cross-Encoder on top-K candidate pool |
| Grounded generation | XML-wrapped context with strict hallucination refusal instructions |
| Async task queue | Thread-safe FIFO queue decoupling ingestion from query workloads |
| Exponential backoff | Jittered retry decorator protecting all external API calls |
| Nanosecond telemetry | `perf_counter_ns` profiling across retrieval, fusion, and generation phases |

---
## System Architecture

The system is organized into four primary layers. Documents are ingested asynchronously, indexed across multiple storage engines, retrieved in parallel, fused using Reciprocal Rank Fusion (RRF), reranked with a Cross-Encoder, and finally passed to the LLM for grounded response generation.

```text
┌───────────────────────────────────────────────────────────────────────────────┐
│                               INGESTION LAYER                                │
├───────────────────────────────────────────────────────────────────────────────┤
│ Docling AST Parser → Hierarchical Chunker → Tokenizer → Embedder             │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                                STORAGE LAYER                                 │
├───────────────────────────────────────────────────────────────────────────────┤
│  Qdrant Vector DB   │   SQLite BM25 Index   │   SQLite Knowledge Graph       │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                               RETRIEVAL LAYER                                │
├───────────────────────────────────────────────────────────────────────────────┤
│ Dense Retriever │ Sparse Retriever │ Graph Retriever (Parallel Execution)    │
│                                  │                                            │
│                     Reciprocal Rank Fusion (RRF)                             │
│                                  │                                            │
│                      Cross-Encoder Neural Reranker                           │
└───────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                               GENERATION LAYER                               │
├───────────────────────────────────────────────────────────────────────────────┤
│ Semantic Query Processor → XML Prompt Builder → Groq Streaming Response      │
└───────────────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

<details>
<summary><strong>Project Structure</strong></summary>

```text
RAG_pipeline/
│
├── config/
│   └── ontologies/
│       ├── software.json
│       └── aliases.json
│
├── data/
│   └── raw/
│
├── src/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── tokenizer.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   ├── sparse_store.py
│   ├── graph_store.py
│   ├── retrieval.py
│   ├── retrievers.py
│   ├── fusion.py
│   ├── generation.py
│   ├── semantic_processor.py
│   ├── queue_worker.py
│   ├── telemetry.py
│   ├── utils.py
│   │
│   └── graph/
│       ├── ontology.py
│       ├── prompt_builder.py
│       └── graph_extractor.py
│
├── requirements.txt
├── stopwords.txt
├── LICENSE
└── README.md
```

</details>

---

## Component Architecture

### `src/main.py` - Dependency Injection Container & Entry Point

`main.py` serves as the composition root of the application. It initializes all infrastructure components, constructs the dependency graph, injects shared instances into downstream modules, boots the background ingestion worker, validates ontology configuration, and starts the retrieval pipeline.

### `src/config.py` - Immutable Environment Configuration

Loads runtime configuration from `.env`, exposes immutable application constants, and configures global logging.

### `src/models.py` - Data Contract Layer

Defines the core Pydantic models and dataclasses used throughout the pipeline, including `ParentChunk`, `ChildChunk`, and `EmbeddingCache`.

### `src/chunker.py` - Layout-Aware Hierarchical Chunker

Uses Docling's AST parser to preserve document structure while producing hierarchical parent chunks and token-bounded child chunks for downstream indexing.

### `src/tokenizer.py` - Text Normalization Pipeline

Performs Unicode normalization, punctuation removal, tokenization, and stopword filtering prior to BM25 indexing.

### `src/embedder.py` - Local Batch Embedding Engine

Generates dense embeddings using SentenceTransformers, batching requests and caching vectors using deterministic content hashes.

### `src/vector_store.py` - Dense Vector Storage

Manages embedding persistence and similarity search through Qdrant while avoiding duplicate uploads via hash-based filtering.

### `src/sparse_store.py` - BM25 Search Engine

Implements a native SQLite-backed BM25 inverted index with document frequency tracking and statistical ranking.

### `src/graph_store.py` - Knowledge Graph Database

Stores entity-relation triples inside SQLite and supports multi-hop Breadth-First Search traversal over indexed graph edges.

### `src/retrievers.py` - Independent Retrieval Workers

Contains independent Dense, Sparse, and Graph retrievers that execute concurrently and remain fully decoupled.

### `src/semantic_processor.py` - Query Understanding

Uses an LLM to rewrite user queries, identify technical intent, and extract important entities before retrieval.

### `src/retrieval.py` - Parallel Retrieval Engine

Coordinates concurrent retrieval using `ThreadPoolExecutor`, applies Reciprocal Rank Fusion, and performs Cross-Encoder reranking over the top candidates.

### `src/fusion.py` - Reciprocal Rank Fusion

Combines heterogeneous ranking signals into a unified score using:

```text
RRF(d) = Σ 1 / (k + rankᵢ(d))
```

where **k = 60**.

### `src/generation.py` - Grounded Response Generation

Constructs XML-based prompts from retrieved evidence and streams grounded responses from Groq while minimizing hallucinations.

---

## Knowledge Graph Extraction Pipeline

```text
                Parent Chunk
                     │
                     ▼
        Ontology Validation (Pydantic)
                     │
                     ▼
      Dynamic Prompt Builder (Ontology)
                     │
                     ▼
         Groq LLM (JSON Mode, temp=0)
                     │
                     ▼
        JSON Schema Validation
                     │
      ┌──────────────┼──────────────┐
      ▼              ▼              ▼
 Alias Mapping   Deduplication   Filtering
      │
      ▼
 SQLite Knowledge Graph Store
      │
      ▼
 Multi-hop BFS Retrieval
```

---

## Hybrid Retrieval Pipeline

The retrieval engine combines three complementary retrieval strategies.

| Engine | Strength | Limitation |
|---------|----------|------------|
| Dense Retrieval | Semantic similarity | Misses exact identifiers |
| Sparse Retrieval | Exact keyword matching | Weak semantic understanding |
| Graph Retrieval | Entity relationships | Requires extracted graph structure |

All three retrievers execute concurrently.

Their results are merged using **Reciprocal Rank Fusion (RRF)** and then passed through a **Cross-Encoder** reranker before grounded generation.

```text
User Query
     │
     ▼
Semantic Query Processing
     │
     ├───────────────┬───────────────┐
     ▼               ▼               ▼
 Dense          Sparse          Graph
Retriever      Retriever      Retriever
     └───────────────┬───────────────┘
                     ▼
        Reciprocal Rank Fusion
                     ▼
      Cross-Encoder Reranker
                     ▼
       Final Context Window
                     ▼
      Grounded LLM Generation
```

---

## Technology Stack

| Category | Library / Tool | Version Notes |
|---|---|---|
| LLM Inference | Groq (`llama3-70b-8192`) | Remote API, `temperature=0.0` for extraction |
| Embedding Model | SentenceTransformers `all-MiniLM-L6-v2` | Local CPU, 384-dim output |
| Cross-Encoder Reranker | SentenceTransformers `CrossEncoder` | MiniLM architecture |
| Document Parser | Docling `DocumentConverter` | AST-level layout parsing |
| Dense Vector Store | Qdrant (local file backend) | `qdrant-client` |
| Sparse Store | SQLite (`sqlite3`) | Native BM25 implementation |
| Knowledge Graph Store | SQLite (`sqlite3`) | Adjacency matrix schema |
| Token Counting | `tiktoken` (`cl100k_base` encoder) | Deterministic tokenisation |
| Text Normalization | NLTK | Word tokenization + stopwords |
| Data Validation | Pydantic (`BaseModel`) | Runtime schema enforcement |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | GIL-released parallel I/O |
| Task Queue | `queue.Queue` + `threading.Thread` | Thread-safe FIFO |
| Configuration | `python-dotenv` | `.env`-based secret loading |
| Profiling | `time.perf_counter_ns` | Nanosecond-resolution |

---

## Installation

**Prerequisites:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/Akashpg-M/RAG-System
cd RAG-pipeline

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the NLTK stopwords corpus (one-time)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

---

## Configuration

Copy `.env.example` to `.env` and populate the following variables:

```env
# ── LLM Provider ─────────────────────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# ── Chunking Hyperparameters ──────────────────────────────────────────────────
CHUNK_SIZE=512          # Maximum token count per Child chunk
CHUNK_OVERLAP=64        # Sliding window overlap between adjacent Child chunks

# ── Retrieval Settings ────────────────────────────────────────────────────────
TOP_K_DENSE=20          # Candidates returned from Qdrant
TOP_K_SPARSE=20         # Candidates returned from BM25
TOP_K_GRAPH=10          # Candidates returned from graph traversal
RERANK_TOP_N=10         # Cross-Encoder input pool size after RRF fusion

# ── Storage Paths ─────────────────────────────────────────────────────────────
QDRANT_PATH=./data/qdrant
SQLITE_PATH=./data/store.db
```

**Ontology Configuration**

Edit `config/ontologies/software.json` to define the entity types the graph extractor should look for:

```json
{
  "entity_types": ["SOFTWARE", "LIBRARY", "API", "DATABASE", "PROTOCOL"],
  "relation_types": ["DEPENDS_ON", "INTEGRATES_WITH", "REPLACES", "EXTENDS"]
}
```

Edit `config/ontologies/aliases.json` to map surface forms to canonical names:

```json
{
  "postgres": "PostgreSQL",
  "pg": "PostgreSQL",
  "k8s": "Kubernetes",
  "tf": "TensorFlow"
}
```

---

## Running the Project

**Start the application:**

```bash
python src/main.py
```

The boot sequence will:
1. Validate ontology configuration files via Pydantic
2. Initialize storage connection pools (Qdrant, SQLite BM25, SQLite Graph)
3. Start the background ingestion worker thread
4. Enter the interactive query loop

**Ingest a document:**

```
> ingest data/raw/architecture_guide.pdf
[INFO] Task queued: task_3f9a1c2d
[INFO] Background worker: parsing document...
[INFO] Background worker: extracted 847 child chunks, 112 parent chunks
[INFO] Background worker: embedded and indexed in 14.3s
[INFO] Background worker: extracted 203 KG triples
```

**Issue a query:**

```
> query: What are the known connection pool limits for PostgreSQL in this system?
[Retrieving...]
[Dense: 18 candidates | Sparse: 12 candidates | Graph: 6 candidates]
[RRF fusion: 28 unique candidates → Cross-Encoder rerank top 10]

Answer: Based on the documentation, the PostgreSQL connection pool is configured
with a maximum of 20 concurrent connections per worker process...
```

---

## Example Workflow

**Scenario:** An infrastructure team wants to query a collection of internal architecture decision records (ADRs) and system design documents.

```bash
# Batch-ingest all documents in the raw folder
for f in data/raw/*.pdf; do
  python src/main.py --ingest "$f"
done

# Issue a multi-hop relational query
python src/main.py --query "Which services depend on the message broker, and what failover strategy do they use?"
```

**What happens internally:**

1. `semantic_processor.py` identifies `["message broker", "failover strategy"]` as the key entities.
2. `DenseRetriever` finds chunks semantically similar to message queue architecture patterns.
3. `SparseRetriever` finds chunks containing exact tokens `message`, `broker`, `failover`.
4. `GraphRetriever` identifies `MessageBroker` as a seed entity and BFS-walks to `ServiceA → DEPENDS_ON → MessageBroker`, `ServiceA → USES → CircuitBreaker`, surfacing relational context that neither dense nor sparse retrieval would find.
5. RRF fusion promotes chunks that appear in multiple result sets.
6. Cross-Encoder reranks the top candidates by direct pairwise relevance scoring.
7. `generation.py` synthesises a grounded answer across all three evidence streams.

---

## Future Improvements

| Area | Planned Improvement |
|---|---|
| **Graph Store** | Migrate to a native graph database (Neo4j, Kuzu) for richer Cypher query support and native graph algorithms |
| **Embedding Model** | Evaluate `bge-large-en-v1.5` or `e5-mistral-7b-instruct` for higher recall on technical text |
| **Sparse Index** | Add field-weighted BM25F to score matches in headings higher than body text |
| **Reranker** | Replace MiniLM Cross-Encoder with a larger reranker model for higher reranking precision |
| **Multimodal Ingestion** | Add table-extraction and diagram-captioning via vision models to handle complex PDF layouts |
| **Graph Extraction** | Implement multi-document co-reference resolution to link entities mentioned across separate files |
| **API Layer** | Expose retrieval and ingestion as a FastAPI service with async endpoints for production HTTP integration |
| **Persistent Cache** | Move embedding cache from in-process memory to Redis for persistence across restarts |
| **Evaluation Framework** | Integrate RAGAS or a custom NDCG evaluation harness for retrieval quality benchmarking |

---

## License

This project is licensed under the [MIT License](LICENSE).

---

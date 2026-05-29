import random

from src.models import RawDocument
from src.engine import IngestionEngine
from src.vector_store import VectorStoreManager


def load_file(file_path: str) -> str:
    """
    Reads raw text from a file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def generate_mock_embeddings(
    num_chunks: int,
    dimensions: int = 1536
) -> list:
    """
    Generates mock embedding vectors for testing
    the vector database pipeline.
    """

    return [
        [random.uniform(-1.0, 1.0) for _ in range(dimensions)]
        for _ in range(num_chunks)
    ]


def main():

    print("\n--- Starting AI Ingestion Pipeline ---\n")

    # 1. Read raw document
    file_path = "data/raw/system_design.md"

    raw_text = load_file(file_path)

    # 2. Create structured document object
    doc = RawDocument.from_text(
        filename="system_design.md",
        raw_text=raw_text
    )

    print(f"Loaded Document ID: {doc.document_id}")

    # 3. Initialize ingestion engine
    engine = IngestionEngine(
        chunk_size=30,
        chunk_overlap=5
    )

    # 4. Process and chunk document
    chunks = engine.process_document(doc)

    print(f"Generated {len(chunks)} chunks.\n")

    # 5. Preview chunks
    print("Previewing First 2 Chunks:\n")

    for chunk in chunks[:2]:

        print("=" * 60)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.text)
        print()

    # 6. Generate mock embeddings
    print("Generating mock embeddings...\n")

    mock_embeddings = generate_mock_embeddings(
        num_chunks=len(chunks),
        dimensions=1536
    )

    print(f"Generated {len(mock_embeddings)} embedding vectors.")

    # 7. Initialize vector database manager
    db_manager = VectorStoreManager(
        collection_name="production_kb",
        vector_size=1536
    )

    # 8. Store chunks + embeddings in Qdrant
    db_manager.upsert_chunks(
        chunks=chunks,
        embeddings=mock_embeddings
    )

    print("\n--- Ingestion Pipeline Completed Successfully ---\n")


if __name__ == "__main__":
    main()
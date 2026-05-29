from src.models import RawDocument
from src.engine import IngestionEngine


def load_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def main():

    # 1. Read document from external file
    file_path = "data/raw/system_design.md"
    raw_text = load_file(file_path)

    # 2. Load Document
    doc = RawDocument.from_text(
        filename="system_design.md",
        raw_text=raw_text
    )

    print(f"Loaded Document ID: {doc.document_id}")

    # 3. Initialize Engine
    engine = IngestionEngine(
        chunk_size=30,
        chunk_overlap=5
    )

    # 4. Process Document
    chunks = engine.process_document(doc)

    print(f"Generated {len(chunks)} chunks.\n")

    # 5. Preview Chunks
    for chunk in chunks[:2]:
        print("=" * 50)
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Tokens: {chunk.token_count}")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.text)


if __name__ == "__main__":
    main()
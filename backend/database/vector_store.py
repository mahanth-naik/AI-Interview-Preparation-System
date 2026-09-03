import chromadb
from pathlib import Path
from uuid import uuid4

client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma"))

collection = client.get_or_create_collection(
    name="interview_documents"
)


def add_chunks(chunks: list[str], filename: str):
    ids = [
        f"{uuid4().hex}_{i}"
        for i in range(len(chunks))
    ]

    if not chunks:
        return 0

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[
            {
                "filename": filename,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]
    )

    return len(chunks)


def search_chunks(query: str, n_results: int = 3):
    if collection.count() == 0:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results
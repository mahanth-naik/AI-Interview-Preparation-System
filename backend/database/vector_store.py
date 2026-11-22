
import chromadb
from pathlib import Path
from uuid import uuid4

client = chromadb.PersistentClient(
    path=str(Path(__file__).parent / "chroma")
)

collection = client.get_or_create_collection(
    name="interview_documents"
)


def add_chunks(
    chunks: list[str], filename: str, user_id: str | None = None
):
    if not chunks:
        return 0

    document_id = uuid4().hex

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": filename,
            "document_id": document_id,
            "chunk_index": i
        }
        for i in range(len(chunks))
    ]
    if user_id is not None:
        for metadata in metadatas:
            metadata["user_id"] = user_id

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )

    return {
        "document_id": document_id,
        "chunks_stored": len(chunks)
    }


def search_chunks(
    query: str,
    n_results: int = 3,
    document_id: str | None = None,
    user_id: str | None = None,
):
    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    filters = []
    if document_id:
        filters.append({"document_id": document_id})
    if user_id:
        filters.append({"user_id": user_id})
    where = (
        {"$and": filters}
        if len(filters) > 1
        else filters[0]
        if filters
        else None
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where
    )

    return results

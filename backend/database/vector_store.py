
import chromadb
from pathlib import Path
from uuid import uuid4

client = chromadb.PersistentClient(
    path=str(Path(__file__).parent / "chroma")
)

collection = client.get_or_create_collection(
    name="interview_documents"
)


def _empty_search_results() -> dict:
    return {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def add_chunks(
    chunks: list[str],
    filename: str,
    user_id: str | None = None,
    document_type: str = "resume",
    document_id: str | None = None,
):
    if not chunks:
        return 0

    document_id = document_id or uuid4().hex

    ids = [
        f"{document_id}_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "filename": filename,
            "document_id": document_id,
            "document_type": document_type,
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


def delete_document_chunks(document_id: str, user_id: str) -> None:
    collection.delete(
        where={
            "$and": [
                {"document_id": document_id},
                {"user_id": user_id},
            ]
        }
    )


def search_chunks(
    query: str,
    n_results: int = 3,
    document_id: str | None = None,
    user_id: str | None = None,
    document_type: str | None = None,
):
    if not isinstance(user_id, str) or not user_id.strip():
        return _empty_search_results()

    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        return _empty_search_results()

    total_chunks = collection.count()

    if total_chunks == 0:
        return _empty_search_results()

    filters = []
    if document_id is not None:
        filters.append({"document_id": document_id})
    filters.append({"user_id": user_id})
    if document_type is not None:
        filters.append({"document_type": document_type})
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

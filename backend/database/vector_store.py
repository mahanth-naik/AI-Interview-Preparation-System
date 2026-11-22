
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
    chunks: list[str | dict[str, str | int]],
    filename: str,
    user_id: str | None = None,
    document_type: str = "resume",
    document_id: str | None = None,
):
    if not chunks:
        return 0

    document_id = document_id or uuid4().hex

    prepared_chunks = []
    seen_chunk_ids = set()
    for index, chunk in enumerate(chunks):
        if isinstance(chunk, str):
            text = chunk
            chunk_id = f"chunk-{index:06d}"
            page_number = None
        else:
            text = chunk.get("text")
            chunk_id = chunk.get("chunk_id")
            page_number = chunk.get("page_number")
            if not isinstance(text, str):
                raise ValueError("Each chunk must include text")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError("Each chunk must include a chunk_id")
            if page_number is not None and (
                not isinstance(page_number, int)
                or isinstance(page_number, bool)
                or page_number < 1
            ):
                raise ValueError("page_number must be a positive integer")

        if chunk_id in seen_chunk_ids:
            raise ValueError("chunk_id values must be unique within a document")
        seen_chunk_ids.add(chunk_id)
        prepared_chunks.append((text, chunk_id, page_number))

    ids = [
        f"{document_id}_{i}"
        for i in range(len(prepared_chunks))
    ]

    metadatas = []
    for index, (text, chunk_id, page_number) in enumerate(prepared_chunks):
        metadata = {
            "filename": filename,
            "document_id": document_id,
            "document_type": document_type,
            "chunk_index": index,
            "chunk_id": chunk_id,
        }
        if page_number is not None:
            metadata["page_number"] = page_number
        if user_id is not None:
            metadata["user_id"] = user_id
        metadatas.append(metadata)

    collection.add(
        documents=[text for text, _, _ in prepared_chunks],
        ids=ids,
        metadatas=metadatas
    )

    return {
        "document_id": document_id,
        "chunks_stored": len(prepared_chunks)
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

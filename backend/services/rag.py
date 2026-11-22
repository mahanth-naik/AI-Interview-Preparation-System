
from database.vector_store import search_chunks


def retrieve_context(
    query: str,
    n_results: int = 3,
    document_id: str | None = None,
    user_id: str | None = None,
    document_type: str | None = None,
) -> list[dict]:

    query = query.strip()

    if not query:
        raise ValueError("query must not be empty")

    if n_results < 1 or n_results > 20:
        raise ValueError("n_results must be between 1 and 20")

    if not isinstance(user_id, str) or not user_id.strip():
        return []

    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        return []

    filters = {"user_id": user_id}
    if document_id is not None:
        filters["document_id"] = document_id
    if document_type is not None:
        filters["document_type"] = document_type

    results = search_chunks(query, n_results, **filters)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    evidence = []
    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        metadata = metadata or {}
        page_number = metadata.get("page_number")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            page_number = None

        chunk_id = metadata.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = ids[index] if index < len(ids) else None
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            chunk_id = None

        evidence.append(
            {
                "document_id": metadata.get("document_id"),
                "filename": metadata.get("filename"),
                "document_type": metadata.get("document_type"),
                "page_number": page_number,
                "chunk_id": chunk_id,
                "text": document,
                "distance": distances[index] if index < len(distances) else None,
                "metadata": metadata,
            }
        )

    return evidence


def build_context(
    documents: list[str] | list[dict],
    context_mode: str | None = None,
) -> str:

    if not documents:
        return "No relevant information was found."

    heading = {
        "resume": "Resume evidence",
        "study_material": "Study-material evidence",
    }.get(context_mode, "Context")

    context_parts = []

    for i, document in enumerate(documents, start=1):
        source_details = []
        if isinstance(document, dict):
            text = document["text"]
            for label, key in (
                ("source", "filename"),
                ("page", "page_number"),
                ("chunk", "chunk_id"),
            ):
                value = document.get(key)
                if value is not None:
                    source_details.append(f"{label}={value}")
        else:
            text = document

        reference = (
            f"; {'; '.join(source_details)}"
            if source_details
            else ""
        )
        context_parts.append(
            f"[{heading} {i}{reference}]\n{text}"
        )

    return "\n\n".join(context_parts)

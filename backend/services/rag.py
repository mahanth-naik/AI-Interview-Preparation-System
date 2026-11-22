
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

    return [
        {
            "text": document,
            "metadata": metadata or {},
            "distance": distances[index] if index < len(distances) else None,
        }
        for index, (document, metadata) in enumerate(
            zip(documents, metadatas)
        )
    ]


def build_context(documents: list[str]) -> str:

    if not documents:
        return "No relevant information was found."

    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Context {i}]\n{document}"
        )

    return "\n\n".join(context_parts)

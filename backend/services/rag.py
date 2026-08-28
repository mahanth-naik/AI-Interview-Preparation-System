def build_context(documents: list[str]) -> str:
    if not documents:
        return "No relevant information was found."

    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"[Context {i}]\n{document}"
        )

    return "\n\n".join(context_parts)
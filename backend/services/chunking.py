
from collections.abc import Iterable


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between zero and chunk_size - 1")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


def chunk_pages(
    pages: Iterable[tuple[int, str]],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict[str, str | int]]:
    chunks = []

    for page_number, text in pages:
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError("page_number must be a positive integer")

        for page_chunk in chunk_text(text, chunk_size, overlap):
            chunks.append(
                {
                    "text": page_chunk,
                    "page_number": page_number,
                    "chunk_id": f"chunk-{len(chunks):06d}",
                }
            )

    return chunks

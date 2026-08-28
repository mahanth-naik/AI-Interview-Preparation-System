import chromadb

client = chromadb.PersistentClient(path="./database/chroma")

collection = client.get_or_create_collection(
    name="interview_documents"
)


def add_chunks(chunks: list[str], filename: str):
    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    # Remove existing chunks for this file
    existing = collection.get(
        where={"filename": filename}
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=[
            {"filename": filename}
            for _ in chunks
        ]
    )

    return len(chunks)


def search_chunks(query: str, n_results: int = 3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results
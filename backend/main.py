from database.vector_store import add_chunks, search_chunks
from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
from pypdf import PdfReader

from services.chunking import chunk_text
import services.rag

app = FastAPI(title="AI Interview Preparation System")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "AI Interview Preparation System API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / file.filename

    content = await file.read()
    file_path.write_bytes(content)

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    chunks = chunk_text(text)
    chunks_stored = add_chunks(chunks, file.filename)

    return {
    "filename": file.filename,
    "pages": len(reader.pages),
    "characters_extracted": len(text),
    "chunks_created": len(chunks),
    "chunks_stored": chunks_stored,
    "message": "PDF uploaded, text extracted, chunked, and stored successfully"
}

@app.get("/documents/search")
def search_documents(query: str, n_results: int = 3):
    results = search_chunks(query, n_results)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    matches = []

    for document, metadata in zip(documents, metadatas):
        matches.append({
            "text": document,
            "source": metadata.get("filename")
        })

    return {
        "query": query,
        "results": matches
    }
@app.get("/ask")
def ask_question(query: str, n_results: int = 3):
    results = search_chunks(query, n_results)

    documents = results.get("documents", [[]])[0]

    context = services.rag.build_context(documents)

    return {
        "question": query,
        "retrieved_chunks": len(documents),
        "context": context
    }

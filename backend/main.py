from database.vector_store import add_chunks
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pathlib import Path
from pypdf import PdfReader
from pydantic import BaseModel, Field

from services.chunking import chunk_text
from services.interview import InterviewService
from services.rag import build_context, retrieve_context

app = FastAPI(title="AI Interview Preparation System")

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
interview_service = InterviewService()


class StartInterviewRequest(BaseModel):
    role: str = Field(min_length=2, max_length=100)
    interview_type: str = Field(min_length=2, max_length=50)
    difficulty: str = Field(default="medium", min_length=3, max_length=20)
    number_of_questions: int = Field(default=5, ge=1, le=20)


class QuestionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class EvaluationRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)


@app.get("/")
def root():
    return {"message": "AI Interview Preparation System API is running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    file_path = UPLOAD_DIR / Path(filename).name

    content = await file.read()
    file_path.write_bytes(content)

    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable PDF") from exc

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise HTTPException(status_code=422, detail="The PDF contains no extractable text")

    chunks = chunk_text(text)
    chunks_stored = add_chunks(chunks, filename)

    return {
    "filename": filename,
    "pages": len(reader.pages),
    "characters_extracted": len(text),
    "chunks_created": len(chunks),
    "chunks_stored": chunks_stored,
    "message": "PDF uploaded, text extracted, chunked, and stored successfully"
}

@app.get("/documents/search")
def search_documents(query: str = Query(min_length=1), n_results: int = Query(default=3, ge=1, le=20)):
    try:
        matches = retrieve_context(query, n_results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "query": query,
        "results": [
            {"text": item["text"], "source": item["metadata"].get("filename"), "distance": item["distance"]}
            for item in matches
        ]
    }


@app.get("/ask")
def ask_question(query: str = Query(min_length=1), n_results: int = Query(default=3, ge=1, le=20)):
    try:
        matches = retrieve_context(query, n_results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "question": query,
        "retrieved_chunks": len(matches),
        "context": build_context([item["text"] for item in matches]),
    }


@app.post("/interview/start")
def start_interview(request: StartInterviewRequest):
    try:
        session = interview_service.start(
            request.role, request.interview_type, request.difficulty, request.number_of_questions
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"session_id": session.session_id, "role": session.role, "interview_type": session.interview_type,
            "difficulty": session.difficulty, "number_of_questions": len(session.questions), "provider": "local"}


@app.post("/interview/question")
def get_question(request: QuestionRequest):
    try:
        return interview_service.next_question(request.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Interview session not found") from exc


@app.post("/interview/evaluate")
def evaluate_answer(request: EvaluationRequest):
    try:
        return interview_service.evaluate(request.session_id, request.question, request.answer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Interview session not found") from exc


@app.get("/interview/{session_id}")
def get_interview(session_id: str):
    try:
        return interview_service.get(session_id).__dict__
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Interview session not found") from exc

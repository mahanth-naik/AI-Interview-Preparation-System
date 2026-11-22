
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from database.vector_store import add_chunks
from auth import get_current_user, router as auth_router
from services.chunking import chunk_text
from services.rag import build_context, retrieve_context
from services.interview import InterviewService
from database.document_store import SQLiteDocumentStore
from services.ai_provider import (
    LLMConfigurationError,
    LLMProviderError,
)


app = FastAPI(title="AI Interview Preparation System")
app.include_router(auth_router)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

interview_service = InterviewService()
document_store = SQLiteDocumentStore()


def _require_owned_document(document_id: str, user_id: str) -> None:
    if document_store.get_for_user(document_id, user_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )


# Request models

class StartInterviewRequest(BaseModel):
    role: str = Field(min_length=1)
    interview_type: str = Field(min_length=1)
    difficulty: str = Field(default="medium")
    number_of_questions: int = Field(default=5, ge=1, le=20)
    document_id: str | None = None


class QuestionRequest(BaseModel):
    session_id: str = Field(min_length=1)


class EvaluationRequest(BaseModel):
    session_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)

# Basic routes

@app.get("/")
def root():
    return {
        "message": "AI Interview Preparation System API is running"
    }


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# Document upload

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    filename = Path(file.filename).name

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    try:
        reader = PdfReader(BytesIO(content))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded PDF"
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF"
        )

    chunks = chunk_text(text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Unable to create text chunks from the PDF"
        )

    user_id = current_user["user_id"]
    storage_result = add_chunks(chunks, filename, user_id=user_id)
    document_store.register(
        storage_result["document_id"],
        user_id,
        filename,
    )

    return {
        "filename": filename,
        "document_id": storage_result["document_id"],
        "pages": len(reader.pages),
        "characters_extracted": len(text),
        "chunks_created": len(chunks),
        "chunks_stored": storage_result["chunks_stored"],
        "message": "PDF uploaded, text extracted, chunked, and stored successfully"
    }


# Document search

@app.get("/documents/search")
def search_documents(
    query: str = Query(min_length=1),
    n_results: int = Query(default=3, ge=1, le=20),
    document_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    if document_id is not None:
        _require_owned_document(document_id, current_user["user_id"])

    try:
        matches = retrieve_context(
            query,
            n_results,
            document_id=document_id,
            user_id=current_user["user_id"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    return {
        "query": query,
        "document_id": document_id,
        "results": matches
    }


# Ask questions using RAG

@app.get("/ask")
def ask_question(
    query: str = Query(min_length=1),
    n_results: int = Query(default=3, ge=1, le=20),
    document_id: str | None = None,
    current_user: dict = Depends(get_current_user),
):
    if document_id is not None:
        _require_owned_document(document_id, current_user["user_id"])

    try:
        matches = retrieve_context(
            query,
            n_results,
            document_id=document_id,
            user_id=current_user["user_id"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    return {
        "question": query,
        "document_id": document_id,
        "retrieved_chunks": len(matches),
        "context": build_context(
            [item["text"] for item in matches]
        )
    }


# Interview routes

@app.post("/interview/start")
def start_interview(
    request: StartInterviewRequest,
    current_user: dict = Depends(get_current_user),
):
    if request.document_id is not None:
        _require_owned_document(request.document_id, current_user["user_id"])

    try:
        session = interview_service.start(
            request.role,
            request.interview_type,
            request.difficulty,
            request.number_of_questions,
            request.document_id,
            user_id=current_user["user_id"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    return {
        "session_id": session.session_id,
        "role": session.role,
        "interview_type": session.interview_type,
        "difficulty": session.difficulty,
        "number_of_questions": session.target_question_count,
        "document_id": session.document_id,
        "provider": interview_service.provider.name
    }


@app.post("/interview/question")
def get_question(
    request: QuestionRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return interview_service.next_question(
            request.session_id,
            user_id=current_user["user_id"],
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Interview session not found"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except (LLMConfigurationError, LLMProviderError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc


@app.post("/interview/evaluate")
def evaluate_answer(
    request: EvaluationRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        return interview_service.evaluate(
            request.session_id,
            request.question,
            request.answer,
            user_id=current_user["user_id"],
        )

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Interview session not found"
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc

    except (LLMConfigurationError, LLMProviderError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc)
        ) from exc


@app.get("/interviews")
def list_interviews(
    limit: int = Query(default=50, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
):
    try:
        return {
            "sessions": interview_service.list_sessions(
                limit,
                user_id=current_user["user_id"],
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/interview/{session_id}/report")
def get_interview_report(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        return interview_service.get_report(
            session_id,
            user_id=current_user["user_id"],
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Interview session not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/interview/{session_id}")
def get_interview(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        session = interview_service.get(
            session_id,
            user_id=current_user["user_id"],
        )
        return session.__dict__

    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Interview session not found"
        ) from exc

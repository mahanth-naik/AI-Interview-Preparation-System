
import os
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Query,
    Depends,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pypdf import PdfReader

from database.vector_store import add_chunks, delete_document_chunks
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
MAX_DOCUMENT_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024


def _require_owned_document(document_id: str, user_id: str) -> dict:
    document = document_store.get_for_user(document_id, user_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )
    return document


def _public_document(document: dict) -> dict:
    return {
        "document_id": document["document_id"],
        "filename": document["filename"],
        "document_type": document["document_type"],
        "processing_status": document["processing_status"],
        "created_at": document["created_at"],
    }


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
    document_type: Literal["resume", "study_material"] = Form(
        default="resume"
    ),
    current_user: dict = Depends(get_current_user),
):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    filename = Path(file.filename).name

    content = await file.read(MAX_DOCUMENT_UPLOAD_SIZE_BYTES + 1)

    if len(content) > MAX_DOCUMENT_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MiB upload limit",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty"
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid PDF",
        )

    user_id = current_user["user_id"]
    document_id = uuid4().hex
    document_store.register(
        document_id,
        user_id,
        filename,
        document_type=document_type,
        processing_status="processing",
    )

    try:
        reader = PdfReader(BytesIO(content))
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except Exception as exc:
        document_store.set_processing_status(
            document_id,
            user_id,
            "failed",
        )
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded PDF"
        ) from exc

    if not text.strip():
        document_store.set_processing_status(
            document_id,
            user_id,
            "failed",
        )
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in the PDF"
        )

    chunks = chunk_text(text)

    if not chunks:
        document_store.set_processing_status(
            document_id,
            user_id,
            "failed",
        )
        raise HTTPException(
            status_code=400,
            detail="Unable to create text chunks from the PDF"
        )

    processing_complete = False
    try:
        storage_result = add_chunks(
            chunks,
            filename,
            user_id=user_id,
            document_type=document_type,
            document_id=document_id,
        )
        document_store.set_processing_status(
            document_id,
            user_id,
            "ready",
        )
        processing_complete = True
    finally:
        if not processing_complete:
            document_store.set_processing_status(
                document_id,
                user_id,
                "failed",
            )

    return {
        "filename": filename,
        "document_id": document_id,
        "document_type": document_type,
        "processing_status": "ready",
        "pages": len(reader.pages),
        "characters_extracted": len(text),
        "chunks_created": len(chunks),
        "chunks_stored": storage_result["chunks_stored"],
        "message": "PDF uploaded, text extracted, chunked, and stored successfully"
    }


@app.get("/documents")
def list_documents(
    current_user: dict = Depends(get_current_user),
):
    return {
        "documents": [
            _public_document(document)
            for document in document_store.list_for_user(
                current_user["user_id"]
            )
        ]
    }


# Document search

@app.get("/documents/search")
def search_documents(
    query: str = Query(min_length=1),
    n_results: int = Query(default=3, ge=1, le=20),
    document_id: str | None = None,
    document_type: Literal["resume", "study_material"] | None = None,
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
            document_type=document_type,
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
    document_type: Literal["resume", "study_material"] | None = None,
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
            document_type=document_type,
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


@app.get("/documents/{document_id}")
def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    document = document_store.get_for_user(
        document_id,
        current_user["user_id"],
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return _public_document(document)


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
):
    user_id = current_user["user_id"]
    _require_owned_document(document_id, user_id)
    delete_document_chunks(document_id, user_id)
    if not document_store.delete_for_user(document_id, user_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return Response(status_code=204)


# Interview routes

@app.post("/interview/start")
def start_interview(
    request: StartInterviewRequest,
    current_user: dict = Depends(get_current_user),
):
    if request.document_id is not None:
        document = _require_owned_document(
            request.document_id,
            current_user["user_id"],
        )
        if document["processing_status"] != "ready":
            raise HTTPException(
                status_code=409,
                detail="Document is not ready for interview use",
            )

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

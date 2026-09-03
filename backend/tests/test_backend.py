from fastapi.testclient import TestClient

import main
from database import vector_store
from services import interview
from services.chunking import chunk_text
from services import rag


client = TestClient(main.app)


def test_chunking_has_expected_overlap_and_validation():
    assert chunk_text("abcdefghij", chunk_size=6, overlap=2) == ["abcdef", "efghij", "ij"]

    try:
        chunk_text("text", chunk_size=2, overlap=2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap should raise ValueError")


def test_retrieval_returns_text_metadata_and_distance(monkeypatch):
    monkeypatch.setattr(
        rag,
        "search_chunks",
        lambda query, n_results: {
            "documents": [["FastAPI experience"]],
            "metadatas": [[{"filename": "resume.pdf", "chunk_index": 0}]],
            "distances": [[0.2]],
        },
    )
    results = rag.retrieve_context("FastAPI", 1)
    assert results == [{
        "text": "FastAPI experience",
        "metadata": {"filename": "resume.pdf", "chunk_index": 0},
        "distance": 0.2,
    }]


def test_empty_vector_collection_returns_empty_results(monkeypatch):
    class EmptyCollection:
        def count(self):
            return 0

    monkeypatch.setattr(vector_store, "collection", EmptyCollection())
    assert vector_store.search_chunks("anything") == {
        "documents": [[]], "metadatas": [[]], "distances": [[]]
    }


def test_upload_rejects_non_pdf():
    response = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )
    assert response.status_code == 400


def test_interview_session_and_evaluation(monkeypatch):
    monkeypatch.setattr(interview, "retrieve_context", lambda query, n_results: [])
    main.interview_service = interview.InterviewService()

    response = client.post(
        "/interview/start",
        json={"role": "Python Developer", "interview_type": "technical", "number_of_questions": 2},
    )
    assert response.status_code == 200
    session_id = response.json()["session_id"]
    assert response.json()["provider"] == "local"

    question_response = client.post("/interview/question", json={"session_id": session_id})
    assert question_response.status_code == 200
    question = question_response.json()["question"]

    evaluation_response = client.post(
        "/interview/evaluate",
        json={"session_id": session_id, "question": question, "answer": "I used tests because they catch regressions."},
    )
    assert evaluation_response.status_code == 200
    assert 1 <= evaluation_response.json()["score"] <= 10
    assert "strengths" in evaluation_response.json()


def test_interview_validation_and_missing_session():
    invalid = client.post("/interview/start", json={"role": "", "interview_type": "technical"})
    assert invalid.status_code == 422

    missing = client.post("/interview/question", json={"session_id": "missing"})
    assert missing.status_code == 404
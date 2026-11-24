import pytest

import main
from database.session_store import SQLiteSessionStore
from services.ai_provider import EvaluationResult
from services.interview import InterviewService


class APIProvider:
    name = "test"

    def generate_question(
        self,
        role,
        interview_type,
        difficulty,
        history,
        context,
        context_mode="general",
    ):
        return f"{difficulty} question {len(history) + 1}"

    def evaluate_answer(
        self,
        role,
        question,
        answer,
        context,
        difficulty,
        context_mode="general",
    ):
        return EvaluationResult(
            technical_correctness=8,
            relevance=8,
            completeness=8,
            clarity=8,
            strengths=["Clear reasoning"],
            weaknesses=["Add more detail"],
            improvement_suggestions=["Use a concrete example"],
            follow_up_required=False,
            next_difficulty="hard",
        )


def test_http_flow_persists_session_and_returns_completed_report(
    tmp_path, monkeypatch, authenticated_client
):
    monkeypatch.setattr(
        "services.interview.retrieve_context",
        lambda *args, **kwargs: [],
    )
    main.interview_service = InterviewService(
        store=SQLiteSessionStore(tmp_path / "api.sqlite3"),
        provider=APIProvider(),
    )
    started = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Engineer",
            "interview_type": "technical",
            "difficulty": "medium",
            "number_of_questions": 1,
        },
    )
    assert started.status_code == 200
    session_id = started.json()["session_id"]

    question_response = authenticated_client.post(
        "/interview/question",
        json={"session_id": session_id},
    )
    question = question_response.json()["question"]
    assert authenticated_client.get(
        f"/interview/{session_id}/report"
    ).status_code == 409
    evaluated = authenticated_client.post(
        "/interview/evaluate",
        json={
            "session_id": session_id,
            "question": question,
            "answer": "I measured the result.",
        },
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["next_question_available"] is False

    status = authenticated_client.get(f"/interview/{session_id}")
    report = authenticated_client.get(f"/interview/{session_id}/report")
    history = authenticated_client.get("/interviews")

    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["active_question"] is None
    assert report.status_code == 200
    assert report.json()["overall_score"] == 8
    assert report.json()["question_results"][0]["question"] == question
    assert history.status_code == 200
    assert history.json()["sessions"][0]["session_id"] == session_id


@pytest.mark.parametrize(
    ("document_type", "document_id", "filename"),
    [
        ("resume", "api-resume", "resume.pdf"),
        ("study_material", "api-study", "notes.pdf"),
    ],
)
def test_http_flow_returns_and_persists_document_evidence(
    monkeypatch,
    authenticated_client,
    document_type,
    document_id,
    filename,
):
    user_id = authenticated_client.get("/auth/me").json()["user"]["user_id"]
    main.document_store.register(
        document_id,
        user_id,
        filename,
        document_type=document_type,
    )
    evidence = {
        "document_id": document_id,
        "filename": filename,
        "document_type": document_type,
        "page_number": 2,
        "chunk_id": "api-evidence-chunk",
        "text": "Deterministic evidence for this interview.",
        "distance": 0.1,
    }
    retrieval_calls = []

    def fake_retrieve_context(query, n_results, **filters):
        retrieval_calls.append((query, filters))
        return [evidence]

    monkeypatch.setattr(
        "services.interview.retrieve_context",
        fake_retrieve_context,
    )
    main.interview_service = InterviewService(
        store=SQLiteSessionStore(main.document_store.database_path),
        provider=APIProvider(),
    )

    started = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Engineer",
            "interview_type": "technical",
            "number_of_questions": 1,
            "document_id": document_id,
        },
    )
    assert started.status_code == 200
    session_id = started.json()["session_id"]

    question_response = authenticated_client.post(
        "/interview/question",
        json={"session_id": session_id},
    )
    assert question_response.status_code == 200
    question = question_response.json()
    assert question["evidence"] == [evidence]

    evaluated = authenticated_client.post(
        "/interview/evaluate",
        json={
            "session_id": session_id,
            "question": question["question"],
            "answer": "A deterministic answer.",
        },
    )
    assert evaluated.status_code == 200
    assert evaluated.json()["evidence"] == [evidence]

    report = authenticated_client.get(f"/interview/{session_id}/report")
    assert report.status_code == 200
    question_result = report.json()["question_results"][0]
    assert question_result["question_evidence"] == [evidence]
    assert question_result["evaluation_evidence"] == [evidence]
    assert len(retrieval_calls) == 2
    assert all(
        filters
        == {
            "document_id": document_id,
            "user_id": user_id,
            "document_type": document_type,
        }
        for _, filters in retrieval_calls
    )

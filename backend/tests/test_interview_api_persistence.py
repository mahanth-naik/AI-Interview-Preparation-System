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

import pytest

from database.session_store import SQLiteSessionStore
from services.ai_provider import EvaluationResult
from services.interview import InterviewService


class FakeProvider:
    name = "fake"

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
            relevance=9,
            completeness=7,
            clarity=8,
            strengths=["Clear explanation"],
            weaknesses=["Missing a metric"],
            improvement_suggestions=["Add a measured result"],
            follow_up_required=False,
            next_difficulty="hard",
        )


def test_session_and_active_question_survive_service_restart(tmp_path, monkeypatch):
    monkeypatch.setattr("services.interview.retrieve_context", lambda *args, **kwargs: [])
    database_path = tmp_path / "interviews.sqlite3"
    provider = FakeProvider()
    service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    )
    session = service.start("Engineer", "technical", "medium", 2, "resume-123")

    first_question = service.next_question(session.session_id)
    resumed_service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    )

    assert resumed_service.get(session.session_id).active_question == first_question["question"]
    assert resumed_service.next_question(session.session_id) == first_question

    evaluation = resumed_service.evaluate(
        session.session_id,
        first_question["question"],
        "I measured the result and explained the trade-off.",
    )
    assert evaluation["next_question_available"] is True

    second_question = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    ).next_question(session.session_id)

    assert second_question["question"] == "hard question 2"
    restored = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    ).get(session.session_id)
    assert restored.document_id == "resume-123"
    assert restored.current_question == 1
    assert restored.current_difficulty == "hard"
    assert restored.answers[0]["question"] == first_question["question"]
    assert restored.active_question == second_question["question"]


def test_completed_session_produces_persisted_final_report(tmp_path, monkeypatch):
    monkeypatch.setattr("services.interview.retrieve_context", lambda *args, **kwargs: [])
    database_path = tmp_path / "reports.sqlite3"
    service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=FakeProvider(),
    )
    session = service.start("Engineer", "technical", "medium", 1)

    question = service.next_question(session.session_id)["question"]
    with pytest.raises(ValueError, match="after the interview is completed"):
        service.get_report(session.session_id)

    service.evaluate(session.session_id, question, "A detailed answer.")
    resumed = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=FakeProvider(),
    )
    report = resumed.get_report(session.session_id)

    assert report["status"] == "completed"
    assert report["question_count"] == 1
    assert report["overall_score"] == 8
    assert report["question_results"][0]["question"] == question
    assert report["strengths"] == ["Clear explanation"]
    assert report["weaknesses"] == ["Missing a metric"]
    assert report["recommendations"] == ["Add a measured result"]
    assert resumed.list_sessions()[0]["session_id"] == session.session_id

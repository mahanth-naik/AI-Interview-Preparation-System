import main
from database import vector_store
from services import interview
from services.ai_provider import LocalInterviewProvider
from services.chunking import chunk_text
from services import rag


def test_chunking_has_expected_overlap_and_validation():
    assert chunk_text("abcdefghij", chunk_size=6, overlap=2) == [
        "abcdef",
        "efghij",
        "ij",
    ]

    try:
        chunk_text("text", chunk_size=2, overlap=2)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid overlap should raise ValueError")


def test_retrieval_returns_text_metadata_and_distance(monkeypatch):
    captured = {}

    def fake_search_chunks(query, n_results, **filters):
        captured.update(query=query, n_results=n_results, filters=filters)
        return {
            "documents": [["FastAPI experience"]],
            "metadatas": [[{"filename": "resume.pdf", "chunk_index": 0}]],
            "distances": [[0.2]],
        }

    monkeypatch.setattr(
        rag,
        "search_chunks",
        fake_search_chunks,
    )

    results = rag.retrieve_context("FastAPI", 1, user_id="user-1")

    assert results == [
        {
            "text": "FastAPI experience",
            "metadata": {
                "filename": "resume.pdf",
                "chunk_index": 0,
            },
            "distance": 0.2,
        }
    ]
    assert captured == {
        "query": "FastAPI",
        "n_results": 1,
        "filters": {"user_id": "user-1"},
    }


def test_empty_vector_collection_returns_empty_results(monkeypatch):
    class EmptyCollection:
        def count(self):
            return 0

    monkeypatch.setattr(
        vector_store,
        "collection",
        EmptyCollection(),
    )

    assert vector_store.search_chunks("anything") == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }


def test_upload_rejects_non_pdf(authenticated_client):
    response = authenticated_client.post(
        "/documents/upload",
        files={
            "file": (
                "notes.txt",
                b"not a PDF",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_interview_session_and_evaluation(monkeypatch, authenticated_client):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results, **kwargs: [],
    )

    main.interview_service = interview.InterviewService()
    main.interview_service.provider = LocalInterviewProvider()

    response = authenticated_client.post(
        "/interview/start",
        json={
            "role": "Python Developer",
            "interview_type": "technical",
            "number_of_questions": 2,
        },
    )

    assert response.status_code == 200

    session_id = response.json()["session_id"]

    assert response.json()["provider"] == "local"

    question_response = authenticated_client.post(
        "/interview/question",
        json={
            "session_id": session_id,
        },
    )

    assert question_response.status_code == 200

    question = question_response.json()["question"]

    evaluation_response = authenticated_client.post(
        "/interview/evaluate",
        json={
            "session_id": session_id,
            "question": question,
            "answer": "I used tests because they catch regressions.",
        },
    )

    assert evaluation_response.status_code == 200
    assert 1 <= evaluation_response.json()["score"] <= 10
    assert "strengths" in evaluation_response.json()


def test_interview_validation_and_missing_session(authenticated_client):
    invalid = authenticated_client.post(
        "/interview/start",
        json={
            "role": "",
            "interview_type": "technical",
        },
    )

    assert invalid.status_code == 422

    missing = authenticated_client.post(
        "/interview/question",
        json={
            "session_id": "missing",
        },
    )

    assert missing.status_code == 404


def test_interview_adapts_difficulty_and_generates_next_question(
    monkeypatch,
):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results: [],
    )

    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.generated_difficulties = []

        def generate_question(
            self,
            role,
            interview_type,
            difficulty,
            history,
            context,
            context_mode="general",
        ):
            self.generated_difficulties.append(difficulty)
            return f"{difficulty} question"

        def evaluate_answer(
            self,
            role,
            question,
            answer,
            context,
            difficulty,
            context_mode="general",
        ):
            from services.ai_provider import EvaluationResult

            return EvaluationResult(
                technical_correctness=8,
                relevance=8,
                completeness=7,
                clarity=8,
                strengths=["Correct concept"],
                weaknesses=["Could explain more"],
                improvement_suggestions=["Add an example"],
                follow_up_required=False,
                next_difficulty="hard",
            )

    service = interview.InterviewService()

    provider = FakeProvider()
    service.provider = provider

    session = service.start(
        role="Python Developer",
        interview_type="technical",
        difficulty="medium",
        number=2,
    )

    first_question = service.next_question(
        session.session_id,
    )

    assert first_question["question"] == "medium question"
    assert first_question["difficulty"] == "medium"

    evaluation = service.evaluate(
        session.session_id,
        first_question["question"],
        "FastAPI uses dependency injection.",
    )

    assert evaluation["next_difficulty"] == "hard"

    second_question = service.next_question(
        session.session_id,
    )

    assert second_question["question"] == "hard question"
    assert second_question["difficulty"] == "hard"

    assert provider.generated_difficulties == [
        "medium",
        "hard",
    ]


def test_evaluation_rejects_wrong_question(monkeypatch):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results: [],
    )

    service = interview.InterviewService()
    service.provider = LocalInterviewProvider()

    session = service.start(
        "Python Developer", "technical", "medium", 2
    )

    service.next_question(session.session_id)

    try:
        service.evaluate(
            session.session_id,
            "Some other question",
            "My answer",
        )
    except ValueError as error:
        assert "Invalid question" in str(error)
    else:
        raise AssertionError("Wrong question should be rejected")

    assert session.current_question == 0
    assert len(session.answers) == 0
    assert len(session.evaluations) == 0


def test_evaluation_rejects_duplicate_submission(monkeypatch):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results: [],
    )

    service = interview.InterviewService()
    service.provider = LocalInterviewProvider()

    session = service.start(
        "Python Developer", "technical", "medium", 2
    )

    question = service.next_question(session.session_id)["question"]

    service.evaluate(
        session.session_id,
        question,
        "First answer",
    )

    try:
        service.evaluate(
            session.session_id,
            question,
            "Duplicate answer",
        )
    except ValueError as error:
        assert "No active question" in str(error)
    else:
        raise AssertionError("Duplicate submission should be rejected")

    assert session.current_question == 1
    assert len(session.answers) == 1
    assert len(session.evaluations) == 1


def test_evaluation_rejects_when_no_question_was_generated():
    service = interview.InterviewService()
    service.provider = LocalInterviewProvider()

    session = service.start(
        "Python Developer", "technical", "medium", 2
    )

    try:
        service.evaluate(
            session.session_id,
            "Any question",
            "My answer",
        )
    except ValueError as error:
        assert "No active question" in str(error)
    else:
        raise AssertionError("Evaluation without a question should fail")

    assert session.current_question == 0
    assert len(session.answers) == 0


def test_evaluation_rejects_empty_answer(monkeypatch):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results: [],
    )

    service = interview.InterviewService()
    service.provider = LocalInterviewProvider()

    session = service.start(
        "Python Developer", "technical", "medium", 2
    )

    question = service.next_question(session.session_id)["question"]

    try:
        service.evaluate(
            session.session_id,
            question,
            "   ",
        )
    except ValueError as error:
        assert "Answer cannot be empty" in str(error)
    else:
        raise AssertionError("Empty answer should be rejected")

    assert session.current_question == 0
    assert len(session.answers) == 0


def test_evaluation_rejects_after_interview_completion(monkeypatch):
    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda query, n_results: [],
    )

    service = interview.InterviewService()
    service.provider = LocalInterviewProvider()

    session = service.start(
        "Python Developer", "technical", "medium", 1
    )

    question = service.next_question(session.session_id)["question"]

    service.evaluate(
        session.session_id,
        question,
        "My answer",
    )

    assert session.status == "completed"

    try:
        service.evaluate(
            session.session_id,
            question,
            "Another answer",
        )
    except ValueError as error:
        assert "not active" in str(error)
    else:
        raise AssertionError("Completed interview should reject evaluation")

    assert session.current_question == 1
    assert len(session.answers) == 1
    assert len(session.evaluations) == 1


def test_local_provider_generates_different_questions():
    from services.ai_provider import LocalInterviewProvider

    provider = LocalInterviewProvider()

    first = provider.generate_question(
        "Java Developer", "Technical", "medium", [], ""
    )

    history = [{"question": first, "answer": "My answer"}]

    second = provider.generate_question(
        "Java Developer", "Technical", "medium", history, ""
    )

    assert first != second


def test_local_provider_changes_question_by_difficulty():
    from services.ai_provider import LocalInterviewProvider

    provider = LocalInterviewProvider()

    easy = provider.generate_question(
        "Java Developer", "Technical", "easy", [], ""
    )

    hard = provider.generate_question(
        "Java Developer", "Technical", "hard", [], ""
    )

    assert easy != hard


def test_local_provider_avoids_multiple_previous_questions():
    from services.ai_provider import LocalInterviewProvider

    provider = LocalInterviewProvider()

    questions = provider.generate_questions(
        "Java Developer", "Technical", "medium", 5, ""
    )

    history = [
        {"question": question, "answer": "Previous answer"}
        for question in questions[:3]
    ]

    next_question = provider.generate_question(
        "Java Developer", "Technical", "medium", history, ""
    )

    assert next_question not in questions[:3]

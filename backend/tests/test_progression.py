import pytest

from database.session_store import SQLiteSessionStore
from services.ai_provider import (
    EvaluationResult,
    LLMProviderError,
    LocalInterviewProvider,
)
from services.interview import InterviewService


def _evaluation(
    score=7,
    *,
    follow_up=False,
    weaknesses=None,
    improvements=None,
    next_difficulty="hard",
):
    return EvaluationResult(
        technical_correctness=score,
        relevance=score,
        completeness=score,
        clarity=score,
        strengths=["Clear response"],
        weaknesses=weaknesses or [],
        improvement_suggestions=improvements or [],
        follow_up_required=follow_up,
        next_difficulty=next_difficulty,
    )


class ScriptedProvider:
    name = "test"

    def __init__(self, questions, evaluations=None):
        self.questions = list(questions)
        self.evaluations = list(evaluations or [_evaluation()])
        self.generated_history = []

    def generate_question(
        self,
        role,
        interview_type,
        difficulty,
        history,
        context,
        context_mode="general",
    ):
        self.generated_history.append(history)
        if self.questions:
            return self.questions.pop(0)
        return "What is caching?"

    def evaluate_answer(
        self,
        role,
        question,
        answer,
        context,
        difficulty,
        context_mode="general",
    ):
        if len(self.evaluations) > 1:
            return self.evaluations.pop(0)
        return self.evaluations[0]


def _service(tmp_path, provider, count=3):
    service = InterviewService(
        store=SQLiteSessionStore(tmp_path / "progression.sqlite3"),
        provider=provider,
    )
    session = service.start("Engineer", "technical", "medium", count)
    return service, session


@pytest.mark.parametrize(
    ("provider_name", "first_question", "duplicate"),
    [
        ("local", "What is caching?", "What is caching?"),
        (
            "local",
            "Explain how caching improves performance in web applications.",
            "How does caching improve web application performance?",
        ),
        ("gemini", "What is caching?", "What is caching?"),
    ],
)
def test_duplicate_questions_retry_across_provider_outputs(
    tmp_path,
    provider_name,
    first_question,
    duplicate,
):
    provider = ScriptedProvider(
        [first_question, duplicate, "How would you measure cache hit rate?"]
    )
    provider.name = provider_name
    service, session = _service(tmp_path, provider)

    first = service.next_question(session.session_id)
    service.evaluate(session.session_id, first["question"], "A useful answer.")
    second = service.next_question(session.session_id)

    assert second["question"] == "How would you measure cache hit rate?"
    assert second["question_number"] == 2
    assert len(provider.generated_history) == 3
    assert duplicate in provider.generated_history[-1][-1][
        "_generation_metadata"
    ]["rejected_questions"]


def test_duplicate_retries_are_bounded_and_do_not_activate_stale_question(
    tmp_path,
):
    provider = ScriptedProvider(
        ["What is caching?"] * 4,
        [_evaluation(follow_up=False)],
    )
    service, session = _service(tmp_path, provider)
    first = service.next_question(session.session_id)
    service.evaluate(session.session_id, first["question"], "A useful answer.")

    with pytest.raises(LLMProviderError, match="after 3 attempts"):
        service.next_question(session.session_id)

    restored = service.get(session.session_id)
    assert len(provider.generated_history) == 4
    assert restored.questions == ["What is caching?"]
    assert restored.current_question == 1
    assert restored.active_question is None


def test_stored_duplicate_question_is_replaced_before_activation(tmp_path):
    provider = ScriptedProvider(
        [
            "What is caching?",
            "How would you measure cache performance?",
        ]
    )
    service, session = _service(tmp_path, provider)
    first = service.next_question(session.session_id)
    service.evaluate(session.session_id, first["question"], "A useful answer.")

    session.questions.append(first["question"])
    session.question_evidence.append([{"chunk_id": "stale-evidence"}])
    session.question_followups.append(False)
    session.active_question = first["question"]
    service.store.save(session)

    replacement = service.next_question(session.session_id)

    assert replacement["question"] == "How would you measure cache performance?"
    assert replacement["question_number"] == 2
    assert session.active_question == replacement["question"]
    assert replacement["evidence"] == []
    with pytest.raises(ValueError, match="Invalid question"):
        service.evaluate(
            session.session_id,
            first["question"],
            "A stale answer.",
        )


def test_local_template_exhaustion_fails_without_repeating_a_question(
    tmp_path,
):
    provider = LocalInterviewProvider()
    service, session = _service(tmp_path, provider, count=7)
    previous_questions = provider.generate_questions(
        "Engineer",
        "technical",
        "medium",
        5,
        "",
    )
    session.questions = previous_questions
    session.question_followups = [False] * len(previous_questions)
    session.question_evidence = [[] for _ in previous_questions]
    session.answers = [
        {"question": question, "answer": "Previously answered."}
        for question in previous_questions
    ]
    session.evaluations = [
        {"follow_up_required": False} for _ in previous_questions
    ]
    session.current_question = len(previous_questions)
    service.store.save(session)

    sixth = service.next_question(session.session_id)
    assert sixth["question"] not in previous_questions
    long_answer = " ".join(["reasoning"] * 100)
    service.evaluate(session.session_id, sixth["question"], long_answer)

    with pytest.raises(LLMProviderError, match="after 3 attempts"):
        service.next_question(session.session_id)


def test_follow_up_uses_previous_answer_and_counts_toward_limit(
    tmp_path,
):
    provider = ScriptedProvider(
        [
            "How did you measure latency improvements?",
            "You mentioned latency measurements. What trade-off informed "
            "your chosen measurement approach?",
        ],
        [
            _evaluation(
                6,
                follow_up=True,
                weaknesses=["Explain the measurement choice."],
                improvements=["Explain why you chose that measurement."],
            ),
            _evaluation(8, follow_up=True),
        ],
    )
    service, session = _service(tmp_path, provider, count=2)

    first = service.next_question(session.session_id)
    assert first["is_follow_up"] is False
    first_evaluation = service.evaluate(
        session.session_id,
        first["question"],
        "I measured latency and compared the results.",
    )
    assert first_evaluation["follow_up_required"] is True
    assert first_evaluation["next_question_available"] is True

    follow_up = service.next_question(session.session_id)
    assert follow_up["question_number"] == 2
    assert follow_up["is_follow_up"] is True
    assert follow_up["question"] != first["question"]
    generation_metadata = provider.generated_history[-1][-1][
        "_generation_metadata"
    ]
    assert generation_metadata["follow_up_requested"] is True
    assert generation_metadata["follow_up_reason"]
    assert provider.generated_history[-1][-1]["answer"] == (
        "I measured latency and compared the results."
    )

    final_evaluation = service.evaluate(
        session.session_id,
        follow_up["question"],
        "The traffic pattern influenced that trade-off.",
    )
    assert final_evaluation["follow_up_required"] is False
    assert final_evaluation["next_question_available"] is False
    assert session.status == "completed"
    assert session.current_question == session.target_question_count == 2

    report = service.get_report(session.session_id)
    assert [item["is_follow_up"] for item in report["question_results"]] == [
        False,
        True,
    ]
    assert report["question_count"] == 2


def test_follow_up_progression_survives_service_restart(tmp_path):
    database_path = tmp_path / "restart.sqlite3"
    provider = ScriptedProvider(
        [
            "How did you validate the result?",
            "You said tests caught the regression. How did you choose the cases?",
        ],
        [
            _evaluation(follow_up=True),
            _evaluation(),
        ],
    )
    service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    )
    session = service.start("Engineer", "technical", "medium", 2)
    first = service.next_question(session.session_id)
    service.evaluate(
        session.session_id,
        first["question"],
        "Tests caught the regression.",
    )

    resumed_service = InterviewService(
        store=SQLiteSessionStore(database_path),
        provider=provider,
    )
    follow_up = resumed_service.next_question(session.session_id)

    assert follow_up["is_follow_up"] is True
    assert follow_up["question_number"] == 2
    restored = resumed_service.get(session.session_id)
    assert restored.question_followups == [False, True]
    assert restored.answers[0]["question"] == first["question"]


def test_question_limit_suppresses_unavailable_follow_up(tmp_path):
    provider = ScriptedProvider(
        ["Only question"],
        [_evaluation(follow_up=True)],
    )
    service, session = _service(tmp_path, provider, count=1)
    question = service.next_question(session.session_id)["question"]

    result = service.evaluate(session.session_id, question, "An answer.")

    assert result["follow_up_required"] is False
    assert result["next_question_available"] is False
    assert service.next_question(session.session_id)["status"] == "completed"
    assert len(provider.generated_history) == 1


@pytest.mark.parametrize(
    ("current", "score", "weaknesses", "improvements", "expected"),
    [
        ("medium", 9, [], [], "hard"),
        ("medium", 3, [], [], "easy"),
        ("medium", 6, ["Needs evidence"], [], "easy"),
        ("medium", 7, ["Needs evidence"], [], "medium"),
        ("medium", 9, ["Minor omission"], [], "medium"),
        ("easy", 3, [], [], "easy"),
        ("hard", 9, [], [], "hard"),
    ],
)
def test_difficulty_policy_uses_score_feedback_and_boundaries(
    tmp_path,
    current,
    score,
    weaknesses,
    improvements,
    expected,
):
    provider = ScriptedProvider(
        ["A distinct question"],
        [
            _evaluation(
                score,
                weaknesses=weaknesses,
                improvements=improvements,
                next_difficulty="hard",
            )
        ],
    )
    service, session = _service(tmp_path, provider, count=2)
    session.current_difficulty = current
    session.difficulty = current
    question = service.next_question(session.session_id)["question"]

    evaluation = service.evaluate(session.session_id, question, "An answer.")

    assert evaluation["next_difficulty"] == expected
    assert session.current_difficulty == expected

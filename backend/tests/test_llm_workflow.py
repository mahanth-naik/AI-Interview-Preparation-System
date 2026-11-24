import json

import pytest

from services.ai_provider import (
    EvaluationResult,
    GeminiProvider,
    LLMConfigurationError,
    LLMProviderError,
    LocalInterviewProvider,
    OpenAIProvider,
)
from services.autogen_orchestration import AutoGenProvider
from services import interview
from services.rag import build_context
from database.session_store import SQLiteSessionStore


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(self.content)


class FakeClient:
    def __init__(self, content):
        self.completions = FakeCompletions(content)
        self.chat = type(
            "Chat",
            (),
            {"completions": self.completions},
        )()


def valid_evaluation():
    return {
        "technical_correctness": 8,
        "relevance": 9,
        "completeness": 7,
        "clarity": 8,
        "strengths": ["Explains the trade-off"],
        "weaknesses": ["Needs a concrete metric"],
        "improvement_suggestions": ["Add a measured result"],
        "follow_up_required": True,
        "next_difficulty": "hard",
    }


def test_openai_provider_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError):
        OpenAIProvider()


def test_openai_provider_generates_question_with_rag_context():
    provider = OpenAIProvider(api_key="test-key", client=FakeClient("Ask about caching."))
    question = provider.generate_question("Python Developer", "technical", "medium", [], "[Context 1]\nFastAPI")
    assert question == "Ask about caching."


def test_openai_provider_validates_structured_evaluation():
    import json

    provider = OpenAIProvider(
        api_key="test-key", client=FakeClient(json.dumps(valid_evaluation()))
    )
    result = provider.evaluate_answer("Python Developer", "What is FastAPI?", "A framework", "FastAPI", "medium")
    assert isinstance(result, EvaluationResult)
    assert result.score == 8
    assert result.next_difficulty == "hard"


def test_openai_provider_rejects_malformed_evaluation():
    provider = OpenAIProvider(api_key="test-key", client=FakeClient("not-json"))
    with pytest.raises(LLMProviderError):
        provider.evaluate_answer("Python Developer", "Question", "Answer", "Context", "medium")


def test_gemini_provider_surfaces_rate_limit_errors():
    class RateLimitError(Exception):
        status_code = 429

    client = FakeClient("")

    def rate_limited(**kwargs):
        raise RateLimitError("rate limit")

    client.completions.create = rate_limited
    provider = GeminiProvider(api_key="test-key", client=client)

    with pytest.raises(LLMProviderError) as error:
        provider.generate_question(
            "Engineer",
            "technical",
            "medium",
            [],
            "Synthetic context.",
        )

    assert getattr(error.value.__cause__, "status_code", None) == 429


@pytest.mark.parametrize("provider_type", [OpenAIProvider, GeminiProvider])
def test_llm_provider_prompts_distinguish_resume_and_study_modes(provider_type):
    resume_client = FakeClient("Ask about the project.")
    resume_provider = provider_type(api_key="test-key", client=resume_client)
    resume_context = build_context(
        ["Built a Python reporting service."],
        "resume",
    )
    resume_provider.generate_question(
        "Python Developer",
        "technical",
        "medium",
        [],
        resume_context,
        context_mode="resume",
    )
    resume_messages = resume_client.completions.requests[-1]["messages"]

    assert "explicitly present" in resume_messages[0]["content"]
    assert "Resume evidence" in resume_messages[1]["content"]
    assert "Study-material evidence" not in resume_messages[1]["content"]
    resume_client.completions.content = json.dumps(valid_evaluation())

    study_client = FakeClient(json.dumps(valid_evaluation()))
    study_provider = provider_type(api_key="test-key", client=study_client)
    study_context = build_context(
        ["TCP uses a congestion window to control transmission."],
        "study_material",
    )
    study_provider.generate_question(
        "Network Engineer",
        "technical",
        "medium",
        [],
        study_context,
        context_mode="study_material",
    )
    study_question_messages = study_client.completions.requests[-1]["messages"]
    assert "not candidate experience" in study_question_messages[0]["content"]
    assert "Study-material evidence" in study_question_messages[1]["content"]

    study_provider.evaluate_answer(
        "Network Engineer",
        "What does the congestion window control?",
        "It limits in-flight data.",
        study_context,
        "medium",
        context_mode="study_material",
    )
    study_evaluation_messages = study_client.completions.requests[-1]["messages"]
    assert "unsupported claims" in study_evaluation_messages[0]["content"]
    assert "Study-material evidence" in study_evaluation_messages[1]["content"]

    resume_provider.evaluate_answer(
        "Python Developer",
        "What is a transaction?",
        "A unit of work.",
        resume_context,
        "medium",
        context_mode="resume",
    )
    resume_evaluation_messages = resume_client.completions.requests[-1]["messages"]
    assert "every correct technical answer" in resume_evaluation_messages[0]["content"]


def test_local_provider_questions_quote_the_selected_source():
    provider = LocalInterviewProvider()
    resume_context = build_context(
        ["Developed a Python reporting service for finance teams."],
        "resume",
    )
    study_context = build_context(
        ["TCP uses a congestion window to control transmission."],
        "study_material",
    )

    resume_question = provider.generate_question(
        "Python Developer",
        "technical",
        "medium",
        [],
        resume_context,
        context_mode="resume",
    )
    study_question = provider.generate_question(
        "Network Engineer",
        "technical",
        "medium",
        [],
        study_context,
        context_mode="study_material",
    )

    assert "Developed a Python reporting service" in resume_question
    assert "TCP uses a congestion window" in study_question
    assert "candidate experience" not in study_question.lower()


def test_llm_question_prompt_only_requests_explicit_follow_ups():
    history = [
        {
            "question": "How did you measure latency?",
            "answer": "I compared the p95 measurements.",
            "_generation_metadata": {
                "follow_up_requested": True,
                "follow_up_reason": "Explain the measurement choice.",
                "rejected_questions": ["Repeat the original question?"],
            },
        }
    ]
    client = FakeClient("What trade-off led you to that measurement?")
    provider = OpenAIProvider(api_key="test-key", client=client)
    provider.generate_question(
        "Engineer",
        "technical",
        "medium",
        history,
        "",
    )

    prompt = client.completions.requests[-1]["messages"][1]["content"]
    assert "distinct follow-up" in prompt
    assert "Explain the measurement choice." in prompt
    assert "Repeat the original question?" in prompt


def test_local_study_evaluation_does_not_reward_unmatched_claims():
    provider = LocalInterviewProvider()
    context = build_context(
        ["TCP uses a congestion window to control transmission."],
        "study_material",
    )

    result = provider.evaluate_answer(
        "Network Engineer",
        "What controls transmission?",
        "Quantum teleportation controls photosynthesis.",
        context,
        "medium",
        context_mode="study_material",
    )

    assert result.technical_correctness == 1
    assert result.score <= 5
    assert "term overlap only" in result.improvement_suggestions[0]


def test_autogen_provider_includes_mode_specific_source_instructions():
    class FakeOrchestrator:
        def __init__(self):
            self.tasks = []

        def run(self, role, task):
            self.tasks.append((role, task))
            if role == "evaluator":
                return json.dumps(valid_evaluation())
            return "Explain the concept in the passage."

    orchestrator = FakeOrchestrator()
    provider = AutoGenProvider(
        api_key="test-key",
        orchestrator=orchestrator,
    )
    context = build_context(
        ["TCP uses a congestion window to control transmission."],
        "study_material",
    )
    provider.generate_question(
        "Network Engineer",
        "technical",
        "medium",
        [],
        context,
        context_mode="study_material",
    )
    provider.generate_question(
        "Network Engineer",
        "technical",
        "medium",
        [
            {
                "question": "What does the congestion window control?",
                "answer": "It limits unacknowledged data.",
                "_generation_metadata": {
                    "follow_up_requested": True,
                    "follow_up_reason": "Clarify the reasoning.",
                    "rejected_questions": [],
                },
            }
        ],
        context,
        context_mode="study_material",
    )
    provider.evaluate_answer(
        "Network Engineer",
        "What does the congestion window control?",
        "It limits unacknowledged data.",
        context,
        "medium",
        context_mode="study_material",
    )

    assert "not candidate experience" in orchestrator.tasks[0][1]
    assert "Study-material evidence" in orchestrator.tasks[0][1]
    assert "distinct follow-up" in orchestrator.tasks[1][1]
    assert "unsupported claims" in orchestrator.tasks[2][1]
    assert "Study-material evidence" in orchestrator.tasks[2][1]


def test_interview_service_retrieves_mode_specific_context_for_question_and_evaluation(
    tmp_path,
    monkeypatch,
):
    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.modes = []

        def generate_question(
            self,
            role,
            interview_type,
            difficulty,
            history,
            context,
            context_mode="general",
        ):
            self.modes.append(context_mode)
            assert context_mode == "resume"
            assert "Resume evidence" in context
            return "What did you build?"

        def evaluate_answer(
            self,
            role,
            question,
            answer,
            context,
            difficulty,
            context_mode="general",
        ):
            self.modes.append(context_mode)
            assert context_mode == "resume"
            assert "Resume evidence" in context
            return EvaluationResult(**valid_evaluation())

    retrieval_calls = []

    def fake_retrieve_context(query, n_results, **filters):
        retrieval_calls.append((query, filters))
        return [{
            "document_id": "document-1",
            "filename": "resume.pdf",
            "document_type": "resume",
            "page_number": 2,
            "chunk_id": "chunk-000001",
            "text": "Built a Python reporting service.",
            "distance": 0.1,
        }]

    monkeypatch.setattr(interview, "retrieve_context", fake_retrieve_context)
    provider = FakeProvider()
    service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / "resume.sqlite3"),
        provider=provider,
    )
    service.document_store.register(
        "document-1",
        "user-1",
        "resume.pdf",
        document_type="resume",
    )
    session = service.start(
        "Python Developer",
        "technical",
        "medium",
        2,
        document_id="document-1",
        user_id="user-1",
    )
    question = service.next_question(session.session_id)
    result = service.evaluate(
        session.session_id,
        question["question"],
        "A detailed answer",
    )
    assert result["technical_correctness"] == 8
    assert result["next_question_available"] is True
    assert "resume skills technologies projects" in retrieval_calls[0][0]
    assert "resume skills technologies projects" in retrieval_calls[1][0]
    assert "A detailed answer" not in retrieval_calls[1][0]
    assert all(
        filters
        == {
            "document_id": "document-1",
            "user_id": "user-1",
            "document_type": "resume",
        }
        for _, filters in retrieval_calls
    )
    assert provider.modes == ["resume", "resume"]
    assert question["evidence"][0]["filename"] == "resume.pdf"
    assert question["evidence"][0]["page_number"] == 2
    assert result["evidence"][0]["chunk_id"] == "chunk-000001"


def test_study_material_interview_uses_source_only_context_and_question_query(
    tmp_path,
    monkeypatch,
):
    class FakeProvider:
        name = "fake"

        def __init__(self):
            self.question_context = None
            self.evaluation_context = None

        def generate_question(
            self,
            role,
            interview_type,
            difficulty,
            history,
            context,
            context_mode="general",
        ):
            self.question_context = (context_mode, context)
            assert context_mode == "study_material"
            assert "Study-material evidence" in context
            return "What does the congestion window control?"

        def evaluate_answer(
            self,
            role,
            question,
            answer,
            context,
            difficulty,
            context_mode="general",
        ):
            self.evaluation_context = (context_mode, context)
            assert context_mode == "study_material"
            assert "Study-material evidence" in context
            return EvaluationResult(**valid_evaluation())

    retrieval_calls = []

    def fake_retrieve_context(query, n_results, **filters):
        retrieval_calls.append((query, filters))
        return [
            {
                "document_id": "study-document",
                "filename": "networking-notes.pdf",
                "document_type": "study_material",
                "page_number": 6,
                "chunk_id": "chunk-000004",
                "text": (
                    "TCP uses a congestion window to control the amount "
                    "of unacknowledged data."
                ),
                "distance": 0.12,
            }
        ]

    monkeypatch.setattr(interview, "retrieve_context", fake_retrieve_context)
    provider = FakeProvider()
    service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / "study.sqlite3"),
        provider=provider,
    )
    service.document_store.register(
        "study-document",
        "user-1",
        "networking-notes.pdf",
        document_type="study_material",
    )
    session = service.start(
        "Network Engineer",
        "technical",
        "medium",
        1,
        document_id="study-document",
        user_id="user-1",
    )

    question = service.next_question(session.session_id)
    evaluation = service.evaluate(
        session.session_id,
        question["question"],
        "The congestion window limits data in transit.",
    )

    assert "key concepts definitions principles" in retrieval_calls[0][0]
    assert "congestion window" in retrieval_calls[1][0]
    assert "The congestion window limits" not in retrieval_calls[1][0]
    assert all(
        filters
        == {
            "document_id": "study-document",
            "user_id": "user-1",
            "document_type": "study_material",
        }
        for _, filters in retrieval_calls
    )
    assert provider.question_context[0] == "study_material"
    assert "TCP uses a congestion window" in provider.question_context[1]
    assert provider.evaluation_context[0] == "study_material"
    assert "TCP uses a congestion window" in provider.evaluation_context[1]
    assert evaluation["technical_correctness"] == 8
    expected_evidence = [{
        "document_id": "study-document",
        "filename": "networking-notes.pdf",
        "document_type": "study_material",
        "page_number": 6,
        "chunk_id": "chunk-000004",
        "text": (
            "TCP uses a congestion window to control the amount "
            "of unacknowledged data."
        ),
        "distance": 0.12,
    }]
    assert question["evidence"] == expected_evidence
    assert evaluation["evidence"] == expected_evidence

    resumed_service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / "study.sqlite3"),
        provider=provider,
    )
    report = resumed_service.get_report(
        session.session_id,
        user_id="user-1",
    )
    assert report["question_results"][0]["question_evidence"] == expected_evidence
    assert report["question_results"][0]["evaluation_evidence"] == expected_evidence


@pytest.mark.parametrize("document_type", ["resume", "study_material"])
def test_document_interview_does_not_generate_without_retrieved_evidence(
    tmp_path,
    monkeypatch,
    document_type,
):
    class UnexpectedProvider:
        name = "unexpected"

        def generate_question(self, *args, **kwargs):
            pytest.fail("A question must not be generated without evidence")

        def evaluate_answer(self, *args, **kwargs):
            pytest.fail("An answer must not be evaluated without evidence")

    monkeypatch.setattr(
        interview,
        "retrieve_context",
        lambda *args, **kwargs: [],
    )
    service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / f"{document_type}.sqlite3"),
        provider=UnexpectedProvider(),
    )
    service.document_store.register(
        "selected-document",
        "user-1",
        f"{document_type}.pdf",
        document_type=document_type,
    )
    session = service.start(
        "Engineer",
        "technical",
        "medium",
        1,
        document_id="selected-document",
        user_id="user-1",
    )

    with pytest.raises(ValueError, match="No relevant content"):
        service.next_question(session.session_id)

    assert session.questions == []


def test_autogen_provider_uses_interviewer_and_evaluator_roles():
    class FakeOrchestrator:
        def __init__(self):
            self.roles = []

        def run(self, role, task):
            self.roles.append(role)
            if role == "evaluator":
                import json

                return json.dumps(valid_evaluation())
            return "Generated by interviewer"

    orchestrator = FakeOrchestrator()
    provider = AutoGenProvider(api_key="test-key", orchestrator=orchestrator)
    assert provider.generate_question("Engineer", "technical", "medium", [], "context") == "Generated by interviewer"
    evaluation = provider.evaluate_answer("Engineer", "Question", "Answer", "context", "medium")
    assert isinstance(evaluation, EvaluationResult)
    assert orchestrator.roles == ["interviewer", "evaluator"]


def test_autogen_agents_have_distinct_role_prompts():
    from services.autogen_orchestration import AutoGenOrchestrator

    assert AutoGenOrchestrator.INTERVIEWER_SYSTEM_MESSAGE != AutoGenOrchestrator.EVALUATOR_SYSTEM_MESSAGE
    assert "exactly one interview question" in AutoGenOrchestrator.INTERVIEWER_SYSTEM_MESSAGE
    assert "valid JSON" in AutoGenOrchestrator.EVALUATOR_SYSTEM_MESSAGE


def test_autogen_orchestrator_builds_two_distinct_agents(monkeypatch):
    import sys
    import types

    class FakeAssistantAgent:
        def __init__(self, name, model_client, system_message):
            self.name = name
            self.model_client = model_client
            self.system_message = system_message

    class FakeSelectorGroupChat:
        def __init__(self, participants, **kwargs):
            self.participants = participants
            self.kwargs = kwargs

    class FakeModelClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    agents_module = types.ModuleType("autogen_agentchat.agents")
    agents_module.AssistantAgent = FakeAssistantAgent
    conditions_module = types.ModuleType("autogen_agentchat.conditions")
    conditions_module.MaxMessageTermination = lambda count: count
    teams_module = types.ModuleType("autogen_agentchat.teams")
    teams_module.SelectorGroupChat = FakeSelectorGroupChat
    models_module = types.ModuleType("autogen_ext.models.openai")
    models_module.OpenAIChatCompletionClient = FakeModelClient
    monkeypatch.setitem(sys.modules, "autogen_agentchat.agents", agents_module)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.conditions", conditions_module)
    monkeypatch.setitem(sys.modules, "autogen_agentchat.teams", teams_module)
    monkeypatch.setitem(sys.modules, "autogen_ext.models.openai", models_module)

    from services.autogen_orchestration import AutoGenOrchestrator

    orchestrator = AutoGenOrchestrator("test-model", "test-key")
    assert orchestrator.workflow.participants == [
        orchestrator.interviewer_agent,
        orchestrator.evaluator_agent,
    ]
    assert orchestrator.interviewer_agent is not orchestrator.evaluator_agent
    assert orchestrator.interviewer_agent.name == "interviewer_agent"
    assert orchestrator.evaluator_agent.name == "evaluator_agent"


def test_autogen_provider_rejects_malformed_evaluation():
    class FakeOrchestrator:
        def run(self, role, task):
            assert role == "evaluator"
            return "not-json"

    provider = AutoGenProvider(api_key="test-key", orchestrator=FakeOrchestrator())
    with pytest.raises(LLMProviderError, match="malformed evaluation JSON"):
        provider.evaluate_answer("Engineer", "Question", "Answer", "context", "medium")


class FakeWorkflow:
    def __init__(self, interviewer_agent, evaluator_agent):
        self.interviewer_agent = interviewer_agent
        self.evaluator_agent = evaluator_agent
        self.selected_agents = []

    async def run(self, task):
        selected_agent = (
            self.evaluator_agent if "Requested role: evaluator" in task else self.interviewer_agent
        )
        self.selected_agents.append(selected_agent)
        response = (
            "{\"technical_correctness\": 8, \"relevance\": 8, \"completeness\": 8, "
            "\"clarity\": 8, \"strengths\": [], \"weaknesses\": [], "
            "\"improvement_suggestions\": [], \"follow_up_required\": false, "
            "\"next_difficulty\": \"hard\"}"
            if selected_agent is self.evaluator_agent
            else "Ask about the candidate's caching design."
        )
        task_message = type(
            "Message",
            (),
            {"content": task, "source": "user"},
        )()
        agent_message = type(
            "Message",
            (),
            {"content": response, "source": selected_agent.name},
        )()
        return type(
            "Result",
            (),
            {"messages": [task_message, agent_message]},
        )()


class FakeAgent:
    def __init__(self, name):
        self.name = name


def fake_orchestrator():
    from services.autogen_orchestration import AutoGenOrchestrator

    orchestrator = object.__new__(AutoGenOrchestrator)
    orchestrator.model_client = None

    def build_workflow():
        interviewer_agent = FakeAgent("interviewer_agent")
        evaluator_agent = FakeAgent("evaluator_agent")
        workflow = FakeWorkflow(
            interviewer_agent,
            evaluator_agent,
        )
        return interviewer_agent, evaluator_agent, workflow

    orchestrator._build_workflow = build_workflow
    orchestrator._replace_workflow()
    return orchestrator


def test_autogen_orchestrator_routes_roles_and_returns_selected_response():
    import asyncio

    orchestrator = fake_orchestrator()
    interviewer_response = asyncio.run(orchestrator.run_async("interviewer", "Generate a question."))
    evaluator_response = asyncio.run(orchestrator.run_async("evaluator", "Evaluate an answer."))

    assert interviewer_response == "Ask about the candidate's caching design."
    assert "technical_correctness" in evaluator_response
    assert orchestrator.workflow.selected_agents == [
        orchestrator.interviewer_agent,
        orchestrator.evaluator_agent,
    ]


def test_autogen_orchestrator_rebuilds_workflow_after_failure():
    import asyncio

    orchestrator = fake_orchestrator()
    failed_workflow = orchestrator.workflow
    orchestrator._event_loop = asyncio.new_event_loop()

    async def fail_once(task):
        assert "Requested role: interviewer" in task
        raise RuntimeError("synthetic workflow failure")

    failed_workflow.run = fail_once

    try:
        with pytest.raises(
            LLMProviderError,
            match="synthetic workflow failure",
        ):
            orchestrator.run(
                "interviewer",
                "Generate a question.",
            )

        assert orchestrator.workflow is not failed_workflow
        assert orchestrator.run(
            "interviewer",
            "Generate a question after recovery.",
        ) == "Ask about the candidate's caching design."
    finally:
        orchestrator._event_loop.close()


def test_autogen_orchestrator_preserves_cancellation_when_reset_fails():
    import asyncio

    orchestrator = fake_orchestrator()
    cancellation = asyncio.CancelledError("synthetic cancellation")
    reset_error = RuntimeError("synthetic reset failure")

    async def cancel_run(task):
        raise cancellation

    def fail_reset():
        raise reset_error

    orchestrator.workflow.run = cancel_run
    orchestrator._replace_workflow = fail_reset

    with pytest.raises(asyncio.CancelledError) as error:
        asyncio.run(
            orchestrator.run_async(
                "interviewer",
                "Generate a question.",
            )
        )

    assert error.value is cancellation
    assert error.value.__cause__ is reset_error


def test_autogen_orchestrator_preserves_workflow_error_when_reset_fails():
    import asyncio

    orchestrator = fake_orchestrator()
    workflow_error = RuntimeError("synthetic workflow failure")
    reset_error = RuntimeError("synthetic reset failure")

    async def fail_run(task):
        raise workflow_error

    def fail_reset():
        raise reset_error

    orchestrator.workflow.run = fail_run
    orchestrator._replace_workflow = fail_reset

    with pytest.raises(
        LLMProviderError,
        match="synthetic workflow failure.*synthetic reset failure",
    ) as error:
        asyncio.run(
            orchestrator.run_async(
                "interviewer",
                "Generate a question.",
            )
        )

    assert error.value.__cause__ is workflow_error


@pytest.mark.parametrize(
    ("role_name", "expected_source"),
    [
        ("interviewer", "interviewer_agent"),
        ("evaluator", "evaluator_agent"),
    ],
)
def test_autogen_orchestrator_rejects_routing_task_as_response(
    role_name,
    expected_source,
):
    import asyncio

    orchestrator = fake_orchestrator()

    class RoutingTaskOnlyWorkflow:
        async def run(self, task):
            task_message = type(
                "Message",
                (),
                {"content": task, "source": "user"},
            )()
            return type(
                "Result",
                (),
                {"messages": [task_message]},
            )()

    orchestrator.workflow = RoutingTaskOnlyWorkflow()

    with pytest.raises(
        LLMProviderError,
        match=f"no response from {expected_source}",
    ):
        asyncio.run(
            orchestrator.run_async(
                role_name,
                "Perform the requested task.",
            )
        )


def test_autogen_orchestrator_rejects_unsupported_role():
    import asyncio

    with pytest.raises(LLMProviderError, match="Unsupported AutoGen role"):
        asyncio.run(fake_orchestrator().run_async("moderator", "Do something."))


def test_autogen_orchestrator_run_uses_sync_adapter_without_running_loop():
    orchestrator = fake_orchestrator()

    assert orchestrator.run("interviewer", "Generate a question.") == (
        "Ask about the candidate's caching design."
    )


def test_autogen_orchestrator_run_rejects_active_event_loop():
    import asyncio

    async def invoke_run():
        with pytest.raises(LLMProviderError, match="active event loop"):
            fake_orchestrator().run("interviewer", "Generate a question.")

    asyncio.run(invoke_run())


def test_provider_selection_supports_local_openai_and_autogen(monkeypatch):
    from services.ai_provider import get_interview_provider

    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert get_interview_provider().name == "local"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert get_interview_provider().name == "openai"

    import services.autogen_orchestration as autogen_module

    class FakeAutoGenProvider:
        name = "autogen"

        def __init__(self):
            pass

    monkeypatch.setattr(autogen_module, "AutoGenProvider", FakeAutoGenProvider)
    monkeypatch.setenv("LLM_PROVIDER", "autogen")
    assert get_interview_provider().name == "autogen"

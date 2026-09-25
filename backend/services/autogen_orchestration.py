import asyncio
import os
from typing import Any

from services.ai_provider import (
    EvaluationResult,
    LLMConfigurationError,
    LLMProviderError,
    _strip_json_fence,
)


class AutoGenOrchestrator:
    """Routes interview tasks through a two-agent AutoGen workflow."""

    INTERVIEWER_SYSTEM_MESSAGE = """You are InterviewerAgent in an interview workflow.
Generate exactly one interview question. Use the supplied candidate/RAG context,
role, interview type, difficulty, and interview history. Ask an appropriate
follow-up when history or the previous answer requires it. Never invent candidate
facts. Return only the question, with no explanation or numbering."""

    EVALUATOR_SYSTEM_MESSAGE = """You are EvaluatorAgent in an interview workflow.
Evaluate the candidate answer using the supplied question, candidate/RAG context,
role, and difficulty. Return only valid JSON compatible with EvaluationResult,
including technical_correctness, relevance, completeness, clarity, strengths,
weaknesses, improvement_suggestions, follow_up_required, and next_difficulty.
Use integer scores from 1 to 10, and next_difficulty must be easy, medium, or hard.
Never invent candidate facts."""

    def __init__(self, model: str, api_key: str):
        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import SelectorGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError as exc:
            raise LLMConfigurationError("autogen-agentchat and autogen-ext[openai] are required for the AutoGen provider") from exc

        self.model_client = OpenAIChatCompletionClient(model=model, api_key=api_key)
        self.interviewer_agent = AssistantAgent(
            name="interviewer_agent",
            model_client=self.model_client,
            system_message=self.INTERVIEWER_SYSTEM_MESSAGE,
        )
        self.evaluator_agent = AssistantAgent(
            name="evaluator_agent",
            model_client=self.model_client,
            system_message=self.EVALUATOR_SYSTEM_MESSAGE,
        )
        self.workflow = SelectorGroupChat(
            [self.interviewer_agent, self.evaluator_agent],
            model_client=self.model_client,
            termination_condition=MaxMessageTermination(1),
            selector_prompt=(
                "Select exactly one agent for each task. Select interviewer_agent "
                "when the task requests a question. Select evaluator_agent when "
                "the task requests evaluation JSON. Do not select the other role."
            ),
        )

    async def run_async(self, role_name: str, task: str) -> str:
        if role_name not in {"interviewer", "evaluator"}:
            raise LLMProviderError(f"Unsupported AutoGen role: {role_name}")

        routing_task = (
            f"Requested role: {role_name}. Route this task to the matching agent.\n\n{task}"
        )
        try:
            result = await self.workflow.run(task=routing_task)
            return str(result.messages[-1].content)
        except Exception as exc:
            raise LLMProviderError(f"AutoGen workflow failed: {exc}") from exc

    def run(self, role_name: str, task: str) -> str:
        """Synchronous adapter for the existing InterviewProvider contract.

        FastAPI currently calls the synchronous provider methods from synchronous
        routes, so no event loop is active here. Async callers should use
        ``run_async`` instead of nesting an event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(role_name, task))
        raise LLMProviderError("AutoGen async workflow cannot run inside an active event loop; use run_async")


class AutoGenProvider:
    name = "autogen"

    def __init__(self, api_key: str | None = None, model: str | None = None, orchestrator: Any = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        if not self.api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required when using the AutoGen provider")
        self.orchestrator = orchestrator or AutoGenOrchestrator(self.model, self.api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.orchestrator.run("interviewer", f"{system_prompt}\n\n{user_prompt}")

    def generate_question(
        self, role: str, interview_type: str, difficulty: str, history: list[dict], context: str
    ) -> str:
        return self.orchestrator.run(
            "interviewer",
            f"Generate exactly one {difficulty} {interview_type} interview question for a {role}.\n"
            f"Candidate/RAG context:\n{context}\nInterview history:\n{history}",
        )

    def evaluate_answer(self, role: str, question: str, answer: str, context: str, difficulty: str) -> EvaluationResult:
        raw = self.orchestrator.run(
            "evaluator",
            f"Evaluate this answer as JSON. Role: {role}. Question: {question}. Answer: {answer}. "
            f"Candidate/RAG context: {context}. Current difficulty: {difficulty}. "
            "Return every EvaluationResult field.",
        )
        try:
            return EvaluationResult.model_validate_json(_strip_json_fence(raw))
        except (ValueError, TypeError) as exc:
            raise LLMProviderError("AutoGen evaluator returned malformed evaluation JSON") from exc
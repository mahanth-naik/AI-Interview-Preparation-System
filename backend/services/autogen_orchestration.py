import asyncio
import os
from typing import Any, Sequence

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
role, and difficulty.

Return ONLY one valid JSON object.
Do not return markdown.
Do not use code fences.
Do not return any explanation outside the JSON object.

The JSON MUST contain exactly these fields:

technical_correctness: integer from 1 to 10
relevance: integer from 1 to 10
completeness: integer from 1 to 10
clarity: integer from 1 to 10
strengths: array of strings
weaknesses: array of strings
improvement_suggestions: array of strings
follow_up_required: boolean
next_difficulty: exactly one of "easy", "medium", or "hard"

IMPORTANT:
- strengths MUST be an array of strings, even if there is only one strength.
- weaknesses MUST be an array of strings, even if there is only one weakness.
- improvement_suggestions MUST be an array of strings, even if there is only one suggestion.
- follow_up_required MUST be true or false, not a string.
- technical_correctness, relevance, completeness, and clarity MUST be integers.
- next_difficulty MUST be exactly "easy", "medium", or "hard".
- Never invent candidate facts.

Example of the required JSON structure:
{
  "technical_correctness": 8,
  "relevance": 9,
  "completeness": 7,
  "clarity": 8,
  "strengths": [
    "Correctly explains the main concept"
  ],
  "weaknesses": [
    "Does not provide an example"
  ],
  "improvement_suggestions": [
    "Add a concrete code example"
  ],
  "follow_up_required": true,
  "next_difficulty": "medium"
}
"""

    @staticmethod
    def _select_agent(messages: Sequence[Any]) -> str | None:
        """Select the agent requested by the routing task."""
        if not messages:
            return "interviewer_agent"

        last_message = messages[-1]

        content = getattr(last_message, "content", "")
        if not isinstance(content, str):
            content = str(content)

        if "Requested role: evaluator" in content:
            return "evaluator_agent"

        if "Requested role: interviewer" in content:
            return "interviewer_agent"

        return None

    def __init__(self, model: str, api_key: str):
        # AutoGen's workflow contains asyncio objects such as queues.
        # Keep one event loop for the lifetime of this orchestrator so that
        # multiple interview calls reuse the same loop.
        self._event_loop = asyncio.new_event_loop()

        try:
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.conditions import MaxMessageTermination
            from autogen_agentchat.teams import SelectorGroupChat
            from autogen_ext.models.openai import OpenAIChatCompletionClient
        except ImportError as exc:
            raise LLMConfigurationError(
                "autogen-agentchat and autogen-ext[openai] are required "
                "for the AutoGen provider"
            ) from exc

        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
        )

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
            termination_condition=MaxMessageTermination(2),
            selector_func=self._select_agent,
            selector_prompt=(
                "Select exactly one agent for each task. Select interviewer_agent "
                "when the task requests a question. Select evaluator_agent when "
                "the task requests evaluation JSON. Do not select the other role."
            ),
        )

    async def run_async(self, role_name: str, task: str) -> str:
        if role_name not in {"interviewer", "evaluator"}:
            raise LLMProviderError(
                f"Unsupported AutoGen role: {role_name}"
            )

        routing_task = (
            f"Requested role: {role_name}. "
            f"Route this task to the matching agent.\n\n{task}"
        )

        try:
            result = await self.workflow.run(task=routing_task)
            return str(result.messages[-1].content)
        except Exception as exc:
            raise LLMProviderError(
                f"AutoGen workflow failed: {exc}"
            ) from exc

    def run(self, role_name: str, task: str) -> str:
        """Run AutoGen using one persistent event loop.

        The synchronous InterviewProvider interface may call this method
        multiple times during one interview session. AutoGen's workflow
        contains asyncio objects that must remain associated with the same
        event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Some unit tests construct the orchestrator with __new__ and
            # therefore bypass __init__. Create the loop lazily in that case.
            if not hasattr(self, "_event_loop"):
                self._event_loop = asyncio.new_event_loop()

            try:
                return self._event_loop.run_until_complete(
                    self.run_async(role_name, task)
                )
            except Exception as exc:
                if isinstance(exc, LLMProviderError):
                    raise

                raise LLMProviderError(
                    f"AutoGen workflow failed: {exc}"
                ) from exc

        raise LLMProviderError(
            "AutoGen async workflow cannot run inside an active event loop; "
            "use run_async"
        )


class AutoGenProvider:
    name = "autogen"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        orchestrator: Any = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3-flash-preview",
        )

        if not self.api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is required when using the AutoGen provider"
            )

        self.orchestrator = orchestrator or AutoGenOrchestrator(
            self.model,
            self.api_key,
        )

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        return self.orchestrator.run(
            "interviewer",
            f"{system_prompt}\n\n{user_prompt}",
        )

    def generate_question(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        history: list[dict],
        context: str,
    ) -> str:
        return self.orchestrator.run(
            "interviewer",
            f"Generate exactly one {difficulty} "
            f"{interview_type} interview question for a {role}.\n"
            f"Candidate/RAG context:\n{context}\n"
            f"Interview history:\n{history}",
        )

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str,
        context: str,
        difficulty: str,
    ) -> EvaluationResult:
        raw = self.orchestrator.run(
            "evaluator",
            f"Evaluate this answer as JSON. "
            f"Role: {role}. "
            f"Question: {question}. "
            f"Answer: {answer}. "
            f"Candidate/RAG context: {context}. "
            f"Current difficulty: {difficulty}. "
            "Return every EvaluationResult field.",
        )

        try:
            return EvaluationResult.model_validate_json(
                _strip_json_fence(raw)
            )
        except (ValueError, TypeError) as exc:
            raise LLMProviderError(
                "AutoGen evaluator returned malformed evaluation JSON"
            ) from exc
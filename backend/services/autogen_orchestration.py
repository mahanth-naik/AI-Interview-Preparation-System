
import asyncio
import os
from typing import Any, Sequence

from services.ai_provider import (
    EvaluationResult,
    InterviewContextMode,
    LLMConfigurationError,
    LLMProviderError,
    _evaluation_mode_prompt,
    _question_mode_prompt,
    _strip_json_fence,
)


class AutoGenOrchestrator:
    """Routes interview tasks through a two-agent AutoGen workflow."""

    INTERVIEWER_SYSTEM_MESSAGE = """
You are InterviewerAgent in an interview workflow.

Generate exactly one interview question.
Use the supplied source context, role, interview type,
difficulty, and interview history.

Do not repeat previous questions.
Ask an appropriate follow-up when history or the previous answer
requires it.
Follow the mode-specific source instructions in the task.
Return only the question, without explanation or numbering.
"""

    EVALUATOR_SYSTEM_MESSAGE = """
You are EvaluatorAgent in an interview workflow.

Evaluate the candidate answer using the supplied question,
source context, role, and difficulty.

Return ONLY one valid JSON object.
Do not return markdown or code fences.
Do not return explanations outside the JSON.

The JSON must contain exactly these fields:

technical_correctness: integer from 1 to 10
relevance: integer from 1 to 10
completeness: integer from 1 to 10
clarity: integer from 1 to 10
strengths: array of strings
weaknesses: array of strings
improvement_suggestions: array of strings
follow_up_required: boolean
next_difficulty: "easy", "medium", or "hard"

Follow the mode-specific source instructions in the task.
"""

    @staticmethod
    def _select_agent(messages: Sequence[Any]) -> str:
        """Select the agent based on the latest routing task."""

        if not messages:
            return "interviewer_agent"

        content = getattr(messages[-1], "content", "")

        if not isinstance(content, str):
            content = str(content)

        if "Requested role: evaluator" in content:
            return "evaluator_agent"

        if "Requested role: interviewer" in content:
            return "interviewer_agent"

        return "interviewer_agent"

    def __init__(self, model: str, api_key: str):
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
            base_url=(
                "https://generativelanguage.googleapis.com/"
                "v1beta/openai/"
            ),
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
                "Select interviewer_agent for question generation. "
                "Select evaluator_agent for evaluation JSON."
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

            if not result.messages:
                raise LLMProviderError(
                    "AutoGen returned no messages"
                )

            # The final message contains the agent's response
            content = result.messages[-1].content

            if not isinstance(content, str) or not content.strip():
                raise LLMProviderError(
                    "AutoGen returned an empty response"
                )

            return content.strip()

        except LLMProviderError:
            raise

        except Exception as exc:
            raise LLMProviderError(
                f"AutoGen workflow failed: {exc}"
            ) from exc



    def run(self, role_name: str, task: str) -> str:
        """Run AutoGen using the persistent event loop."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if not hasattr(self, "_event_loop"):
                self._event_loop = asyncio.new_event_loop()

            try:
                return self._event_loop.run_until_complete(
                    self.run_async(role_name, task)
                )
            except LLMProviderError:
                raise
            except Exception as exc:
                raise LLMProviderError(
                    f"AutoGen workflow failed: {exc}"
                ) from exc

        raise LLMProviderError(
            "AutoGen cannot run synchronously inside an active "
            "event loop; use run_async instead"
        )



class AutoGenProvider:
    """Interview provider backed by AutoGen and Gemini."""

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
                "GEMINI_API_KEY is required when using AutoGen"
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
        context_mode: InterviewContextMode = "general",
    ) -> str:
        mode_instructions, context_label = _question_mode_prompt(context_mode)

        return self.orchestrator.run(
            "interviewer",
            f"""
{mode_instructions}

Generate exactly one {difficulty} {interview_type}
interview question for a {role}.

Do not repeat any question in the history.

{context_label}:
{context}

Interview history:
{history}
""",
        )

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str,
        context: str,
        difficulty: str,
        context_mode: InterviewContextMode = "general",
    ) -> EvaluationResult:
        mode_instructions, context_label = _evaluation_mode_prompt(context_mode)

        raw = self.orchestrator.run(
            "evaluator",
            f"""
{mode_instructions}

Evaluate this interview answer.

Role: {role}
Question: {question}
Answer: {answer}
{context_label}: {context}
Current difficulty: {difficulty}

Return every required EvaluationResult field as valid JSON.
""",
        )

        try:
            return EvaluationResult.model_validate_json(
                _strip_json_fence(raw)
            )

        except (ValueError, TypeError) as exc:
            raise LLMProviderError(
                "AutoGen evaluator returned malformed evaluation JSON"
            ) from exc

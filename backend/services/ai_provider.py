import json
import os
import re
from dataclasses import dataclass
from typing import Any, Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


class LLMConfigurationError(RuntimeError):
    """Raised when an LLM provider is not configured correctly."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM request fails or returns an unusable response."""


class EvaluationResult(BaseModel):
    technical_correctness: int = Field(ge=1, le=10)
    relevance: int = Field(ge=1, le=10)
    completeness: int = Field(ge=1, le=10)
    clarity: int = Field(ge=1, le=10)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)
    follow_up_required: bool
    next_difficulty: str = Field(pattern="^(easy|medium|hard)$")

    @property
    def score(self) -> int:
        return round(
            (
                self.technical_correctness
                + self.relevance
                + self.completeness
                + self.clarity
            )
            / 4
        )

    @property
    def feedback(self) -> str:
        return (
            " ".join(self.improvement_suggestions)
            or "The answer was assessed across correctness, relevance, completeness, and clarity."
        )

    @property
    def improvements(self) -> list[str]:
        return self.improvement_suggestions


class InterviewProvider(Protocol):
    name: str

    def generate_question(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        history: list[dict],
        context: str,
    ) -> str: ...

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str,
        context: str,
        difficulty: str,
    ) -> EvaluationResult: ...


@dataclass
class Evaluation:
    score: int
    feedback: str
    strengths: list[str]
    improvements: list[str]


class LocalInterviewProvider:
    """Deterministic offline provider used until an external LLM is configured."""

    name = "local"

    def generate_question(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        history: list[dict],
        context: str,
    ) -> str:
        question = self.generate_questions(
            role, interview_type, difficulty, 1, context
        )[0]

        if history:
            question = (
                f"As a follow-up, "
                f"{question[0].lower() + question[1:]}"
            )

        return question

    def generate_questions(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        number: int,
        context: str,
    ) -> list[str]:
        technologies = self._technologies(context)
        focus = ", ".join(technologies[:3]) or role

        templates = [
            f"How would you explain your experience with {focus} in a {role} project?",
            f"What design or implementation trade-offs did you make while working with {focus}?",
            f"How would you test and troubleshoot a {interview_type} solution for {role}?",
            f"Describe a difficult problem related to {focus} and how you solved it.",
            f"How would you improve the reliability and maintainability of your {role} work?",
        ]

        return [
            templates[index % len(templates)]
            for index in range(number)
        ]

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str | None = None,
        context: str = "",
        difficulty: str = "medium",
    ) -> EvaluationResult:

        if answer is None:
            answer = question
            question = role

        words = answer.split()

        score = min(
            10,
            max(
                1,
                3 + min(4, len(words) // 20),
            ),
        )

        strengths = []
        improvements = []

        if len(words) >= 20:
            strengths.append(
                "The answer provides enough detail to assess the approach."
            )
        else:
            improvements.append(
                "Add a concrete example and explain the implementation steps."
            )

        if any(
            marker in answer.lower()
            for marker in (
                "because",
                "trade-off",
                "test",
                "example",
            )
        ):
            strengths.append(
                "The answer includes reasoning or supporting detail."
            )
        else:
            improvements.append(
                "Explain why the chosen approach was appropriate."
            )

        return EvaluationResult(
            technical_correctness=score,
            relevance=score,
            completeness=score,
            clarity=score,
            strengths=strengths,
            weaknesses=improvements,
            improvement_suggestions=improvements,
            follow_up_required=score < 7,
            next_difficulty=(
                "hard"
                if score >= 8
                else "medium"
                if score >= 5
                else "easy"
            ),
        )

    @staticmethod
    def _technologies(context: str) -> list[str]:
        known = [
            "Python",
            "FastAPI",
            "ChromaDB",
            "RAG",
            "PostgreSQL",
            "Docker",
            "JavaScript",
            "React",
            "SQL",
            "Git",
            "AWS",
            "Machine Learning",
        ]

        return [
            technology
            for technology in known
            if re.search(
                re.escape(technology),
                context,
                re.I,
            )
        ]


class OpenAIProvider:
    """Provider implementation using the OpenAI-compatible chat completions API."""

    name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv(
            "LLM_MODEL",
            "gpt-4o-mini",
        )

        if not self.api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when using the OpenAI provider"
            )

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The openai package is required for the OpenAI provider"
                ) from exc

            client = OpenAI(api_key=self.api_key)

        self.client = client

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = response.choices[0].message.content

        except Exception as exc:
            raise LLMProviderError(
                f"LLM request failed: {exc}"
            ) from exc

        if not content or not content.strip():
            raise LLMProviderError(
                "LLM returned an empty response"
            )

        return content.strip()

    def generate_question(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        history: list[dict],
        context: str,
    ) -> str:
        return self.complete(
            "You are a rigorous interviewer. "
            "Use the supplied candidate context where relevant. "
            "Do not invent candidate facts.",
            f"Generate exactly one {difficulty} "
            f"{interview_type} interview question for a {role}. "
            f"Ask a relevant follow-up when history exists. "
            f"Candidate context:\n{context}\n"
            f"History:\n{history}",
        )

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str,
        context: str,
        difficulty: str,
    ) -> EvaluationResult:

        raw = self.complete(
            "You are an interview evaluator. "
            "Return only valid JSON matching the requested schema. "
            "Do not invent candidate facts.",
            f"Evaluate this {role} interview answer. "
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Candidate context:\n{context}\n"
            f"Current difficulty: {difficulty}\n"
            "Use integer scores from 1 to 10 and "
            "next_difficulty of easy, medium, or hard. "
            "Return keys: technical_correctness, relevance, "
            "completeness, clarity, strengths, weaknesses, "
            "improvement_suggestions, follow_up_required, "
            "next_difficulty.",
        )

        try:
            payload = json.loads(
                _strip_json_fence(raw)
            )

            return EvaluationResult.model_validate(
                payload
            )

        except (
            json.JSONDecodeError,
            ValidationError,
            TypeError,
        ) as exc:
            raise LLMProviderError(
                "LLM returned malformed evaluation JSON"
            ) from exc


class GeminiProvider:
    """
    Provider implementation using Google's Gemini API
    through its OpenAI-compatible endpoint.
    """

    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
    ):
        self.api_key = api_key or os.getenv(
            "GEMINI_API_KEY"
        )

        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3-flash-preview",
        )

        if not self.api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is required when using the Gemini provider"
            )

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMConfigurationError(
                    "The openai package is required for the Gemini provider"
                ) from exc

            client = OpenAI(
                api_key=self.api_key,
                base_url=(
                    "https://generativelanguage.googleapis.com/"
                    "v1beta/openai/"
                ),
            )

        self.client = client

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
            )

            content = response.choices[0].message.content

        except Exception as exc:
            raise LLMProviderError(
                f"Gemini request failed: {exc}"
            ) from exc

        if not content or not content.strip():
            raise LLMProviderError(
                "Gemini returned an empty response"
            )

        return content.strip()

    def generate_question(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        history: list[dict],
        context: str,
    ) -> str:
        return self.complete(
            "You are a rigorous interviewer. "
            "Use the supplied candidate context where relevant. "
            "Do not invent candidate facts.",
            f"Generate exactly one {difficulty} "
            f"{interview_type} interview question for a {role}. "
            f"Ask a relevant follow-up when history exists. "
            f"Candidate context:\n{context}\n"
            f"History:\n{history}",
        )

    def evaluate_answer(
        self,
        role: str,
        question: str,
        answer: str,
        context: str,
        difficulty: str,
    ) -> EvaluationResult:

        raw = self.complete(
    "You are an interview evaluator. "
    "Return ONLY valid JSON. "
    "Do not use markdown or code fences. "
    "The JSON must exactly match the requested schema. "
    "technical_correctness, relevance, completeness, and clarity "
    "must be integers from 1 to 10. "
    "strengths MUST be an array of strings. "
    "weaknesses MUST be an array of strings. "
    "improvement_suggestions MUST be an array of strings. "
    "follow_up_required MUST be a boolean true or false. "
    "next_difficulty MUST be exactly one of: easy, medium, hard. "
    "Do not return explanatory text outside the JSON object.",
    f"Evaluate this {role} interview answer.\n"
    f"Question: {question}\n"
    f"Answer: {answer}\n"
    f"Candidate context:\n{context}\n"
    f"Current difficulty: {difficulty}\n\n"
    "Return exactly these JSON keys:\n"
    "technical_correctness, relevance, completeness, clarity, "
    "strengths, weaknesses, improvement_suggestions, "
    "follow_up_required, next_difficulty.",
)
        try:
            payload = json.loads(
                _strip_json_fence(raw)
            )

            return EvaluationResult.model_validate(
                payload
            )

        except (
            json.JSONDecodeError,
            ValidationError,
            TypeError,
        ) as exc:
            raise LLMProviderError(
                "Gemini returned malformed evaluation JSON"
            ) from exc


def _strip_json_fence(value: str) -> str:
    value = value.strip()

    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            value,
            flags=re.IGNORECASE,
        )

    return value.strip()


def get_interview_provider() -> InterviewProvider:
    provider_name = os.getenv(
        "LLM_PROVIDER",
        "local",
    ).lower()

    if provider_name == "local":
        return LocalInterviewProvider()

    if provider_name == "openai":
        return OpenAIProvider()

    if provider_name == "gemini":
        return GeminiProvider()

    if provider_name == "autogen":
        from services.autogen_orchestration import AutoGenProvider

        return AutoGenProvider()

    raise LLMConfigurationError(
        f"Unsupported LLM_PROVIDER: {provider_name}"
    )
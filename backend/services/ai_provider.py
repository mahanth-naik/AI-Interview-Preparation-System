import re
from dataclasses import dataclass


@dataclass
class Evaluation:
    score: int
    feedback: str
    strengths: list[str]
    improvements: list[str]


class LocalInterviewProvider:
    """Deterministic offline provider used until an external LLM is configured."""

    name = "local"

    def generate_questions(
        self, role: str, interview_type: str, difficulty: str, number: int, context: str
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
        return [templates[index % len(templates)] for index in range(number)]

    def evaluate_answer(self, question: str, answer: str) -> Evaluation:
        words = answer.split()
        score = min(10, max(1, 3 + min(4, len(words) // 20)))
        strengths = []
        improvements = []
        if len(words) >= 20:
            strengths.append("The answer provides enough detail to assess the approach.")
        else:
            improvements.append("Add a concrete example and explain the implementation steps.")
        if any(marker in answer.lower() for marker in ("because", "trade-off", "test", "example")):
            strengths.append("The answer includes reasoning or supporting detail.")
        else:
            improvements.append("Explain why the chosen approach was appropriate.")
        feedback = f"Offline evaluation for '{question}': the response shows a {score}/10 level of detail."
        return Evaluation(score, feedback, strengths, improvements)

    @staticmethod
    def _technologies(context: str) -> list[str]:
        known = [
            "Python", "FastAPI", "ChromaDB", "RAG", "PostgreSQL", "Docker",
            "JavaScript", "React", "SQL", "Git", "AWS", "Machine Learning",
        ]
        return [technology for technology in known if re.search(re.escape(technology), context, re.I)]


def get_interview_provider() -> LocalInterviewProvider:
    return LocalInterviewProvider()
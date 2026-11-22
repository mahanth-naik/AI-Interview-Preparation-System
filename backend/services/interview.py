
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
import re
from uuid import uuid4

from services.ai_provider import (
    EvaluationResult,
    InterviewContextMode,
    InterviewProvider,
    LLMProviderError,
    get_interview_provider,
)
from database.document_store import SQLiteDocumentStore
from database.session_store import SQLiteSessionStore
from services.rag import build_context, retrieve_context


_MAX_QUESTION_GENERATION_ATTEMPTS = 3
_QUESTION_STOP_WORDS = {
    "a", "an", "and", "are", "can", "describe", "does", "do", "explain",
    "for", "how", "in", "is", "it", "of", "on", "the", "to", "what",
    "when", "where", "which", "who", "why", "with", "would", "you", "your",
}


def _normalized_question(question: str) -> str:
    words = re.findall(r"[a-z0-9]+", question.casefold())
    normalized = []
    for word in words:
        if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
            word = word[:-1]
        if word not in _QUESTION_STOP_WORDS:
            normalized.append(word)
    return " ".join(normalized)


def _is_duplicate_question(question: str, previous: list[str]) -> bool:
    candidate = _normalized_question(question)
    if not candidate:
        return True

    candidate_words = set(candidate.split())
    for previous_question in previous:
        previous_normalized = _normalized_question(previous_question)
        if not previous_normalized:
            continue
        if candidate == previous_normalized:
            return True

        previous_words = set(previous_normalized.split())
        shared_words = candidate_words.intersection(previous_words)
        smaller_size = min(len(candidate_words), len(previous_words))
        union_size = len(candidate_words.union(previous_words))
        if (
            smaller_size >= 3
            and len(shared_words) / smaller_size >= 0.8
            and len(shared_words) / union_size >= 0.65
        ):
            return True
        if SequenceMatcher(
            None,
            candidate,
            previous_normalized,
        ).ratio() >= 0.9:
            return True
    return False


def _difficulty_after_evaluation(current: str, evaluation: object) -> str:
    """Raise at score >= 8 without concerns; lower at <= 4 or <= 6 with concerns."""
    levels = ("easy", "medium", "hard")
    current = current.casefold() if isinstance(current, str) else "medium"
    if current not in levels:
        current = "medium"

    score = getattr(evaluation, "score", None)
    if not isinstance(score, (int, float)):
        return current

    weaknesses = getattr(evaluation, "weaknesses", [])
    improvements = getattr(
        evaluation,
        "improvement_suggestions",
        getattr(evaluation, "improvements", []),
    )
    has_concerns = any(
        isinstance(item, str) and item.strip()
        for items in (weaknesses, improvements)
        if isinstance(items, list)
        for item in items
    )

    index = levels.index(current)
    if score <= 4 or (score <= 6 and has_concerns):
        index = max(0, index - 1)
    elif score >= 8 and not has_concerns:
        index = min(len(levels) - 1, index + 1)
    return levels[index]


@dataclass
class InterviewSession:
    session_id: str
    role: str
    interview_type: str
    difficulty: str
    questions: list[str]

    document_id: str | None = None

    current_question: int = 0
    answers: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)

    status: str = "active"
    current_difficulty: str = "medium"
    target_question_count: int = 5

    active_question: str | None = None
    user_id: str | None = None
    question_evidence: list[list[dict]] = field(default_factory=list)
    question_followups: list[bool] = field(default_factory=list)


class InterviewService:

    def __init__(
        self,
        store: SQLiteSessionStore | None = None,
        provider: InterviewProvider | None = None,
    ):
        self.sessions: dict[str, InterviewSession] = {}
        self.provider = provider or get_interview_provider()
        self.store = store or SQLiteSessionStore()
        self.document_store = SQLiteDocumentStore(self.store.database_path)

    def start(
        self,
        role: str,
        interview_type: str,
        difficulty: str,
        number: int,
        document_id: str | None = None,
        user_id: str | None = None,
    ) -> InterviewSession:

        session = InterviewSession(
            session_id=uuid4().hex,
            role=role,
            interview_type=interview_type,
            difficulty=difficulty,
            questions=[],
            document_id=document_id,
            user_id=user_id,
        )

        session.current_difficulty = difficulty
        session.target_question_count = number

        self.sessions[session.session_id] = session
        self.store.save(session)

        return session

    def get(
        self, session_id: str, user_id: str | None = None
    ) -> InterviewSession:

        stored_session = self.store.get(session_id, user_id=user_id)

        if stored_session is None:
            raise KeyError(session_id)

        session = self.sessions.get(session_id)
        if session is None:
            session = InterviewSession(**stored_session)
            self.sessions[session_id] = session
        else:
            session.__dict__.update(stored_session)
        session.question_followups = (
            session.question_followups
            + [False] * max(
                0,
                len(session.questions) - len(session.question_followups),
            )
        )[:len(session.questions)]
        return session

    def _retrieve_session_context(
        self,
        session: InterviewSession,
        query: str,
        document_type: InterviewContextMode | None = None,
    ) -> list[dict]:
        if (
            not isinstance(session.user_id, str)
            or not session.user_id.strip()
            or not isinstance(session.document_id, str)
            or not session.document_id.strip()
            or document_type not in {"resume", "study_material"}
        ):
            return []

        return retrieve_context(
            query,
            3,
            document_id=session.document_id,
            user_id=session.user_id,
            document_type=document_type,
        )

    @staticmethod
    def _evidence_references(context_items: list[dict]) -> list[dict]:
        fields = (
            "document_id",
            "filename",
            "document_type",
            "page_number",
            "chunk_id",
            "text",
            "distance",
        )
        return [
            {field: item.get(field) for field in fields}
            for item in context_items
        ]

    def _session_document_type(
        self,
        session: InterviewSession,
    ) -> InterviewContextMode | None:
        if not session.document_id or not session.user_id:
            return None

        document = self.document_store.get_for_user(
            session.document_id,
            session.user_id,
        )
        if document is None:
            raise ValueError("The selected document is no longer available.")
        if document["processing_status"] != "ready":
            raise ValueError("The selected document is not ready for interviews.")

        if document["document_type"] == "resume":
            return "resume"
        if document["document_type"] == "study_material":
            return "study_material"
        raise ValueError("The selected document has an unsupported type.")

    @staticmethod
    def _question_retrieval_query(
        session: InterviewSession,
        document_type: InterviewContextMode | None,
        follow_up: bool = False,
    ) -> str:
        previous = session.answers[-1] if session.answers else {}
        previous_question = previous.get("question", "")
        previous_answer = previous.get("answer", "")
        follow_up_context = (
            f" Follow-up to question: {previous_question}. "
            f"Candidate answer: {previous_answer}"
            if follow_up
            else ""
        )
        if document_type == "resume":
            return (
                f"resume skills technologies projects and work experience "
                f"relevant to {session.role} {session.interview_type}"
                f"{follow_up_context}"
            )
        if document_type == "study_material":
            return (
                "key concepts definitions principles processes and "
                "explanations in the selected study material"
                f"{follow_up_context}"
            )
        return f"{session.role} {session.interview_type}{follow_up_context}"

    @staticmethod
    def _generation_history(
        session: InterviewSession,
        follow_up: bool,
        rejected_questions: list[str],
    ) -> list[dict]:
        history = [dict(item) for item in session.answers]
        if not follow_up and not rejected_questions:
            return history

        reason = ""
        if session.evaluations:
            evaluation = session.evaluations[-1]
            suggestions = evaluation.get(
                "improvement_suggestions",
                evaluation.get("improvements", []),
            )
            if isinstance(suggestions, list):
                reason = " ".join(
                    item for item in suggestions if isinstance(item, str)
                )
            if not reason and isinstance(evaluation.get("feedback"), str):
                reason = evaluation["feedback"]

        metadata = {
            "follow_up_requested": follow_up,
            "follow_up_reason": reason,
            "rejected_questions": list(rejected_questions),
        }
        if history:
            history[-1]["_generation_metadata"] = metadata
        else:
            history.append({"_generation_metadata": metadata})
        return history

    def _generate_distinct_question(
        self,
        session: InterviewSession,
        context: str,
        document_type: InterviewContextMode | None,
        follow_up: bool,
    ) -> str:
        rejected_questions: list[str] = []
        for _ in range(_MAX_QUESTION_GENERATION_ATTEMPTS):
            history = self._generation_history(
                session,
                follow_up,
                rejected_questions,
            )
            generated = self.provider.generate_question(
                session.role,
                session.interview_type,
                session.current_difficulty,
                history,
                context,
                context_mode=document_type or "general",
            )
            if not isinstance(generated, str) or not generated.strip():
                raise LLMProviderError(
                    "The interview provider returned an empty question."
                )
            question = generated.strip()
            if not _is_duplicate_question(
                question,
                [*session.questions, *rejected_questions],
            ):
                return question
            rejected_questions.append(question)

        raise LLMProviderError(
            "Unable to generate a distinct interview question after "
            f"{_MAX_QUESTION_GENERATION_ATTEMPTS} attempts."
        )

    @staticmethod
    def _evaluation_retrieval_query(
        session: InterviewSession,
        question: str,
        document_type: InterviewContextMode | None,
    ) -> str:
        if document_type == "resume":
            return (
                f"resume skills technologies projects and experience "
                f"relevant to the {session.role} interview question: {question}"
            )
        if document_type == "study_material":
            return (
                "study-material evidence relevant to this interview question: "
                f"{question}"
            )
        return f"{session.role} {question}"

    def next_question(
        self, session_id: str, user_id: str | None = None
    ) -> dict:

        session = self.get(session_id, user_id=user_id)

        if session.status == "completed":
            return {
                "question": None,
                "question_number": None,
                "status": "completed"
            }

        if session.current_question >= session.target_question_count:

            session.status = "completed"
            session.active_question = None
            self.store.save(session)

            return {
                "question": None,
                "question_number": None,
                "status": "completed"
            }

        if session.current_question > len(session.questions):
            raise ValueError("Interview question progression is inconsistent.")
        while len(session.question_followups) < len(session.questions):
            session.question_followups.append(False)

        if (
            session.current_question < len(session.questions)
            and _is_duplicate_question(
                session.questions[session.current_question],
                session.questions[:session.current_question],
            )
        ):
            session.questions = session.questions[:session.current_question]
            session.question_evidence = session.question_evidence[
                :session.current_question
            ]
            session.question_followups = session.question_followups[
                :session.current_question
            ]
            session.active_question = None
            self.store.save(session)

        if (
            session.current_question == len(session.questions)
            and session.active_question is not None
        ):
            session.active_question = None
            self.store.save(session)

        # Generate a question only when one does not already exist.
        if session.current_question == len(session.questions):
            is_follow_up = bool(
                session.evaluations
                and session.evaluations[-1].get("follow_up_required") is True
                and (
                    not session.question_followups
                    or not session.question_followups[-1]
                )
            )
            document_type = self._session_document_type(session)

            context_items = self._retrieve_session_context(
                session,
                self._question_retrieval_query(
                    session,
                    document_type,
                    follow_up=is_follow_up,
                ),
                document_type,
            )

            if document_type is not None and not context_items:
                raise ValueError(
                    "No relevant content was found in the selected document."
                )

            context = build_context(context_items, document_type)

            question = self._generate_distinct_question(
                session,
                context,
                document_type,
                is_follow_up,
            )

            while len(session.question_evidence) < len(session.questions):
                session.question_evidence.append([])
            session.questions.append(question)
            session.question_evidence.append(
                self._evidence_references(context_items)
            )
            session.question_followups.append(is_follow_up)

        question = session.questions[session.current_question]
        is_follow_up = (
            session.question_followups[session.current_question]
            if session.current_question < len(session.question_followups)
            else False
        )
        evidence = (
            session.question_evidence[session.current_question]
            if session.current_question < len(session.question_evidence)
            else []
        )

        if session.active_question != question:
            session.active_question = question
        self.store.save(session)

        return {
            "question": question,
            "question_number": session.current_question + 1,
            "status": session.status,
            "difficulty": session.current_difficulty,
            "evidence": evidence,
            "is_follow_up": is_follow_up,
        }

    def evaluate(
        self,
        session_id: str,
        question: str,
        answer: str,
        user_id: str | None = None,
    ) -> dict:

        session = self.get(session_id, user_id=user_id)

        # Check whether the interview is active.
        if session.status != "active":
            raise ValueError(
                "Cannot evaluate an answer because the interview is not active."
            )

        # Check whether a question is waiting for an answer.
        if session.active_question is None:
            raise ValueError(
                "No active question found. Generate a question first."
            )

        if session.current_question >= len(session.questions):
            raise ValueError(
                "No question is available for evaluation."
            )

        # Verify the submitted question.
        expected_question = session.questions[session.current_question]

        if question != expected_question or question != session.active_question:
            raise ValueError(
                "Invalid question. Please submit the currently active interview question."
            )

        # Prevent empty answers.
        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        document_type = self._session_document_type(session)
        context_items = self._retrieve_session_context(
            session,
            self._evaluation_retrieval_query(
                session,
                question,
                document_type,
            ),
            document_type,
        )

        if document_type == "study_material" and not context_items:
            raise ValueError(
                "No relevant study-material evidence was found for this question."
            )

        context = build_context(context_items, document_type)
        evidence = self._evidence_references(context_items)

        # Evaluate the answer.
        evaluation = self.provider.evaluate_answer(
            session.role,
            question,
            answer,
            context,
            session.current_difficulty,
            context_mode=document_type or "general",
        )

        # Convert evaluation into a dictionary.
        if isinstance(evaluation, EvaluationResult):
            result = evaluation.model_dump()
        else:
            result = asdict(evaluation)

        result["score"] = evaluation.score
        result["feedback"] = evaluation.feedback
        result["improvements"] = evaluation.improvements
        result["evidence"] = evidence
        result["next_difficulty"] = _difficulty_after_evaluation(
            session.current_difficulty,
            evaluation,
        )
        current_is_follow_up = (
            session.question_followups[session.current_question]
            if session.current_question < len(session.question_followups)
            else False
        )

        # Update the session only after successful evaluation.
        session.answers.append({
            "question": question,
            "answer": answer
        })

        session.evaluations.append(result)

        session.active_question = None
        session.current_question += 1

        session.current_difficulty = result["next_difficulty"]
        follow_up_requested = getattr(
            evaluation,
            "follow_up_required",
            evaluation.score <= 4,
        )
        result["follow_up_required"] = bool(
            follow_up_requested
            and not current_is_follow_up
            and session.current_question < session.target_question_count
        )

        # Check whether the interview is completed.
        if session.current_question >= session.target_question_count:
            session.status = "completed"

        result["next_question_available"] = (
            session.status != "completed"
        )

        self.store.save(session)
        return result

    def list_sessions(
        self, limit: int = 50, user_id: str | None = None
    ) -> list[dict]:
        return self.store.list(limit, user_id=user_id)

    def get_report(
        self, session_id: str, user_id: str | None = None
    ) -> dict:
        session = self.get(session_id, user_id=user_id)
        if session.status != "completed":
            raise ValueError("The final report is available after the interview is completed.")

        question_results = []
        all_strengths = []
        all_weaknesses = []
        recommendations = []

        def append_unique(target: list[str], values: list[str]) -> None:
            existing = {value.casefold() for value in target}
            for value in values:
                if isinstance(value, str) and value.strip() and value.casefold() not in existing:
                    target.append(value.strip())
                    existing.add(value.casefold())

        for index, evaluation in enumerate(session.evaluations):
            question = session.questions[index] if index < len(session.questions) else ""
            answer = session.answers[index]["answer"] if index < len(session.answers) else ""
            dimension_scores = [
                evaluation[key]
                for key in (
                    "technical_correctness",
                    "relevance",
                    "completeness",
                    "clarity",
                )
                if isinstance(evaluation.get(key), (int, float))
            ]
            score = evaluation.get("score")
            if not isinstance(score, (int, float)):
                score = round(sum(dimension_scores) / len(dimension_scores)) if dimension_scores else 0

            strengths = evaluation.get("strengths", [])
            weaknesses = evaluation.get("weaknesses", [])
            suggestions = evaluation.get(
                "improvement_suggestions",
                evaluation.get("improvements", []),
            )
            if not isinstance(strengths, list):
                strengths = []
            if not isinstance(weaknesses, list):
                weaknesses = []
            if not isinstance(suggestions, list):
                suggestions = []

            append_unique(all_strengths, strengths)
            append_unique(all_weaknesses, weaknesses)
            append_unique(recommendations, suggestions or weaknesses)
            question_results.append(
                {
                    "question_number": index + 1,
                    "question": question,
                    "is_follow_up": (
                        session.question_followups[index]
                        if index < len(session.question_followups)
                        else False
                    ),
                    "answer": answer,
                    "score": score,
                    "technical_correctness": evaluation.get("technical_correctness"),
                    "relevance": evaluation.get("relevance"),
                    "completeness": evaluation.get("completeness"),
                    "clarity": evaluation.get("clarity"),
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "recommendations": suggestions,
                    "question_evidence": (
                        session.question_evidence[index]
                        if index < len(session.question_evidence)
                        else []
                    ),
                    "evaluation_evidence": (
                        evaluation.get("evidence", [])
                        if isinstance(evaluation.get("evidence", []), list)
                        else []
                    ),
                }
            )

        scores = [item["score"] for item in question_results]
        return {
            "session_id": session.session_id,
            "role": session.role,
            "interview_type": session.interview_type,
            "status": session.status,
            "question_count": len(question_results),
            "overall_score": round(sum(scores) / len(scores)) if scores else 0,
            "question_results": question_results,
            "strengths": all_strengths,
            "weaknesses": all_weaknesses,
            "recommendations": recommendations,
        }


from dataclasses import asdict, dataclass, field
from uuid import uuid4

from services.ai_provider import (
    EvaluationResult,
    InterviewContextMode,
    InterviewProvider,
    get_interview_provider,
)
from database.document_store import SQLiteDocumentStore
from database.session_store import SQLiteSessionStore
from services.rag import build_context, retrieve_context


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
    ) -> str:
        if document_type == "resume":
            return (
                f"resume skills technologies projects and work experience "
                f"relevant to {session.role} {session.interview_type}"
            )
        if document_type == "study_material":
            return (
                "key concepts definitions principles processes and "
                "explanations in the selected study material"
            )
        return f"{session.role} {session.interview_type}"

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

        # Generate a question only when one does not already exist.
        if session.current_question == len(session.questions):
            document_type = self._session_document_type(session)

            context_items = self._retrieve_session_context(
                session,
                self._question_retrieval_query(session, document_type),
                document_type,
            )

            if document_type is not None and not context_items:
                raise ValueError(
                    "No relevant content was found in the selected document."
                )

            context = build_context(
                [item["text"] for item in context_items],
                document_type,
            )

            question = self.provider.generate_question(
                session.role,
                session.interview_type,
                session.current_difficulty,
                session.answers,
                context,
                context_mode=document_type or "general",
            )

            session.questions.append(question)

        question = session.questions[session.current_question]

        session.active_question = question
        self.store.save(session)

        return {
            "question": question,
            "question_number": session.current_question + 1,
            "status": session.status,
            "difficulty": session.current_difficulty
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

        context = build_context(
            [item["text"] for item in context_items],
            document_type,
        )

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

        # Update the session only after successful evaluation.
        session.answers.append({
            "question": question,
            "answer": answer
        })

        session.evaluations.append(result)

        session.active_question = None
        session.current_question += 1

        # Adapt difficulty for the next question.
        session.current_difficulty = evaluation.next_difficulty

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
                    "answer": answer,
                    "score": score,
                    "technical_correctness": evaluation.get("technical_correctness"),
                    "relevance": evaluation.get("relevance"),
                    "completeness": evaluation.get("completeness"),
                    "clarity": evaluation.get("clarity"),
                    "strengths": strengths,
                    "weaknesses": weaknesses,
                    "recommendations": suggestions,
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

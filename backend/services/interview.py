from dataclasses import asdict, dataclass, field
from uuid import uuid4

from services.ai_provider import EvaluationResult, get_interview_provider
from services.rag import build_context, retrieve_context


@dataclass
class InterviewSession:
    session_id: str
    role: str
    interview_type: str
    difficulty: str
    questions: list[str]
    current_question: int = 0
    answers: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    status: str = "active"
    current_difficulty: str = "medium"
    target_question_count: int = 5


class InterviewService:
    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}
        self.provider = get_interview_provider()

    def start(self, role: str, interview_type: str, difficulty: str, number: int) -> InterviewSession:
        session = InterviewSession(uuid4().hex, role, interview_type, difficulty, [])
        session.current_difficulty = difficulty
        session.target_question_count = number
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> InterviewSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def next_question(self, session_id: str) -> dict:
        session = self.get(session_id)
        if session.current_question >= getattr(session, "target_question_count", len(session.questions)):
            session.status = "completed"
            return {"question": None, "question_number": None, "status": session.status}
        if session.current_question == len(session.questions):
            context_items = retrieve_context(f"{session.role} {session.interview_type}", 3)
            context = build_context([item["text"] for item in context_items])
            question = self.provider.generate_question(
                session.role,
                session.interview_type,
                session.current_difficulty,
                session.answers,
                context,
            )
            session.questions.append(question)
        question = session.questions[session.current_question]
        return {
            "question": question,
            "question_number": session.current_question + 1,
            "status": session.status,
            "difficulty": session.current_difficulty,
        }

    def evaluate(self, session_id: str, question: str, answer: str) -> dict:
        session = self.get(session_id)
        context_items = retrieve_context(f"{session.role} {question} {answer}", 3)
        context = build_context([item["text"] for item in context_items])
        evaluation = self.provider.evaluate_answer(
            session.role, question, answer, context, session.current_difficulty
        )
        result = evaluation.model_dump() if isinstance(evaluation, EvaluationResult) else asdict(evaluation)
        result["score"] = evaluation.score
        result["feedback"] = evaluation.feedback
        result["improvements"] = evaluation.improvements
        session.answers.append({"question": question, "answer": answer})
        session.evaluations.append(result)
        session.current_question += 1
        session.current_difficulty = evaluation.next_difficulty
        if session.current_question >= getattr(session, "target_question_count", len(session.questions)):
            session.status = "completed"
        result["next_question_available"] = session.status != "completed"
        return result
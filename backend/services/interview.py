from dataclasses import asdict, dataclass, field
from uuid import uuid4

from services.ai_provider import get_interview_provider
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


class InterviewService:
    def __init__(self):
        self.sessions: dict[str, InterviewSession] = {}
        self.provider = get_interview_provider()

    def start(self, role: str, interview_type: str, difficulty: str, number: int) -> InterviewSession:
        context_items = retrieve_context(f"{role} {interview_type}", 3)
        context = build_context([item["text"] for item in context_items])
        questions = self.provider.generate_questions(role, interview_type, difficulty, number, context)
        session = InterviewSession(uuid4().hex, role, interview_type, difficulty, questions)
        self.sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> InterviewSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def next_question(self, session_id: str) -> dict:
        session = self.get(session_id)
        if session.current_question >= len(session.questions):
            session.status = "completed"
            return {"question": None, "question_number": None, "status": session.status}
        question = session.questions[session.current_question]
        return {
            "question": question,
            "question_number": session.current_question + 1,
            "status": session.status,
        }

    def evaluate(self, session_id: str, question: str, answer: str) -> dict:
        session = self.get(session_id)
        evaluation = self.provider.evaluate_answer(question, answer)
        result = asdict(evaluation)
        session.answers.append({"question": question, "answer": answer})
        session.evaluations.append(result)
        session.current_question += 1
        if session.current_question >= len(session.questions):
            session.status = "completed"
        return result
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from services.interview import InterviewSession


class SQLiteSessionStore:
    """Persist interview session state in a local SQLite database."""

    def __init__(self, database_path: str | Path | None = None):
        configured_path = os.getenv("INTERVIEW_DB_PATH")
        self.database_path = Path(
            database_path
            or configured_path
            or Path(__file__).with_name("interviews.sqlite3")
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS interview_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    role TEXT NOT NULL,
                    interview_type TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    document_id TEXT,
                    current_question INTEGER NOT NULL,
                    questions_json TEXT NOT NULL,
                    answers_json TEXT NOT NULL,
                    evaluations_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_difficulty TEXT NOT NULL,
                    target_question_count INTEGER NOT NULL,
                    active_question TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(interview_sessions)"
                )
            }
            if "user_id" not in columns:
                connection.execute(
                    "ALTER TABLE interview_sessions ADD COLUMN user_id TEXT"
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_interview_sessions_user_updated
                ON interview_sessions(user_id, updated_at DESC)
                """
            )

    def save(self, session: "InterviewSession") -> None:
        values = asdict(session)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO interview_sessions (
                    session_id, user_id, role, interview_type, difficulty, document_id,
                    current_question, questions_json, answers_json,
                    evaluations_json, status, current_difficulty,
                    target_question_count, active_question
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    role = excluded.role,
                    interview_type = excluded.interview_type,
                    difficulty = excluded.difficulty,
                    document_id = excluded.document_id,
                    current_question = excluded.current_question,
                    questions_json = excluded.questions_json,
                    answers_json = excluded.answers_json,
                    evaluations_json = excluded.evaluations_json,
                    status = excluded.status,
                    current_difficulty = excluded.current_difficulty,
                    target_question_count = excluded.target_question_count,
                    active_question = excluded.active_question,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    values["session_id"],
                    values["user_id"],
                    values["role"],
                    values["interview_type"],
                    values["difficulty"],
                    values["document_id"],
                    values["current_question"],
                    json.dumps(values["questions"]),
                    json.dumps(values["answers"]),
                    json.dumps(values["evaluations"]),
                    values["status"],
                    values["current_difficulty"],
                    values["target_question_count"],
                    values["active_question"],
                ),
            )

    def get(self, session_id: str, user_id: str | None = None) -> dict | None:
        with self._connect() as connection:
            if user_id is None:
                row = connection.execute(
                    "SELECT * FROM interview_sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT * FROM interview_sessions
                    WHERE session_id = ? AND user_id = ?
                    """,
                    (session_id, user_id),
                ).fetchone()

        if row is None:
            return None

        return {
            "session_id": row["session_id"],
            "user_id": row["user_id"],
            "role": row["role"],
            "interview_type": row["interview_type"],
            "difficulty": row["difficulty"],
            "document_id": row["document_id"],
            "current_question": row["current_question"],
            "questions": json.loads(row["questions_json"]),
            "answers": json.loads(row["answers_json"]),
            "evaluations": json.loads(row["evaluations_json"]),
            "status": row["status"],
            "current_difficulty": row["current_difficulty"],
            "target_question_count": row["target_question_count"],
            "active_question": row["active_question"],
        }

    def list(
        self, limit: int = 50, user_id: str | None = None
    ) -> list[dict]:
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        with self._connect() as connection:
            query = """
                SELECT session_id, role, interview_type, status,
                       current_question, target_question_count,
                       current_difficulty, questions_json, evaluations_json,
                       updated_at
                FROM interview_sessions
            """
            parameters: tuple[str | int, ...]
            if user_id is None:
                parameters = (limit,)
            else:
                query += " WHERE user_id = ?"
                parameters = (user_id, limit)
            query += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
            rows = connection.execute(query, parameters).fetchall()

        sessions = []
        for row in rows:
            questions = json.loads(row["questions_json"])
            evaluations = json.loads(row["evaluations_json"])
            scores = [
                evaluation["score"]
                for evaluation in evaluations
                if isinstance(evaluation.get("score"), (int, float))
            ]
            sessions.append(
                {
                    "session_id": row["session_id"],
                    "role": row["role"],
                    "interview_type": row["interview_type"],
                    "status": row["status"],
                    "current_question": row["current_question"],
                    "target_question_count": row["target_question_count"],
                    "current_difficulty": row["current_difficulty"],
                    "question_count": len(questions),
                    "overall_score": (
                        round(sum(scores) / len(scores)) if scores else None
                    ),
                    "updated_at": row["updated_at"],
                }
            )
        return sessions

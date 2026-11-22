import json
import sqlite3

from database import vector_store
from database.session_store import SQLiteSessionStore


def test_legacy_sessions_are_preserved_but_unowned(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE interview_sessions (
                session_id TEXT PRIMARY KEY,
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
        connection.execute(
            """
            INSERT INTO interview_sessions (
                session_id, role, interview_type, difficulty, document_id,
                current_question, questions_json, answers_json,
                evaluations_json, status, current_difficulty,
                target_question_count, active_question
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-session",
                "Engineer",
                "technical",
                "medium",
                "legacy-resume",
                0,
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                "active",
                "medium",
                2,
                None,
            ),
        )

    store = SQLiteSessionStore(database_path)
    legacy = store.get("legacy-session")

    assert legacy is not None
    assert legacy["role"] == "Engineer"
    assert legacy["user_id"] is None
    assert store.get("legacy-session", user_id="new-account") is None
    assert store.list(user_id="new-account") == []


def test_vector_search_combines_document_and_owner_filters(monkeypatch):
    class Collection:
        def __init__(self):
            self.arguments = None

        def count(self):
            return 1

        def query(self, **kwargs):
            self.arguments = kwargs
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    collection = Collection()
    monkeypatch.setattr(vector_store, "collection", collection)

    vector_store.search_chunks(
        "experience",
        document_id="resume-1",
        user_id="user-1",
    )

    assert collection.arguments["where"] == {
        "$and": [{"document_id": "resume-1"}, {"user_id": "user-1"}]
    }


def test_new_vector_chunks_include_owner_metadata(monkeypatch):
    class Collection:
        def add(self, **kwargs):
            self.arguments = kwargs

    collection = Collection()
    monkeypatch.setattr(vector_store, "collection", collection)

    result = vector_store.add_chunks(
        ["candidate experience"],
        "resume.pdf",
        user_id="user-1",
    )

    assert result["chunks_stored"] == 1
    assert collection.arguments["metadatas"][0]["user_id"] == "user-1"

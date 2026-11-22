import json
import sqlite3

import pytest

from database import vector_store
from database.document_store import SQLiteDocumentStore
from database.session_store import SQLiteSessionStore
from services import interview, rag


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


def test_vector_search_without_owner_fails_closed(monkeypatch):
    class Collection:
        def __init__(self):
            self.called = False

        def count(self):
            self.called = True
            return 1

        def query(self, **kwargs):
            self.called = True
            raise AssertionError("unscoped vector query must not run")

    collection = Collection()
    monkeypatch.setattr(vector_store, "collection", collection)

    assert vector_store.search_chunks("query") == {
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    assert collection.called is False


def test_rag_retrieval_without_owner_fails_closed(monkeypatch):
    def unexpected_search(*args, **kwargs):
        pytest.fail("search_chunks must not run without an owner")

    monkeypatch.setattr(rag, "search_chunks", unexpected_search)

    assert rag.retrieve_context("query") == []


@pytest.mark.parametrize(
    ("user_id", "document_id"),
    [
        (None, "document-1"),
        ("user-1", None),
        ("", "document-1"),
        ("user-1", " "),
    ],
)
def test_interview_retrieval_requires_owner_and_document(
    tmp_path,
    monkeypatch,
    user_id,
    document_id,
):
    def unexpected_retrieval(*args, **kwargs):
        pytest.fail("interview retrieval must require owner and document")

    monkeypatch.setattr(interview, "retrieve_context", unexpected_retrieval)
    service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / "interview.sqlite3")
    )
    session = service.start(
        "Engineer",
        "technical",
        "medium",
        1,
        document_id=document_id,
        user_id=user_id,
    )

    assert service._retrieve_session_context(session, "query") == []


def test_interview_retrieval_passes_owner_and_selected_document(
    tmp_path,
    monkeypatch,
):
    calls = []

    def fake_retrieve_context(query, n_results, **filters):
        calls.append((query, n_results, filters))
        return [{"text": "selected document chunk"}]

    monkeypatch.setattr(interview, "retrieve_context", fake_retrieve_context)
    service = interview.InterviewService(
        store=SQLiteSessionStore(tmp_path / "interview.sqlite3")
    )
    session = service.start(
        "Engineer",
        "technical",
        "medium",
        1,
        document_id="document-1",
        user_id="user-1",
    )

    assert service._retrieve_session_context(session, "query") == [
        {"text": "selected document chunk"}
    ]
    assert calls == [
        (
            "query",
            3,
            {"document_id": "document-1", "user_id": "user-1"},
        )
    ]


@pytest.mark.parametrize("document_type", ["resume", "study_material"])
def test_new_vector_chunks_include_owner_metadata(
    monkeypatch,
    document_type,
):
    class Collection:
        def add(self, **kwargs):
            self.arguments = kwargs

    collection = Collection()
    monkeypatch.setattr(vector_store, "collection", collection)

    result = vector_store.add_chunks(
        ["candidate experience"],
        "resume.pdf",
        user_id="user-1",
        document_type=document_type,
    )

    assert result["chunks_stored"] == 1
    assert collection.arguments["metadatas"][0]["user_id"] == "user-1"
    assert (
        collection.arguments["metadatas"][0]["document_type"]
        == document_type
    )


def test_legacy_documents_migrate_to_resume_and_ready(tmp_path):
    database_path = tmp_path / "legacy-documents.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO documents (document_id, user_id, filename)
            VALUES ('legacy-document', 'user-1', 'resume.pdf')
            """
        )

    document = SQLiteDocumentStore(database_path).get_for_user(
        "legacy-document",
        "user-1",
    )

    assert document is not None
    assert document["document_type"] == "resume"
    assert document["processing_status"] == "ready"


def test_vector_search_filters_by_owner_and_document_type(monkeypatch):
    class Collection:
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
        "operating systems",
        user_id="user-1",
        document_type="study_material",
    )

    assert collection.arguments["where"] == {
        "$and": [
            {"user_id": "user-1"},
            {"document_type": "study_material"},
        ]
    }


def test_vector_document_deletion_requires_owner_filter(monkeypatch):
    class Collection:
        def delete(self, **kwargs):
            self.arguments = kwargs

    collection = Collection()
    monkeypatch.setattr(vector_store, "collection", collection)

    vector_store.delete_document_chunks("doc-1", "user-1")

    assert collection.arguments["where"] == {
        "$and": [
            {"document_id": "doc-1"},
            {"user_id": "user-1"},
        ]
    }

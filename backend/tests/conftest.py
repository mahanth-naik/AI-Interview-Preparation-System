import pytest
from fastapi.testclient import TestClient

import auth as auth_module
import main
from database.document_store import SQLiteDocumentStore
from database.session_store import SQLiteSessionStore
from database.user_store import SQLiteUserStore
from services.ai_provider import LocalInterviewProvider
from services.interview import InterviewService


@pytest.fixture
def authenticated_client(tmp_path, monkeypatch):
    database_path = tmp_path / "test.sqlite3"
    monkeypatch.setenv(
        "JWT_SECRET_KEY",
        "test-only-secret-key-that-is-at-least-32-chars",
    )
    monkeypatch.setenv("JWT_ACCESS_TOKEN_MINUTES", "30")
    monkeypatch.setattr(
        auth_module,
        "user_store",
        SQLiteUserStore(database_path),
    )
    monkeypatch.setattr(
        main,
        "document_store",
        SQLiteDocumentStore(database_path),
    )
    monkeypatch.setattr(
        main,
        "interview_service",
        InterviewService(
            store=SQLiteSessionStore(database_path),
            provider=LocalInterviewProvider(),
        ),
    )

    client = TestClient(main.app)
    registration = client.post(
        "/auth/register",
        json={
            "full_name": "Test User",
            "email": "test@example.com",
            "password": "a-very-long-test-password",
        },
    )
    assert registration.status_code == 201
    login = client.post(
        "/auth/login",
        json={
            "email": "test@example.com",
            "password": "a-very-long-test-password",
        },
    )
    assert login.status_code == 200
    client.headers.update(
        {"Authorization": f"Bearer {login.json()['access_token']}"}
    )
    return client

import sqlite3

import jwt
import pytest
from fastapi.testclient import TestClient

import auth as auth_module
import main
from database.user_store import SQLiteUserStore


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        auth_module,
        "user_store",
        SQLiteUserStore(tmp_path / "auth.sqlite3"),
    )
    monkeypatch.setenv("JWT_SECRET_KEY", "test-only-secret-key-that-is-at-least-32-chars")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_MINUTES", "30")
    return TestClient(main.app)


def test_registration_persists_argon2_hash_and_normalizes_email(auth_client):
    response = auth_client.post(
        "/auth/register",
        json={
            "full_name": "  Ada Lovelace  ",
            "email": " ADA@example.com ",
            "password": "a-very-long-test-password",
        },
    )

    assert response.status_code == 201
    assert response.json()["user"]["full_name"] == "Ada Lovelace"
    assert response.json()["user"]["email"] == "ada@example.com"
    assert "password" not in response.json()["user"]

    stored = auth_module.user_store.get_user_by_email("ada@example.com")
    assert stored is not None
    assert stored["password_hash"].startswith("$argon2")
    assert stored["password_hash"] != "a-very-long-test-password"
    assert auth_module.password_hasher.verify(
        "a-very-long-test-password",
        stored["password_hash"],
    )


def test_duplicate_registration_returns_conflict(auth_client):
    payload = {
        "full_name": "Ada",
        "email": "ada@example.com",
        "password": "a-very-long-test-password",
    }
    assert auth_client.post("/auth/register", json=payload).status_code == 201
    duplicate = auth_client.post(
        "/auth/register",
        json={**payload, "email": "ADA@example.com"},
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["detail"] == "An account with this email already exists."


@pytest.mark.parametrize(
    "payload",
    [
        {"full_name": "Ada", "email": "not-an-email", "password": "a-very-long-test-password"},
        {"full_name": "Ada", "email": "ada@example.com", "password": "short"},
        {"full_name": "   ", "email": "ada@example.com", "password": "a-very-long-test-password"},
    ],
)
def test_registration_rejects_invalid_input(auth_client, payload):
    response = auth_client.post("/auth/register", json=payload)
    assert response.status_code == 422


def test_login_me_and_logout_revoke_access_token(auth_client):
    auth_client.post(
        "/auth/register",
        json={
            "full_name": "Ada Lovelace",
            "email": "ada@example.com",
            "password": "a-very-long-test-password",
        },
    )
    login = auth_client.post(
        "/auth/login",
        json={
            "email": "ADA@example.com",
            "password": "a-very-long-test-password",
        },
    )

    assert login.status_code == 200
    token = login.json()["access_token"]
    assert login.json()["token_type"] == "bearer"
    assert login.json()["expires_in"] == 1800
    assert "password" not in login.json()["user"]

    headers = {"Authorization": f"Bearer {token}"}
    me = auth_client.get("/auth/me", headers=headers)
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "ada@example.com"

    logged_out = auth_client.post("/auth/logout", headers=headers)
    assert logged_out.status_code == 204
    assert auth_client.get("/auth/me", headers=headers).status_code == 401


def test_login_uses_same_error_for_unknown_email_and_bad_password(auth_client):
    auth_client.post(
        "/auth/register",
        json={
            "full_name": "Ada",
            "email": "ada@example.com",
            "password": "a-very-long-test-password",
        },
    )
    wrong_password = auth_client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "incorrect-password"},
    )
    unknown_email = auth_client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "incorrect-password"},
    )

    assert wrong_password.status_code == unknown_email.status_code == 401
    assert wrong_password.json() == unknown_email.json()


def test_me_rejects_missing_malformed_and_expired_tokens(auth_client):
    assert auth_client.get("/auth/me").status_code == 401
    assert auth_client.get(
        "/auth/me",
        headers={"Authorization": "Bearer not-a-jwt"},
    ).status_code == 401

    registered = auth_client.post(
        "/auth/register",
        json={
            "full_name": "Ada",
            "email": "ada@example.com",
            "password": "a-very-long-test-password",
        },
    ).json()["user"]
    expired_token = jwt.encode(
        {
            "sub": registered["user_id"],
            "jti": "expired-test-token",
            "iat": 1,
            "exp": 2,
            "iss": "ai-interview-preparation-system",
        },
        "test-only-secret-key-that-is-at-least-32-chars",
        algorithm="HS256",
    )
    response = auth_client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired access token."


def test_login_reports_missing_jwt_configuration(auth_client, monkeypatch):
    auth_client.post(
        "/auth/register",
        json={
            "full_name": "Ada",
            "email": "ada@example.com",
            "password": "a-very-long-test-password",
        },
    )
    monkeypatch.delenv("JWT_SECRET_KEY")

    response = auth_client.post(
        "/auth/login",
        json={"email": "ada@example.com", "password": "a-very-long-test-password"},
    )

    assert response.status_code == 503
    assert "JWT_SECRET_KEY" in response.json()["detail"]


def test_users_and_revoked_tokens_share_existing_sqlite_database(tmp_path):
    database_path = tmp_path / "existing-interviews.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE interview_sessions (session_id TEXT PRIMARY KEY)"
        )

    store = SQLiteUserStore(database_path)
    user = store.create_user(
        "user-1",
        "ada@example.com",
        "Ada",
        "argon2-hash-placeholder",
    )

    assert user["user_id"] == "user-1"
    with sqlite3.connect(database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "interview_sessions" in tables
    assert "users" in tables
    assert "revoked_tokens" in tables

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SQLiteUserStore:
    """Store user credentials and revoked JWT IDs in SQLite."""

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
    def _connect(self) -> Iterator[sqlite3.Connection]:
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
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS revoked_tokens (
                    token_id TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL
                )
                """
            )

    def create_user(
        self,
        user_id: str,
        email: str,
        full_name: str,
        password_hash: str,
    ) -> dict:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (user_id, email, full_name, password_hash)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, email, full_name, password_hash),
            )
            row = connection.execute(
                """
                SELECT user_id, email, full_name, created_at
                FROM users WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row)

    def get_user_by_email(self, email: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (email,),
            ).fetchone()
        return dict(row) if row is not None else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, email, full_name, created_at
                FROM users WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def revoke_token(self, token_id: str, expires_at: int) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM revoked_tokens WHERE expires_at <= ?",
                (expires_at,),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO revoked_tokens (token_id, expires_at)
                VALUES (?, ?)
                """,
                (token_id, expires_at),
            )

    def is_token_revoked(self, token_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM revoked_tokens WHERE token_id = ?",
                (token_id,),
            ).fetchone()
        return row is not None

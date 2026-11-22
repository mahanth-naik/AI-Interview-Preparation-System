import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class SQLiteDocumentStore:
    """Track resume ownership separately from the ChromaDB vector data."""

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
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_user_created
                ON documents(user_id, created_at DESC)
                """
            )

    def register(
        self, document_id: str, user_id: str, filename: str
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (document_id, user_id, filename)
                VALUES (?, ?, ?)
                """,
                (document_id, user_id, filename),
            )

    def get_for_user(
        self, document_id: str, user_id: str
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, user_id, filename, created_at
                FROM documents
                WHERE document_id = ? AND user_id = ?
                """,
                (document_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

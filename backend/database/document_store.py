import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


class SQLiteDocumentStore:
    """Track uploaded document ownership and processing state in SQLite."""

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
                    document_type TEXT NOT NULL DEFAULT 'resume',
                    processing_status TEXT NOT NULL DEFAULT 'ready',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(documents)")
            }
            if "document_type" not in columns:
                connection.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN document_type TEXT NOT NULL DEFAULT 'resume'
                    """
                )
            if "processing_status" not in columns:
                connection.execute(
                    """
                    ALTER TABLE documents
                    ADD COLUMN processing_status TEXT NOT NULL DEFAULT 'ready'
                    """
                )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_documents_user_created
                ON documents(user_id, created_at DESC)
                """
            )

    def register(
        self,
        document_id: str,
        user_id: str,
        filename: str,
        document_type: str = "resume",
        processing_status: str = "ready",
    ) -> None:
        if document_type not in {"resume", "study_material"}:
            raise ValueError("Unsupported document type")
        if processing_status not in {"processing", "ready", "failed"}:
            raise ValueError("Unsupported document processing status")

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id, user_id, filename, document_type,
                    processing_status
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    user_id,
                    filename,
                    document_type,
                    processing_status,
                ),
            )

    def set_processing_status(
        self,
        document_id: str,
        user_id: str,
        processing_status: str,
    ) -> None:
        if processing_status not in {"processing", "ready", "failed"}:
            raise ValueError("Unsupported document processing status")

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE documents
                SET processing_status = ?
                WHERE document_id = ? AND user_id = ?
                """,
                (processing_status, document_id, user_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(document_id)

    def get_for_user(
        self, document_id: str, user_id: str
    ) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT document_id, user_id, filename, document_type,
                       processing_status, created_at
                FROM documents
                WHERE document_id = ? AND user_id = ?
                """,
                (document_id, user_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_for_user(self, user_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT document_id, user_id, filename, document_type,
                       processing_status, created_at
                FROM documents
                WHERE user_id = ?
                ORDER BY created_at DESC, rowid DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_for_user(self, document_id: str, user_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE document_id = ? AND user_id = ?
                """,
                (document_id, user_id),
            )
        return cursor.rowcount == 1

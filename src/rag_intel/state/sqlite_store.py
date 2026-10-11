"""Small durable metadata store for document/chunk identity."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class MetadataStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.init_db()

    @contextmanager
    def connection(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def init_db(self) -> None:
        with self.connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT,
                    status TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    stage_progress INTEGER NOT NULL DEFAULT 0,
                    error_message TEXT,
                    extra_metadata TEXT NOT NULL DEFAULT '{}',
                    file_path TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id INTEGER UNIQUE,
                    doc_id TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    section TEXT,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id)
                );
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);
                """
            )

    @staticmethod
    def _encode(value: dict[str, Any] | None) -> str:
        return json.dumps(value or {}, sort_keys=True)

    @staticmethod
    def _document(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        result = dict(row)
        result["extra_metadata"] = json.loads(result.get("extra_metadata") or "{}")
        return result

    def queue_document(
        self,
        doc_id: str,
        *,
        filename: str,
        source_type: str,
        file_path: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        persisted_metadata = {"filename": filename, "source_type": source_type}
        if extra_metadata:
            persisted_metadata.update(extra_metadata)
        title = str(persisted_metadata.get("title") or Path(filename).stem)
        self.upsert_document_stub(
            doc_id,
            filename=filename,
            source_type=source_type,
            title=title,
            status="queued",
            current_stage="queued",
            stage_progress=0,
            file_path=file_path,
            extra_metadata=persisted_metadata,
        )

    def upsert_document_stub(
        self,
        doc_id: str,
        *,
        filename: str,
        source_type: str,
        title: str,
        status: str,
        current_stage: str,
        stage_progress: int,
        file_path: str | None = None,
        error_message: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, source_type, title, status, current_stage,
                    stage_progress, error_message, extra_metadata, file_path, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename=excluded.filename,
                    source_type=excluded.source_type,
                    title=excluded.title,
                    status=excluded.status,
                    current_stage=excluded.current_stage,
                    stage_progress=excluded.stage_progress,
                    error_message=excluded.error_message,
                    extra_metadata=excluded.extra_metadata,
                    file_path=excluded.file_path,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    doc_id,
                    filename,
                    source_type,
                    title,
                    status,
                    current_stage,
                    stage_progress,
                    error_message,
                    self._encode(extra_metadata),
                    file_path,
                ),
            )

    def update_document_state(self, doc_id: str, **fields: Any) -> None:
        allowed = {
            "status",
            "current_stage",
            "stage_progress",
            "error_message",
            "file_path",
            "extra_metadata",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unsupported document fields: {sorted(unknown)}")
        if not fields:
            return
        values = dict(fields)
        if "extra_metadata" in values:
            values["extra_metadata"] = self._encode(values["extra_metadata"])
        assignments = ", ".join(f"{field} = ?" for field in values)
        query = f"UPDATE documents SET {assignments}, updated_at=CURRENT_TIMESTAMP WHERE doc_id = ?"
        with self.connection() as connection:
            connection.execute(query, [*values.values(), doc_id])

    def insert_chunks(self, doc_id: str, chunks: list[dict]) -> list[int]:
        vector_ids: list[int] = []
        with self.connection() as connection:
            for chunk in chunks:
                cursor = connection.execute(
                    "INSERT INTO chunks (doc_id, chunk_id, text, section) VALUES (?, ?, ?, ?)",
                    (
                        doc_id,
                        int(chunk["chunk_id"]),
                        str(chunk["text"]),
                        str(chunk.get("metadata", {}).get("section", "")),
                    ),
                )
                vector_id = int(cursor.lastrowid)
                connection.execute(
                    "UPDATE chunks SET vector_id = ? WHERE id = ?",
                    (vector_id, vector_id),
                )
                vector_ids.append(vector_id)
        return vector_ids

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        return self._document(row)

    def get_documents_by_status(self, statuses: list[str]) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents WHERE status IN ({placeholders}) ORDER BY updated_at",
                statuses,
            ).fetchall()
        documents: list[dict[str, Any]] = []
        for row in rows:
            document = self._document(row)
            if document is not None:
                documents.append(document)
        return documents

    def get_documents_by_ids(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        with self.connection() as connection:
            rows = connection.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
        documents = [self._document(row) for row in rows]
        return {str(document["doc_id"]): document for document in documents if document is not None}

    def list_chunks(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT vector_id, doc_id, chunk_id, text, section FROM chunks ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chunks_by_vector_ids(self, vector_ids: list[int]) -> list[dict[str, Any]]:
        if not vector_ids:
            return []
        placeholders = ",".join("?" for _ in vector_ids)
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT vector_id, doc_id, chunk_id, text, section "
                f"FROM chunks WHERE vector_id IN ({placeholders})",
                vector_ids,
            ).fetchall()
        return [dict(row) for row in rows]

    def get_vector_ids_for_doc(self, doc_id: str) -> list[int]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT vector_id FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
            ).fetchall()
        return [int(row["vector_id"]) for row in rows if row["vector_id"] is not None]

    def stats(self) -> dict[str, int]:
        with self.connection() as connection:
            documents = int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunks = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            failed = int(
                connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE status = 'failed'"
                ).fetchone()[0]
            )
        return {"documents": documents, "chunks": chunks, "failed": failed}

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

    def delete_document(self, doc_id: str) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

"""Restart recovery policy for durable ingestion state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rag_intel.state.sqlite_store import MetadataStore

Enqueue = Callable[[str, str, str, dict[str, Any]], Awaitable[None]]


async def recover_stuck_documents(store: MetadataStore, enqueue: Enqueue) -> dict[str, int]:
    """Requeue recoverable queued jobs and fail orphaned processing jobs."""
    counts = {"requeued": 0, "orphaned": 0}
    for document in store.get_documents_by_status(["queued"]):
        file_path = document.get("file_path")
        if file_path and Path(file_path).exists():
            await enqueue(
                str(file_path),
                str(document["filename"]),
                str(document["doc_id"]),
                dict(document.get("extra_metadata") or {}),
            )
            counts["requeued"] += 1
        else:
            store.update_document_state(
                str(document["doc_id"]),
                status="failed",
                current_stage="failed",
                stage_progress=100,
                error_message="source_missing_on_restart",
            )
            counts["orphaned"] += 1
    for document in store.get_documents_by_status(["processing"]):
        store.update_document_state(
            str(document["doc_id"]),
            status="failed",
            current_stage="failed",
            stage_progress=100,
            error_message="orphaned_on_restart",
        )
        counts["orphaned"] += 1
    return counts

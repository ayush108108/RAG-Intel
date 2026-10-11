"""Integration tests for local ephemeral database ingestion."""

import tempfile
from pathlib import Path

import pytest

from rag_intel.state.sqlite_store import MetadataStore


@pytest.mark.integration
def test_ephemeral_sqlite_store_ingestion():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_store.db"
        store = MetadataStore(db_path)

        doc_id = "test_doc_001"
        store.queue_document(
            doc_id,
            filename="sample.txt",
            source_type="text",
            file_path=str(Path(tmp_dir) / "sample.txt"),
        )

        doc = store.get_document(doc_id)
        assert doc is not None
        assert doc["doc_id"] == doc_id
        assert doc["status"] == "queued"

        chunks = [{"chunk_id": 0, "text": "Ephemeral test chunk."}]
        vector_ids = store.insert_chunks(doc_id, chunks)
        assert len(vector_ids) == 1

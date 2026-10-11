import pytest

from rag_intel.state.recovery import recover_stuck_documents
from rag_intel.state.sqlite_store import MetadataStore


def test_queue_document_persists_caller_metadata(tmp_path):
    store = MetadataStore(tmp_path / "docs.db")
    source = tmp_path / "paper.pdf"
    source.write_text("x")
    store.queue_document(
        "doc-1",
        filename="paper.pdf",
        source_type="pdf",
        file_path=str(source),
        extra_metadata={"title": "Recovered title", "category": "cs.AI"},
    )
    document = store.get_document("doc-1")
    assert document is not None
    assert document["extra_metadata"]["title"] == "Recovered title"
    assert document["extra_metadata"]["category"] == "cs.AI"
    assert document["extra_metadata"]["filename"] == "paper.pdf"


@pytest.mark.asyncio
async def test_restart_recovery_preserves_metadata_and_orphans_processing(tmp_path):
    store = MetadataStore(tmp_path / "docs.db")
    source = tmp_path / "queued.txt"
    source.write_text("recover me")
    store.queue_document(
        "queued",
        filename="queued.txt",
        source_type="text",
        file_path=str(source),
        extra_metadata={"title": "Durable title"},
    )
    store.upsert_document_stub(
        "processing",
        filename="lost.txt",
        source_type="text",
        title="lost",
        status="processing",
        current_stage="embedding",
        stage_progress=70,
        extra_metadata={"source_type": "text"},
    )
    seen = []

    async def enqueue(file_path, filename, doc_id, extra_metadata):
        seen.append((file_path, filename, doc_id, extra_metadata))

    counts = await recover_stuck_documents(store, enqueue)
    assert counts == {"requeued": 1, "orphaned": 1}
    assert seen[0][2] == "queued"
    assert seen[0][3]["title"] == "Durable title"
    processing = store.get_document("processing")
    assert processing is not None
    assert processing["status"] == "failed"
    assert processing["error_message"] == "orphaned_on_restart"

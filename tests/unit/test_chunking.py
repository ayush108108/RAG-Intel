"""Unit tests for text chunking without network calls."""

from rag_intel.chunking import chunk_text


def test_basic_text_chunking():
    text = "RAG-Intel provides deterministic chunking boundaries for document processing."
    chunks = chunk_text(text, chunk_size=40, chunk_overlap=10)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert len(chunk["text"]) <= 40


def test_empty_text_returns_empty_list():
    assert chunk_text("", chunk_size=100) == []
    assert chunk_text("   ", chunk_size=100) == []

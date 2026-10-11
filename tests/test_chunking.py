from rag_intel.chunking import chunk_text


def test_recursive_chunks_respect_total_size_budget():
    text = "alpha beta gamma delta epsilon " * 30
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=20)
    assert chunks
    assert max(len(chunk["text"]) for chunk in chunks) <= 80

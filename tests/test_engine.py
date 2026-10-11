from pathlib import Path

from rag_intel.config import EngineSettings
from rag_intel.engine import RAGIntelEngine
from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    ReciprocalRankFusion,
    RetrievalPolicy,
    TopKSelector,
)


def build_engine(tmp_path: Path) -> RAGIntelEngine:
    settings = EngineSettings(
        data_dir=tmp_path / "runtime",
        retrieval_mode="lexical",
        chunk_size=120,
        chunk_overlap=12,
        max_ingest_size_mb=2,
        embedding_model=None,
        embedding_batch_size=8,
        enable_reranking=False,
        reranker_model=None,
    )
    policy = RetrievalPolicy(
        candidate_depth=20,
        fusion=ReciprocalRankFusion(rank_constant=20),
        aggregate=MaxScoreAggregator(),
        select=TopKSelector(limit=5),
    )
    return RAGIntelEngine(settings, policy)


def test_local_engine_ingest_search_restart_and_delete(tmp_path: Path):
    source = tmp_path / "community.md"
    source.write_text(
        "# Community retrieval\n\nTransparent retrieval systems are easier to inspect and extend.",
        encoding="utf-8",
    )
    engine = build_engine(tmp_path)

    ingested = engine.ingest(source, doc_id="community-doc")
    assert ingested["chunks"] >= 1
    assert engine.status() == {"documents": 1, "chunks": ingested["chunks"], "failed": 0}

    result = engine.search("transparent retrieval")
    assert result[0]["doc_id"] == "community-doc"
    assert result[0]["matching_chunks"]

    restarted = build_engine(tmp_path)
    assert restarted.search("inspect")[0]["doc_id"] == "community-doc"
    assert restarted.delete("community-doc") is True
    assert restarted.delete("community-doc") is False
    assert restarted.status()["documents"] == 0

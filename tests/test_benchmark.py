import json
from pathlib import Path

import pytest

from rag_intel.benchmark import BenchmarkError, load_manifest, run_benchmark
from rag_intel.config import EngineSettings
from rag_intel.engine import RAGIntelEngine
from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    ReciprocalRankFusion,
    RetrievalPolicy,
    TopKSelector,
)


def _engine(tmp_path: Path) -> RAGIntelEngine:
    settings = EngineSettings(
        data_dir=tmp_path / "benchmark-state",
        retrieval_mode="lexical",
        chunk_size=640,
        chunk_overlap=64,
        max_ingest_size_mb=25,
        embedding_model=None,
        embedding_batch_size=16,
        enable_reranking=False,
        reranker_model=None,
    )
    policy = RetrievalPolicy(
        candidate_depth=40,
        fusion=ReciprocalRankFusion(rank_constant=60),
        aggregate=MaxScoreAggregator(),
        select=TopKSelector(limit=8),
    )
    return RAGIntelEngine(settings, policy)


def test_public_lexical_benchmark_meets_quality_floor(tmp_path: Path):
    manifest = load_manifest(Path(__file__).parents[1] / "benchmarks" / "lexical-smoke-v1.json")

    report = run_benchmark(_engine(tmp_path), manifest, k=3, runs=2, warmup_runs=1)

    assert report["corpus"]["documents"] == 4
    assert report["quality"]["recall@3"] == 1.0
    assert report["quality"]["ndcg@3"] == 1.0
    assert report["quality"]["reciprocal_rank"] == 1.0
    assert report["query_latency_ms"]["samples"] == 8


def test_manifest_rejects_unknown_relevance_ids(tmp_path: Path):
    source = tmp_path / "document.txt"
    source.write_text("benchmark evidence", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "name": "invalid",
                "documents": [{"doc_id": "known", "path": "document.txt"}],
                "queries": [
                    {
                        "query_id": "q1",
                        "text": "evidence",
                        "relevant_ids": ["unknown"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkError, match="unknown documents"):
        load_manifest(manifest)

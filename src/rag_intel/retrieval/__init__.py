"""Retrieval primitives and explicitly selected reference strategies."""

from rag_intel.retrieval.bm25 import BM25Retriever
from rag_intel.retrieval.fusion import l2_distance_to_similarity
from rag_intel.retrieval.reranker import rerank
from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    MeanScoreAggregator,
    ReciprocalRankFusion,
    RetrievalPolicy,
    TopKSelector,
)

__all__ = [
    "BM25Retriever",
    "MaxScoreAggregator",
    "MeanScoreAggregator",
    "ReciprocalRankFusion",
    "RetrievalPolicy",
    "TopKSelector",
    "l2_distance_to_similarity",
    "rerank",
]

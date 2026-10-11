"""Composable retrieval policy contracts and neutral reference strategies.

Nothing in this module is an implicit recommendation. Applications must select
and configure every strategy explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

ScoreMap = dict[int, float]
RankedScores = list[tuple[int, float]]


class FusionStrategy(Protocol):
    """Combine dense and sparse rankings into one chunk ranking."""

    def __call__(self, dense_scores: ScoreMap, sparse_scores: ScoreMap) -> RankedScores: ...


class DocumentAggregator(Protocol):
    """Reduce matching chunk scores to one document score."""

    def __call__(self, chunk_scores: Sequence[float]) -> float: ...


class CandidateSelector(Protocol):
    """Select document candidates after aggregation."""

    def __call__(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]: ...


class EmbeddingProvider(Protocol):
    """Provider-neutral dense embedding boundary."""

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, query: str) -> np.ndarray: ...


class Reranker(Protocol):
    """Provider-neutral candidate reranking boundary."""

    def __call__(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


ChunkingStrategy = Callable[[str], list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ReciprocalRankFusion:
    """Standard reciprocal-rank fusion over dense and sparse result order."""

    rank_constant: int

    def __post_init__(self) -> None:
        if self.rank_constant < 1:
            raise ValueError("rank_constant must be positive")

    def __call__(self, dense_scores: ScoreMap, sparse_scores: ScoreMap) -> RankedScores:
        fused: dict[int, float] = {}
        for scores in (dense_scores, sparse_scores):
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            for rank, (identifier, _) in enumerate(ranked, start=1):
                fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (self.rank_constant + rank)
        return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True, slots=True)
class MaxScoreAggregator:
    """Use the strongest matching chunk as the document score."""

    def __call__(self, chunk_scores: Sequence[float]) -> float:
        return max(chunk_scores, default=0.0)


@dataclass(frozen=True, slots=True)
class MeanScoreAggregator:
    """Use the arithmetic mean of matching chunk scores."""

    def __call__(self, chunk_scores: Sequence[float]) -> float:
        return sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0


@dataclass(frozen=True, slots=True)
class TopKSelector:
    """Keep an explicitly configured number of document candidates."""

    limit: int

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("candidate limit must be positive")

    def __call__(self, ranked: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return ranked[: self.limit]


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    """All consequential ranking choices required by the local engine."""

    candidate_depth: int
    fusion: FusionStrategy
    aggregate: DocumentAggregator
    select: CandidateSelector

    def __post_init__(self) -> None:
        if self.candidate_depth < 1:
            raise ValueError("candidate_depth must be positive")

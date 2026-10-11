"""Generic ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math


def precision_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    observed = ranked[:k]
    if not observed:
        return 0.0
    return sum(identifier in relevant for identifier in observed) / k


def recall_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return sum(identifier in relevant for identifier in ranked[:k]) / len(relevant)


def reciprocal_rank(relevant: set[str], ranked: list[str]) -> float:
    for rank, identifier in enumerate(ranked, start=1):
        if identifier in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant: set[str], ranked: list[str], k: int) -> float:
    """Return binary normalized discounted cumulative gain at *k*."""
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, identifier in enumerate(ranked[:k], start=1)
        if identifier in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal


def evaluate_ranking(relevant: set[str], ranked: list[str], k: int = 10) -> dict[str, float]:
    return {
        f"precision@{k}": precision_at_k(relevant, ranked, k),
        f"recall@{k}": recall_at_k(relevant, ranked, k),
        f"ndcg@{k}": ndcg_at_k(relevant, ranked, k),
        "reciprocal_rank": reciprocal_rank(relevant, ranked),
    }

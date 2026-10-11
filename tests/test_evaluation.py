import pytest

from rag_intel.evaluation import (
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_ranking_metrics_known_example():
    relevant = {"b", "d"}
    ranked = ["a", "b", "c", "d"]
    assert precision_at_k(relevant, ranked, 2) == 0.5
    assert recall_at_k(relevant, ranked, 2) == 0.5
    assert reciprocal_rank(relevant, ranked) == 0.5
    assert ndcg_at_k(relevant, ranked, 4) == pytest.approx(0.6509, rel=1e-3)
    assert evaluate_ranking(relevant, ranked, 2)["reciprocal_rank"] == 0.5


def test_ndcg_empty_relevance_and_invalid_k():
    assert ndcg_at_k(set(), ["a"], 1) == 0.0
    assert precision_at_k({"a"}, ["a"], 3) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="k must be positive"):
        ndcg_at_k({"a"}, ["a"], 0)

import pytest

from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    MeanScoreAggregator,
    ReciprocalRankFusion,
    TopKSelector,
)


def test_reciprocal_rank_fusion_combines_rankings():
    fusion = ReciprocalRankFusion(rank_constant=10)
    ranking = fusion({1: 0.9, 2: 0.5}, {2: 9.0, 3: 2.0})

    assert ranking[0][0] == 2
    assert {identifier for identifier, _ in ranking} == {1, 2, 3}


def test_reference_strategies_require_explicit_valid_settings():
    with pytest.raises(ValueError, match="rank_constant"):
        ReciprocalRankFusion(rank_constant=0)
    with pytest.raises(ValueError, match="candidate limit"):
        TopKSelector(limit=0)

    assert MaxScoreAggregator()([0.1, 0.8, 0.4]) == 0.8
    assert MeanScoreAggregator()([0.2, 0.4]) == pytest.approx(0.3)
    assert TopKSelector(limit=1)([("a", 1.0), ("b", 0.5)]) == [("a", 1.0)]

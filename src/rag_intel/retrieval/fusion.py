"""Generic score conversions shared by retrieval adapters."""

from __future__ import annotations


def l2_distance_to_similarity(distance: float) -> float:
    """Convert squared-L2 distance of normalized vectors into [0, 1] similarity."""
    similarity = 1.0 - float(distance) / 2.0
    return max(0.0, min(1.0, similarity))

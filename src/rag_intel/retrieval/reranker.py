"""Cross-encoder reranker using sentence-transformers.

The model name is resolved from the RAG_INTEL_RERANKER_MODEL environment
variable (default: cross-encoder/ms-marco-MiniLM-L-6-v2).

Usage::

    from rag_intel.retrieval.reranker import rerank

    candidates = [{"text": "...", ...}, ...]
    reranked = rerank(query, candidates)
    # candidates now have a "rerank_score" key and are sorted best-first.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_reranker = None
_reranker_name: str | None = None

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class CrossEncoderReranker:
    """Explicit cross-encoder adapter for the local engine."""

    model_name: str
    text_key: str = "text"
    _instance: Any | None = field(default=None, init=False, repr=False)

    def _load(self):
        if self._instance is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("Reranking requires the 'reranking' extra") from exc
            self._instance = CrossEncoder(self.model_name)
        return self._instance

    def __call__(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        scores = self._load().predict(
            [(query, str(candidate[self.text_key])) for candidate in candidates]
        )
        enriched = [dict(candidate) for candidate in candidates]
        for candidate, score in zip(enriched, scores, strict=True):
            candidate["rerank_score"] = float(score)
        return sorted(enriched, key=lambda item: item["rerank_score"], reverse=True)


def get_reranker():
    """Return a cached CrossEncoder, reloading if the model name changed."""
    global _reranker, _reranker_name
    target = os.environ.get("RAG_INTEL_RERANKER_MODEL", DEFAULT_RERANKER)
    if _reranker is None or _reranker_name != target:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Reranking requires sentence-transformers. Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading reranker model: %s", target)
        _reranker = CrossEncoder(target)
        _reranker_name = target
        logger.info("Reranker loaded")
    return _reranker


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    text_key: str = "text",
) -> list[dict[str, Any]]:
    """Score and sort *candidates* by cross-encoder relevance to *query*.

    Each candidate dict must contain a ``text_key`` field (default ``"text"``).
    The function adds a ``"rerank_score"`` key to every candidate and returns
    the list sorted best-first.

    If the reranker model cannot be loaded the original order is preserved and
    a warning is emitted rather than raising.
    """
    if not candidates:
        return candidates

    try:
        reranker = get_reranker()
    except Exception as exc:
        logger.warning("Reranker unavailable, skipping: %s", exc)
        return candidates

    try:
        pairs = [(query, c[text_key]) for c in candidates]
        scores = reranker.predict(pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        logger.debug("Reranked %d candidates", len(candidates))
    except Exception as exc:  # noqa: BLE001
        logger.error("Reranking failed, returning original order: %s", exc)

    return candidates

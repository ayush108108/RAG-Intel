"""Lazy-loading dense embedding wrapper using sentence-transformers.

The model name is resolved from the RAG_INTEL_EMBEDDING_MODEL environment
variable (default: all-MiniLM-L6-v2). Embeddings are L2-normalised so that
squared-L2 distance is equivalent to cosine distance.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_model = None
_model_name: str | None = None

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32


@dataclass
class SentenceTransformerProvider:
    """Explicit SentenceTransformers adapter for :class:`RAGIntelEngine`."""

    model_name: str
    batch_size: int
    _instance: Any | None = field(default=None, init=False, repr=False)

    def _load(self):
        if self._instance is None:
            try:
                sentence_transformers = __import__("sentence_transformers")
            except ImportError as exc:
                raise RuntimeError("Dense embedding requires the 'semantic' extra") from exc
            self._instance = sentence_transformers.SentenceTransformer(self.model_name)
        return self._instance

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        result = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(result, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        if not query.strip():
            raise ValueError("query must not be empty")
        return self.embed_documents([query])[0]


def get_embedder():
    """Return a cached SentenceTransformer, reloading if the model name changed."""
    global _model, _model_name
    target = os.environ.get("RAG_INTEL_EMBEDDING_MODEL", DEFAULT_MODEL)
    if _model is None or _model_name != target:
        try:
            sentence_transformers = __import__("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "Dense embedding requires sentence-transformers. "
                "Run: pip install sentence-transformers"
            ) from exc
        logger.info("Loading embedding model: %s", target)
        t0 = time.perf_counter()
        _model = sentence_transformers.SentenceTransformer(target)
        _model_name = target
        logger.info("Embedding model loaded in %.2fs", time.perf_counter() - t0)
    return _model


def embed_texts(
    texts: Sequence[str],
    *,
    batch_size: int | None = None,
) -> np.ndarray:
    """Embed a list of strings, returning an (N, D) float32 array.

    Embeddings are L2-normalised so inner-product equals cosine similarity.
    """
    if not texts:
        return np.empty((0,), dtype=np.float32)

    bs = batch_size or int(os.environ.get("RAG_INTEL_BATCH_SIZE", DEFAULT_BATCH_SIZE))
    model = get_embedder()

    t0 = time.perf_counter()
    result: np.ndarray = model.encode(
        list(texts),
        batch_size=bs,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.debug("Embedded %d texts in %.3fs", len(texts), time.perf_counter() - t0)
    return result.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, returning a 1-D float32 vector."""
    return embed_texts([query])[0]


def embedding_dim() -> int:
    """Return the output dimension of the current embedding model."""
    model = get_embedder()
    sample = model.encode(["probe"], normalize_embeddings=True, convert_to_numpy=True)
    return int(sample.shape[1])

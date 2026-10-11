"""Validated persistent FAISS store with stable external IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class FaissStore:
    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        embedding_model: str,
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model = embedding_model
        self.index: Any | None = None
        self.ids: list[int] = []
        self.dimension: int | None = None

    @property
    def count(self) -> int:
        """Number of vectors tracked by the persisted index."""
        return len(self.ids)

    @staticmethod
    def _faiss():
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("FAISS support requires the 'vector' extra") from exc
        return faiss

    def add(self, embeddings: np.ndarray, ids: list[int]) -> None:
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")
        if len(vectors) != len(ids):
            raise ValueError("embeddings and ids must have the same length")
        if len(set(ids)) != len(ids) or set(ids).intersection(self.ids):
            raise ValueError("vector ids must be unique")
        if self.index is None and self.index_path.exists():
            self.load()
        faiss = self._faiss()
        if self.index is None:
            self.dimension = int(vectors.shape[1])
            self.index = faiss.IndexIDMap2(faiss.IndexFlatL2(self.dimension))
        elif int(self.index.d) != int(vectors.shape[1]):
            raise ValueError(
                f"embedding dimension mismatch: index={self.index.d}, incoming={vectors.shape[1]}"
            )
        self.index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
        self.ids.extend(ids)
        self.persist()

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.index is None:
            self.load()
        if self.index is None or self.index.ntotal == 0 or top_k < 1:
            return []
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.index.d:
            raise ValueError("query embedding dimension does not match index")
        distances, identifiers = self.index.search(query, min(top_k, self.index.ntotal))
        return [
            (int(identifier), float(distance))
            for identifier, distance in zip(identifiers[0], distances[0], strict=True)
            if identifier != -1
        ]

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        if self.index is None:
            self.load()
        if self.index is None:
            return
        remove_set = set(ids)
        self.index.remove_ids(np.asarray(sorted(remove_set), dtype=np.int64))
        self.ids = [identifier for identifier in self.ids if identifier not in remove_set]
        self.persist()

    def delete(self, ids: list[int]) -> None:
        """Backward-compatible alias for :meth:`remove`."""
        self.remove(ids)

    def persist(self) -> None:
        if self.index is None:
            return
        faiss = self._faiss()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        index_tmp = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        metadata_tmp = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
        faiss.write_index(self.index, str(index_tmp))
        metadata_tmp.write_text(
            json.dumps(
                {
                    "ids": self.ids,
                    "total": len(self.ids),
                    "dimension": int(self.index.d),
                    "embedding_model": self.embedding_model,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        os.replace(index_tmp, self.index_path)
        os.replace(metadata_tmp, self.metadata_path)

    def load(self) -> None:
        if not self.index_path.exists() and not self.metadata_path.exists():
            self.index = None
            self.ids = []
            self.dimension = None
            return
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise RuntimeError("incomplete persisted vector state")
        faiss = self._faiss()
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != self.embedding_model:
            raise ValueError("Index model mismatch: persisted index uses a different model")
        index = faiss.read_index(str(self.index_path))
        ids = [int(identifier) for identifier in metadata.get("ids", [])]
        if int(index.ntotal) != len(ids) or int(metadata.get("total", -1)) != len(ids):
            raise RuntimeError("persisted vector count does not match metadata")
        if int(index.d) != int(metadata.get("dimension", -1)):
            raise RuntimeError("persisted vector dimension does not match metadata")
        self.index = index
        self.ids = ids
        self.dimension = int(index.d)

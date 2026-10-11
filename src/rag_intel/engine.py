"""Small synchronous composition root for local retrieval workflows."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rag_intel.chunking import chunk_text
from rag_intel.config import EngineSettings
from rag_intel.index.faiss_store import FaissStore
from rag_intel.ingestion import route_file
from rag_intel.normalization import normalize_text
from rag_intel.retrieval.bm25 import BM25Retriever
from rag_intel.retrieval.fusion import l2_distance_to_similarity
from rag_intel.retrieval.strategies import EmbeddingProvider, Reranker, RetrievalPolicy
from rag_intel.state.sqlite_store import MetadataStore


class EngineError(RuntimeError):
    """Raised when a local engine operation cannot complete consistently."""


class RAGIntelEngine:
    """Compose public retrieval primitives without production orchestration.

    Ranking policy is always injected. Dense retrieval is enabled only when an
    embedding provider is supplied and the configured mode is ``hybrid``.
    """

    def __init__(
        self,
        settings: EngineSettings,
        policy: RetrievalPolicy,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        reranker: Reranker | None = None,
        chunker: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        if settings.retrieval_mode == "hybrid" and embedding_provider is None:
            raise ValueError("hybrid retrieval requires an embedding provider")
        if settings.enable_reranking and reranker is None:
            raise ValueError("reranking is enabled but no reranker was supplied")

        self.settings = settings
        self.policy = policy
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.chunker = chunker or self._chunk
        self.store = MetadataStore(settings.database_path)
        self.sparse = BM25Retriever()
        self.vectors = (
            FaissStore(
                settings.index_path,
                settings.index_metadata_path,
                embedding_model=embedding_provider.model_name,
            )
            if embedding_provider is not None
            else None
        )
        self._rebuild_sparse()

    def _chunk(self, text: str) -> list[dict[str, Any]]:
        return chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def _rebuild_sparse(self) -> None:
        chunks = self.store.list_chunks()
        self.sparse.index(
            [str(chunk["text"]) for chunk in chunks],
            [int(chunk["vector_id"]) for chunk in chunks],
        )

    def ingest(self, file_path: str | Path, *, doc_id: str | None = None) -> dict[str, Any]:
        """Extract, chunk, and index one document with visible failure state."""
        path = Path(file_path)
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
        except OSError as exc:
            raise EngineError(f"cannot read input file: {path}") from exc
        if size_mb > self.settings.max_ingest_size_mb:
            raise EngineError(
                f"input is {size_mb:.2f} MB; configured limit is "
                f"{self.settings.max_ingest_size_mb:.2f} MB"
            )

        document = route_file(path, doc_id=doc_id)
        identifier = str(document["doc_id"])
        previous_ids = self.store.get_vector_ids_for_doc(identifier)
        if previous_ids and self.vectors is not None:
            self.vectors.remove(previous_ids)
        self.store.delete_chunks_for_doc(identifier)
        metadata = dict(document["metadata"])
        self.store.queue_document(
            identifier,
            filename=str(metadata["filename"]),
            source_type=str(metadata["source_type"]),
            file_path=str(path),
            extra_metadata=metadata,
        )
        self.store.update_document_state(
            identifier, status="processing", current_stage="chunking", stage_progress=25
        )

        vector_ids: list[int] = []
        try:
            chunks = self.chunker(normalize_text(str(document["content"])))
            if not chunks:
                raise EngineError("extraction produced no indexable text")
            vector_ids = self.store.insert_chunks(identifier, chunks)
            if self.vectors is not None and self.embedding_provider is not None:
                embeddings = self.embedding_provider.embed_documents(
                    [str(chunk["text"]) for chunk in chunks]
                )
                self.vectors.add(embeddings, vector_ids)
            self.store.update_document_state(
                identifier,
                status="completed",
                current_stage="indexed",
                stage_progress=100,
                error_message=None,
            )
            self._rebuild_sparse()
            return {
                "doc_id": identifier,
                "filename": metadata["filename"],
                "source_type": metadata["source_type"],
                "chunks": len(chunks),
            }
        except Exception as exc:
            if vector_ids and self.vectors is not None:
                self.vectors.remove(vector_ids)
            self.store.delete_chunks_for_doc(identifier)
            self.store.update_document_state(
                identifier,
                status="failed",
                current_stage="failed",
                stage_progress=100,
                error_message=str(exc),
            )
            self._rebuild_sparse()
            if isinstance(exc, EngineError):
                raise
            raise EngineError(f"ingestion failed for {path.name}: {exc}") from exc

    def search(self, query: str) -> list[dict[str, Any]]:
        """Search chunks, aggregate documents, and optionally rerank results."""
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        sparse_hits = self.sparse.search(query, self.policy.candidate_depth)
        sparse_scores = dict(sparse_hits)
        dense_scores: dict[int, float] = {}
        if self.vectors is not None and self.embedding_provider is not None:
            query_vector = self.embedding_provider.embed_query(query)
            dense_scores = {
                identifier: l2_distance_to_similarity(distance)
                for identifier, distance in self.vectors.search(
                    query_vector, self.policy.candidate_depth
                )
            }
        fused = self.policy.fusion(dense_scores, sparse_scores)
        if not fused:
            return []

        score_by_vector = dict(fused)
        chunk_rows = self.store.get_chunks_by_vector_ids(list(score_by_vector))
        scores_by_doc: dict[str, list[float]] = defaultdict(list)
        chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for chunk in chunk_rows:
            vector_id = int(chunk["vector_id"])
            doc_id = str(chunk["doc_id"])
            score = score_by_vector[vector_id]
            scores_by_doc[doc_id].append(score)
            chunks_by_doc[doc_id].append(
                {
                    "vector_id": vector_id,
                    "text": chunk["text"],
                    "section": chunk["section"],
                    "score": score,
                }
            )

        ranked_docs = sorted(
            ((doc_id, self.policy.aggregate(scores)) for doc_id, scores in scores_by_doc.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        selected = self.policy.select(ranked_docs)
        documents = self.store.get_documents_by_ids([doc_id for doc_id, _ in selected])
        results: list[dict[str, Any]] = []
        for doc_id, score in selected:
            metadata = documents.get(doc_id, {})
            matching_chunks = sorted(
                chunks_by_doc[doc_id], key=lambda chunk: chunk["score"], reverse=True
            )
            results.append(
                {
                    "doc_id": doc_id,
                    "title": metadata.get("title") or metadata.get("filename") or doc_id,
                    "filename": metadata.get("filename") or doc_id,
                    "source_type": metadata.get("source_type") or "unknown",
                    "score": score,
                    "matching_chunks": matching_chunks,
                    "text": "\n\n".join(str(chunk["text"]) for chunk in matching_chunks),
                }
            )
        if self.reranker is not None:
            results = self.reranker(query, results)
        return results

    def delete(self, doc_id: str) -> bool:
        """Delete a document and its vectors from every local store."""
        if self.store.get_document(doc_id) is None:
            return False
        vector_ids = self.store.get_vector_ids_for_doc(doc_id)
        if self.vectors is not None:
            self.vectors.remove(vector_ids)
        self.store.delete_document(doc_id)
        self._rebuild_sparse()
        return True

    def status(self) -> dict[str, int]:
        """Return local metadata counts suitable for CLI output."""
        return self.store.stats()

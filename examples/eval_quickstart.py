"""Offline, deterministic retrieval evaluation on the bundled sample corpus.

Runs the public library end to end: bounded chunking, BM25 lexical retrieval,
deterministic stand-in dense ordering, reciprocal-rank fusion, and
precision@k / recall@k / reciprocal-rank metrics.

No network access, no embedding model, no API key, no configuration file.
The dense ordering is a documented stand-in (token-overlap count) so the
fusion and metric plumbing can be verified reproducibly; use the engine with
a real embedding provider (see README) to measure actual retrieval quality.

Usage:
    uv run python examples/eval_quickstart.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rag_intel.chunking import chunk_text
from rag_intel.evaluation import evaluate_ranking
from rag_intel.retrieval.bm25 import BM25Retriever
from rag_intel.retrieval.strategies import ReciprocalRankFusion

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "sample_data" / "eval"
QRELS_PATH = CORPUS_DIR / "qrels.json"
CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
TOP_K = 5

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def stand_in_dense_order(
    query_tokens: set[str], chunks: dict[int, dict]
) -> list[tuple[int, float]]:
    """Deterministic stand-in for dense ordering (no model, no network)."""
    scored: list[tuple[int, float]] = []
    for chunk_id, chunk in chunks.items():
        chunk_tokens = set(tokens(chunk["text"]))
        overlap = len(query_tokens & chunk_tokens)
        if overlap:
            scored.append((chunk_id, float(overlap)))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def main() -> None:
    qrels = json.loads(QRELS_PATH.read_text())["queries"]

    documents: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.glob("doc_*.txt")):
        documents[path.stem] = path.read_text()

    chunks: dict[int, dict] = {}
    chunk_doc: dict[int, str] = {}
    next_chunk_id = 0
    for doc_id, text in documents.items():
        for chunk in chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
            chunks[next_chunk_id] = chunk
            chunk_doc[next_chunk_id] = doc_id
            next_chunk_id += 1

    retriever = BM25Retriever()
    retriever.index([c["text"] for c in chunks.values()], list(chunks.keys()))
    fusion = ReciprocalRankFusion(rank_constant=60)

    header = f"{'query':<52} {'p@1':>6} {f'p@{TOP_K}':>6} {f'r@{TOP_K}':>6} {'MRR':>6}"
    print(f"corpus: {len(documents)} documents, {len(chunks)} chunks")
    print(f"queries: {len(qrels)} (each query has exactly one relevant document)\n")
    print(header)
    print("-" * 80)

    rows = []
    for query, relevant_docs in qrels.items():
        query_tokens = set(tokens(query))
        sparse = dict(retriever.search(query, top_k=TOP_K * 4))
        dense = dict(stand_in_dense_order(query_tokens, chunks))
        ranked = fusion(dense, sparse)

        ranked_docs: list[str] = []
        for chunk_id, _ in ranked:
            doc_id = chunk_doc[chunk_id]
            if doc_id not in ranked_docs:
                ranked_docs.append(doc_id)

        metrics = evaluate_ranking(set(relevant_docs), ranked_docs, k=TOP_K)
        metrics["precision@1"] = evaluate_ranking(
            set(relevant_docs), ranked_docs, k=1
        )["precision@1"]
        rows.append(metrics)
        print(
            f"{query:<52}"
            f" {metrics['precision@1']:>6.3f}"
            f" {metrics[f'precision@{TOP_K}']:>6.3f}"
            f" {metrics[f'recall@{TOP_K}']:>6.3f}"
            f" {metrics['reciprocal_rank']:>6.3f}"
        )
    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }
    print("-" * 80)
    print(
        f"{'mean':<52}"
        f" {means['precision@1']:>6.3f}"
        f" {means[f'precision@{TOP_K}']:>6.3f}"
        f" {means[f'recall@{TOP_K}']:>6.3f}"
        f" {means['reciprocal_rank']:>6.3f}"
    )


if __name__ == "__main__":
    main()

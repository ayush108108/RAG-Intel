"""Foundation v0.4 BEIR qualification runner for RAG-Intel."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import statistics
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from rag_intel.config import load_config
from rag_intel.embedding import SentenceTransformerProvider
from rag_intel.engine import RAGIntelEngine
from rag_intel.retrieval.reranker import CrossEncoderReranker


BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"


def download_nfcorpus(root: Path) -> Path:
    archive = root / "nfcorpus.zip"
    urllib.request.urlretrieve(BEIR_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return root / "nfcorpus"


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["_id"])] = row
    return rows


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            result.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return result


def dcg(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    import math

    total = 0.0
    for rank, doc_id in enumerate(ranking[:k], start=1):
        rel = relevance.get(doc_id, 0)
        total += (2**rel - 1) / math.log2(rank + 1)
    return total


def metrics(ranking: list[str], relevance: dict[str, int], k: int) -> dict[str, float]:
    relevant = {doc_id for doc_id, score in relevance.items() if score > 0}
    top = ranking[:k]
    hits = [doc_id for doc_id in top if doc_id in relevant]
    precision = len(hits) / k
    recall = len(hits) / len(relevant) if relevant else 0.0
    rr = next((1.0 / rank for rank, doc_id in enumerate(ranking, 1) if doc_id in relevant), 0.0)
    ideal = sorted(relevance.values(), reverse=True)
    ideal_dcg = 0.0
    import math
    for rank, rel in enumerate(ideal[:k], start=1):
        ideal_dcg += (2**rel - 1) / math.log2(rank + 1)
    ndcg = dcg(ranking, relevance, k) / ideal_dcg if ideal_dcg else 0.0
    return {"precision": precision, "recall": recall, "mrr": rr, "ndcg": ndcg}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmarks/foundation_v04_config.toml")
    parser.add_argument("--output", default="benchmarks/results/foundation-v04-rag-intel.json")
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--corpus-limit", type=int, default=1200)
    args = parser.parse_args()

    loaded = load_config(args.config)
    if loaded.settings.data_dir.exists():
        shutil.rmtree(loaded.settings.data_dir)

    provider = SentenceTransformerProvider(
        model_name=loaded.settings.embedding_model or "BAAI/bge-base-en-v1.5",
        batch_size=loaded.settings.embedding_batch_size,
    )
    reranker = CrossEncoderReranker(model_name=loaded.settings.reranker_model or "")
    engine = RAGIntelEngine(
        loaded.settings,
        loaded.policy,
        embedding_provider=provider,
        reranker=reranker,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        dataset = download_nfcorpus(root)
        corpus = read_jsonl(dataset / "corpus.jsonl")
        queries = read_jsonl(dataset / "queries.jsonl")
        qrels = read_qrels(dataset / "qrels" / "test.tsv")

        rng = random.Random(20260903)
        all_query_ids = sorted(qrels)
        query_ids = sorted(rng.sample(all_query_ids, min(args.query_limit, len(all_query_ids))))
        relevant_ids = {
            doc_id
            for qid in query_ids
            for doc_id, score in qrels[qid].items()
            if score > 0
        }
        if args.corpus_limit <= 0:
            selected_ids = sorted(corpus)
        else:
            filler = [doc_id for doc_id in sorted(corpus) if doc_id not in relevant_ids]
            needed = max(0, args.corpus_limit - len(relevant_ids))
            sampled_filler = rng.sample(filler, min(needed, len(filler)))
            selected_ids = sorted(relevant_ids | set(sampled_filler))

        docs_dir = root / "docs"
        docs_dir.mkdir()

        # Quality indexing deliberately defers the sparse rebuild so retrieval quality
        # is not dominated by repeated O(corpus) BM25 rebuilds during every ingest.
        native_rebuild = engine._rebuild_sparse
        engine._rebuild_sparse = lambda: None

        ingest_started = time.perf_counter()
        for doc_id in selected_ids:
            row = corpus[doc_id]
            path = docs_dir / f"{doc_id}.txt"
            path.write_text(f"{row.get('title', '')}\n\n{row.get('text', '')}", encoding="utf-8")
            engine.ingest(path, doc_id=doc_id)
        native_rebuild()
        ingest_seconds = time.perf_counter() - ingest_started

        reports = []
        latencies = []
        aggregate = {f"{metric}@{k}": [] for metric in ("precision", "recall", "ndcg") for k in (5, 10, 20)}
        aggregate["mrr"] = []

        for query_id in query_ids:
            query = str(queries[query_id].get("text") or queries[query_id].get("title") or "")
            started = time.perf_counter()
            results = engine.search(query)
            latencies.append((time.perf_counter() - started) * 1000)
            ranking = [str(item["doc_id"]) for item in results]
            row_metrics = {}
            for k in (5, 10, 20):
                current = metrics(ranking, qrels[query_id], k)
                for metric in ("precision", "recall", "ndcg"):
                    key = f"{metric}@{k}"
                    aggregate[key].append(current[metric])
                    row_metrics[key] = current[metric]
            current_mrr = metrics(ranking, qrels[query_id], 20)["mrr"]
            aggregate["mrr"].append(current_mrr)
            row_metrics["mrr"] = current_mrr
            reports.append({"query_id": query_id, "query": query, "ranking": ranking[:20], "metrics": row_metrics})

        first_doc = selected_ids[0]
        delete_ok = engine.delete(first_doc)
        delete_search_clean = all(item["doc_id"] != first_doc for item in engine.search(str(queries[query_ids[0]].get("text") or "")))

        summary = {key: statistics.mean(values) if values else 0.0 for key, values in aggregate.items()}
        ordered = sorted(latencies)
        def pct(frac: float) -> float:
            return ordered[min(len(ordered)-1, max(0, int(frac * len(ordered)) - 1))] if ordered else 0.0

        output = {
            "engine": "rag-intel",
            "profile": "foundation-v0.4-parity",
            "intent": "oss-reference-engine",
            "dataset": "BEIR/nfcorpus:test deterministic qualification subset",
            "query_count": len(query_ids),
            "corpus_count": len(selected_ids),
            "config": {
                "embedding": loaded.settings.embedding_model,
                "reranker": loaded.settings.reranker_model,
                "chunk_size": loaded.settings.chunk_size,
                "chunk_overlap": loaded.settings.chunk_overlap,
                "candidate_depth": loaded.policy.candidate_depth,
                "fusion": "rrf",
                "quality_indexing_mode": "native ingest with BM25 rebuild deferred until corpus complete",
            },
            "quality": summary,
            "performance": {
                "ingest_seconds": ingest_seconds,
                "query_p50_ms": statistics.median(latencies) if latencies else 0.0,
                "query_p95_ms": pct(0.95),
                "query_max_ms": max(latencies) if latencies else 0.0,
            },
            "correctness": {
                "delete_returned_true": delete_ok,
                "deleted_doc_absent_after_search": delete_search_clean,
            },
            "queries": reports,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(json.dumps(output["quality"], indent=2))


if __name__ == "__main__":
    main()

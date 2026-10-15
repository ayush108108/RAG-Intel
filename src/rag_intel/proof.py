"""Offline, passage-level development evaluation and demo of the real lexical core.

Run from a source checkout: python -m rag_intel.proof --help.
No question labels are used by indexing or search.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import platform
import random
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag_intel.config import EngineSettings
from rag_intel.engine import RAGIntelEngine
from rag_intel.normalization import normalize_text
from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    ReciprocalRankFusion,
    RetrievalPolicy,
    TopKSelector,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "benchmarks/python-docs-v1"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_dataset(dataset: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    texts = {}
    for doc in manifest["documents"]:
        data = (dataset / doc["path"]).read_bytes()
        if digest(data) != doc["sha256"]:
            raise ValueError(f"corpus checksum mismatch: {doc['id']}")
        texts[doc["id"]] = normalize_text(data.decode("utf-8"))
    questions = [
        json.loads(line)
        for line in (dataset / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen = set()
    for question in questions:
        if question["id"] in seen:
            raise ValueError("duplicate question ID")
        seen.add(question["id"])
        if bool(question["gold"]) != question["answerable"]:
            raise ValueError("answerability and gold passages disagree")
        for gold in question["gold"]:
            text = texts[gold["doc_id"]]
            if not gold["quote"] or text[gold["start"] : gold["end"]] != gold["quote"]:
                raise ValueError(f"invalid gold span: {question['id']}")
    return manifest, questions


def fingerprint(dataset: Path) -> str:
    """Include source, dependencies, manifest and labels, never a path or mtime."""
    files = sorted((ROOT / "src/rag_intel").rglob("*.py"))
    files += [ROOT / "uv.lock", dataset / "manifest.json", dataset / "questions.jsonl"]
    return digest(
        b"".join(path.name.encode() + path.read_bytes().replace(b"\r\n", b"\n") for path in files)
    )


class Collection:
    def __init__(self, dataset: Path, variant: str, cache: Path) -> None:
        self.manifest, _ = read_dataset(dataset)
        self.settings = self.manifest[variant]
        self.variant = variant
        key = digest((fingerprint(dataset) + variant).encode())[:24]
        directory = cache / key
        settings = EngineSettings(
            data_dir=directory,
            retrieval_mode="lexical",
            chunk_size=self.settings["chunk_size"],
            chunk_overlap=self.settings["chunk_overlap"],
            max_ingest_size_mb=1,
            embedding_model=None,
            embedding_batch_size=16,
            enable_reranking=False,
            reranker_model=None,
        )
        policy = RetrievalPolicy(
            candidate_depth=40,
            fusion=ReciprocalRankFusion(rank_constant=60),
            aggregate=MaxScoreAggregator(),
            select=TopKSelector(limit=8),
        )
        self.engine = RAGIntelEngine(settings, policy)
        marker = directory / "complete.json"
        self.cache_hit = marker.exists()
        if not self.cache_hit:
            # Incomplete caches are reingested with stable document identities.
            for doc in self.manifest["documents"]:
                self.engine.ingest(dataset / doc["path"], doc_id=doc["id"])
            marker.write_text(json.dumps({"fingerprint": key}), encoding="utf-8")
        self.documents = {doc["id"]: doc for doc in self.manifest["documents"]}
        self.chunks = {}
        for row in self.engine.store.list_chunks():
            text = str(row["text"])
            self.chunks[int(row["vector_id"])] = {
                "id": f"{row['doc_id']}:{row['chunk_id']}:{digest(text.encode())[:12]}",
                "doc_id": row["doc_id"],
                "text": text,
                "source_url": self.documents[row["doc_id"]]["source_url"],
            }

    def search(self, query: str, limit: int | None = None) -> list[dict]:
        if not query.strip():
            raise ValueError("query must not be empty")
        # The repair is query-only stopword removal. No synonyms or gold lookup.
        stopwords = set(self.settings["query_stopwords"])
        tokens = re.findall(r"\b\w+\b", query.lower())
        effective = " ".join(token for token in tokens if token not in stopwords)
        hits = self.engine.sparse.search(effective, len(self.chunks) if limit is None else limit)
        return [{**self.chunks[identifier], "score": score} for identifier, score in hits]


def relevant(chunk: dict, gold: list[dict]) -> bool:
    """Require the complete supporting passage, not merely its parent document."""
    return any(g["doc_id"] == chunk["doc_id"] and g["quote"] in chunk["text"] for g in gold)


def assess(ranked: list[dict], gold: list[dict]) -> dict:
    rank = next((i for i, chunk in enumerate(ranked, 1) if relevant(chunk, gold)), None)
    return {
        "first_gold_rank": rank,
        "reciprocal_rank": 1 / rank if rank else 0.0,
        **{f"hit@{k}": rank is not None and rank <= k for k in (1, 3, 5)},
    }


def git_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "execution_commit": git("rev-parse", "HEAD"),
        "worktree_dirty": bool(git("status", "--porcelain")),
    }


def evaluate(dataset: Path, variant: str, cache: Path) -> dict:
    started = time.perf_counter()
    manifest, questions = read_dataset(dataset)
    random.seed(manifest["seed"])
    collection = Collection(dataset, variant, cache)
    rows = []
    for question in questions:
        ranked = collection.search(question["question"])
        coverage = [c["id"] for c in collection.chunks.values() if relevant(c, question["gold"])]
        rows.append(
            {
                **question,
                **assess(ranked, question["gold"]),
                "supporting_chunk_ids": coverage,
                "top5": [dict(c, gold_hit=relevant(c, question["gold"])) for c in ranked[:5]],
                "retrieved_count": len(ranked),
            }
        )
    positives = [r for r in rows if r["answerable"]]
    negatives = [r for r in rows if not r["answerable"]]
    metrics = {
        f"hit@{k}": sum(r[f"hit@{k}"] for r in positives) / len(positives) for k in (1, 3, 5)
    }
    metrics["MRR"] = sum(r["reciprocal_rank"] for r in positives) / len(positives)
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **git_info(),
        "source_and_dataset_sha256": fingerprint(dataset),
        "corpus": manifest,
        "variant": variant,
        "settings": {
            **manifest[variant],
            "retriever": "RAGIntelEngine.sparse / BM25Retriever",
            "k1": 1.5,
            "b": 0.75,
            "tokenizer": r"\b\w+\b lowercase",
            "ranking_unit": "chunk (before document aggregation and RRF)",
            "tie_break": "fixed corpus ingestion order, then chunk order",
            "embedding_model": None,
            "reranker": None,
            "generation": False,
            "seed": manifest["seed"],
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": {p: version(p) for p in ("rag-intel", "numpy", "typer")},
            "uv_lock_sha256": digest((ROOT / "uv.lock").read_bytes()),
        },
        "cache_hit": collection.cache_hit,
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {
            "documents": len(manifest["documents"]),
            "chunks": len(collection.chunks),
            "answerable": len(positives),
            "negative": len(negatives),
            "answerable_without_covering_chunk": sum(
                not r["supporting_chunk_ids"] for r in positives
            ),
        },
        "metrics_answerable_only": metrics,
        "negative_diagnostics": {
            "returned_any_passage": sum(bool(r["top5"]) for r in negatives),
            "total": len(negatives),
            "interpretation": (
                "Retrieval is not answerability detection. Returned passages may not answer "
                "the question; no abstention or generation claim."
            ),
        },
        "questions": rows,
    }


def markdown(report: dict) -> str:
    metrics = report["metrics_answerable_only"]
    lines = [
        f"# {report['variant']} — Python documentation development eval",
        "",
        f"Date (UTC): {report['generated_at_utc']}",
        f"Execution commit: `{report['execution_commit']}`",
        f"Working tree dirty: {report['worktree_dirty']}",
        f"Source/dataset SHA256: `{report['source_and_dataset_sha256']}`",
        "",
        "All 36 answerable questions are the denominator; negatives are separate.",
        "Full passage containment is required. MRR uses the full positive-score ranking.",
        "This is a development set used to inspect the repair, with no held-out test set.",
        "No embeddings, reranking, generated answers, or generation-quality claims.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {key} | {value:.6f} |" for key, value in metrics.items()],
        "",
        f"Counts: `{json.dumps(report['counts'], sort_keys=True)}`",
        f"Settings: `{json.dumps(report['settings'], sort_keys=True)}`",
        f"Negative diagnostics: `{json.dumps(report['negative_diagnostics'], sort_keys=True)}`",
        "",
        "See adjacent JSON for exact gold spans, scores, sources and top-five text.",
        "",
        "| Question | Kind | First gold rank | Hit@1 | Hit@3 | Hit@5 | Gold covering chunks |",
        "|---|---|---:|---|---|---|---:|",
    ]
    for row in report["questions"]:
        values = [str(row[f"hit@{k}"]) if row["answerable"] else "N/A" for k in (1, 3, 5)]
        label = f"{row['id']}: {row['question']}".replace("|", "\\|")
        lines.append(
            f"| {label} | {row['kind']} | {row['first_gold_rank'] or '—'} | "
            + " | ".join(values)
            + f" | {len(row['supporting_chunk_ids'])} |"
        )
    return "\n".join(lines) + "\n"


def page(collection: Collection, query: str) -> str:
    esc = html.escape
    cards = []
    if query.strip():
        for rank, chunk in enumerate(collection.search(query, 5), 1):
            cards.append(
                f"<article><h2>#{rank} · {esc(chunk['doc_id'])}</h2>"
                f"<p>BM25 score <b>{chunk['score']:.4f}</b> · {esc(chunk['id'])}</p>"
                f"<pre>{esc(chunk['text'])}</pre>"
                f'<a href="{esc(chunk["source_url"], quote=True)}">'
                "Pinned source document</a></article>"
            )
        if not cards:
            cards.append(
                "<article>No lexical matches. "
                "This does not establish that the answer is absent.</article>"
            )
    return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG-Intel · Public corpus proof</title>
<style>body{{font:16px/1.5 system-ui;background:#f4f6f8;color:#172c3c;
margin:32px auto;max-width:1050px;padding:0 24px}}
h1{{font-size:34px;margin-bottom:0}}header p{{color:#475b6c}}
form,article{{background:white;border:1px solid #ccd6de;border-radius:10px;
padding:20px;margin:18px 0}}
input{{width:80%;font:inherit;padding:10px}}
button{{font:inherit;padding:10px;background:#145c64;color:white;border:0;border-radius:5px}}
pre{{white-space:pre-wrap;font:15px/1.6 ui-monospace,monospace}}
h2{{font-size:20px;margin:0}}a{{color:#095f9c}}
.note{{border-left:4px solid #b26a00;padding:12px;background:#fff7df}}</style>
<header><p>RAG-INTEL / INSPECTABLE RETRIEVAL</p><h1>Question → passages → sources</h1>
<p>CPython 3.12.7 · 4 public tutorial chapters · PSF-2.0 · {len(collection.chunks)} chunks<br>
Policy: {esc(collection.variant)} · BM25 · CPU · no embeddings or generated answers</p></header>
<p class="note">These are retrieved passages, not an answer. A high score does not prove support.
The eval checks full gold-passage containment; unanswerable questions can still return results.</p>
<form method="get"><label for="q">Ask about this collection</label><br>
<input id="q" name="q" value="{esc(query, quote=True)}" maxlength="1000" required>
<button>Retrieve</button></form>{"".join(cards)}
<footer>Local-only proof · development evaluation · no client data<br>
Corpus source commit {collection.manifest["source_commit"]}</footer></html>'''


def serve(collection: Collection, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path != "/":
                self.send_error(404)
                return
            query = parse_qs(url.query).get("q", [""])[0][:1000]
            body = page(collection, query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"Retrieval demo: http://127.0.0.1:{port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("eval", "search", "serve"))
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--cache", type=Path, default=ROOT / ".rag-intel/proof-cache")
    parser.add_argument("--variant", choices=("baseline", "repaired"), default="repaired")
    parser.add_argument("--output", type=Path, default=ROOT / ".rag-intel/proof-runs")
    parser.add_argument("--query", default="")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if args.command == "eval":
        args.output.mkdir(parents=True, exist_ok=True)
        for variant in ("baseline", "repaired"):
            report = evaluate(args.dataset, variant, args.cache)
            stem = f"{report['generated_at_utc'][:10]}-{variant}"
            (args.output / f"{stem}.json").write_text(
                json.dumps(report, indent=2) + "\n", encoding="utf-8"
            )
            (args.output / f"{stem}.md").write_text(markdown(report), encoding="utf-8")
            print(variant, json.dumps(report["metrics_answerable_only"], sort_keys=True))
    else:
        collection = Collection(args.dataset, args.variant, args.cache)
        if args.command == "serve":
            serve(collection, args.port)
        else:
            print(json.dumps(collection.search(args.query, 5), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

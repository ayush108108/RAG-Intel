"""Reproducible benchmark runner for public retrieval evidence."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from rag_intel import __version__
from rag_intel.engine import RAGIntelEngine
from rag_intel.evaluation import evaluate_ranking


class BenchmarkError(ValueError):
    """Raised when a benchmark manifest or run is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkDocument:
    doc_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class BenchmarkQuery:
    query_id: str
    text: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    name: str
    documents: tuple[BenchmarkDocument, ...]
    queries: tuple[BenchmarkQuery, ...]


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field} must be a non-empty string")
    return value.strip()


def load_manifest(path: str | Path) -> BenchmarkManifest:
    """Load and validate a benchmark manifest with paths relative to its file."""
    manifest_path = Path(path)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BenchmarkError(f"cannot read benchmark manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"invalid benchmark JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise BenchmarkError("benchmark manifest must be a JSON object")
    unknown = set(raw) - {"name", "documents", "queries"}
    if unknown:
        raise BenchmarkError(f"unknown benchmark fields: {sorted(unknown)}")

    name = _non_empty_string(raw.get("name"), "name")
    document_rows = raw.get("documents")
    query_rows = raw.get("queries")
    if not isinstance(document_rows, list) or not document_rows:
        raise BenchmarkError("documents must be a non-empty list")
    if not isinstance(query_rows, list) or not query_rows:
        raise BenchmarkError("queries must be a non-empty list")

    documents: list[BenchmarkDocument] = []
    document_ids: set[str] = set()
    for index, row in enumerate(document_rows):
        if not isinstance(row, dict) or set(row) != {"doc_id", "path"}:
            raise BenchmarkError(f"documents[{index}] must contain only doc_id and path")
        doc_id = _non_empty_string(row["doc_id"], f"documents[{index}].doc_id")
        if doc_id in document_ids:
            raise BenchmarkError(f"duplicate document id: {doc_id}")
        source = (manifest_path.parent / _non_empty_string(row["path"], "path")).resolve()
        if not source.is_file():
            raise BenchmarkError(f"benchmark document does not exist: {source}")
        document_ids.add(doc_id)
        documents.append(BenchmarkDocument(doc_id=doc_id, path=source))

    queries: list[BenchmarkQuery] = []
    query_ids: set[str] = set()
    for index, row in enumerate(query_rows):
        if not isinstance(row, dict) or set(row) != {"query_id", "text", "relevant_ids"}:
            raise BenchmarkError(
                f"queries[{index}] must contain only query_id, text, and relevant_ids"
            )
        query_id = _non_empty_string(row["query_id"], f"queries[{index}].query_id")
        relevant = row["relevant_ids"]
        if query_id in query_ids:
            raise BenchmarkError(f"duplicate query id: {query_id}")
        if not isinstance(relevant, list) or not relevant:
            raise BenchmarkError(f"queries[{index}].relevant_ids must be non-empty")
        relevant_ids = frozenset(
            _non_empty_string(identifier, f"queries[{index}].relevant_ids")
            for identifier in relevant
        )
        unknown_ids = relevant_ids - document_ids
        if unknown_ids:
            raise BenchmarkError(
                f"query {query_id} references unknown documents: {sorted(unknown_ids)}"
            )
        query_ids.add(query_id)
        queries.append(
            BenchmarkQuery(
                query_id=query_id,
                text=_non_empty_string(row["text"], f"queries[{index}].text"),
                relevant_ids=relevant_ids,
            )
        )

    return BenchmarkManifest(name=name, documents=tuple(documents), queries=tuple(queries))


def _percentile(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((percentile * len(ordered) + 0.9999) - 1)))
    return ordered[index]


def run_benchmark(
    engine: RAGIntelEngine,
    manifest: BenchmarkManifest,
    *,
    k: int,
    runs: int,
    warmup_runs: int = 1,
) -> dict[str, Any]:
    """Ingest a manifest and measure deterministic quality and query latency."""
    if k < 1:
        raise BenchmarkError("k must be positive")
    if runs < 1:
        raise BenchmarkError("runs must be positive")
    if warmup_runs < 0:
        raise BenchmarkError("warmup_runs must be non-negative")

    ingest_started = time.perf_counter()
    chunk_count = 0
    for document in manifest.documents:
        result = engine.ingest(document.path, doc_id=document.doc_id)
        chunk_count += int(result["chunks"])
    ingest_ms = (time.perf_counter() - ingest_started) * 1000

    query_reports: list[dict[str, Any]] = []
    all_latencies: list[float] = []
    metric_samples: dict[str, list[float]] = {}
    for query in manifest.queries:
        for _ in range(warmup_runs):
            engine.search(query.text)

        expected_ranking: list[str] | None = None
        latencies: list[float] = []
        for _ in range(runs):
            started = time.perf_counter()
            results = engine.search(query.text)
            latencies.append((time.perf_counter() - started) * 1000)
            ranking = [str(result["doc_id"]) for result in results]
            if expected_ranking is None:
                expected_ranking = ranking
            elif ranking != expected_ranking:
                raise BenchmarkError(f"non-deterministic ranking for query: {query.query_id}")

        ranking = expected_ranking or []
        metrics = evaluate_ranking(set(query.relevant_ids), ranking, k)
        for metric, value in metrics.items():
            metric_samples.setdefault(metric, []).append(value)
        all_latencies.extend(latencies)
        query_reports.append(
            {
                "query_id": query.query_id,
                "query": query.text,
                "relevant_ids": sorted(query.relevant_ids),
                "ranked_ids": ranking,
                "metrics": metrics,
                "latency_ms": {
                    "min": round(min(latencies), 3),
                    "median": round(median(latencies), 3),
                    "max": round(max(latencies), 3),
                },
            }
        )

    return {
        "schema_version": 1,
        "benchmark": manifest.name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "software": {"rag_intel": __version__, "python": platform.python_version()},
        "system": {"platform": platform.platform(), "implementation": sys.implementation.name},
        "configuration": {
            "retrieval_mode": engine.settings.retrieval_mode,
            "chunk_size": engine.settings.chunk_size,
            "chunk_overlap": engine.settings.chunk_overlap,
            "candidate_depth": engine.policy.candidate_depth,
            "fusion": type(engine.policy.fusion).__name__,
            "aggregation": type(engine.policy.aggregate).__name__,
            "selection": type(engine.policy.select).__name__,
            "k": k,
            "runs": runs,
            "warmup_runs": warmup_runs,
        },
        "corpus": {
            "documents": len(manifest.documents),
            "chunks": chunk_count,
            "ingest_ms": round(ingest_ms, 3),
        },
        "quality": {metric: round(mean(values), 6) for metric, values in metric_samples.items()},
        "query_latency_ms": {
            "samples": len(all_latencies),
            "min": round(min(all_latencies), 3),
            "p50": round(median(all_latencies), 3),
            "p95": round(_percentile(all_latencies, 0.95), 3),
            "max": round(max(all_latencies), 3),
        },
        "queries": query_reports,
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Write a stable, human-reviewable JSON report."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

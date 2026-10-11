# Reproducibility

The public package is designed to be testable without a private dataset or hosted service.

## Deterministic pieces

- normalization;
- recursive chunking for fixed parameters;
- BM25 indexing/search for a fixed corpus;
- reciprocal-rank fusion for fixed rankings;
- SQLite identity/state transitions;
- vector-store reload for fixed embeddings and IDs;
- ranking metrics, including nDCG;
- benchmark rankings and quality for a fixed manifest/configuration.

## Local verification

```bash
uv sync --extra dev --frozen
uv run ruff check src tests examples
uv run pytest -q
uv run python -m compileall -q src
uv run pip-audit
```

The synthetic example uses no external model or network call. Parser adapters are optional and therefore are not required for the core test suite.

## Benchmark reproduction

```bash
uv run rag-intel benchmark benchmarks/lexical-smoke-v1.json \
	--config config/benchmark.lexical.toml \
	--k 3 --runs 30 --warmup-runs 3 \
	--output benchmarks/results/lexical-smoke-v1.json
```

The manifest fixes document IDs, source files, queries, and relevance labels. The runner fails if repeated searches return different rankings. Quality metrics should match the committed report; latency and timestamp fields are expected to vary by run.

## What is not reproducible from this repo

Private corpora, tuned ranking parameters, reranker/model routing, production telemetry, and application/deployment behavior are intentionally not included.

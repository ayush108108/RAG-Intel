# Results and evidence

RAG-Intel publishes reproducible engineering evidence, not private workload metrics.

## Verified invariants

The automated suite checks that:

- chunk size and overlap validation are deterministic;
- zero-overlap BM25 candidates are omitted;
- explicit RRF combines independent rankings;
- required TOML policy fails fast when incomplete;
- local ingest, search, restart, and deletion work end to end;
- queued metadata survives restart recovery;
- interrupted processing is marked failed rather than completed;
- optional FAISS persistence validates vector identity and model compatibility;
- ranking metrics match known examples.

## Benchmark policy

A quality or performance result belongs here only when its corpus is redistributable and the command, configuration, dependency versions, hardware, and raw output are published. Until then, the project describes itself as a modern, modular reference engine—not universally state of the art.

## Lexical smoke benchmark v1

This repository now publishes one deliberately small regression benchmark:

- manifest: `benchmarks/lexical-smoke-v1.json`;
- configuration: `config/benchmark.lexical.toml`;
- raw report: `benchmarks/results/lexical-smoke-v1.json`;
- corpus: 4 local TXT/Markdown/CSV/JSON documents;
- judgments: 4 queries with explicit relevant document IDs;
- execution: 3 warmups followed by 30 measured runs per query.

Measured on CPython 3.12.1, Linux 6.8 Azure x86_64:

| Metric | Value |
|---|---:|
| Precision@3 | 0.333333 |
| Recall@3 | 1.000000 |
| nDCG@3 | 1.000000 |
| Reciprocal rank | 1.000000 |
| Query latency p50 | 0.567 ms |
| Query latency p95 | 0.781 ms |
| Query latency samples | 120 |

Latency is hardware- and load-specific. CI asserts deterministic ranking and quality floors, but intentionally does not enforce latency thresholds. The corpus tests exact lexical retrieval behavior; it is too small and too direct to support comparisons with other systems or a state-of-the-art claim.

### Reproduce

```bash
uv sync --extra dev --frozen
uv run rag-intel benchmark benchmarks/lexical-smoke-v1.json \
	--config config/benchmark.lexical.toml \
	--k 3 --runs 30 --warmup-runs 3 \
	--output benchmarks/results/lexical-smoke-v1.json
```

Quality and rankings should reproduce. Timing will vary, and `generated_at_utc` will change.

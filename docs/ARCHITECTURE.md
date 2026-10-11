# Architecture

## Scope

The public package exposes a small set of retrieval-engineering primitives. It is intentionally not the architecture of the full private application.

```text
                 ┌──────────────────────┐
file ───────────>│ ingestion boundary   │
                 └──────────┬───────────┘
                            v
                 ┌──────────────────────┐
                 │ normalization        │
                 └──────────┬───────────┘
                            v
                 ┌──────────────────────┐
                 │ bounded chunking     │
                 └──────┬─────────┬─────┘
                        │         │
                        v         v
                 caller vectors  BM25
                        │         │
                        v         │
                 persistent      │
                 FAISS store     │
                        └────┬────┘
                             v
                                                 explicit policy
                             |
                             v
                                          aggregation + selection

                 SQLite metadata state
                    ↕ queue/recovery
```

## Boundaries

### Ingestion

`ingestion.py` converts supported file formats to text and metadata. Optional parser dependencies are loaded lazily. Parser configuration errors are surfaced rather than converted into successful empty documents.

### Chunking

`chunking.py` provides deterministic recursive splitting and an optional semantic grouping hook.

- **Recursive splitting** follows a separator hierarchy (paragraph → line → sentence → word → character). Overlap is accounted for inside the chunk-size budget to keep token footprints predictable. Overlap start boundaries are aligned to word edges.
- **Semantic splitting** groups consecutive sentences whose dense-embedding cosine similarity exceeds a caller-supplied threshold. The recommended local backend is `sentence-transformers` (`all-MiniLM-L6-v2`), which is CPU-capable and requires no API key. Any callable with signature `(list[str]) -> list[list[float]]` is accepted as the embedding function.
- **Resource guard** at the ingestion layer validates file size against a caller-supplied (or default 100 MB) threshold before loading into memory. Oversized files are rejected with a structured log entry rather than allowed to trigger an OOM condition.

The public API accepts all configuration as parameters; it does not import private runtime configuration.

### Retrieval

BM25 is a conventional lexical baseline. Optional dense retrieval uses a caller-selected embedding provider and validated FAISS index. `RetrievalPolicy` requires fusion, document aggregation, candidate depth, and final selection. Standard RRF/max/mean/top-k implementations are references, not implicit defaults.

### Persistence

`FaissStore` wraps a flat ID-mapped FAISS index to demonstrate identity management, crash-aware writes, compatibility checks, deletion, and reload. It deliberately avoids exposing private index tuning.

`MetadataStore` keeps document/chunk identity and durable queue metadata in SQLite. Recovery logic treats queued and processing jobs differently so restarts do not create false-completed states.

### Engine

`RAGIntelEngine` is a synchronous local composition root used by both Python callers and the Typer CLI. It owns lifecycle state and cleanup across metadata, sparse, and optional dense stores. Production scheduling, hosted APIs, and product orchestration remain application concerns.

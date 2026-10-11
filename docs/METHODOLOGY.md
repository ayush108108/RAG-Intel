# Methodology

## Retrieval quality starts before retrieval

Files are extracted through format-specific adapters, normalized deterministically, and rejected explicitly when an optional parser is unavailable. Empty extraction never becomes a successful index operation.

## Bounded chunking

Recursive splitting follows structural separators from paragraphs down to characters. Overlap consumes part of the configured chunk budget instead of extending it, and overlap starts are aligned to whitespace where possible.

Semantic grouping is an optional caller-controlled hook. Callers provide both the embedding function and threshold; the engine does not encode a recommended corpus-specific value.

## Lexical and dense evidence

BM25 supplies conventional exact-term evidence. Dense retrieval uses normalized vectors and an ID-mapped flat FAISS index. Persisted vector metadata records model identity, count, and dimension so incompatible indexes fail visibly.

## Explicit ranking

The local engine requires four choices:

1. candidate depth;
2. fusion strategy;
3. document aggregation strategy;
4. final candidate selection.

The public RRF implementation combines result order rather than incomparable raw score scales. Max and mean aggregation are intentionally simple baselines. More specialized policy belongs in an injected strategy.

## Durable local state

SQLite stores documents, chunks, vector IDs, and lifecycle state. Ingestion transitions through queued, processing, completed, or failed. If vector indexing fails, inserted chunks are removed and the document remains visibly failed.

## Evaluation

Precision@k, Recall@k, and reciprocal rank are independent of ranking implementation. Results should always name the corpus, labels, configuration, model versions, hardware, and seed. This repository does not generalize one workload into a universal quality claim.

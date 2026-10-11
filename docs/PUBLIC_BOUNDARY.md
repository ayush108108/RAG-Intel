# Public boundary

RAG-Intel separates reusable retrieval mechanisms from application-specific policy. The project is pre-1.0; listed APIs are supported but may evolve with documented release notes.

## Public mechanisms

- extraction and deterministic normalization;
- bounded recursive chunking and a caller-supplied semantic hook;
- BM25, FAISS, and SQLite adapters;
- provider-neutral embedding and reranking protocols;
- `RAGIntelEngine` for synchronous local ingest, search, deletion, and status;
- Precision@k, Recall@k, and reciprocal-rank metrics.

## Reference policy

`rag_intel.retrieval.strategies` includes standard, explicitly selected building blocks:

- reciprocal-rank fusion;
- maximum and arithmetic-mean document aggregation;
- fixed top-k candidate selection.

These are reference implementations, not hidden defaults or claims of optimality. The engine requires a `RetrievalPolicy`.

## Deliberately outside the project

- corpus-tuned ranking and model-selection policy;
- prompts, agents, generation, and provider routing;
- HTTP services, authentication, background scheduling, and product UI;
- private corpora, relevance labels, deployment telemetry, and benchmark results.

Applications may implement the public protocols in private or separate packages. Dependency direction is one-way: applications import RAG-Intel; RAG-Intel never imports application policy.

## Compatibility

Breaking changes to `RAGIntelEngine`, `EngineSettings`, strategy protocols, CLI commands, or persisted-state format require release notes. Private helpers and underscored symbols are not compatibility contracts.

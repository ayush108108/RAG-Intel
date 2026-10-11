# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) principles. The project uses semantic versioning after the pre-1.0 API settles.

## Unreleased

### Added

- `RAGIntelEngine` for synchronous local ingest, search, restart, and deletion.
- Required typed TOML configuration with actionable validation errors.
- Provider-neutral strategy, embedding, and reranking protocols.
- Explicit RRF, max/mean aggregation, and top-k reference strategies.
- Complete Typer commands for ingestion, search, deletion, status, and evaluation.
- Versioned benchmark manifests with Precision/Recall/MRR/nDCG and latency reports.
- Checked-in lexical smoke evidence plus deterministic quality regression tests.
- Core and full-adapter CI lanes.

### Changed

- Ranking policy must be selected explicitly; no tuned fusion recipe is a library default.
- Documentation now distinguishes verified engineering invariants from benchmark claims.
- Docker launches the discoverable CLI rather than an example application.

### Removed

- Hardcoded terminal dashboard and duplicated retrieval orchestration.
- Placeholder tests for prompts and agent routing that are outside the OSS boundary.
- Environment variables containing ranking-policy defaults.

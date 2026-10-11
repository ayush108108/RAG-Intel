"""Typer CLI for the local RAG-Intel reference engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from rag_intel import __version__
from rag_intel.benchmark import BenchmarkError, load_manifest, run_benchmark, write_report
from rag_intel.config import ConfigurationError, LoadedConfiguration, load_config
from rag_intel.embedding import SentenceTransformerProvider
from rag_intel.engine import EngineError, RAGIntelEngine
from rag_intel.evaluation import evaluate_ranking
from rag_intel.retrieval.reranker import CrossEncoderReranker

app = typer.Typer(
    name="rag-intel",
    help="Transparent local retrieval, assembled by explicit policy.",
    add_completion=False,
    no_args_is_help=True,
)


def _build_engine(config_path: Path) -> RAGIntelEngine:
    loaded: LoadedConfiguration = load_config(config_path)
    settings = loaded.settings
    provider = (
        SentenceTransformerProvider(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )
        if settings.embedding_model
        else None
    )
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model)
        if settings.enable_reranking and settings.reranker_model
        else None
    )
    return RAGIntelEngine(
        settings,
        loaded.policy,
        embedding_provider=provider,
        reranker=reranker,
    )


def _fail(exc: Exception) -> None:
    typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


@app.command()
def version() -> None:
    """Display the installed version."""
    typer.echo(f"rag-intel {__version__}")


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Extract and index one local document."""
    try:
        result = _build_engine(config).ingest(source)
    except (ConfigurationError, EngineError, OSError, RuntimeError, ValueError) as exc:
        _fail(exc)
    typer.secho("✓ Indexed", fg=typer.colors.GREEN, bold=True)
    typer.echo(json.dumps(result, indent=2))


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Search the local corpus and print ranked JSON results."""
    try:
        results = _build_engine(config).search(query)
    except (ConfigurationError, EngineError, OSError, RuntimeError, ValueError) as exc:
        _fail(exc)
    if not results:
        typer.secho("No matches found.", fg=typer.colors.YELLOW)
        return
    typer.echo(json.dumps(results, indent=2))


@app.command()
def delete(
    doc_id: Annotated[str, typer.Argument()],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Delete a document from local metadata and indexes."""
    try:
        deleted = _build_engine(config).delete(doc_id)
    except (ConfigurationError, EngineError, OSError, RuntimeError, ValueError) as exc:
        _fail(exc)
    if not deleted:
        _fail(ValueError(f"document not found: {doc_id}"))
    typer.secho(f"✓ Deleted {doc_id}", fg=typer.colors.GREEN)


@app.command()
def status(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
) -> None:
    """Show local corpus health and counts."""
    try:
        stats = _build_engine(config).status()
    except (ConfigurationError, EngineError, OSError, RuntimeError, ValueError) as exc:
        _fail(exc)
    typer.secho("RAG-Intel local index", fg=typer.colors.CYAN, bold=True)
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def evaluate(
    relevant: Annotated[str, typer.Option(help="Comma-separated relevant document IDs.")],
    ranked: Annotated[str, typer.Option(help="Comma-separated ranked document IDs.")],
    k: Annotated[int, typer.Option(min=1)] = 10,
) -> None:
    """Calculate transparent IR metrics for one ranking."""
    relevant_ids = {item.strip() for item in relevant.split(",") if item.strip()}
    ranked_ids = [item.strip() for item in ranked.split(",") if item.strip()]
    typer.echo(json.dumps(evaluate_ranking(relevant_ids, ranked_ids, k), indent=2))


@app.command()
def benchmark(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option("--output", "-o")] = None,
    k: Annotated[int, typer.Option(min=1)] = 3,
    runs: Annotated[int, typer.Option(min=1)] = 5,
    warmup_runs: Annotated[int, typer.Option(min=0)] = 1,
) -> None:
    """Run a versioned public benchmark and emit reproducible JSON evidence."""
    try:
        report = run_benchmark(
            _build_engine(config),
            load_manifest(manifest),
            k=k,
            runs=runs,
            warmup_runs=warmup_runs,
        )
        if output is not None:
            write_report(report, output)
    except (
        BenchmarkError,
        ConfigurationError,
        EngineError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        _fail(exc)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()

"""Strict configuration for the local reference engine.

Retrieval choices are required in TOML. The library intentionally does not
silently select ranking policy on behalf of applications.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_intel.retrieval.strategies import (
    MaxScoreAggregator,
    MeanScoreAggregator,
    ReciprocalRankFusion,
    RetrievalPolicy,
    TopKSelector,
)


class ConfigurationError(ValueError):
    """Raised when engine configuration is incomplete or invalid."""


@dataclass(frozen=True, slots=True)
class EngineSettings:
    """Validated operational settings; retrieval policy is stored separately."""

    data_dir: Path
    retrieval_mode: str
    chunk_size: int
    chunk_overlap: int
    max_ingest_size_mb: float
    embedding_model: str | None
    embedding_batch_size: int
    enable_reranking: bool
    reranker_model: str | None

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"lexical", "hybrid"}:
            raise ConfigurationError("retrieval.mode must be 'lexical' or 'hybrid'")
        if self.chunk_size < 1:
            raise ConfigurationError("chunking.size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ConfigurationError("chunking.overlap must be >= 0 and smaller than size")
        if self.max_ingest_size_mb <= 0:
            raise ConfigurationError("ingestion.max_size_mb must be positive")
        if self.embedding_batch_size < 1:
            raise ConfigurationError("models.embedding_batch_size must be positive")
        if self.retrieval_mode == "hybrid" and not self.embedding_model:
            raise ConfigurationError("hybrid mode requires models.embedding")
        if self.enable_reranking and not self.reranker_model:
            raise ConfigurationError("enabled reranking requires models.reranker")

    @property
    def database_path(self) -> Path:
        return self.data_dir / "metadata.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "vectors.faiss"

    @property
    def index_metadata_path(self) -> Path:
        return self.data_dir / "vectors.json"


@dataclass(frozen=True, slots=True)
class LoadedConfiguration:
    settings: EngineSettings
    policy: RetrievalPolicy


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigurationError(f"missing [{name}] section")
    return section


def _required(section: dict[str, Any], section_name: str, key: str) -> Any:
    if key not in section:
        raise ConfigurationError(f"missing required setting: {section_name}.{key}")
    return section[key]


def _reject_unknown(section: dict[str, Any], name: str, allowed: set[str]) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigurationError(f"unknown settings in [{name}]: {sorted(unknown)}")


def _string(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{key} must be a string")
    return value


def _integer(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{key} must be an integer")
    return value


def _number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{key} must be a number")
    return float(value)


def _boolean(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{key} must be a boolean")
    return value


def load_config(path: str | Path) -> LoadedConfiguration:
    """Load strict engine settings and explicitly selected reference policy."""
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigurationError(f"cannot read configuration: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigurationError(f"invalid TOML in {config_path}: {exc}") from exc

    storage = _section(raw, "storage")
    chunking = _section(raw, "chunking")
    ingestion = _section(raw, "ingestion")
    retrieval = _section(raw, "retrieval")
    models = _section(raw, "models")

    expected_sections = {"storage", "chunking", "ingestion", "retrieval", "models"}
    unknown_sections = set(raw) - expected_sections
    if unknown_sections:
        raise ConfigurationError(f"unknown configuration sections: {sorted(unknown_sections)}")
    _reject_unknown(storage, "storage", {"data_dir"})
    _reject_unknown(chunking, "chunking", {"size", "overlap"})
    _reject_unknown(ingestion, "ingestion", {"max_size_mb"})
    _reject_unknown(
        retrieval,
        "retrieval",
        {
            "mode",
            "fusion",
            "rrf_rank_constant",
            "aggregation",
            "candidate_depth",
            "result_limit",
            "rerank",
        },
    )
    _reject_unknown(models, "models", {"embedding", "embedding_batch_size", "reranker"})

    fusion_name = _string(_required(retrieval, "retrieval", "fusion"), "retrieval.fusion")
    if fusion_name != "rrf":
        raise ConfigurationError("retrieval.fusion must explicitly select 'rrf'")
    aggregation_name = _string(
        _required(retrieval, "retrieval", "aggregation"), "retrieval.aggregation"
    )
    if aggregation_name not in {"max", "mean"}:
        raise ConfigurationError("retrieval.aggregation must be 'max' or 'mean'")
    aggregator = MaxScoreAggregator() if aggregation_name == "max" else MeanScoreAggregator()

    settings = EngineSettings(
        data_dir=Path(
            _string(_required(storage, "storage", "data_dir"), "storage.data_dir")
        ).expanduser(),
        retrieval_mode=_string(_required(retrieval, "retrieval", "mode"), "retrieval.mode"),
        chunk_size=_integer(_required(chunking, "chunking", "size"), "chunking.size"),
        chunk_overlap=_integer(_required(chunking, "chunking", "overlap"), "chunking.overlap"),
        max_ingest_size_mb=_number(
            _required(ingestion, "ingestion", "max_size_mb"), "ingestion.max_size_mb"
        ),
        embedding_model=(
            _string(models["embedding"], "models.embedding") if models.get("embedding") else None
        ),
        embedding_batch_size=_integer(
            _required(models, "models", "embedding_batch_size"),
            "models.embedding_batch_size",
        ),
        enable_reranking=_boolean(_required(retrieval, "retrieval", "rerank"), "retrieval.rerank"),
        reranker_model=(
            _string(models["reranker"], "models.reranker") if models.get("reranker") else None
        ),
    )
    policy = RetrievalPolicy(
        candidate_depth=_integer(
            _required(retrieval, "retrieval", "candidate_depth"),
            "retrieval.candidate_depth",
        ),
        fusion=ReciprocalRankFusion(
            rank_constant=_integer(
                _required(retrieval, "retrieval", "rrf_rank_constant"),
                "retrieval.rrf_rank_constant",
            )
        ),
        aggregate=aggregator,
        select=TopKSelector(
            limit=_integer(
                _required(retrieval, "retrieval", "result_limit"),
                "retrieval.result_limit",
            )
        ),
    )
    return LoadedConfiguration(settings=settings, policy=policy)

from pathlib import Path

import pytest

from rag_intel.config import ConfigurationError, load_config

VALID_CONFIG = """
[storage]
data_dir = "./runtime"

[chunking]
size = 640
overlap = 64

[ingestion]
max_size_mb = 25

[retrieval]
mode = "lexical"
fusion = "rrf"
rrf_rank_constant = 42
aggregation = "max"
candidate_depth = 30
result_limit = 5
rerank = false

[models]
embedding = ""
embedding_batch_size = 8
reranker = ""
"""


def test_load_config_requires_and_builds_explicit_policy(tmp_path: Path):
    path = tmp_path / "rag-intel.toml"
    path.write_text(VALID_CONFIG, encoding="utf-8")

    loaded = load_config(path)

    assert loaded.settings.chunk_size == 640
    assert loaded.settings.retrieval_mode == "lexical"
    assert loaded.policy.candidate_depth == 30
    assert loaded.policy.select([("a", 1.0)] * 10) == [("a", 1.0)] * 5


def test_load_config_rejects_missing_policy(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text("[storage]\ndata_dir = './runtime'\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"missing \[chunking\]"):
        load_config(path)


def test_load_config_rejects_unknown_and_wrong_type(tmp_path: Path):
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(VALID_CONFIG + "\n[private_policy]\nweight = 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="unknown configuration sections"):
        load_config(unknown)

    wrong_type = tmp_path / "wrong-type.toml"
    wrong_type.write_text(
        VALID_CONFIG.replace("rerank = false", 'rerank = "false"'), encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="retrieval.rerank must be a boolean"):
        load_config(wrong_type)

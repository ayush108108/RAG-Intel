"""Check passage-level scoring, provenance, cold/warm reproducibility and demo safety."""

import json
import shutil

import pytest

from rag_intel.proof import DATASET, Collection, assess, evaluate, page, read_dataset


def test_scoring_requires_full_passage_and_correct_document():
    gold = [{"doc_id": "a", "quote": "supporting passage"}]
    ranked = [
        {"doc_id": "a", "text": "supporting"},
        {"doc_id": "b", "text": "supporting passage"},
        {"doc_id": "a", "text": "the supporting passage is here"},
    ]
    assert assess(ranked, gold) == {
        "first_gold_rank": 3,
        "reciprocal_rank": 1 / 3,
        "hit@1": False,
        "hit@3": True,
        "hit@5": True,
    }
    assert assess(ranked, []) == {
        "first_gold_rank": None,
        "reciprocal_rank": 0,
        "hit@1": False,
        "hit@3": False,
        "hit@5": False,
    }


def test_dataset_provenance_and_integrity(tmp_path):
    manifest, questions = read_dataset(DATASET)
    assert len(questions) == 40
    assert sum(q["answerable"] for q in questions) == 36
    assert len(manifest["documents"]) == 4
    assert all(q["provenance"]["date"] == "2026-10-15" for q in questions)
    dataset = tmp_path / "dataset"
    shutil.copytree(DATASET, dataset)
    path = dataset / manifest["documents"][0]["path"]
    path.write_bytes(path.read_bytes() + b"tampering")
    with pytest.raises(ValueError, match="checksum"):
        read_dataset(dataset)


def test_bad_gold_span_is_rejected(tmp_path):
    dataset = tmp_path / "dataset"
    shutil.copytree(DATASET, dataset)
    path = dataset / "questions.jsonl"
    questions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    questions[0]["gold"][0]["start"] += 1
    path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gold span"):
        read_dataset(dataset)


def test_cold_and_warm_rankings_and_metrics_match(tmp_path):
    before = evaluate(DATASET, "baseline", tmp_path)
    after = evaluate(DATASET, "repaired", tmp_path)
    warm = evaluate(DATASET, "repaired", tmp_path)
    assert not after["cache_hit"]
    assert warm["cache_hit"]
    assert after["questions"] == warm["questions"]
    assert after["metrics_answerable_only"] == warm["metrics_answerable_only"]
    positives = [q for q in after["questions"] if q["answerable"]]
    assert after["metrics_answerable_only"]["hit@1"] == sum(q["hit@1"] for q in positives) / 36
    assert after["metrics_answerable_only"]["MRR"] > before["metrics_answerable_only"]["MRR"]
    # Keep the known regression and unindexable support visible, too.
    assert after["metrics_answerable_only"]["hit@3"] < before["metrics_answerable_only"]["hit@3"]
    assert after["counts"]["answerable_without_covering_chunk"] == 3
    assert after["negative_diagnostics"]["returned_any_passage"] == 4


def test_demo_uses_live_retrieval_and_escapes_query(tmp_path):
    collection = Collection(DATASET, "repaired", tmp_path)
    query = 'What does dir return for a module? <script>alert("x")</script>'
    result = collection.search(query, 5)
    document = page(collection, query)
    assert result[0]["text"] in document
    assert "<script>" not in document
    assert "&lt;script&gt;" in document
    assert result[0]["source_url"] in document
    with pytest.raises(ValueError, match="empty"):
        collection.search(" ")

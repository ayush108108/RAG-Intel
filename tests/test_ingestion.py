import json

from rag_intel.ingestion import extract_csv, extract_json, route_file


def test_route_text_file(tmp_path):
    source = tmp_path / "hello-world.txt"
    source.write_text("hello retrieval", encoding="utf-8")
    document = route_file(source)
    assert document["content"] == "hello retrieval"
    assert document["metadata"]["source_type"] == "text"
    assert document["metadata"]["title"] == "hello world"


def test_json_and_csv_extraction(tmp_path):
    json_path = tmp_path / "doc.json"
    json_path.write_text(json.dumps({"name": "rag", "items": [1, 2]}), encoding="utf-8")
    assert "name: rag" in extract_json(json_path)

    csv_path = tmp_path / "doc.csv"
    csv_path.write_text("name,value\nalpha,1\n", encoding="utf-8")
    assert "name is alpha" in extract_csv(csv_path)

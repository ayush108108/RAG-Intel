import numpy as np
import pytest

from rag_intel.index.faiss_store import FaissStore


def test_faiss_round_trip_preserves_external_ids(tmp_path):
    pytest.importorskip("faiss")

    index_path = tmp_path / "vectors.faiss"
    metadata_path = tmp_path / "vectors.json"
    vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    store = FaissStore(index_path, metadata_path, embedding_model="synthetic-v1")
    store.add(vectors, [10, 12])

    reloaded = FaissStore(index_path, metadata_path, embedding_model="synthetic-v1")
    reloaded.load()
    assert reloaded.ids == [10, 12]
    result = reloaded.search(np.asarray([1.0, 0.0], dtype=np.float32), top_k=2)
    assert result[0][0] == 10

"""Deterministic RRF example with no model or network dependency."""

from rag_intel.retrieval.bm25 import BM25Retriever
from rag_intel.retrieval.strategies import ReciprocalRankFusion

CORPUS = [
    "hybrid retrieval combines lexical and semantic evidence",
    "durable ingestion queues preserve document metadata",
    "vector indexes map embeddings to stable external identifiers",
]

retriever = BM25Retriever()
retriever.index(CORPUS, [1, 2, 3])
sparse = dict(retriever.search("retrieval lexical", top_k=3))

# Stand-in dense similarities: in a real application they come from the caller's model/index.
dense = {1: 0.82, 2: 0.16, 3: 0.30}
ranking = ReciprocalRankFusion(rank_constant=60)(dense, sparse)

for identifier, score in ranking:
    print(identifier, round(score, 4), CORPUS[identifier - 1])

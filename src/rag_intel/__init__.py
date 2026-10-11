"""Public retrieval-engineering primitives for RAG-Intel."""

from rag_intel import config
from rag_intel.chunking import chunk_text
from rag_intel.engine import RAGIntelEngine
from rag_intel.ingestion import route_file
from rag_intel.normalization import normalize_text

__version__ = "0.1.0"

__all__ = [
    "config",
    "RAGIntelEngine",
    "chunk_text",
    "route_file",
    "normalize_text",
]

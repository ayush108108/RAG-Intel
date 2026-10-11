"""Bounded deterministic chunking with an optional semantic hook."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 80,
    semantic: bool = False,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    semantic_threshold: float = 0.75,
) -> list[dict]:
    """Split input text into bounded, overlap-budgeted chunks.

    Supports recursive structural splitting (paragraphs, sentences, words)
    or semantic sentence grouping via cosine similarity of embeddings.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if not text or not text.strip():
        return []
    if semantic:
        if embedding_fn is None:
            raise ValueError("semantic chunking requires embedding_fn")
        return _semantic_chunk(text, embedding_fn, semantic_threshold, chunk_size)
    return _recursive_chunk(text, chunk_size, chunk_overlap)


def _recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> list[dict]:
    separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    payload_size = max(1, chunk_size - overlap)
    pieces = _split_recursive(text, separators, payload_size)
    result: list[dict] = []
    previous = ""
    for index, piece in enumerate(pieces):
        if overlap and previous:
            tail = previous[-overlap:]
            space_idx = tail.find(" ")
            if space_idx != -1 and space_idx < len(tail) - 1:
                tail = tail[space_idx + 1 :]
            piece = tail + piece
        cleaned = piece.strip()
        if cleaned:
            result.append(
                {
                    "chunk_id": index,
                    "text": cleaned,
                    "metadata": {"section": _detect_section(cleaned), "source_type": "text"},
                }
            )
        previous = piece
    return result


def _split_recursive(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]
    for position, separator in enumerate(separators):
        if not separator or separator not in text:
            continue
        parts = text.split(separator)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (separator if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                chunks.extend(_split_recursive(part, separators[position + 1 :], chunk_size))
                current = ""
            else:
                current = part
        if current:
            chunks.append(current)
        return chunks
    return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]


def _semantic_chunk(
    text: str,
    embedding_fn: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    threshold: float,
    max_size: int,
) -> list[dict]:
    # Split by sentence endings (.!?), pipe delimiters (|), semicolons, or newlines
    sentences = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|(?<=\|)\s+|\n+|\s*;\s*", text)
        if part.strip()
    ]
    if not sentences:
        return []
    embeddings = np.asarray(embedding_fn(sentences), dtype=float)
    if len(embeddings) != len(sentences):
        raise ValueError("embedding_fn must return one vector per sentence")
    groups: list[list[str]] = [[sentences[0]]]
    group_length = len(sentences[0])
    for index in range(1, len(sentences)):
        add_length = len(sentences[index]) + 1
        similarity = _cosine_similarity(embeddings[index - 1], embeddings[index])
        if similarity >= threshold and group_length + add_length <= max_size:
            groups[-1].append(sentences[index])
            group_length += add_length
        else:
            groups.append([sentences[index]])
            group_length = len(sentences[index])
    return [
        {
            "chunk_id": index,
            "text": " ".join(group),
            "metadata": {"section": _detect_section(group[0]), "source_type": "semantic"},
        }
        for index, group in enumerate(groups)
    ]


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _detect_section(text: str) -> str:
    first_line = text.split("\n", maxsplit=1)[0].strip()
    named_section = (
        r"^(?:abstract|introduction|conclusion|summary|references|background|"
        r"methodology|results|discussion)\b"
    )
    patterns = [
        (named_section, "named_section"),
        (r"^\d+\.\s+\w", "numbered_section"),
        (r"^#+\s+\w", "markdown_header"),
    ]
    for pattern, label in patterns:
        if re.match(pattern, first_line, re.IGNORECASE):
            return label
    return "body"

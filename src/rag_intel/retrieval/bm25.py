"""Small conventional BM25 lexical retriever."""

from __future__ import annotations

import math
import re
from collections import defaultdict


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.ids: list[int] = []
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_doc_len = 0.0
        self._tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def index(self, texts: list[str], ids: list[int]) -> None:
        if len(texts) != len(ids):
            raise ValueError("texts and ids must have the same length")
        self.corpus = list(texts)
        self.ids = list(ids)
        self.doc_freqs = defaultdict(int)
        self.doc_len = []
        self._tokenized_corpus = []
        for text in texts:
            tokens = self._tokenize(text)
            self._tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for token in set(tokens):
                self.doc_freqs[token] += 1
        count = len(texts)
        self.avg_doc_len = sum(self.doc_len) / max(1, count)
        self.idf = {
            term: math.log((count - frequency + 0.5) / (frequency + 0.5) + 1)
            for term, frequency in self.doc_freqs.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k < 1 or not self.corpus:
            return []
        query_tokens = self._tokenize(query)
        scores: list[tuple[int, float]] = []
        for index, tokens in enumerate(self._tokenized_corpus):
            tf_map: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_map[token] += 1
            score = 0.0
            doc_length = self.doc_len[index]
            for token in query_tokens:
                if token not in self.idf:
                    continue
                tf = tf_map.get(token, 0)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_length / max(1.0, self.avg_doc_len)
                )
                score += self.idf[token] * (tf * (self.k1 + 1) / denominator)
            if score > 0:
                scores.append((self.ids[index], score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]

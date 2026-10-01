"""Dependency-free BM25 baseline; later retrievers use the same search contract."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

from .models import Passage, RankedPassage

TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 retrieval with deterministic tie-breaking by chunk ID."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75) -> None:
        if not passages:
            raise ValueError("Cannot index an empty passage collection")
        self.passages, self.k1, self.b = passages, k1, b
        self.frequencies = [Counter(tokenize(passage.text)) for passage in passages]
        self.lengths = [sum(frequencies.values()) for frequencies in self.frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        self.document_frequency: dict[str, int] = defaultdict(int)
        for frequencies in self.frequencies:
            for term in frequencies:
                self.document_frequency[term] += 1

    def search(self, query: str, top_k: int) -> list[RankedPassage]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        scores: list[tuple[Passage, float]] = []
        for passage, frequencies, length in zip(self.passages, self.frequencies, self.lengths):
            score = 0.0
            for term in tokenize(query):
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                df = self.document_frequency[term]
                idf = math.log(1 + (len(self.passages) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append((passage, score))
        ordered = sorted(scores, key=lambda item: (-item[1], item[0].chunk_id))[:top_k]
        return [RankedPassage(passage, score, rank) for rank, (passage, score) in enumerate(ordered, 1)]

"""Dense, sparse, hybrid, and reranking primitives behind a common API."""

from __future__ import annotations

import math
from collections import Counter

from .models import Passage, RankedPassage
from .retrieval import BM25Retriever, tokenize


class TfidfDenseRetriever:
    """Local dependency-free semantic fallback; swap for sentence-transformers in production."""
    def __init__(self, passages: list[Passage]) -> None:
        self.passages, self.docs = passages, [Counter(tokenize(p.text)) for p in passages]
        self.df = Counter(term for doc in self.docs for term in doc)

    def _vector(self, terms: Counter[str]) -> dict[str, float]:
        return {term: count * math.log((len(self.docs) + 1) / (self.df.get(term, 0) + 1)) for term, count in terms.items()}

    def search(self, query: str, top_k: int) -> list[RankedPassage]:
        query_vector = self._vector(Counter(tokenize(query)))
        scores = []
        for passage, document in zip(self.passages, self.docs):
            vector = self._vector(document)
            denominator = math.sqrt(sum(x*x for x in query_vector.values())) * math.sqrt(sum(x*x for x in vector.values()))
            scores.append((passage, sum(query_vector.get(term, 0) * value for term, value in vector.items()) / denominator if denominator else 0.0))
        ordered = sorted(scores, key=lambda item: (-item[1], item[0].chunk_id))[:top_k]
        return [RankedPassage(passage, score, rank) for rank, (passage, score) in enumerate(ordered, 1)]


def reciprocal_rank_fusion(*rankings: list[RankedPassage], top_k: int, constant: int = 60) -> list[RankedPassage]:
    scores, passages = Counter(), {}
    for ranking in rankings:
        for item in ranking:
            scores[item.passage.chunk_id] += 1 / (constant + item.rank)
            passages[item.passage.chunk_id] = item.passage
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [RankedPassage(passages[chunk_id], scores[chunk_id], rank) for rank, chunk_id in enumerate(ordered, 1)]


class HybridRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.sparse, self.dense = BM25Retriever(passages), TfidfDenseRetriever(passages)

    def search(self, query: str, top_k: int) -> list[RankedPassage]:
        candidate_k = max(top_k * 4, 20)
        return reciprocal_rank_fusion(self.sparse.search(query, candidate_k), self.dense.search(query, candidate_k), top_k=top_k)


def lexical_rerank(query: str, ranked: list[RankedPassage], top_k: int) -> list[RankedPassage]:
    terms = set(tokenize(query))
    ordered = sorted(ranked, key=lambda item: (-len(terms & set(tokenize(item.passage.text))), item.rank, item.passage.chunk_id))[:top_k]
    return [RankedPassage(item.passage, item.score, rank) for rank, item in enumerate(ordered, 1)]

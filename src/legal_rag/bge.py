"""Local BGE-M3 embedding and cross-encoder reranking adapters."""

from __future__ import annotations

from .models import Passage, RankedPassage


class BGEM3Retriever:
    """Dense retrieval using the local `BAAI/bge-m3` SentenceTransformer model."""

    def __init__(self, passages: list[Passage], model_name: str = "BAAI/bge-m3") -> None:
        if not passages:
            raise ValueError("Cannot index an empty passage collection")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError("BGE-M3 requires `pip install -e '.[dense]'`") from error
        self.passages = passages
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode([passage.text for passage in passages], normalize_embeddings=True)

    def search(self, query: str, top_k: int) -> list[RankedPassage]:
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        ordered = sorted(zip(self.passages, scores), key=lambda item: (-float(item[1]), item[0].chunk_id))[:top_k]
        return [RankedPassage(passage, float(score), rank) for rank, (passage, score) in enumerate(ordered, 1)]


def bge_rerank(query: str, ranked: list[RankedPassage], top_k: int, model_name: str = "BAAI/bge-reranker-v2-m3") -> list[RankedPassage]:
    """Cross-encode a candidate set using BGE reranker v2 M3."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError("BGE reranking requires `pip install -e '.[dense]'`") from error
    model = CrossEncoder(model_name)
    scores = model.predict([(query, item.passage.text) for item in ranked])
    ordered = sorted(zip(ranked, scores), key=lambda item: (-float(item[1]), item[0].passage.chunk_id))[:top_k]
    return [RankedPassage(item.passage, float(score), rank) for rank, (item, score) in enumerate(ordered, 1)]

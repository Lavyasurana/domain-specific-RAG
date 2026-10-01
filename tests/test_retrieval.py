from legal_rag.models import Passage
from legal_rag.retrieval import BM25Retriever


def test_bm25_returns_relevant_passage_first() -> None:
    passages = [Passage("a:1", "a", "Anticipatory bail under Section 438 protects personal liberty."), Passage("b:1", "b", "Property possession requires proof of continuous occupation.")]
    ranking = BM25Retriever(passages).search("Section 438 anticipatory bail", top_k=2)
    assert [item.passage.chunk_id for item in ranking] == ["a:1", "b:1"]

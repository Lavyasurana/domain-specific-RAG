from legal_rag.metrics import evaluate_retrieval, mrr_at_k, ndcg_at_k, recall_at_k
from legal_rag.models import BenchmarkQuestion, Passage, RankedPassage


def question() -> BenchmarkQuestion:
    return BenchmarkQuestion("q1", "question", True, ("case_a:one", "case_b:one"), {"case_a:one": 3, "case_b:one": 2}, "holding")


def ranked() -> list[RankedPassage]:
    return [RankedPassage(Passage("case_z:one", "case_z", "noise"), 0.9, 1), RankedPassage(Passage("case_b:one", "case_b", "relevant"), 0.8, 2), RankedPassage(Passage("case_a:one", "case_a", "best"), 0.7, 3)]


def test_rank_metrics_use_gold_passages_and_grades() -> None:
    assert recall_at_k(question(), ranked(), 1) == 0.0
    assert recall_at_k(question(), ranked(), 2) == 1.0
    assert mrr_at_k(question(), ranked()) == 0.5
    assert 0 < ndcg_at_k(question(), ranked()) < 1


def test_evaluation_produces_macro_averages() -> None:
    rows, summary = evaluate_retrieval([question()], {"q1": ranked()})
    assert len(rows) == 1
    assert summary["passage_recall_at_3"] == 1.0

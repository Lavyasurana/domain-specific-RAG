"""Auditable retrieval metrics for answerable benchmark questions."""

from __future__ import annotations

import math
from collections import defaultdict

from .models import BenchmarkQuestion, RankedPassage


def _is_answerable(question: BenchmarkQuestion) -> bool:
    return question.answerable


def recall_at_k(question: BenchmarkQuestion, ranked: list[RankedPassage], k: int) -> float | None:
    if not _is_answerable(question):
        return None
    return float(bool({item.passage.chunk_id for item in ranked[:k]} & set(question.gold_passage_ids)))


def mrr_at_k(question: BenchmarkQuestion, ranked: list[RankedPassage], k: int = 10) -> float | None:
    if not _is_answerable(question):
        return None
    gold = set(question.gold_passage_ids)
    return next((1 / item.rank for item in ranked[:k] if item.passage.chunk_id in gold), 0.0)


def ndcg_at_k(question: BenchmarkQuestion, ranked: list[RankedPassage], k: int = 10) -> float | None:
    if not _is_answerable(question):
        return None
    actual = sum((2 ** question.relevance_grades.get(item.passage.chunk_id, 0) - 1) / math.log2(i + 2) for i, item in enumerate(ranked[:k]))
    ideal = sum((2**grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(question.relevance_grades.values(), reverse=True)[:k]))
    return actual / ideal if ideal else 0.0


def context_precision_at_k(question: BenchmarkQuestion, ranked: list[RankedPassage], k: int) -> float | None:
    if not _is_answerable(question):
        return None
    hits = sum(item.passage.chunk_id in set(question.gold_passage_ids) for item in ranked[:k])
    return hits / len(ranked[:k]) if ranked[:k] else 0.0


def document_recall_at_k(question: BenchmarkQuestion, ranked: list[RankedPassage], k: int) -> float | None:
    if not _is_answerable(question):
        return None
    gold_docs = {chunk_id.split(":", 1)[0] for chunk_id in question.gold_passage_ids}
    return float(bool(gold_docs & {item.passage.document_id for item in ranked[:k]}))


def evaluate_retrieval(questions: list[BenchmarkQuestion], rankings: dict[str, list[RankedPassage]], cutoffs: tuple[int, ...] = (1, 3, 5, 10, 20)) -> tuple[list[dict], dict]:
    rows, aggregate = [], defaultdict(list)
    for question in questions:
        ranked = rankings[question.question_id]
        row = {"question_id": question.question_id, "question_type": question.question_type, "answerable": question.answerable, "ranked_chunk_ids": [item.passage.chunk_id for item in ranked]}
        if question.answerable:
            for cutoff in cutoffs:
                values = {f"passage_recall_at_{cutoff}": recall_at_k(question, ranked, cutoff), f"document_recall_at_{cutoff}": document_recall_at_k(question, ranked, cutoff), f"context_precision_at_{cutoff}": context_precision_at_k(question, ranked, cutoff)}
                row.update(values)
                for name, value in values.items(): aggregate[name].append(value)
            for name, value in {"mrr_at_10": mrr_at_k(question, ranked), "ndcg_at_10": ndcg_at_k(question, ranked)}.items():
                row[name] = value; aggregate[name].append(value)
        rows.append(row)
    summary = {name: sum(values) / len(values) for name, values in aggregate.items() if values}
    summary.update(answerable_questions=sum(q.answerable for q in questions), unanswerable_questions=sum(not q.answerable for q in questions))
    return rows, summary

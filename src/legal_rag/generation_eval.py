"""Deterministic generation checks; LLM judging can be added as a provider adapter."""

from __future__ import annotations

import re

from .generation import GeneratedAnswer
from .models import BenchmarkQuestion, RankedPassage


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def evaluate_answer(question: BenchmarkQuestion, answer: GeneratedAnswer, context: list[RankedPassage]) -> dict:
    context_by_id = {item.passage.chunk_id: _norm(item.passage.text) for item in context}
    citations_valid = all(citation in context_by_id for citation in answer.citations)
    fact_recall = None
    if question.key_facts:
        fact_recall = sum(_norm(fact) in _norm(answer.answer) for fact in question.key_facts) / len(question.key_facts)
    abstention_correct = answer.abstained == (not question.answerable)
    return {"question_id": question.question_id, "abstained": answer.abstained, "abstention_correct": abstention_correct, "citation_precision_proxy": float(citations_valid), "key_fact_recall": fact_recall}

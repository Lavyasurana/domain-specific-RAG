"""Stable schemas shared by ingestion, retrieval, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """A source-traceable span of a judgment."""

    chunk_id: str
    document_id: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A human-reviewed retrieval target; unanswerable questions have no gold."""

    question_id: str
    question: str
    answerable: bool
    gold_passage_ids: tuple[str, ...]
    relevance_grades: dict[str, int]
    question_type: str
    reference_answer: str = ""
    key_facts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedPassage:
    passage: Passage
    score: float
    rank: int

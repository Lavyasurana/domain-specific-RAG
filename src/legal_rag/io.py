"""JSONL readers with validation at the data boundary."""

from __future__ import annotations

import json
from pathlib import Path

from .models import BenchmarkQuestion, Passage


def _records(path: str | Path) -> list[dict]:
    file_path = Path(path)
    records: list[dict] = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {file_path}:{line_number}") from error
    return records


def load_passages(path: str | Path) -> list[Passage]:
    passages = []
    for record in _records(path):
        required = {"chunk_id", "document_id", "text"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Passage missing fields: {sorted(missing)}")
        passages.append(Passage(**{key: record.get(key) for key in Passage.__dataclass_fields__}))
    if len({passage.chunk_id for passage in passages}) != len(passages):
        raise ValueError("chunk_id values must be unique")
    return passages


def load_benchmark(path: str | Path) -> list[BenchmarkQuestion]:
    questions = []
    for record in _records(path):
        required = {"id", "question", "answerable", "type"}
        missing = required - record.keys()
        if missing:
            raise ValueError(f"Benchmark item missing fields: {sorted(missing)}")
        gold = record.get("gold_passages", [])
        gold_ids = tuple(item["chunk_id"] for item in gold)
        if record["answerable"] and not gold_ids:
            raise ValueError(f"Answerable question {record['id']} needs gold evidence")
        questions.append(BenchmarkQuestion(
            question_id=record["id"], question=record["question"], answerable=bool(record["answerable"]),
            gold_passage_ids=gold_ids, relevance_grades={item["chunk_id"]: int(item.get("relevance", 3)) for item in gold},
            question_type=record["type"], reference_answer=record.get("reference_answer", ""),
            key_facts=tuple(record.get("key_facts", [])),
        ))
    return questions

"""LLM-as-judge adapter for faithfulness and key-fact correctness checks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .generation import ResponsesClient
from .models import BenchmarkQuestion, RankedPassage


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: float
    key_fact_recall: float
    citation_support: float
    notes: str
    model: str
    same_family_as_generator: bool


def judge_answer(question: BenchmarkQuestion, answer: str, citations: tuple[str, ...], context: list[RankedPassage], client: ResponsesClient | None = None, model: str = "gpt-4o-mini", generator_model: str = "gpt-4o-mini") -> JudgeResult:
    """Judge only provided evidence; caller records the same-family caveat."""
    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("LLM judging requires `pip install -e '.[openai]'`") from error
        client = OpenAI()
    evidence = "\n\n".join(f"[{item.passage.chunk_id}] {item.passage.text}" for item in context)
    prompt = f"""Evaluate a legal-RAG answer using only the evidence. Return JSON only with numeric fields faithfulness, key_fact_recall, citation_support (each 0 to 1), and string notes.
Question: {question.question}
Reference answer: {question.reference_answer}
Key facts: {list(question.key_facts)}
Answer: {answer}
Citations: {list(citations)}
Evidence: {evidence}"""
    response = client.responses.create(model=model, input=prompt, temperature=0, max_output_tokens=350)
    try:
        payload = json.loads(response.output_text)
        return JudgeResult(*(float(payload[key]) for key in ("faithfulness", "key_fact_recall", "citation_support")), str(payload["notes"]), model, model == generator_model)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Judge returned invalid JSON") from error

"""LLM-as-judge adapter for faithfulness and key-fact correctness checks."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .generation import ChatModel
from .models import BenchmarkQuestion, RankedPassage


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: float
    key_fact_recall: float
    citation_support: float
    notes: str
    model: str
    same_family_as_generator: bool


def judge_answer(question: BenchmarkQuestion, answer: str, citations: tuple[str, ...], context: list[RankedPassage], llm: ChatModel | None = None, model: str = "gpt-4o-mini", generator_model: str = "gpt-4o-mini") -> JudgeResult:
    """Judge only provided evidence; caller records the same-family caveat."""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as error:
        raise RuntimeError("LLM judging requires `pip install -e '.[openai]'`") from error
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("LLM judging requires `pip install -e '.[openai]'`") from error
        llm = ChatOpenAI(model=model, temperature=0, max_tokens=350)
    evidence = "\n\n".join(f"[{item.passage.chunk_id}] {item.passage.text}" for item in context)
    prompt = ChatPromptTemplate.from_template("""Evaluate a legal-RAG answer using only the evidence. Return JSON only with numeric fields faithfulness, key_fact_recall, citation_support (each 0 to 1), and string notes.
Question: {question}
Reference answer: {reference_answer}
Key facts: {key_facts}
Answer: {answer}
Citations: {citations}
Evidence: {evidence}""")
    response = llm.invoke(prompt.invoke({"question": question.question, "reference_answer": question.reference_answer, "key_facts": list(question.key_facts), "answer": answer, "citations": list(citations), "evidence": evidence}))
    try:
        payload = json.loads(StrOutputParser().invoke(response))
        return JudgeResult(*(float(payload[key]) for key in ("faithfulness", "key_fact_recall", "citation_support")), str(payload["notes"]), model, model == generator_model)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Judge returned invalid JSON") from error

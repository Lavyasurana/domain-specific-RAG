"""Grounded, citation-first answer assembly with a safe extractive default."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .models import RankedPassage
from .retrieval import tokenize


@dataclass(frozen=True)
class GeneratedAnswer:
    answer: str
    citations: tuple[str, ...]
    abstained: bool
    reason: str


class ChatModel(Protocol):
    """The LangChain chat-model contract used by the generation pipeline."""

    def invoke(self, input, **kwargs): ...


def generate_grounded_answer(question: str, context: list[RankedPassage], minimum_overlap: int = 1) -> GeneratedAnswer:
    """Extractive baseline: every emitted sentence carries its source chunk ID."""
    terms = set(tokenize(question))
    selected = []
    for item in context:
        sentences = re.split(r"(?<=[.!?])\s+", item.passage.text)
        best = max(sentences, key=lambda sentence: len(terms & set(tokenize(sentence))), default="")
        if len(terms & set(tokenize(best))) >= minimum_overlap:
            selected.append((best.strip(), item.passage.chunk_id))
    if not selected:
        return GeneratedAnswer("I could not find support for that in this corpus.", (), True, "insufficient_retrieval_evidence")
    selected = selected[:3]
    return GeneratedAnswer(" ".join(f"{sentence} [{chunk_id}]" for sentence, chunk_id in selected), tuple(chunk_id for _, chunk_id in selected), False, "grounded_extractive")


def generate_openai_answer(question: str, context: list[RankedPassage], llm: ChatModel | None = None, model: str = "gpt-4o-mini") -> GeneratedAnswer:
    """Generate with a LangChain chat model and validate every returned citation ID."""
    if not context:
        return GeneratedAnswer("I could not find support for that in this corpus.", (), True, "empty_context")
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as error:
        raise RuntimeError("OpenAI generation requires `pip install -e '.[openai]'`") from error
    if llm is None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("OpenAI generation requires `pip install -e '.[openai]'`") from error
        llm = ChatOpenAI(model=model, temperature=0, max_tokens=700)
    evidence = "\n\n".join(f"[{item.passage.chunk_id}]\n{item.passage.text}" for item in context)
    prompt = ChatPromptTemplate.from_messages([("system", """You are a legal-research assistant. Answer only from the supplied evidence.
Do not provide legal advice or claim a judgment is current or controlling law.
Every material factual or legal claim must end with one or more exact evidence IDs in square brackets.
If the evidence does not support an answer, reply exactly: I could not find support for that in this corpus.
Do not cite an ID that is not in the evidence."""), ("human", "Question: {question}\n\nEvidence:\n{evidence}")])
    # Prompt construction, model invocation, and message parsing are all LangChain
    # components; retrieval remains separately auditable for benchmark evaluation.
    response = llm.invoke(prompt.invoke({"question": question, "evidence": evidence}))
    text = StrOutputParser().invoke(response).strip()
    valid_ids = {item.passage.chunk_id for item in context}
    cited_ids = tuple(dict.fromkeys(re.findall(r"\[([^\]]+)\]", text)))
    valid_citations = tuple(citation for citation in cited_ids if citation in valid_ids)
    refusal = text == "I could not find support for that in this corpus."
    if refusal:
        return GeneratedAnswer(text, (), True, "model_insufficient_evidence")
    if not valid_citations or len(valid_citations) != len(cited_ids):
        return GeneratedAnswer("I could not find support for that in this corpus.", (), True, "invalid_or_missing_citations")
    return GeneratedAnswer(text, valid_citations, False, f"langchain-openai:{model}")

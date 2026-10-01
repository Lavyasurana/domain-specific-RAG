"""Optional FastAPI surface for inspection-oriented legal RAG queries."""

from __future__ import annotations

import os

from .generation import generate_openai_answer
from .hybrid import HybridRetriever
from .io import load_passages


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel
    except ImportError as error:
        raise RuntimeError("API requires `pip install -e '.[api]'`") from error
    passages_path = os.environ.get("LEGAL_RAG_PASSAGES")
    if not passages_path:
        raise RuntimeError("Set LEGAL_RAG_PASSAGES to a JSONL passage file")
    retriever = HybridRetriever(load_passages(passages_path))
    app = FastAPI(title="Evaluated Legal RAG", version="0.1.0")

    class Query(BaseModel):
        question: str
        top_k: int = 5

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query")
    def query(request: Query) -> dict:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="question is required")
        context = retriever.search(request.question, min(max(request.top_k, 1), 12))
        answer = generate_openai_answer(request.question, context)
        return {"answer": answer.answer, "citations": answer.citations, "abstained": answer.abstained, "reason": answer.reason, "passages": [{"chunk_id": item.passage.chunk_id, "document_id": item.passage.document_id, "section": item.passage.section, "score": item.score, "text": item.passage.text} for item in context]}

    return app

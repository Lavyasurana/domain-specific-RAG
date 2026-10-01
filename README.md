# Evaluated Legal RAG

A framework-light retrieval-augmented generation research project over Indian
Supreme Court judgments. It measures retrieval separately from answer quality,
validates an LLM judge against human labels, and treats abstention as a
first-class behavior.

The current baseline includes structured passages, deterministic BM25 retrieval,
retrieval metrics, YAML configuration, and a small fixture benchmark. The
production corpus and frozen benchmark are intentionally not included.

> This is a research tool, not legal advice. A retrieved judgment does not prove
> that it is current, controlling, or applicable to a particular situation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
legal-rag-eval evaluate --config configs/baseline.yaml
pytest
```

The evaluation command writes per-question rankings and aggregate metrics to
the configured results directory. Fixture scores are only a pipeline check, not
research findings. See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the full protocol.

## Import, API, and demo

Import approved local source files (text, HTML, or PDF with the `pdf` extra):

```bash
legal-rag-eval ingest --source-dir data/raw --output data/processed/passages.jsonl
```

Run the API after installing `.[api]`:

```bash
LEGAL_RAG_PASSAGES=data/processed/passages.jsonl uvicorn 'legal_rag.api:create_app' --factory
```

Run the inspection UI after installing `.[demo,openai]` and exporting an API key:

```bash
export OPENAI_API_KEY="your_api_key_here"
LEGAL_RAG_PASSAGES=data/processed/passages.jsonl streamlit run app/streamlit_app.py
```

The application uses OpenAI's `gpt-4o-mini` through the Responses API for
generation. Every material claim must cite a retrieved chunk; outputs with
missing or invented citation IDs are rejected into an abstention response.

## Model roles

* **Embeddings:** local `BAAI/bge-m3`, enabled by `pip install -e '.[dense]'`.
* **Reranker:** local `BAAI/bge-reranker-v2-m3`, enabled by the same extra.
* **Generator:** OpenAI `gpt-4o-mini`.
* **Judge:** OpenAI `gpt-4o-mini`, as configured by this project. This is useful
  for development but is marked `same_family_as_generator: true`; validate it
  against human labels before using its score as a final research claim.

"""Config-driven baseline evaluation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .hybrid import HybridRetriever, TfidfDenseRetriever, lexical_rerank
from .ingest import ingest_directory
from .io import load_benchmark, load_passages
from .metrics import evaluate_retrieval
from .retrieval import BM25Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a legal-RAG retrieval evaluation.")
    subcommands = parser.add_subparsers(dest="command")
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    ingest = subcommands.add_parser("ingest")
    ingest.add_argument("--source-dir", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--max-tokens", type=int, default=512)
    ingest.add_argument("--overlap-tokens", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.command == "ingest":
        print(json.dumps({"chunks_written": ingest_directory(arguments.source_dir, arguments.output, arguments.max_tokens, arguments.overlap_tokens)}))
        return
    if arguments.command != "evaluate":
        parser.error("choose `evaluate` or `ingest`")
    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    passages = load_passages(root / config["data"]["passages"])
    mode = config["retrieval"]["mode"]
    retriever = {"bm25": BM25Retriever, "dense": TfidfDenseRetriever, "hybrid": HybridRetriever}.get(mode)
    if mode == "bge_m3":
        from .bge import BGEM3Retriever
        retriever = BGEM3Retriever
    if retriever is None:
        raise ValueError(f"Unsupported retrieval mode: {mode}")
    retriever = retriever(passages)
    questions = load_benchmark(root / config["data"]["benchmark"])
    rankings = {question.question_id: retriever.search(question.question, config["retrieval"]["top_k"]) for question in questions}
    if config["retrieval"].get("rerank") == "lexical":
        question_text = {question.question_id: question.question for question in questions}
        rankings = {
            question_id: lexical_rerank(question_text[question_id], ranking, config["retrieval"]["top_k"])
            for question_id, ranking in rankings.items()
        }
    if config["retrieval"].get("rerank") == "bge-reranker-v2-m3":
        from .bge import bge_rerank
        question_text = {question.question_id: question.question for question in questions}
        rankings = {question_id: bge_rerank(question_text[question_id], ranking, config["retrieval"]["top_k"]) for question_id, ranking in rankings.items()}
    rows, summary = evaluate_retrieval(questions, rankings)
    output_dir = root / config["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "per_question.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

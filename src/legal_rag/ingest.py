"""Local import pipeline for approved PDFs, HTML, and plain text files."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .chunking import chunk_document


def extract_text(path: str | Path) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".html", ".htm"}:
        raw = file_path.read_text(encoding="utf-8", errors="replace")
        return re.sub(r"<[^>]+>", " ", raw) if suffix != ".txt" else raw
    if suffix == ".pdf":
        try:
            import fitz  # type: ignore[import-not-found]
        except ImportError as error:
            raise RuntimeError("PDF ingestion requires `pip install -e '.[pdf]'`") from error
        with fitz.open(file_path) as pdf:
            return "\n\n".join(page.get_text("text") for page in pdf)
    raise ValueError(f"Unsupported source type: {suffix}")


def ingest_directory(source_dir: str | Path, output_path: str | Path, max_tokens: int = 512, overlap_tokens: int = 64) -> int:
    source = Path(source_dir)
    rows = []
    for path in sorted(item for item in source.rglob("*") if item.suffix.lower() in {".pdf", ".txt", ".html", ".htm"}):
        text = extract_text(path).strip()
        if not text:
            continue
        document_id = path.stem
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        for chunk in chunk_document(document_id, text, max_tokens, overlap_tokens):
            rows.append({"chunk_id": chunk.chunk_id, "document_id": document_id, "text": chunk.text, "section": chunk.section, "source_path": str(path), "source_sha256": checksum})
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return len(rows)

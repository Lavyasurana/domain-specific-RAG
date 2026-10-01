"""Structure-preserving chunking with stable, inspectable identifiers."""

from __future__ import annotations

import re

from .models import Passage

# Do not allow newlines inside generic headings: PDF page-letter sequences such
# as "A\nB\nC" are body text, not a section title.
HEADING = re.compile(r"^(?:[A-Z][A-Z ]{3,}|(?:FACTS|ISSUES|REASONING|ANALYSIS|HOLDING|ORDER))$", re.MULTILINE)
WORD = re.compile(r"\S+")


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING.finditer(text))
    if not matches:
        return [("body", text.strip())]
    sections = []
    if text[:matches[0].start()].strip():
        sections.append(("preamble", text[:matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group().strip().lower().replace(" ", "_"), text[match.end():end].strip()))
    return [(name, body) for name, body in sections if body]


def chunk_document(document_id: str, text: str, max_tokens: int = 512, overlap_tokens: int = 64) -> list[Passage]:
    """Chunk only within section boundaries; token positions are deterministic."""
    if max_tokens < 1 or not 0 <= overlap_tokens < max_tokens:
        raise ValueError("max_tokens must be positive and overlap smaller than max_tokens")
    chunks, sequence = [], 0
    for section, body in split_sections(text):
        words = WORD.findall(body)
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            sequence += 1
            chunks.append(Passage(f"{document_id}:{section}:{sequence:04d}", document_id, " ".join(words[start:end]), section=section))
            if end == len(words):
                break
            start = end - overlap_tokens
    return chunks

from legal_rag.chunking import chunk_document
from legal_rag.generation import generate_grounded_answer
from legal_rag.models import Passage, RankedPassage


def test_chunking_preserves_section_and_stable_id() -> None:
    chunks = chunk_document("case", "FACTS\nOne two three.\nHOLDING\nFour five six.", max_tokens=3, overlap_tokens=0)
    assert chunks[0].chunk_id == "case:facts:0001"
    assert chunks[-1].section == "holding"


def test_generator_cites_extracted_evidence() -> None:
    context = [RankedPassage(Passage("case:holding:1", "case", "Section 438 allows anticipatory bail."), 1.0, 1)]
    answer = generate_grounded_answer("What does Section 438 allow?", context)
    assert not answer.abstained
    assert answer.citations == ("case:holding:1",)

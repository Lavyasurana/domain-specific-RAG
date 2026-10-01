from legal_rag.ingest import ingest_directory
from legal_rag.io import load_passages


def test_ingest_text_source_into_traceable_chunks(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "judgment.txt").write_text("HOLDING\nSection 438 protects liberty.", encoding="utf-8")
    destination = tmp_path / "passages.jsonl"
    assert ingest_directory(source, destination, max_tokens=10, overlap_tokens=0) == 1
    passage = load_passages(destination)[0]
    assert passage.document_id == "judgment"
    assert passage.section == "holding"

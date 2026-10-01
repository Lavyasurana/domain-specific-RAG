.PHONY: eval test lint ingest

eval:
	python3 -m legal_rag.cli evaluate --config $(CONFIG)

ingest:
	python3 -m legal_rag.cli ingest --source-dir $(SOURCE_DIR) --output $(OUTPUT)

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check src tests app

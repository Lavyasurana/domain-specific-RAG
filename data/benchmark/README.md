# Pilot annotation queue

The eight-document pilot corpus has been ingested to
`data/processed/pilot_2023_anticipatory_bail.jsonl`. Create questions only after
reading the source PDFs and recording the supporting stable `chunk_id` values.

Use one JSON object per line with this minimum shape:

```json
{"id":"pilot_001","type":"holding","question":"...","answerable":true,"gold_passages":[{"chunk_id":"...","relevance":3}],"reference_answer":"...","key_facts":["..."]}
```

Keep candidate questions in `pilot_review_queue.jsonl`; move only independently
reviewed items into a versioned development or frozen test benchmark.

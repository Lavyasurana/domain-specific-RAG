# Data card

The repository has no production legal corpus. Use only a source whose terms,
license, and permitted access method have been reviewed. For every imported
document retain: source URL, retrieval date, court/date/citation metadata,
original-file checksum, extraction version, and exclusion reason if rejected.

`legal-rag-eval ingest` accepts local `.txt`, `.html`, and `.pdf` files. PDFs
need the optional `pdf` dependency. It does not scrape websites; this is
intentional to keep acquisition subject to explicit source and terms review.

Before using an imported corpus for research claims, manually inspect at least
50 extractions for page order, headings, paragraph continuity, and metadata.
Do not commit raw documents or private annotations unless their license permits
redistribution.

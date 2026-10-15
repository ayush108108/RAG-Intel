# Python documentation passage eval v1

Four complete tutorial source chapters from **CPython v3.12.7**, commit
`0b05ead877f909b7efe712db758012d9dbece7ce`: Data Structures, Modules,
Input and Output, Errors and Exceptions. Total: 93,969 bundled bytes.
`manifest.json` records the immutable upstream URLs and SHA256 checksums.

## Redistribution

Python documentation is licensed under **PSF-2.0**; its code examples are also
available under 0BSD. See the [upstream terms](https://docs.python.org/3.12/license.html)
and the retained complete [upstream licence](LICENSE-PYTHON.txt), including the
PSF copyright notice. These files retain their upstream licence rather than the
repository's MIT licence. Copyright (c) 2001-2024 Python Software Foundation;
All Rights Reserved.

Change summary: the four `.rst` files are saved with a `.txt` extension so the
existing plain-text ingestion adapter can read them. Two `sectionauthor`
directives containing a contributor name/email are removed from Data Structures
to keep contact details out of the corpus. Other source bytes are unchanged.
The normalizer and chunker transform a local indexed copy;
the raw files and notices remain intact. reStructuredText directives are left
visible deliberately: this is source-document retrieval, not rendered HTML.
This collection contains no client documents, email messages or personal data.

## Labels and protocol

`questions.jsonl` contains 40 questions: 36 answerable (including 16 tagged hard)
and 4 negative. Questions and supporting paragraphs were manually authored and
selected by Codex from these files before the first retrieval run, not sampled
from search results. Each positive records a document ID, exact quote and
zero-based half-open character offsets into `normalize_text(raw UTF-8 text)`.
Negative rows have no gold and explain the missing information. No independent
human relevance review has occurred; judgments can be incomplete or mistaken.

The harness validates every source checksum and gold span before evaluation.
Stable chunk IDs combine document ID, chunk ordinal and a content hash. The
reports map gold spans to **all** chunks containing the full gold quote. A hit
requires the correct document and complete quote in a retrieved chunk; finding
some other passage in the same chapter does not count. Long gold paragraphs
may not fit a chunk: these remain misses in the denominator. This conservative
support definition is not a human score of partially useful passages.

Hit@1/3/5 is the fraction of the 36 answerable questions with at least one full
support passage in the first k chunks. MRR averages the reciprocal of the first
full-support rank over the entire positive-score ranking, with zero for a miss.
This is **hit rate**, not recall over all relevant passages. Negative questions
are excluded from these denominators and separately report how many return any
passages. There is no calibrated abstention mechanism or generation evaluation.

## Controlled repair

Baseline: existing starter chunking (640 characters, 64 overlap), existing
lowercase Unicode word tokenizer, BM25 k1=1.5/b=0.75. Repair: remove only the
explicit common-word list in `manifest.json` from the query before BM25 search.
Index text, corpus, labels, chunking, scoring and tie order remain fixed. This
is an explicit example query policy in `rag_intel.proof`, not a silent change to
the engine defaults. Only this one repair candidate was tried. The corpus contact
cleanup happened after a preliminary run; questions and gold quotes were kept
fixed and their character offsets were recalculated before the recorded runs.

Both use the real `RAGIntelEngine.ingest` pipeline and its `sparse.search`.
Evaluation is chunk ranking **before** document aggregation/RRF; it does not
measure the document rankings from the existing `rag-intel search` command.
No embedding model is used (`embedding_model: null`), so model pinning and
embedding downloads do not apply. Dependencies are pinned in `uv.lock`; the
seed is 0 and no randomized component is used. The fingerprinted local SQLite
cache can be discarded by passing a new `--cache` directory. No network access
is needed after dependency installation.

This is a development set inspected during repair, with **no held-out split**.
The small gain and regression are evidence about this one collection only.
For a client collection, collect separate human-reviewed questions and an
untouched test split before quoting an improvement as independent validation.

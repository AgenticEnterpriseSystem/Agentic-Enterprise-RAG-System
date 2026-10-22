# Evaluation files

This folder contains **saved data, not executable code**. Evaluation commands and
metrics live in `src/evaluation/`; automated correctness checks live in `tests/`.
Large corpus/index files and new local experiment outputs live in `.cache/`.
Start with the [README commands](../README.md#everyday-commands).

| File or folder | What it is |
| --- | --- |
| `splits.json` | Which questions are in the development (300), calibration (100), and final (100) splits. Fixed; do not tune on the final split. |
| `corpus_manifest.json` | Record of the exact dataset used: source commit, document counts, and a content fingerprint. Written by `python -m src.data.corpus`. |
| `results/bm25_full_development/` | BM25+ baseline results on the development split |
| `results/dense_full_development/` | BGE-small + FAISS baseline results on the development split |
| `results/dense_titled_full_development/` | BGE-small with the document title on every passage (experiment) |
| `results/dense_base_titled_full_development/` | BGE-base with the document title on every passage (experiment) |
| `results/hybrid_bge-base_*_full_development/` | BM25+ and BGE-base hybrid variants: RRF weights 0.5 and 0.7, and weight 0.7 with each reranker |
| `results/*_full_calibration/` | The one calibration check: BM25+ alone, BGE-base alone, and the chosen hybrid (BM25 weight 0.7) |
| `results/screening/` | Embedding model screening on a 20,364-document sample (development questions): BM25+, BGE-base, GTE, Arctic and Qwen3, alone and as hybrids |

Each results folder contains `summary.json` (all scores), `predictions.jsonl` (the
top 20 documents per question), and `upstream_answers.jsonl` (the top 10 in the
benchmark's own format).

The data comes from Onyx's
[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) (MIT).
Read [Baseline Results](../docs/FULL_CORPUS.md) for measured retrieval scores,
[Evaluation](../docs/EVALUATION.md) for scoring rules and the paused C3 plan,
and the [Data Card](../docs/DATA_CARD.md) for provenance and limitations.

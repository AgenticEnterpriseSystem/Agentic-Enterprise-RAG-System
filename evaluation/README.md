# Saved evaluation artifacts

See [Evaluation and results](../docs/EVALUATION.md) for interpretation, scoring
and reproduction commands.

| Path | Contents |
| --- | --- |
| [splits.json](splits.json) | Frozen development (300), calibration (100) and final (100) question IDs, and the golden subset. Do not tune on the final split. |
| [corpus_manifest.json](corpus_manifest.json) | Dataset revision, counts, coverage and fingerprints. Written by `python -m src.data.corpus`. |
| [results/bm25_full_development/](results/bm25_full_development/) | Full-corpus BM25+ predictions and scores |
| [results/dense_full_development/](results/dense_full_development/) | Full-corpus BGE-small predictions and scores |
| [results/dense_titled_full_development/](results/dense_titled_full_development/) | BGE-small with the document title on every passage (experiment) |
| [results/dense_base_titled_full_development/](results/dense_base_titled_full_development/) | BGE-base with the document title on every passage |
| `results/hybrid_bge-base_*_full_development/` | BM25+ and BGE-base hybrids: RRF weights 0.5 and 0.7, and weight 0.7 with each reranker. Weight 0.7 without a reranker is the chosen setting. |
| `results/*_full_calibration/` | Single calibration checks: BM25+ alone, BGE-base alone, Arctic alone, and the BGE-base and Arctic hybrids (BM25 weight 0.7) |
| [results/screening/](results/screening/) | Embedding model screening on a 20,364-document sample (development questions): BM25+, BGE-base, GTE, Arctic and Qwen3, alone and as hybrids |
| `results/*arctic-embed-l-v2.0*` | Arctic-embed-l-v2.0 on the full corpus: alone and in hybrids with BM25+ |
| [agent_development/20261021.json](agent_development/20261021.json) | Historical live development measurements and provenance |
| [agent_development/critic_before_scope.txt](agent_development/critic_before_scope.txt) | Saved prompt for repeatable development comparisons |
| [../outputs/alex_evaluation/](../outputs/alex_evaluation/) | Answer-evaluation baseline: 20 curated development questions, their BM25 evidence, GPT-OSS answers, LLM-judge scores and the human/judge comparison workbook. See [Answer evaluation](../docs/ANSWER_EVALUATION.md). |

Large corpus/index files and full answer/evidence traces stay in ignored `.cache/`.
Old local reduced-corpus experiments are not the full-corpus results above. Review
selection files and empty review templates do not establish completed human review.

Hybrid, reranker, model-screening and Arctic results are documented in [Retrieval results](../docs/FULL_CORPUS.md).

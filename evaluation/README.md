<a id="saved-evaluation-artifacts"></a>
# Saved evaluation files

This directory contains question splits and saved experiment results. Reading these files does not call a model. For what the numbers mean and how to run a new check, use [Evaluation](../docs/EVALUATION.md). For the current release state, use [Current status](../docs/STATUS.md).

## Where to start

1. Open a result directory's `summary.json` for its measurements and settings.
2. Open `predictions.jsonl` to inspect individual questions and ranked document IDs, where that file is present.
3. Check its manifest/provenance before comparing it with another run. Older unrestricted retrieval results and newer user-permission-aware results use different reference counts.

| File or directory | What it contains |
| --- | --- |
| [splits.json](splits.json) | Frozen question IDs: 300 development, 100 calibration and 100 final, plus a golden development subset |
| [corpus_manifest.json](corpus_manifest.json) | Dataset revision, document counts, coverage and fingerprints |
| [results/bm25_full_development](results/bm25_full_development/) | Historical full-corpus BM25+ search results |
| [results/dense_full_development](results/dense_full_development/) | Historical BGE-small + FAISS search results |
| [results/dense_titled_full_development](results/dense_titled_full_development/) | BGE-small with a title on every passage |
| [results/dense_base_titled_full_development](results/dense_base_titled_full_development/) | BGE-base with a title on every passage |
| `results/hybrid_bge-base_*_full_development/` | BGE-base hybrids at BM25 weights 0.5/0.7 and the tested rerankers; the chosen setting is 0.7 without a reranker |
| `results/*arctic-embed-l-v2.0*` | Arctic alone and combined with BM25+ |
| `results/*_full_calibration/` | Single historical calibration checks for BM25, dense models and hybrids |
| [results/screening](results/screening/) | Embedding-model checks on a 20,364-document sample using development questions |
| [agent_development/20261021.json](agent_development/20261021.json) | Historical BM25/Haiku agent measurements with model, prompt and source identities |
| [agent_development/critic_before_scope.txt](agent_development/critic_before_scope.txt) | Frozen historical comparison prompt; not the current default prompt |
| [outputs/alex_evaluation](../outputs/alex_evaluation/) | Historical 20-question BM25/GPT-OSS answers, machine grades and comparison workbook |

[Full-corpus results](../docs/FULL_CORPUS.md) explains the historical retrieval experiments. [Answer evaluation](../docs/ANSWER_EVALUATION.md) explains the older GPT-OSS workflow.

## Avoid these comparison mistakes

- Do not tune on the final question split.
- A retrieved reference document is not proof of a correct answer.
- A model's `answered` status or critic decision is not a human label.
- Blank review sheets do not mean review is complete.
- A current source fix does not change what an older saved experiment measured.
- New runs need new output directories. Keep failures and unknown costs visible.

Large document/index/model files and private execution/review records are excluded from Git. A fresh checkout therefore does not contain every local experiment. The sanitized [status page](../docs/STATUS.md) describes the current 30-outcome baseline, three-case external-judge pilot, pending human critic review and functional browser checks without requiring private local folders.

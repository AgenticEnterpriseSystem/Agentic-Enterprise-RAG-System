# Saved evaluation artifacts

See [Evaluation](../docs/EVALUATION.md) for commands, scoring and comparison methods.

| Path | Contents |
| --- | --- |
| [splits.json](splits.json) | Frozen development/calibration/final IDs and golden subset |
| [corpus_manifest.json](corpus_manifest.json) | Dataset revision, counts, coverage and fingerprints |
| [results/](results/) | Retrieval predictions, scores and reproduction metadata |

[Full corpus results](../docs/FULL_CORPUS.md) explains the BM25, dense, hybrid,
reranking and model-screening measurements. These are retrieval results, not
independent validation of generated answers.

Corpus/index files and new run outputs belong in ignored `.cache/` storage.
Use a new output directory per run. Review templates and model-generated scores
alone do not establish completed human evaluation.

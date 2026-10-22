# Saved evaluation artifacts

See [Evaluation and results](../docs/EVALUATION.md) for interpretation, scoring
and reproduction commands.

| Path | Contents |
| --- | --- |
| [splits.json](splits.json) | Frozen development/calibration/final IDs and golden subset |
| [corpus_manifest.json](corpus_manifest.json) | Dataset revision, counts, coverage and fingerprints |
| [results/bm25_full_development/](results/bm25_full_development/) | Full-corpus BM25 predictions and scores |
| [results/dense_full_development/](results/dense_full_development/) | Full-corpus BGE-small predictions and scores |
| [agent_development/20261021.json](agent_development/20261021.json) | Historical live development measurements and provenance |
| [agent_development/critic_before_scope.txt](agent_development/critic_before_scope.txt) | Saved prompt for repeatable development comparisons |

Large corpus/index files and full answer/evidence traces stay in ignored `.cache/`.
Old local reduced-corpus experiments are not the full-corpus results above. Review
selection files and empty review templates do not establish completed human review.

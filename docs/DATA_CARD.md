# Data card: EnterpriseRAG-Bench

The app uses a synthetic workplace dataset: documents, messages and questions about the fictional company **Redwood Inference**. It tests enterprise-style search. It does not represent a real company's data or permissions.

## Source and use

- **Source:** Onyx's [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), also on [Hugging Face](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench).
- **Pinned commit:** `d36685e273713975ee20299bbf1ab64165575b3c`.
- **License:** MIT. The authors ask that it not be used for model training. This project uses it only for retrieval, question answering and evaluation, with no training or fine-tuning. Keep upstream attribution and license notices when sharing data.

[Setup](WEB_APP.md) explains how to build the corpus and indexes. [The corpus manifest](../evaluation/corpus_manifest.json) records counts, extraction rules, duplicate handling and fingerprints.

## Contents

**511,958 unique documents** from nine platforms:

| Source | Documents |
| --- | ---: |
| Slack | 285,605 |
| Gmail | 121,390 |
| Linear | 35,308 |
| Google Drive | 25,108 |
| HubSpot | 15,016 |
| Fireflies | 10,173 |
| GitHub | 8,052 |
| Jira | 6,118 |
| Confluence | 5,188 |

The corpus export has five fields: document ID, source type, title, content and source path. PostgreSQL adds revisions, positions and storage metadata. A separate Jira import keeps supported native fields with provenance.

**500 benchmark questions** come with reference answers (the dataset's expected answer, not the app's output). Thirty have no reference-document labels; 93 have several. All 722 distinct referenced documents are in the corpus.

| Question type | Count |
| --- | ---: |
| Basic | 175 |
| Semantic (different wording) | 125 |
| Intra-document reasoning | 40 |
| Project related | 40 |
| Constrained | 30 |
| Conflicting information | 20 |
| Completeness | 20 |
| Miscellaneous | 20 |
| Information not found | 20 |
| High level | 10 |

### Splits

The frozen [split file](../evaluation/splits.json) uses seed 5980: **300 development, 100 calibration, 100 final**. Questions that share reference documents or duplicate text stay together. Related projects can still cross splits.

- Development: inspect failures and choose settings.
- Calibration: already used in historical retrieval comparisons, so not a fresh test set.
- Final: untouched until settings are frozen.

Questions without reference-document labels are left out of document-recall scoring. An empty list does not prove the corpus has nothing relevant. See [evaluation](EVALUATION.md#splits-and-scoring).

## Extraction and limitations

### Duplicate documents

Upstream has 511,962 files. Four document IDs each name two files. The exporter keeps the path chosen by upstream `uuid_index.json`, leaving 511,958 documents. One duplicated ID is referenced by a question. The manifest lists the excluded paths.

Search text is title and content joined by a newline; passage offsets refer to that text. Cleaning a document changes its fingerprint and offsets, so the corpus and both indexes must be rebuilt together.

Extraction follows upstream field labels and does not unify message formats. List-valued content can keep Python-style list formatting (it is converted with `str(...)`), which can affect keyword tokens.

### Who can read each document?

The dataset has **access-policy prose**, organizational scaffolding and folder paths. For example, see the pinned [request controls](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/sources/confluence/policies-and-process/request-lifecycle-and-controls-catalog-2026.json) and [authorization guidance](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/sources/confluence/security-and-compliance/audit-logging/authorization-logging-and-deprovisioning-guidance-2026.json). Some raw records have fields like `confidentiality` and `owner_team`.

It does not include a complete, authoritative **access map** of who can read what. None has been verified or imported, and the five-field export drops those raw fields.

The app instead uses **reviewed explicit user/group grants** in PostgreSQL. They are operator-defined or synthetic. They do not follow managers, departments, projects, folders or instructions inside documents.

### Hierarchy

Folder trees, company/project links and an employee directory exist, but they only describe structure. No permission inheritance is implemented. Upstream warns that high-volume generated documents do not consistently follow its org chart or roles, and folder placement can be noisy. See upstream's [methodology](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/methodology.md) and [source tree](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/source_tree.txt).

### What can mislead

- **Uneven coverage.** Slack is about 56% of documents. Basic and semantic questions are 60% of questions. Averages can hide weak categories.
- **Near-duplicates.** Similar documents can describe different dates, incidents or customers. One live answer cited a source saying April 7; the reference said April 6.
- **Missing dates.** Undated sources cannot show "latest" or "currently deployed". Date filters exclude them.
- **Incomplete labels.** An unlabeled source may also be valid. Finding a labeled document does not prove the excerpt answers the question.
- **Synthetic scope.** Results do not show accuracy, privacy or access-control correctness for a real organization. Real questions, credentials and operator files still need protection.

## Versions and safe evaluation

Record the data commit, question and split hashes, corpus-file hash, ordered catalog hash and model/index settings for every experiment. A file hash identifies exact bytes; a catalog hash identifies content and order. They are not interchangeable.

Reference answers stay outside the runtime and are joined only for scoring. Do not tune on the final split, treat model-generated labels as human review, or train on the dataset. See [status](STATUS.md).

**Related:** [System flow](SYSTEM.md) · [System card](system_card.md) · [Model card](model_card_hybrid_retriever.md) · [Evaluation](EVALUATION.md)

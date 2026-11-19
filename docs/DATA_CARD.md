# Data card: EnterpriseRAG-Bench

This app uses a synthetic workplace dataset: documents, messages and questions about the fictional company **Redwood Inference**. It is useful for testing enterprise-style search without claiming to represent a real company's data or permissions.

## Source and use

The source is Onyx's [EnterpriseRAG-Bench repository](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), also published on [Hugging Face](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench). This project pins the upstream data to commit `d36685e273713975ee20299bbf1ab64165575b3c` so builds can be reproduced.

The dataset is distributed under MIT. The authors separately request that it not be used for model training. This project uses it for retrieval, question answering and evaluation, with no training or fine-tuning. Retain upstream attribution and license notices when sharing data or derived files.

[Setup](WEB_APP.md) explains how to obtain/build the saved corpus and indexes. [The corpus manifest](../evaluation/corpus_manifest.json) records counts, extraction rules, duplicate handling and content fingerprints.

## Contents

There are **511,958 unique documents** from nine source platforms:

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

The shared corpus export stores five fields: document ID, source type, title, content and source path. PostgreSQL adds verified document revisions, positions and other storage metadata. A separate Jira import preserves supported native fields with their source provenance; generic raw metadata is not automatically imported into this five-column export.

There are **500 benchmark questions** with reference answers/facts. A **reference answer** is the dataset's expected answer, not the app's generated output. Thirty questions have no reference-document labels; 93 have multiple reference documents. All 722 distinct referenced documents are present.

| Question type | Count |
| --- | ---: |
| Basic | 175 |
| Semantic: different wording from the source | 125 |
| Intra-document reasoning | 40 |
| Project related | 40 |
| Constrained | 30 |
| Conflicting information | 20 |
| Completeness | 20 |
| Miscellaneous | 20 |
| Information not found | 20 |
| High level | 10 |

### Development, calibration and final questions

The frozen [split file](../evaluation/splits.json) divides questions into 300 development, 100 calibration and 100 final questions, using seed 5980. Questions sharing reference documents or grouped duplicate text stay together. Related projects can still appear across splits, so this is not complete topic isolation.

Development questions are used to inspect failures and choose settings. Calibration has already been used in historical retrieval comparisons; it is not an untouched test set for repeated tuning. Critic calibration additionally needs real review labels, not just model predictions. The final 100 questions remain untouched until settings are frozen.

Questions with no reference-document labels are excluded from document-recall scoring. An empty label list does not prove that the entire corpus lacks relevant material. Missing-information answers still need review. See [evaluation](EVALUATION.md#splits-and-scoring) for the exact scoring rules.

## Extraction and limitations

### Which duplicate document is kept?

Upstream contains 511,962 files. Four document IDs each name two different files. The exporter keeps the canonical path selected by upstream `uuid_index.json`, leaving 511,958 documents. One duplicated ID is referenced by a question. The manifest records excluded paths so this choice is reproducible.

**Canonical text** means the fixed text used for a document revision. Search combines title and content with a newline; passage offsets refer to that exact text. Reformatting or cleaning a document changes its fingerprint and possibly its offsets, so the corpus and both indexes must be rebuilt together.

Extraction follows upstream title/content field labels. It does not reconstruct every source into a uniform message format. In particular, the single-field branch converts a value with `str(...)`; list-valued content can retain Python-style list formatting. This can affect keyword tokens. A cleanup would be a new data/index version, not an invisible change to the measured dataset.

### Does the dataset define who can read each document?

It contains **access-policy prose**, organizational scaffolding and nested folder paths. For example, pinned source documents discuss [request controls](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/sources/confluence/policies-and-process/request-lifecycle-and-controls-catalog-2026.json) and [authorization/deprovisioning](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/sources/confluence/security-and-compliance/audit-logging/authorization-logging-and-deprovisioning-guidance-2026.json). Some raw records have fields such as `confidentiality` and `owner_team`.

That is different from a complete, authoritative **access map** saying which account/group can read every document. Such a map has not been verified or imported. The five-column export does not retain every raw classification, sharing, personnel or date field as a separate column.

The app therefore uses **reviewed explicit user/group document grants** in PostgreSQL. These local mappings are operator-defined or synthetic. They do not automatically follow managers, departments, project names, folders or instructions inside documents. A policy document saying “managers approve access” does not itself give a manager access.

### Does it contain a hierarchy?

There are source/folder trees, company/project relationships and an employee-directory/organization scaffold. They are descriptive structure, not an implemented permission-inheritance tree. Upstream warns that high-volume generated documents are not consistently grounded in its organization chart or role definitions. Folder placement can also be noisy. See the pinned [methodology](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/methodology.md) and [source tree](https://github.com/onyx-dot-app/EnterpriseRAG-Bench/blob/d36685e273713975ee20299bbf1ab64165575b3c/generated_data/source_tree.txt).

### What can make answers misleading?

- **Uneven coverage.** Slack is about 56% of documents. Basic and semantic questions are 60% of questions. An overall average can hide weak categories.
- **Near-duplicates and conflicting versions.** Similar documents can describe different dates, incidents or customers. A live answer cited a source supporting April 7 while the question's reference document said April 6. Correct citation identity alone does not identify the intended target.
- **Missing dates/metadata.** A source with no reliable timestamp cannot establish “latest” or “currently deployed.” Date filters exclude undated evidence instead of guessing.
- **Incomplete reference labels.** Another source might support a valid answer even if it is not in the label list. Conversely, merely finding a labeled document does not prove that the selected excerpt answers the question.
- **Synthetic scope.** Results do not establish accuracy, privacy or access-control correctness for a real organization. Real user questions, credentials and operator files still need protection even though the corpus is fictional.

## Versions and safe evaluation

Keep the data commit, question/split hashes, physical corpus-file hash, ordered catalog hash and model/index settings with every experiment. A file hash identifies exact bytes; a content/catalog hash identifies its declared representation/order. They are not interchangeable, and matching a friendly snapshot name alone is insufficient.

Reference answers stay outside the runtime and are joined only after generation for scoring. Do not tune on the final split, treat model-generated labels as independent human review, or train on the dataset against the authors' request. [Current status](STATUS.md) separates functional checks, research results and unfinished evaluation.

**Related:** [System flow](SYSTEM.md) · [System card](system_card.md) · [Model card](model_card_hybrid_retriever.md) · [Evaluation](EVALUATION.md)

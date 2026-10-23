# Data card: EnterpriseRAG-Bench

## Source and use

Onyx's [EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench)
contains synthetic workplace documents for the fictional company **Redwood
Inference**. It is also published on
[Hugging Face](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench).
This project pins upstream commit `d36685e273713975ee20299bbf1ab64165575b3c`.

The authors distribute the dataset under MIT and request that it not be used for
model training. We use it for retrieval, question answering and evaluation, not
training/fine-tuning. Retain upstream attribution and license notices when sharing
artifacts. See [setup](WEB_APP.md#get-the-artifacts) for obtaining/building them.

## Contents

The [corpus manifest](../evaluation/corpus_manifest.json) records **511,958 unique
documents**, extraction rules, source counts, reference coverage and fingerprints.
Each extracted document has an ID, source type, title and text.

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

There are **500 questions** with reference answers/facts. Thirty have no reference
document labels; 93 have multiple references. All 722 distinct referenced documents
are present. A reference answer is a benchmark target, not a generated system answer.

| Question type | Count |
| --- | ---: |
| Basic | 175 |
| Semantic | 125 |
| Intra-document reasoning | 40 |
| Project related | 40 |
| Constrained | 30 |
| Conflicting info | 20 |
| Completeness | 20 |
| Miscellaneous | 20 |
| Info not found | 20 |
| High level | 10 |

The [evaluation guide](EVALUATION.md#splits-and-scoring) owns split definitions,
scoring and safeguards against using final questions during development.

## Extraction and limitations

- Upstream has 511,962 files: four IDs each refer to two different files. We keep
  the canonical file selected by upstream `uuid_index.json`; one duplicated ID
  is referenced by a question. Skipped paths are recorded in the manifest.
- Dates, roles, project membership and access permissions are not reliable
  structured fields. Text-derived metadata needs verification. Permission tests
  use synthetic scopes, not real enterprise ACLs.
- Slack dominates the corpus; basic and semantic questions comprise 60% of the
  questions. Aggregate results can hide weaker categories.
- Near-duplicates, old drafts and conflicting documents complicate interpretation.
  Reference labels may be incomplete; review against actual evidence rather than
  treating labels or a model judge as unquestionable truth.
- The synthetic company does not establish performance, privacy guarantees or
  access-control correctness for real organizations. Employee-directory grounding
  is limited, and related projects may cross question splits.

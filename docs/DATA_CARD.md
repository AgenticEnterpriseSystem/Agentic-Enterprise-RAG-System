# Data card: EnterpriseRAG-Bench

## 1. Dataset Overview

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench), made by
Onyx, is a synthetic set of internal documents for a fictional AI company,
"Redwood Inference", plus questions about them. It is designed to test search
and question answering over company data.

- **Version:** we build it from the authors' GitHub repository at commit
  `d36685e273713975ee20299bbf1ab64165575b3c`, so everyone gets identical data.
  The same data is also published on
  [Hugging Face](https://huggingface.co/datasets/onyx-dot-app/EnterpriseRAG-Bench).

Build it with:

```sh
python -m src.data.corpus --clone
```

This writes all documents to `.cache/corpus/documents.parquet` and records counts
and a content fingerprint in `evaluation/corpus_manifest.json`.

## 2. Provenance and Collection

EnterpriseRAG-Bench was created by Onyx as a synthetic benchmark for evaluating
search and question-answering systems over enterprise-style data. The dataset
models a fictional AI company called "Redwood Inference" and contains simulated
content from common workplace systems, including Slack, Gmail, Linear, Google
Drive, HubSpot, Fireflies, GitHub, Jira, and Confluence.

The dataset is entirely synthetic rather than collected from a real company's
internal systems. This avoids the direct use of real employee or company data
that would otherwise introduce privacy and personally identifiable information
concerns.

For reproducibility, this project builds the dataset from the upstream
EnterpriseRAG-Bench repository at commit
`d36685e273713975ee20299bbf1ab64165575b3c`. During processing, the source
documents are converted into a single corpus stored at
`.cache/corpus/documents.parquet`. Corpus counts and a content fingerprint are
recorded in `evaluation/corpus_manifest.json`.

## 3. Licensing and Usage

EnterpriseRAG-Bench is distributed under the MIT License, as declared by the
dataset authors. The license permits use, modification, and redistribution,
provided that the required copyright and license notice is retained.

The dataset authors request that EnterpriseRAG-Bench not be used to train
models. This project follows that guidance and uses the dataset only for search,
retrieval, question answering, and evaluation. No model is trained or
fine-tuned on the EnterpriseRAG-Bench corpus.

## 4. Data Structure and Splits

### Documents

**511,958 documents** from 9 sources:

| Source | What it contains | Documents |
| --- | --- | ---: |
| Slack | Team chat threads | 285,605 |
| Gmail | Email threads | 121,390 |
| Linear | Project tickets | 35,308 |
| Google Drive | Shared documents | 25,108 |
| HubSpot | Sales CRM records | 15,016 |
| Fireflies | Meeting transcripts | 10,173 |
| GitHub | Pull requests | 8,052 |
| Jira | Support tickets | 6,118 |
| Confluence | Wiki pages and runbooks | 5,188 |
| **Total** | | **511,958** |

Each document has an ID, a source, a title, and text. An average document is
about 770 words long.

### Questions

**500 questions**, each labeled with a reference answer and the documents that
contain it.

| Type | Count | What it tests |
| --- | ---: | --- |
| Basic | 175 | One document holds the answer |
| Semantic | 125 | Like basic, but reworded to avoid the document's key words |
| Intra-document reasoning | 40 | Combining distant parts of one long document |
| Project related | 40 | Combining several documents about one project |
| Constrained | 30 | Several documents look relevant; details rule out all but one |
| Conflicting info | 20 | Documents contradict each other |
| Completeness | 20 | Every relevant document (up to 10) is needed |
| Miscellaneous | 20 | Informal or loosely organized documents |
| Info not found | 20 | The answer is not in the data |
| High level | 10 | The answer is spread across the company, not in one document |

- 30 questions (info not found and high level) have no reference documents, so
  they cannot be scored on search.
- 93 questions have more than one reference document.
- All 722 reference documents are in the dataset.

### Evaluation Splits

| Split | Questions | Use |
| --- | ---: | --- |
| Development | 300 | Building and comparing methods |
| Calibration | 100 | Tuning the Search Critic later |
| Final | 100 | Final evaluation only, after all settings are frozen |

Each split has a similar mix of question types. Questions that share a reference
document, or have identical wording, stay in the same split, which limits overlap
between development and final (related questions about the same project can still
land in different splits). We do not train a model, so there is no training split. The split is
fixed in `evaluation/splits.json`; see the [evaluation plan](EVALUATION.md) for
the exact rules.

### Data Quality Notes

- **Duplicate IDs:** the repository has 511,962 document files, but four IDs are
  each used by two different files. We keep the file that the authors' own index
  (`uuid_index.json`) points to. One of these four is a reference document.
- **No structured metadata:** dates, authors' roles, project membership, and
  access permissions are not provided as fields. Any values we extract from the
  text must be checked. Permissions used in our tests are made up for testing.

## 5. Limitations and Risks

- **Synthetic-to-real domain gap:** The dataset is entirely synthetic, so its
  documents and questions may not fully reflect how employees communicate in
  real organizations. In particular, generated workplace conversations may
  contain less incidental noise, tangents, and misunderstanding than real
  workplace communication. As a result, retrieval performance on this benchmark
  may overestimate performance in a real enterprise environment.

- **Source and question imbalance:** The corpus is heavily imbalanced across
  source types, with more than half of the documents coming from Slack. The
  evaluation questions are also imbalanced: Basic and Semantic questions make
  up 60% of the 500 questions. Aggregate evaluation scores may therefore be
  influenced disproportionately by performance on these categories.

- **Injected noise and conflicting information:** The benchmark intentionally
  contains near-duplicate documents, outdated or conflicting information, and
  misfiled documents. These features make retrieval more realistic but can also
  make it difficult to determine which document contains the most current or
  authoritative information.

- **Reference-label quality:** Some reference-document labels may be incomplete.
  The benchmark's evaluation tooling can revise reference answers when candidate
  evidence indicates that the existing answer should change. Benchmark results
  should therefore be interpreted relative to the provided reference answers
  and documents rather than as absolute ground truth.

- **Employee-directory and metadata limitations:** The corpus does not provide
  structured fields for dates, employee roles, project membership, or access
  permissions. In addition, some generated content is not fully grounded in the
  employee directory. This limits how directly the benchmark can support our
  planned role-based access filtering. Any permissions introduced by this
  project are synthetic and used only for testing.

- **Evaluation generalizability:** Because the benchmark represents one
  fictional company and a fixed set of enterprise source types, results may not
  generalize directly to organizations with different communication patterns,
  document structures, access-control policies, or software ecosystems.
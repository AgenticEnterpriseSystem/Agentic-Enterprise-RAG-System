# Documentation

New to the project? Read [Setup](WEB_APP.md) to run the app, or
[How the system works](SYSTEM.md) to understand it first.

## Choose a guide

| Guide | What it explains |
| --- | --- |
| [Project README](../README.md) | What the app does and where to start |
| [Setup](WEB_APP.md) | Install, prepare data, start Docker, create accounts, grant access and use the app |
| [Current status](STATUS.md) | What was checked, known issues and remaining work |
| [How the system works](SYSTEM.md) | Login, permissions, hybrid search, the critic and final answers |
| [Dataset](DATA_CARD.md) | Where the documents come from and what they contain |
| [Retriever](model_card_hybrid_retriever.md) | Keyword search, embeddings, ranking settings and measured retrieval results |
| [Full-corpus guide](FULL_CORPUS.md) | Prepare the complete dataset and understand earlier retrieval experiments |
| [Evaluation](EVALUATION.md) | Test search, the agent and answers without mixing their results |
| [Human-versus-LLM answer review](ANSWER_REVIEW.md) | Score saved answers yourself, then compare with the machine grades |
| [Historical answer baseline](ANSWER_EVALUATION.md) | Understand the older BM25/GPT-OSS answers and same-model judge |
| [System card](system_card.md) | Intended use, important limits, data flows and safety checks |
| [Milestone review](MILESTONE2.md) | Review milestone evidence and distinguish old results from the current app |
| [Contributing](../CONTRIBUTING.md) | Make a change, check it and share it for review |
| [Browser tests](../frontend/tests/README.md) | Check the interface using fake test responses |
| [Deployment tests](../tests/deployment/README.md) | Check a small real database and Docker installation |
| [Evaluation files](../evaluation/README.md) | Find dataset splits and saved experiment results |

For the current separate grading and citation-review steps, use
[the Evaluation guide](EVALUATION.md#offline-bridge-to-the-rc2-evaluator).

## Common terms

| Term | Plain meaning |
| --- | --- |
| Corpus | The complete collection of documents being searched |
| Index | Prepared files that help the app search documents efficiently |
| Retrieval | Finding documents or passages that may answer a question |
| BM25 / BM25+ | Keyword search: rank documents using matching words |
| Embedding | Numbers representing text so the app can compare meaning |
| Token | A piece of text counted by a model; it may be a word, part of a word or punctuation |
| Dense search | Search by comparing embeddings |
| Hybrid search | Combine keyword-search and dense-search rankings |
| RRF | Reciprocal Rank Fusion: combine ranked lists using document positions |
| Passage | A selected part of a document |
| Evidence | Passages or structured records supplied to the model |
| Structured/native record | Named fields from a source, such as a Jira ticket's status and project |
| RAG | Find evidence, then use a language model to help answer |
| Agent | The workflow that chooses search steps and decides whether to continue |
| Critic | A model step that checks whether the evidence covers the facts requested |
| Query reformulation | Change a search query to help find missing information |
| Grant | An explicit permission allowing a user or group to read a document |
| Scope | The documents and operations allowed for one request |
| RLS | Row-level security: database rules that limit which rows an account can read |
| Snapshot | A named, verified version of the document collection |
| Manifest | A file listing data, settings or results so a run can be checked later |
| Hash / fingerprint | A value used to check whether file contents have changed |
| Artifact | A saved file produced or used by the system, such as an index or result |
| Reference answer | The benchmark's expected answer, used for evaluation |
| Citation | A link from an answer to supplied evidence; it does not prove the answer is correct |
| Calibration | Compare decisions with reviewed examples before choosing settings |
| Observability | Request diagnostics: timings, counts, usage and errors |
| Fixture / mock | Fake test data or responses used to check specific behavior |
| CI | Automated checks that run when code is shared on GitHub |

## How to read the results

A **functional check** asks whether a specific workflow ran correctly. An
**answer-quality check** asks whether the answer was correct, complete and
supported. Passing the first does not imply passing the second.

Historical experiments keep their original model, settings and dataset split.
They describe those experiments, not every later version of the app. Start with
[Current status](STATUS.md) before using an older score in a report or demo.

# Agentic RAG for Enterprise Knowledge Search

A FastAPI + React application for scoped search over EnterpriseRAG-Bench. Real
local accounts receive explicit document grants; PostgreSQL RLS and request-time
checks keep retrieval, native Jira tools, model prompts and responses within the
current account's permissions. The default deployment uses the full corpus in
local Docker and runs without paid model calls.

## Run locally

Follow [Setup](docs/WEB_APP.md) to prepare the complete corpus/index bundle, the
pinned embedding model cache and the pinned upstream source checkout. These large
inputs are not in Git. The bundle is in `files_web_app` in the
[team Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).

Once the three input directories are prepared:

```sh
python3 scripts/local_stack.py --env-file .env.local init \
  --artifacts-dir "$PWD/.cache/hybrid-artifacts" \
  --models-dir "$PWD/.cache/models" \
  --source-repo-dir "$PWD/.cache/local-source" \
  --operator-dir "$PWD/.cache/local-operator" --project redwood --port 8000
python3 scripts/local_stack.py --env-file .env.local bootstrap
```

The helper generates separate private database role secrets, stages verified
Linux-volume copies, migrates/imports the data and starts the application. It does
not seed a login. [Create a human account and apply reviewed grants](docs/WEB_APP.md#local-account-api),
then open <http://127.0.0.1:8000>. Role/department labels grant no access.

The Docker target is a 16 GiB host with about 8 GiB available to the VM. Compose
caps the app at 6 GiB/2 CPUs and PostgreSQL at 384 MiB/1 CPU. Full artifact copies,
images, database storage and backups need additional disk. See Setup for exact
preparation, troubleshooting and [safe backup/restore](docs/WEB_APP.md#backup-restore-and-compatible-rollback).
Never remove database volumes as routine shutdown.

```sh
python3 scripts/local_stack.py --env-file .env.local stop
python3 scripts/local_stack.py --env-file .env.local up
```

`.env.local` and operator credentials are private and ignored by Git. The Docker
`.env.example` uses internal service names and named-volume paths; it is not a
native-host launcher configuration.

## Application behavior

- **Retrieval-only default:** BM25+ and bounded BGE-base dense search combined by
  weighted RRF. It returns scoped sources without generated answers or paid calls.
- **Fixed RAG / adaptive agent:** fixed retrieval provides a comparison path;
  adaptive mode can request follow-up search or source expansion. Live generation
  requires separate explicit provider/model/rate/budget configuration.
- **Typed evidence:** exact passage excerpts, native Jira record fields with
  provenance, and aggregate counts including zero. Counts describe authorized
  canonical source records; Jira keys can be duplicated. Unknown coverage is not
  treated as zero, and ambiguous native-key lookups request clarification.
- **Current authority:** login, logout, expiry, account disable and grant changes
  affect retrieval and release. Evidence is cleared on account/session changes.
  No browser storage holds passwords, session tokens or source evidence.
- **Protected run metadata:** run ownership/scope bindings and bounded tool
  metadata persist without raw question/answer/source-body history. Operator
  capture/export rechecks ownership and provenance; gold answers stay outside
  runtime inputs.

The local deployment has one RAG worker and bounded database/auth workers. Public
readiness checks live dependencies without revealing corpus or permission counts.
The legacy fixture/benchmark profile is a separate development path, not a user
identity or fallback for the authenticated product.

## Validation status

Authentication, scoped retrieval, native lookup/counts, typed evidence, export
boundaries and account lifecycle have automated and full-corpus functional
validation. Permission checks remain independent in the application and database.
This is a local working application; SSO, distributed admission and model-quality
claims for the current scoped agent are outside this delivery.

CI includes a [five-document deployment fixture](tests/deployment/README.md) using
real human Argon2 login, PostgreSQL, native import formats and unchanged RLS. It
checks dependency failure and restart persistence without downloading a model.
This tiny check does not establish full-corpus resource/recovery performance. The
new packaging/recovery acceptance is recorded separately; historical failing runs
are retained rather than replaced by later successes.

## Dataset and earlier retrieval results

[EnterpriseRAG-Bench](https://github.com/onyx-dot-app/EnterpriseRAG-Bench) represents
synthetic company Redwood Inference: 511,958 documents from nine platforms and
500 questions with reference documents. `evaluation/splits.json` declares 300
development, 100 calibration and 100 final questions.

The table below records the earlier unrestricted retrieval protocol on development
questions (282 have reference documents). It is not a new permission-scoped or
answer-quality result.

| Method | Recall@10 | nDCG@10 |
| --- | ---: | ---: |
| BM25+ baseline | 0.692 | 0.604 |
| BGE-small + FAISS baseline | 0.468 | 0.367 |
| BGE-base, title on every passage | 0.487 | 0.416 |
| Hybrid BM25+ / BGE-base, BM25 weight 0.7 | 0.722 | 0.604 |

Hybrid Recall@10 was 0.767 on calibration versus 0.712 for BM25+. These results
motivated the retrieval configuration; current scoped evaluation filters reference
documents by actual permissions and retains denominator/unknown-label provenance.
See [Retrieval results](docs/FULL_CORPUS.md) and [Evaluation](docs/EVALUATION.md).

An earlier 20-question GPT-OSS/BM25 answer baseline scored mean correctness 1.30
and citation support 1.55 on a 0–2 rubric. It does not establish quality for the
current agent. [Answer evaluation](docs/ANSWER_EVALUATION.md) describes the external
judge protocol; no paid evaluation is required to run the app.

## Documentation

- [Setup, accounts, permissions and recovery](docs/WEB_APP.md)
- [System architecture and trust boundaries](docs/SYSTEM.md)
- [Scoped retrieval evaluation](docs/EVALUATION.md)
- [Answer evaluation](docs/ANSWER_EVALUATION.md)
- [Retrieval results](docs/FULL_CORPUS.md)
- [Dataset](docs/DATA_CARD.md)
- [Hybrid retriever model card](docs/model_card_hybrid_retriever.md)
- [System card](docs/system_card.md)

Team: Hira Mian, Jane Choi, Alexander Xykis. [MIT License](LICENSE).

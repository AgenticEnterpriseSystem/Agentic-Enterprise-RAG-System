# Agentic RAG for Enterprise Knowledge Search

This app helps users find answers in a fictional company's documents. You sign
in, ask a question, and see an answer with links to the documents used. The app
only searches documents your account is allowed to read.

## Demo

[Watch the demo video](https://youtu.be/5jo-Nh9HjOg)

**RAG** means *retrieval-augmented generation*: find relevant information first,
then give it to a language model to help answer the question.

<a id="application-behavior"></a>

## What the app does

- **Search:** combines keyword search with search by meaning. This is called
  **hybrid retrieval**.
- **Fixed RAG:** searches once, then asks the language model to answer. That
  search round can include several queries.
- **Adaptive agent:** checks whether the search found the facts needed. It can
  read more of a document or search again before answering.
- **Document permissions:** each account has an explicit list of documents it
  can access. A role name such as `employee` does not grant access by itself.
- **Jira questions:** can look up imported Jira records and count them when the
  required data is available. It does not connect to a live Jira account.
- **Observability:** a separate admin page shows request timings, usage and
  errors. Access to this page does not grant access to documents.

The default setup returns search results **without paid model calls**. Generated
answers and the model-driven agent need the separate live-mode setup.

<a id="run-locally"></a>

## Run it on your computer

Start with the [step-by-step setup guide](docs/WEB_APP.md). You will need:

1. Git, Python 3.11 and Docker with Compose.
2. A computer with 16 GiB of RAM and about 8 GiB available to Docker.
3. The prepared document and search-index bundle, a matching embedding-model
   cache, and the dataset source checkout described in the guide.

The large data files are **not included in Git**. The team's prepared bundle is
in `files_web_app` in the [team Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing).
Team-folder access may be needed. The setup guide explains the additional model
and source files and how to check them.

Setup creates the database and starts the app, but it does not create a login.
You must [create an account and grant document access](docs/WEB_APP.md#local-account-api).
Then open **http://127.0.0.1:8000** on the computer running Docker.

To enable generated answers, follow
[live-mode setup](docs/WEB_APP.md#enable-live-generation). It explains the API
key, document excerpts sent to Anthropic, and spending limits.

## Testing a teammate's version

Use the exact testing branch or commit shared by your teammate. Follow the same
setup guide on your own computer; pulling code does not copy someone else's
database, login, permissions or private settings.

Create your own login, such as `hira`, and grant the documents you need to test.
If you also need the diagnostics page, follow
[Observability access](docs/WEB_APP.md#observability). A normal login is enough
to use Search; Observability requires an extra permission.

<a id="validation-status"></a>

## What is working, and what still needs work?

The integrated app has passed specific real-browser and database checks.
**Answer accuracy still needs evaluation.** One tested answer used a cited
document with a date that disagreed with the reference answer. Working citations
alone do not guarantee a correct answer.

Read [current status and known limitations](docs/STATUS.md) for the verified
features, remaining work and the difference between a functional check and a
quality result.

<a id="dataset-and-earlier-retrieval-results"></a>

Earlier retrieval scores are explained in
[retrieval results](docs/FULL_CORPUS.md).

## Milestone 4 Updates:

- **Release candidate:** the web interface, FastAPI backend, PostgreSQL storage, login,
  per-document permissions, hybrid search, Jira tools, and Fixed and adaptive RAG, packaged
  with Docker Compose (`scripts/local_stack.py`). Setup: [docs/WEB_APP.md](docs/WEB_APP.md);
  a demo video is linked [above](#demo).
- **Adaptive agent:** the critic checks each fact the question asks for (supported, missing,
  conflicting or ambiguous). The agent can read more of a found document, search again, or
  ask the user to clarify, using up to eight documents of evidence.
- **Retrieval results** (300 development questions, 282 scored; calibration checked once on 94):

  | Method | Recall@10 | nDCG@10 | Calibration Recall@10 |
  | --- | ---: | ---: | ---: |
  | BM25+ | 0.692 | 0.604 | 0.712 |
  | Dense (BGE-base, title on every passage) | 0.487 | 0.416 | 0.487 |
  | **Hybrid (BM25+ 0.7, BGE-base 0.3, RRF)** | **0.722** | **0.604** | **0.767** |

- **Latency and cost** (adaptive mode, 25 development questions, 16 GB laptop CPU): median
  28.6 s per question (mean 30.0 s), about $0.013 per question.
- **Live answer check** (same 25 questions, informal comparison with the reference answers):
  17 answered: 7 correct, 2 partly correct, 8 wrong; 7 asked the user to clarify. Every
  answer that cited a reference document was correct or partly correct; the wrong answers
  came from similar documents when search missed the right one.
- **Further experiments on `hira/dev`** (not in this release): longer evidence excerpts
  (6,000 characters) and two agent fixes raised answered questions from 25 to 34 of 70, and
  gave 7 correct and only 2 wrong answers on the 25 questions. Whole-document rerankers and
  path headers were tested and not adopted. Details:
  [docs/HIRA_DEV_EXPERIMENTS.md](docs/HIRA_DEV_EXPERIMENTS.md).
- **Next:** combine the checklist critic with longer excerpts and a rule that answers must
  cite a matching source; validate critic decisions and answer grades with human review.

<a id="documentation"></a>

## Where to read next

| I want to… | Read this |
| --- | --- |
| Install and use the app | [Setup](docs/WEB_APP.md) |
| Understand the search and agent workflow | [How the system works](docs/SYSTEM.md) |
| See completed work and open issues | [Current status](docs/STATUS.md) |
| Understand the dataset | [Dataset guide](docs/DATA_CARD.md) |
| Understand the search models and settings | [Retriever guide](docs/model_card_hybrid_retriever.md) |
| Test retrieval or answer quality | [Evaluation guide](docs/EVALUATION.md) |
| Contribute code | [Contributing](CONTRIBUTING.md) |
| Find a term or another document | [Documentation index and glossary](docs/README.md) |

Team: Hira Mian, Jane Choi and Alexander Xykis. [MIT License](LICENSE).

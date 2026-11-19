# Redwood: agentic RAG for enterprise knowledge search

Redwood is a web app that answers questions from a company's documents. You sign
in, ask a question, and get the matching passages, plus a written answer with
citations if paid mode is on. It only searches documents your account is allowed
to read. The documents belong to a fictional company.

**RAG** (retrieval-augmented generation): find the relevant text first, then give
it to an AI model to write the answer.

[Watch the demo video](https://youtu.be/5jo-Nh9HjOg)

<a id="demo"></a>
<a id="run-locally"></a>

## Start here

Follow the **[quick start](docs/QUICKSTART.md)**: ten copy-paste steps from a
fresh clone to a working search page, with a check after each one.

You need Git, Python 3.11, Docker, a computer with 16 GiB of RAM, and access to
the [team Drive folder](https://drive.google.com/drive/folders/1ltfTpNLGdl_S9dBnozACJj3Eo1_Cc8i7?usp=sharing)
for the large data files, which aren't in Git.

The first run is **free**: it searches and shows sources without calling a paid
AI model. Written answers are an optional extra that costs money
([quick start step 11](docs/QUICKSTART.md#11-optional-turn-on-ai-answers)).

<a id="application-behavior"></a>

## What it does

- **Hybrid search:** keyword search plus search by meaning.
- **Two ways to answer:** *Fixed RAG* searches once and answers. The *adaptive
  agent* checks whether it found everything you asked for, and if not, reads
  more or searches again.
- **Per-account permissions:** each account sees only the documents it was
  granted.
- **Jira questions:** looks up and counts imported Jira records.
- **Observability:** an admin page showing timings, usage and errors.

How it works: [docs/SYSTEM.md](docs/SYSTEM.md).

<a id="validation-status"></a>

## Where things stand

The app works end to end in a real browser against a real database.
**Answer accuracy isn't proven yet**, and we haven't yet measured whether the
adaptive agent beats fixed RAG. Details: [current status](docs/STATUS.md).

<a id="dataset-and-earlier-retrieval-results"></a>

## Milestone 4 results

**Search quality** on 300 development questions (282 scored), with a one-time
check on 94 calibration questions:

| Method | Recall@10 | nDCG@10 | Calibration Recall@10 |
| --- | ---: | ---: | ---: |
| Keyword (BM25+) | 0.692 | 0.604 | 0.712 |
| Meaning (BGE-base, title on every passage) | 0.487 | 0.416 | 0.487 |
| **Hybrid (BM25+ 0.7, BGE-base 0.3, RRF)** | **0.722** | **0.604** | **0.767** |

Recall@10 is the share of the expected documents that appear in the top 10
results; nDCG@10 also rewards ranking them higher. Earlier scores are in [retrieval results](docs/FULL_CORPUS.md).

**Speed and cost** (adaptive agent, 25 development questions, 16 GB laptop CPU):
median 28.6 seconds per question (mean 30.0), about $0.013 per question.

**Answers** (same 25 questions, informally compared with reference answers):

- 17 answered: 7 correct, 2 partly correct, 8 wrong. The other 7 asked the user
  to clarify.
- Every answer that cited a reference document was correct or partly correct.
  The wrong answers came from similar documents when search missed the right one.

**Further experiments** on the `hira/dev` branch (not in this release): longer
evidence excerpts (6,000 characters) and two agent fixes raised answered
questions from 25 to 34 of 70, and gave 7 correct and only 2 wrong on the 25
questions. Whole-document rerankers and path headers were tested and not
adopted. See [Hira's experiments](docs/HIRA_DEV_EXPERIMENTS.md).

**Next:** combine the fact-checking critic with longer excerpts and a rule that
answers must cite a matching source, and have people review the critic's
decisions and the answer grades.

<a id="documentation"></a>

## Where to read next

| I want to... | Read this |
| --- | --- |
| Install and run the app | [Quick start](docs/QUICKSTART.md) |
| Look up a setup detail | [Reference guide](docs/WEB_APP.md) |
| Understand how it works | [How the system works](docs/SYSTEM.md) |
| See what's tested and what's open | [Current status](docs/STATUS.md) |
| Understand the dataset | [Dataset guide](docs/DATA_CARD.md) |
| Understand the search models | [Retriever guide](docs/model_card_hybrid_retriever.md) |
| Test search or answer quality | [Evaluation guide](docs/EVALUATION.md) |
| Contribute code | [Contributing](CONTRIBUTING.md) |
| Look up a term or find another doc | [Documentation index](docs/README.md) |

Team: Hira Mian, Jane Choi and Alexander Xykis. [MIT License](LICENSE).

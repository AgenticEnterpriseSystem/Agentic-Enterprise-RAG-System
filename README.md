# Agentic RAG for Enterprise Knowledge Search

A FastAPI + React web app that searches enterprise documents and answers questions
with cited evidence. It uses Onyx's synthetic **EnterpriseRAG-Bench** dataset.

- **Fixed RAG:** search once, then generate an answer.
- **Adaptive agent:** search, check whether the evidence is sufficient, and search
  again when needed before answering.

## Run the demo

Install and start **Docker Desktop**, or Docker Engine with Compose v2 on Linux.
Clone or download this repository and open a terminal in its root folder. No
separate Python, Node, or `requirements.txt` installation is needed with Docker.

```sh
docker compose up --build
```

Open **http://127.0.0.1:8000**. The first build downloads dependencies.

Fresh defaults use **synthetic fixture responses**: this previews the interface,
not real retrieval or answer quality. Existing `.env` or shell settings can
change the mode. Ask **“How long are artifacts retained?”**, switch workflows,
click a citation, and expand **Search steps**. Enter submits; text wraps automatically.

When finished:

```sh
docker compose down
```

**For real search and generated answers**, follow the [setup guide](docs/WEB_APP.md).
Copy the shared `.env.example` template to your own local `.env`; never commit
`.env`. You need matching corpus/index files and, for live generation, your own
API key and spending cap. A shared artifact download link has **not yet been published**;
the guide includes local build instructions.

## Current status

The web app connects both workflows, displays exact citation excerpts and search
traces, and handles partial answers and errors. Full-corpus BM25 and BGE-small
retrieval baselines and small live BM25 development comparisons are recorded.
**Overall answer-quality improvement has not been established.**

Matching BGE-base hybrid verification, real authentication/access integration,
independent answer review, critic calibration, and final experiments remain pending.

## Documentation

| Need | Read |
| --- | --- |
| Run locally, configure real data or hybrid, troubleshoot | [Setup](docs/WEB_APP.md) |
| Understand the workflow, API, and limitations | [System](docs/SYSTEM.md) |
| Inspect results or reproduce an evaluation | [Evaluation](docs/EVALUATION.md) |
| Check dataset origin, license, and risks | [Data card](docs/DATA_CARD.md) |

Application code is in `src/` and `frontend/`; tests are in `tests/`;
[saved evaluation artifacts](evaluation/README.md) are in `evaluation/`.
Large local data and experiment traces belong in ignored `.cache/`.

## Project

Team: Hira Mian, Jane Choi, and Alexander Xykis.
See [Contributing](CONTRIBUTING.md), [Code of Conduct](CODE_OF_CONDUCT.md),
and the [MIT license](LICENSE).

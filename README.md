# Agentic RAG for Enterprise Knowledge Search

A FastAPI + React app for answering questions over EnterpriseRAG-Bench documents
with cited sources. It combines BM25 and BGE-base retrieval using RRF. An LLM
critic checks the retrieved passages and requests follow-up searches when needed.

## Run the app

Follow [Setup](docs/WEB_APP.md) to install dependencies, build the frontend,
obtain the corpus/index bundle, and configure your local `.env`.
Then, from the repository root:

```sh
source .venv/bin/activate
python scripts/run_web.py
```

Open <http://127.0.0.1:8000> and wait for **Ready**. Stop with Ctrl+C.
Docker is an optional alternative covered in the setup guide.

| Mode | Behavior |
| --- | --- |
| `fixture` | Scripted demo; no artifacts or API key |
| `retrieval` | Real search; no generated answers |
| `live` | Search, evidence critique and cited answers; API key and budget required |

Select **Adaptive agent** for follow-up searches or **Fixed RAG** for one search.
Neither generates answers in retrieval-only mode.

## Status

Working alpha: full-corpus native hybrid search and a live agent request have
been verified. Broader answer-quality testing and full-corpus Docker validation
remain unfinished. Known issues are described in [Setup](docs/WEB_APP.md) and
[System](docs/SYSTEM.md).

Artifacts are supplied separately by the team; a download URL is not yet published.
Never commit `.env` or API keys.

## Documentation

- [Setup](docs/WEB_APP.md)
- [System](docs/SYSTEM.md)
- [Evaluation](docs/EVALUATION.md)
- [Retrieval results](docs/FULL_CORPUS.md)
- [Dataset](docs/DATA_CARD.md)

Team: Hira Mian, Jane Choi, Alexander Xykis. [MIT License](LICENSE).

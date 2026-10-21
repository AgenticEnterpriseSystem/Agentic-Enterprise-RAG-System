# How the system works

Question → retrieve evidence → critic checks sufficiency → search again if needed
→ generate an answer with citations.

## Current status

| Implemented | Verified | Pending |
| --- | --- | --- |
| BM25 and BGE retrieval | Full-corpus development baselines | Final-split experiments |
| Hybrid search, bounded passages, optional reranker | Offline tests and integration checks | Live hybrid/reranker quality |
| Agent, critic, routing, grounded generation | Offline tests and small live BM25 comparisons | Broader answer quality, critic calibration |
| Scope checks and structured-tool interfaces | Synthetic permission tests | Real tools, identity/access adapters, API/UI |

Alex has an initial answer/judge experiment on his separate development branch.
This branch has development comparisons with AI inspection, not completed human
validation or final answer-quality evaluation. See [agent results](AGENT_DEVELOPMENT.md).

## Components

| Module | Purpose |
| --- | --- |
| `src/retrieval/` | Keyword/dense retrieval, fusion, passage selection, reranking |
| `search_agent.py` | Coordinates search, critique, and generation |
| `search_critic.py` | Identifies missing facts and suggests follow-up searches |
| `routing.py` | Chooses text search, registered tools, or both |
| `budget.py` | Supplies remaining time, tokens, and cost allowances |
| `src/generation/generator.py` | Generates grounded answers and validates citations |
| `pilot.py` | Runs a small development-only debugging pilot |

Agent modules live in `src/agents/`; they are supporting parts of one workflow.
The loop stops on sufficient evidence, no new evidence, a limit, or an error.
At normal round-limit or no-new-evidence stops, it attempts a cited partial
answer within the remaining budget and retains insufficient-evidence status.
Errors, empty evidence and exhausted budgets do not trigger fallback generation.
Multi-round contexts share space across original and relevant follow-up evidence;
long passages retain a lead plus a query window with exact offsets. These lexical
heuristics can still omit facts. The final critic context is bounded to leave
room for generation; exact provider token counting enforces the token cap.
The [interactive terminal](../README.md#pilot-options) accepts custom questions.
It defaults to retrieval-only inspection; `--live` enables actual model calls.
Saved cases include per-search new evidence IDs and critic/stop explanations.

## Retrieval and evidence

BM25 scores whole documents; BGE ranks documents by their best passage. Hybrid
search combines rankings with reciprocal rank fusion (RRF), one vote per document
per ranking. Defaults: 50 candidates per retriever, 50-document rerank pool,
RRF constant 60, equal weights. These settings have not been shown optimal.

Shared documents retain dense evidence when available. BM25-only documents use
query-relevant exact slices of at most 3,000 characters. Offsets remain tied to
original text. The heuristic may omit facts, and character limits are not token
limits. Optional cross-encoder reranking uses a pinned model revision and a
recorded 512-token default input limit; longer pairs can still be truncated.

Dense loading adds one shard at a time to exact FAISS search; the hybrid pilot
shares the corpus table. Corpus hashes, document ordering, and dense query model
settings must match the artifacts, including revision and fp16 precision.

## Limits to keep in mind

- Missing identity or permissions denies access. Filters apply before reranking.
  Unknown dates cannot satisfy date constraints; the corpus adapters have no
  structured timestamps. Real authentication and structured adapters are pending.
- Evidence is untrusted text. The generator abstains without usable evidence and
  rejects malformed responses or unknown citations. Valid IDs do not prove claim
  support; critic confidence is uncalibrated. Gold answers stay out of model inputs.
- Cooperating providers preflight estimated tokens/cost and bound timeouts.
  Estimates are not exact billing guarantees. Synchronous retrieval and providers
  that ignore budget context have soft limits checked between operations.
- The pilot uses unrestricted benchmark access, five development questions and
  at most two rounds by default. It uses fusion without the optional reranker.
  This is debugging, not a completed quality evaluation.

See [README](../README.md#pilot-options) for pilot setup, artifact requirements and
commands; [Baseline Results](FULL_CORPUS.md) for measured retrieval scores; and
[Evaluation](EVALUATION.md) for scoring rules and planned answer review.

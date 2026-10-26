# System

## Request flow

```text
React question form → FastAPI → configured retriever
  Fixed RAG: search → generate cited answer
  Agent:     search → check evidence → follow-up searches if needed → answer
```

The app serves the built frontend and `/api/*` from one origin. Both workflows
use the same generator and `search(request) → evidence` retrieval interface.
The backend loads one retriever at startup and shares it across requests.
[Setup](WEB_APP.md) explains runtime modes and artifact configuration.

## Components

| Location | Responsibility |
| --- | --- |
| `src/contracts.py` | Validated requests, evidence, answers and traces |
| `src/api/main.py`, `runtime.py` | HTTP lifecycle, demo scope, retriever/provider construction |
| `src/agents/` | Search loop, evidence critic, routing and per-run budgets |
| `src/generation/generator.py` | Provider calls, evidence-only generation and citation validation |
| `src/retrieval/` | BM25, dense search, fusion, passage selection and optional reranker |
| `frontend/src/` | Question form, answer, exact source excerpts and search steps |
| `src/evaluation/`, `src/agents/pilot.py` | Retrieval scoring and development comparisons |

BM25 scores documents; BGE uses their best matching passage. Hybrid combines
rankings with weighted reciprocal rank fusion, deduplicating documents. The web
runtime shares the corpus table between retrievers; its settings and compatibility
requirements are in the [hybrid handoff](WEB_APP.md#hybrid-handoff).

Passage selection uses bounded exact text slices with original character offsets.
It balances initial and follow-up evidence, but can still omit useful facts.
The optional cross-encoder has a 512-token default input limit; the web app does
not currently enable it.

The agent stops on sufficient evidence, no new evidence, limits, or failure.
Normal round/no-new-evidence stops can produce a cited partial answer; errors and
exhausted budgets do not trigger fallback generation. The critic guides search;
its confidence is **not calibrated answer accuracy**. Clarification dialogue and
answer streaming are not implemented.

The critic and generator receive each excerpt's citation ID, document ID, title,
platform, date when known, and text. Excerpts that
share a document ID can be read together, and titles are context, not proof. In the
final round, the answer is generated from exactly the evidence the critic approved.

## API

| Endpoint | Result |
| --- | --- |
| `GET /api/health` | Liveness, including during index loading |
| `GET /api/ready` | 200 when ready; 503 while loading or on setup failure |
| `POST /api/query` | `QueryRequest` → `QueryResponse`: answer, sources, trace |
| `GET /` | Built React app |
| `GET /docs` | Generated API schema and interactive documentation |

Query mode is `baseline` or `agentic`; questions contain 1–8000 nonblank characters.
Source/date filters may narrow server scope. HTTP statuses distinguish validation
(422), denied scope/origin (403), busy/budget (429), unavailable setup (503), timeout
(504) and provider/agent errors (502). Agent errors retain permitted traces/evidence;
boundary errors use `detail`. Responses do not expose provider exception text.

Defaults: top-k 5, two rounds, 120 seconds, 16,000 total model tokens, 24,000 context
characters, and 1,024 maximum output tokens per model call. Search and loop limits
use `RAG_*` settings in `src/api/runtime.py`; the provider output limit is fixed
in code. These limits are not quality claims. Provider calls include token-count
requests as well as generation.

## Access, spending and concurrency

- `RAG_DEMO_ACCESS=true` enables the server-owned **benchmark-demo** identity.
  It can read indexed synthetic documents; it is not sign-in or a production ACL.
- `RAG_ALLOWED_DOC_IDS_FILE` can restrict IDs; an empty list denies all.
  `RAG_ALLOWED_SOURCES` restricts platforms. Client identity/permission claims are
  rejected. Retrieval, generation and final responses enforce this scope.
- Dates require time zones. Corpus adapters lack reliable structured timestamps,
  so date filters exclude undated evidence. Structured-tool interfaces exist;
  real adapters and identity integration remain pending.
- Citation clicks show the response's exact authorized excerpts. Valid citation
  IDs alone do not establish that a claim is supported. Evidence is untrusted;
  reference answers are never supplied to the running agent.
- One worker bounds expensive work; health checks remain responsive. Timed-out
  synchronous work retains its slot until completion. SDK retries are disabled.
  Token counting and cost reservations guard calls; limits around non-cooperating
  synchronous tools are soft. Spending caps are process-local and reset on restart.

See [Evaluation](EVALUATION.md) for measured outcomes and known answer failures.

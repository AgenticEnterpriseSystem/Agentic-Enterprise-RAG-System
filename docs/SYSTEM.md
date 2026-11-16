<a id="system"></a>
# How the system works

This app searches workplace documents and answers questions using the documents a signed-in user can read. Its dataset describes a fictional company. It is a local course project, not a connection to a real company's Slack, email or Jira.

**RAG** means *retrieval-augmented generation*: find useful evidence first, then give that evidence to a language model to help write an answer. The app can also ask for clarification or say that the available evidence is insufficient.

Start with [setup](WEB_APP.md) to run it. See [current status](STATUS.md) for completed checks and open work, and the [system card](system_card.md) for risks and limitations.

## Request flow

1. **Sign in.** A local account identifies the user. Its name, department or role does not automatically grant document access.
2. **Ask a question.** The browser sends the question, workflow and optional platform/date filters. It cannot supply a trusted identity or add permissions.
3. **Find the documents this user can read.** PostgreSQL checks the current account, session and reviewed document grants. This allowed set is called the request's **scope**.
4. **Search within that scope.** Keyword search and meaning search rank permitted documents. The app fetches selected document text after these checks.
5. **Choose evidence and answer.** Fixed RAG searches once. The adaptive workflow checks missing facts and can read more or search again before answering.
6. **Check and show the result.** The server rechecks access and evidence identity. The UI shows the answer, exact source excerpts, citations and search steps.

```text
Sign in → check document access → search → choose evidence
  Fixed RAG:       → write answer
  Adaptive agent:  → check requested facts → answer, clarify, or follow up
                                              ↑                 │
                                              └── check again ──┘
```

A **round** can contain several searches or a structured operation. Fixed RAG has one round; it does not necessarily issue only one query. The managed adaptive workflow allows two rounds.

### Hybrid search: words and meaning

**Hybrid retrieval** combines two rankings:

- **BM25+** is keyword search. Exact product names, error codes and IDs help it.
- **BGE-base** is meaning search. It represents text as number vectors called **embeddings**, so differently worded questions can match similar text.

Each method supplies up to 50 document candidates. **Reciprocal rank fusion (RRF)** combines their positions in the lists, with weights 0.7 for BM25+ and 0.3 for dense search. Duplicate documents are merged. No reranker is enabled in the serving app. A reranker would be an additional model that reorders candidates.

Document embeddings are built ahead of time and shared across users. At request time the app embeds the question. It does not rebuild the documents' embeddings or keep a separate vector index for each account.

### What the critic and follow-up search do

The **router** turns the question into a search plan. It can use different wording or several queries from the first round. The **critic** checks the evidence against the facts the user asked for. Its checklist marks facts as supported, missing, conflicting or ambiguous; code derives the overall decision.

If information is missing, the workflow can:

- **Reformulate the query:** search with more specific wording, names or entities.
- **Read more:** fetch an adjacent window from an already permitted document when a useful part may have been cut off. An operation expands at most two documents, normally by up to 3,000 characters each.
- **Ask for clarification:** request a distinguishing detail or a missing source.

The critic suggests actions; the router/search agent carries them out. Query reformulation is part of that existing workflow. Follow-up actions cannot widen document access.

### How much evidence reaches the models

The managed app selects **at most eight unique documents** for the shared critic/generator context. This ceiling is configurable to 5, 8 or 10. Several passages from one document count once. A native Jira record and passage from the same document share that slot. A computed count is one atomic evidence item, not a list limited to eight underlying rows.

Eight is a maximum, not a target. Character and token budgets can leave fewer documents. Each text search requests up to the greater of `top_k` and this ceiling: eight hits in the managed profile. Multiple searches can collect a larger pool before shared selection. Earlier `top_k=5` therefore did not mean a global five-document context.

The selector retains exact passages used for previously supported facts and reserves room for a possible answer before every critic call. The selected ordered evidence is then frozen: generation must use that same context. If supported evidence cannot fit, the system stops rather than silently dropping it. `trace.context_selection` reports selected, excluded and shortened passages. Source cards on an early error can show the larger collected pool; their count is not necessarily the model's input count.

## Access, spending and concurrency

### When permission checks happen

The server resolves document access **before searching or sending evidence to a model**. It builds a compact yes/no lookup, called a **permission mask**, in the same order as the shared document index.

- BM25 applies the mask before selecting candidates.
- Dense search checks the scope before encoding the question, then excludes unauthorized vector rows before similarity calculations and candidate selection.
- PostgreSQL checks access again when text or native records are read.
- Current-access checks run around long operations, model dispatch and final release.

This avoids reading or sending denied document bodies. It does not remove their vectors from the shared files. Dense search still walks shared shards, so a smaller scope is not a promise of proportional disk, memory or latency savings. Full-corpus users get no permission-based reduction in eligible vectors.

Expiry, logout, account changes or a policy change can invalidate a request. Missing or unavailable permission state denies access. Checks cannot recall text already sent to a provider or already shown to a user.

### Limits in the managed local deployment

A **token** is a small text piece counted by a model; it is not necessarily a whole word.

| Limit | Value | Meaning |
| --- | ---: | --- |
| Concurrent expensive requests | 1 | Others receive a busy response |
| Agent rounds | 2 | Maximum search/check cycles |
| Shared document ceiling | 8 | Subject to smaller text/token budgets |
| Evidence characters | 24,000 | Evidence bodies/native JSON, not the whole prompt |
| Model tokens per question | 16,000 | Inputs and outputs across stages |
| Critic output | 2,048 tokens | A larger checklist can still fail |
| Router/answer output | 1,024 tokens | Separate from the critic allowance |
| Agent execution | 120 seconds | Inside the search workflow |
| HTTP request wait | 180 seconds | Also allows for account/policy work |
| Question cost | $0.10 | Checked before provider dispatch |
| Provider budget per process | $0.50 in the reviewed local setup | Resets on restart |
| Idle database transaction | 15 seconds managed; 5 seconds library default | Separate from SQL execution time |
| SQL statement / lock wait | 5 seconds / at most 2 seconds | Not raised by the idle-timeout change |

Settings live in [WebSettings](../src/api/runtime.py), [database settings](../src/db/runtime.py) and the explicit deployment configuration. Live operation needs a model/key, positive prices/budgets and paid mode enabled. Configuration alone does not authorize a new paid experiment.

Provider calls count tokens and reserve worst-case cost before generation. API call counters include token-count requests as well as text generation; they are not a count of answers. SDK retries are disabled; unknown cost stays unknown. A timed-out HTTP request does not prove its worker stopped: the worker keeps the busy slot until it ends. Evaluation runners separately keep durable attempt reservations, so a restart does not authorize replaying an uncertain case.

## Components

| Code | What it does |
| --- | --- |
| `frontend/src/` | Login, Search, source cards, traces and restricted Observability |
| `src/api/` | HTTP requests, startup/readiness and authenticated runtime wiring |
| `src/auth/`, `src/access/` | Accounts, sessions, grants and current-access checks |
| `src/retrieval/` | Keyword/dense rankings, fusion and bounded candidate selection |
| `src/agents/` | Search plans, critic checklist, follow-up actions and budgets |
| `src/generation/` | Provider calls, answer writing and citation validation |
| `src/storage/`, `src/db/`, `migrations/` | Canonical documents, native data, RLS and run metadata |
| `src/contracts.py` | Validated request, evidence, answer and trace formats |
| `src/evaluation/` | Reproducible runs and separate scoring inputs |

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `GET /api/ready` | Liveness and whether the app is ready/busy |
| `GET /api/auth/profile`, `GET /api/auth/csrf` | Auth mode and initial CSRF protection |
| `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout` | Sign in, check session and sign out |
| `POST /api/query` | Return answer, sources and trace |
| `GET /api/runs/{run_id}` | Current owner's minimized run metadata |
| `GET /api/observability/*` | Content-free diagnostics with separate admin permission |
| `/`, `/observability`, `/docs` | Search UI, diagnostics UI and generated API docs |

Questions contain 1–8,000 nonblank characters. API workflow values are `baseline` and `agentic`; UI labels are **Fixed RAG** and **Adaptive agent**. Source/date filters narrow results; undated evidence fails date filters. A complete new question is needed after clarification: persistent chat memory and answer streaming are not implemented.

Errors distinguish sign-in failure (401), denied access/CSRF (403), stale scope (409), invalid input (422), busy/budget (429), model/agent failure (502), unavailable policy/setup (503) and HTTP timeout (504). Messages do not expose raw provider exceptions.

## Relational foundation

PostgreSQL stores canonical documents, accounts, sessions, explicit grants, native Jira data and minimized run/count records. The app uses a restricted shared database service role; each transaction binds the current application user. Application accounts are not separate PostgreSQL service-role logins.

**RLS means row-level security:** database rules check which rows the account can read. Python access checks and RLS both enforce the current session, allowed documents and policy version. An empty pooled connection context grants nothing. Runs and computed counts also require the correct owner and scope.

### Reviewed document permissions

A direct grant or explicit group membership can permit a document. Special compound groups require their declared memberships; they are not a manager/folder inheritance system. Roles, departments, projects, folder paths and policy prose grant nothing automatically. Operator restrictions and query filters can only narrow access. There is no imported, verified enterprise access map; see [the data card](DATA_CARD.md#extraction-and-limitations).

Grant pages are fetched in short transactions. Python row conversion and bitmap work happen after commit/pool release. A fresh transaction checks the account, session and policy revision before each page is used, including final/empty pages. This avoids leaving a transaction idle during local processing.

### Local account implementation

Local accounts use password hashes, expiring/revocable server sessions, an HttpOnly cookie and CSRF/origin checks. CSRF protects writes from being triggered by another site. Signing out clears displayed results and revokes the owned session. Failed local login never falls back to the separate benchmark/demo identity.

### Canonical storage and import lifecycle

**Canonical** means the verified text representation used for a document revision. Imports check source hashes, document order and exact spans before publishing a ready dataset. Published bodies/revisions are immutable. Text containing a null character is preserved as UTF-8 bytes rather than silently changed.

### Native Jira records and exact counts

The Jira adapter uses verified fields imported from 6,118 canonical source records; it does not query live Jira. Two source aliases are excluded and 575 native keys repeat, so a key can match several records. Counts mean permitted canonical source records, not necessarily unique live issues.

An exact count queries the complete authorized relation, independently of the eight-document text context. It needs verified field coverage: unknown coverage is not zero. Unsupported filters and ambiguous lookups stop or ask for clarification. Records/counts have their own provenance rather than invented passage text.

## Implementation contracts

These details matter when extending or reproducing the system.

### Dataset and permission binding

A **snapshot** identifies a fixed dataset through its ID, corpus-file hash, ordered catalog hash and document count. A hash is a fingerprint of exact bytes. Matching names alone are insufficient. A scope also binds the user, policy revision and account session version. Browser/model fields cannot create this authority.

### Text and computed evidence

Passages retain document/revision IDs and exact zero-based, end-exclusive Unicode character offsets. For example, `doc:10:20` refers to characters 10 through 19. Counts and typed records have separate evidence formats. Every answer citation must resolve to supplied evidence; this does not prove claim support.

### Versions and source ownership

[The contracts module](../src/contracts.py) is the shared wire definition. Snapshot-bound passage/count responses use version 2; native records require version 3. Legacy fixture formats remain supported. Account, access, storage and evaluation contracts live in their corresponding `src/` packages.

### Document-store interface and reference fixture

The store exposes bounded metadata iteration and authorized exact-text reads/expansion. Its small in-memory test implementation checks the same interface; passing it does not prove full-corpus performance.

### Experiment and external-evaluator interface

Runtime records freeze source/model/prompt/index/settings identities and actual outcomes. **Reference answers** are joined afterward for scoring and never given to the answering agent. Unknown cost and failed attempts remain visible. Compatible settings and valid files do not establish correctness. See [evaluation](EVALUATION.md).

<a id="authentication-http-contract--future-routes"></a>
### Authentication HTTP contract

The implemented cookie is `rag_session`; authenticated writes use `X-CSRF-Token`. The browser holds its session, not database credentials or document grants. Internal server identity excludes the raw cookie. See [setup](WEB_APP.md) for local HTTP, HTTPS cookie settings and operator provisioning.

### Acceptance and adoption

Authenticated PostgreSQL/hybrid serving and the native Jira adapter are implemented, not future contract-only work. Current acceptance is 19 of 23 gates; remaining evaluation/release work is tracked in [status](STATUS.md). Functional UI checks do not establish answer-quality superiority or production readiness.

<a id="system"></a>
# How the system works

Redwood answers questions using a company's documents, and it only uses the
documents the signed-in person is allowed to read. The documents belong to a
fictional company. Redwood is a course project and isn't connected to real
Slack, email or Jira.

To run it, see the [quick start](QUICKSTART.md). For what's been tested, see
[status](STATUS.md).

## What happens when you ask a question

1. **You sign in.** Your account decides which documents you may read. A job
   title like "employee" or "CEO" doesn't grant anything by itself.
2. **The app finds your documents.** Before searching, it works out exactly
   which documents your account may read.
3. **It searches only those documents.** It runs two kinds of search and
   combines the results (see below).
4. **It picks the best evidence.** Up to 8 documents' worth of passages.
5. **It answers.** There are two ways to answer:
   - **Fixed RAG:** search once, then write the answer.
   - **Adaptive agent:** a checker (the **critic**) looks at whether the
     evidence covers what you asked. If something is missing, the agent can read
     more of a document, search again with better wording, or ask you for a
     missing detail.
6. **It shows its work.** You see the answer, the exact passages it used, and
   the search steps. Each citation links to a passage it was given.

```text
Sign in → find your documents → search → pick evidence
  Fixed RAG:       → write answer
  Adaptive agent:  → check facts → answer, ask you, or search again
                         ↑                         │
                         └────── check again ──────┘
```

The adaptive agent searches at most twice. In free mode the app shows the
passages it found but doesn't write an answer.

## The two kinds of search

- **Keyword search (BM25+)** finds exact words. It's good for names, error codes
  and IDs.
- **Meaning search (BGE-base)** finds text that means the same thing even when
  the words differ.

The app merges the two ranked lists, weighting keyword search 70% and meaning
search 30%.

## How your access is protected

- Access is checked **before** any search and **before** any text is read or
  sent to the AI model, and again before the answer is returned.
- The database enforces access too, as a second line of defense behind the
  app's own checks.
- Nothing you type into the app can add permissions.
- If access changes mid-question (you sign out, or a permission is removed),
  the question stops.

## Main limits

| Limit | Value |
| --- | --- |
| Questions running at once | 1 (others are told the app is busy) |
| Search rounds (adaptive agent) | 2 |
| Evidence documents per answer | 8 |
| Time for the agent | 120 seconds |
| Cost per question | $0.10 |
| Spending per app run | $0.50, resets when the app restarts |

There's no chat memory. After the app asks you for a detail, send a new,
complete question.

---

## Engineering notes

Detail for people changing or debugging the system.

### Code map

| Code | What it does |
| --- | --- |
| `frontend/src/` | Login, Search, source cards, traces, restricted Observability |
| `src/api/` | HTTP requests, startup and readiness |
| `src/auth/`, `src/access/` | Accounts, sessions, grants, access checks |
| `src/retrieval/` | Keyword and dense rankings, fusion |
| `src/agents/` | Search plans, critic, follow-up actions, budgets |
| `src/generation/` | Model calls, answer writing, citation validation |
| `src/storage/`, `src/db/`, `migrations/` | Documents, native data, RLS, run metadata |
| `src/contracts.py` | Request, evidence, answer and trace formats |
| `src/evaluation/` | Reproducible runs and scoring inputs |

### Search details

Each search method returns up to 50 candidates. Reciprocal rank fusion (RRF)
merges them with weights 0.7 (BM25+) and 0.3 (dense). No reranker is used.
Document embeddings are precomputed; only the question is embedded at request
time. A round can hold several searches.

The **router** turns the question into a search plan. The **critic** marks each
requested fact as supported, missing, conflicting or ambiguous. Follow-up actions
are **reformulate**, **read more** (up to 2 documents, normally 3,000 characters
each) and **ask for clarification**. None can widen access.

### Evidence

- At most 8 unique documents (configurable to 5, 8 or 10). Several passages from
  one document count once. Character and token budgets can leave fewer.
- The critic and generator see the same frozen evidence. If supported evidence
  no longer fits, the agent stops instead of dropping it.
- `trace.context_selection` lists selected, excluded and shortened passages.
- A computed count is one evidence item, not a list of rows.

### Permission checks

The server builds a yes/no lookup (the **permission mask**) in the same order as
the shared index. BM25 applies it before picking candidates; dense search drops
denied vectors before scoring; PostgreSQL checks again when text is read. Access
is rechecked around long operations, model calls and the final release. Denied
vectors still sit in the shared files, so a small scope doesn't make search
proportionally faster. Missing permission state denies access. Checks can't
recall text already shown or sent to the provider.

PostgreSQL **row-level security (RLS)** and the Python checks both enforce the
session, allowed documents and policy version. The app uses one restricted
database role and binds the current user to each transaction.

A direct grant or group membership can permit a document; special compound
groups need every declared membership. Roles, departments, projects, folders and
policy text grant nothing. Filters only narrow access. No real enterprise access
map is imported; see [the data card](DATA_CARD.md#extraction-and-limitations).

### Accounts

Password hashes, expiring and revocable server sessions, an HttpOnly cookie, and
CSRF/origin checks. Signing out clears displayed results and revokes the
session. A failed login never falls back to the benchmark identity.

<a id="authentication-http-contract--future-routes"></a>
The cookie is `rag_session`; authenticated writes send `X-CSRF-Token`. The
browser holds a session, not database credentials or grants.

### All limits

| Limit | Value |
| --- | --- |
| Evidence characters | 24,000 (bodies and native JSON, not the whole prompt) |
| Model tokens per question | 16,000 (inputs and outputs) |
| Critic output | 2,048 tokens |
| Router / answer output | 1,024 tokens each |
| HTTP wait | 180 seconds |
| Idle database transaction | 15 seconds (library default 5) |
| SQL statement / lock wait | 5 seconds / at most 2 seconds |

Settings live in [WebSettings](../src/api/runtime.py) and
[database settings](../src/db/runtime.py). Provider calls count tokens and
reserve worst-case cost first. SDK retries are off. An HTTP timeout doesn't stop
its worker; the worker keeps the busy slot until it ends.

### API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health`, `GET /api/ready` | Alive; ready or busy |
| `GET /api/auth/profile`, `GET /api/auth/csrf` | Auth mode; CSRF token |
| `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout` | Sign in, check session, sign out |
| `POST /api/query` | Answer, sources and trace |
| `GET /api/runs/{run_id}` | The owner's minimized run metadata |
| `GET /api/observability/*` | Content-free diagnostics (separate admin permission) |
| `/`, `/observability`, `/docs` | Search UI, diagnostics UI, API docs |

- Questions are 1–8,000 nonblank characters.
- Workflow values are `baseline` and `agentic` (UI: **Fixed RAG** and
  **Adaptive agent**). Undated evidence fails date filters. No streaming.
- Errors: 401 sign-in, 403 denied/CSRF, 409 stale scope, 422 invalid input, 429
  busy/budget, 502 model/agent failure, 503 unavailable policy/setup, 504 HTTP
  timeout. Raw provider errors are never shown.

### Jira records and counts

The Jira tools use fields imported from 6,118 records; they don't query live
Jira. Two source aliases are excluded and 575 keys repeat, so a key can match
several records. Counts mean permitted source records, not unique issues. An
exact count covers the full permitted set, not just the 8-document evidence. It
needs verified field coverage; unknown coverage isn't zero. Unsupported filters
and ambiguous lookups stop or ask for clarification.

### Data formats

- A **snapshot** is a fixed dataset identified by ID, corpus hash, ordered
  catalog hash and document count. A scope also binds user, policy revision and
  session version.
- **Canonical** text is the verified text of a document revision. Imports check
  hashes, order and spans; published text never changes.
- Passages carry document and revision IDs and zero-based, end-exclusive
  character offsets (`doc:10:20` is characters 10 to 19). Every citation must
  resolve to supplied evidence, which doesn't prove the claim.
- [`src/contracts.py`](../src/contracts.py) defines the wire format: passage and
  count responses use version 2, native records version 3. Legacy fixture
  formats still work.
- Experiment records freeze source, model, prompt, index and settings
  identities. Reference answers are joined afterward for scoring and never shown
  to the agent. See [evaluation](EVALUATION.md).

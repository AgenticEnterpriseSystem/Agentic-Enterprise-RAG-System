# System card: Agentic RAG for Enterprise Knowledge Search

**Card date:** 2026-10-24 · **App version:** 0.1.0 (working alpha)
**Owners:** Hira Mian, Jane Choi, Alexander Xykis (CIS 5980 capstone team)

## System overview and purpose

A local web app that answers questions about a company's internal documents and cites
the exact passages it used. It runs on the synthetic **EnterpriseRAG-Bench** corpus
(511,958 documents for the fictional company "Redwood Inference"). A person types a
question; the system searches, checks whether the evidence is enough, searches again if
needed, and writes a short answer with citations, or says the evidence is insufficient.

The system is a course project and a research demo. It is **not** a production
service and has not been tested on real company data.

## Intended use cases

- Asking questions about the synthetic company's projects, customers, incidents and
  policies, and checking the cited excerpts.
- Comparing **Fixed RAG** (one search, then answer) with the **Adaptive agent**
  (search → critic → follow-up searches → answer).
- Measuring retrieval and agent behavior, with reproducible settings.

**Out of scope:** real employee or customer data, use as
a source of truth for decisions, public internet hosting, and many users at once.

## Architecture and components

```text
Browser (React)
   │  POST /api/query  {question, mode, optional source/date filters}
   ▼
FastAPI server (one worker, bound to 127.0.0.1)
   ├─ Request checks: schema, length, allowed origin, busy, spending cap
   ├─ Server-owned demo identity → allowed document IDs and platforms
   ├─ Search agent (src/agents/)
   │    ├─ Router: turns critic feedback into ≤4 follow-up queries (scope can't widen)
   │    ├─ Hybrid retriever: BM25+ and BGE-base, fused with RRF (model card)
   │    ├─ Passage selection: exact text slices with original character offsets
   │    ├─ Evidence critic (LLM): "is this enough?" → missing facts, new queries
   │    └─ Grounded generator (LLM): answer + citation IDs from supplied evidence only
   └─ Response checks: schema, citations, scope and filters re-validated
   ▼
Answer, source excerpts, and a trace of every step (latency, tokens, cost)
```

| Component | Location | Role |
| --- | --- | --- |
| Web frontend | `frontend/src/` | Question form, answer, clickable citations, source excerpts, search steps |
| API and runtime | `src/api/main.py`, `runtime.py` | Startup, readiness, limits, demo scope, error handling |
| Contracts | `src/contracts.py` | Validated request, evidence, answer and trace formats |
| Agent loop | `src/agents/search_agent.py`, `routing.py`, `budget.py` | Search rounds, stop rules, per-question budgets |
| Evidence critic | `src/agents/search_critic.py` | LLM judgment of sufficiency, missing facts, follow-up queries |
| Generator | `src/generation/generator.py` | LLM answer, JSON validation, citation validation, cost accounting |
| Retrieval | `src/retrieval/` | BM25+, BGE-base dense search, fusion, passage selection |
| Evaluation | `src/evaluation/`, `src/agents/pilot.py` | Retrieval scoring and agent comparison runs (offline) |

**Models used**

- **Retrieval:** see the [model card](model_card_hybrid_retriever.md).
- **Critic and generator:** an Anthropic Claude model called through the API, chosen by
  the operator with `RAG_MODEL`. Recorded development runs used
  `claude-haiku-4-5-20251001`. Temperature 0, at most 1,024 output tokens per call.

**Run modes**

| Mode | What runs | Cost |
| --- | --- | --- |
| `fixture` | Scripted demo data and a fake model; tests the full UI and API path | None |
| `retrieval` | Real search; no critic or generated answer | None |
| `live` | Real search, critic and generated answers | Paid API calls, capped |

## Data flows

1. **Question in.** The browser sends the question (1–8,000 characters), the workflow,
   and optional platform/date filters. Clients cannot send an identity or permissions.
2. **Scope.** The server attaches its own demo identity, which sets the allowed
   documents and platforms (optionally narrowed by `RAG_ALLOWED_DOC_IDS_FILE` and
   `RAG_ALLOWED_SOURCES`).
3. **Search.** The retriever searches the local indexes and returns the top 5 documents
   per query, filtered to the allowed scope.
4. **Model calls (live mode only).** The question and selected excerpts (up to 24,000
   characters) are sent to the Anthropic API, first to the critic and then to the
   generator. **This is the only time data leaves the computer.** The API key is read
   from the environment and is never written to responses, traces or logs.
5. **Follow-up (agent only).** If the critic finds gaps, the router makes up to four new
   queries within the same scope, for at most two rounds in total.
6. **Answer out.** The server re-checks that every source and citation is in scope and
   was actually supplied, then returns the answer, excerpts and trace.

## Data retention

- **Web app:** keeps no database and writes no questions, answers or excerpts to disk.
  The server log records request lines and error *types* only, not provider messages or
  source text. The spending total is kept in memory and resets on restart.
- **Browser:** nothing is saved in browser storage.
- **Corpus and indexes:** stored locally in `.cache/` (ignored by Git) and mounted
  read-only in Docker.
- **Evaluation and pilot runs:** save full traces, evidence and model outputs to local
  `.cache/` folders by design, for review. Only summaries are committed.
- **Model provider:** in live mode, prompts and excerpts are processed under the
  provider's API data policy, which the operator should review.
- **Secrets:** stay in a local `.env` file that Git ignores and that must never be committed.

## Safety measures

### Guardrails

- **Evidence-only answers.** The generator is told to answer only from supplied evidence,
  not to guess, and to say `insufficient_evidence` when facts are missing or conflict.
- **Prompt-injection defense.** Both prompts treat evidence as untrusted quoted data and
  tell the model to ignore instructions inside it. Evidence and the question are sent as
  JSON fields, separate from the instructions.
- **Citation checking.** An answer that cites anything outside the supplied passages is
  rejected as an error, not shown.
- **Strict output format.** Model output must be valid JSON with exactly the expected
  fields; truncated or malformed output is rejected.
- **Critic consistency.** If the critic says "sufficient" but lists missing facts, the
  system treats it as not sufficient. It also needs confidence ≥ 0.8 to stop searching.
- **Honest partial answers.** When search stops at the round limit or finds nothing new,
  the answer is marked insufficient and ends with a stated search limitation.
- **Scope can't be widened.** The router may only narrow platforms; attempts to widen
  scope, call unknown tools or reuse a citation ID for different text stop the run.

### Filters

- **Access scope** is enforced three times: during search, before each model call, and
  on the final response.
- **Platform and date filters** are applied to every result. Undated documents are
  excluded when a date filter is used, rather than guessed.
- **Origin allowlist.** Queries from browser origins not on the list are refused (403).
- **Input validation.** Questions over 8,000 characters, blank questions, unknown fields
  and invalid date ranges are rejected (422).
- **Not in place:** no content-moderation filter on questions or answers.

### Rate and spending limits

| Limit | Default | Setting |
| --- | --- | --- |
| Queries at once | 1 (others get 429 "busy") | fixed |
| Time per question | 120 s (504 after) | `RAG_MAX_SECONDS` |
| Time per model call | 30 s | fixed |
| Tokens per question | 16,000 | `RAG_MAX_TOKENS` |
| Output tokens per model call | 1,024 | fixed |
| Search rounds | 2 | `RAG_MAX_ROUNDS` |
| Results per search | 5 | `RAG_TOP_K` |
| Evidence sent to the model | 24,000 characters | `RAG_MAX_CONTEXT_CHARS` |
| Cost per question | $0.10 | `RAG_QUESTION_BUDGET_USD` |
| Cost per server session | operator-set; required > 0 | `RAG_BUDGET_USD` |

- Paid calls need `RAG_MODE=live`, `RAG_ALLOW_PAID=true`, a model ID, an API key,
  a budget and both token prices; otherwise the app refuses to start in live mode.
- Before each call the system counts tokens and reserves the worst-case cost; a call
  that would exceed the question or session cap is not made.
- Automatic SDK retries are off, so failures don't silently add cost.

### Deployment hardening

- Served only on `127.0.0.1` (Docker publishes to localhost only).
- Docker runs as a non-root user with the data folder mounted read-only.
- Error responses use fixed messages and never include provider error text,
  credentials or document text.

## Evaluation and monitoring

**Evaluation so far**

- **Retrieval:** measured on the full corpus. The hybrid reaches Recall@10 0.722 on 282
  development questions and 0.767 on 94 calibration questions (details in the
  [model card](model_card_hybrid_retriever.md)).
- **Agent (development check, not a quality result):** 10 development question pairs with
  BM25 and Claude Haiku 4.5 (2026-10-21). Follow-up searches found more reference documents
  in 2 cases. Mean latency was 5.1 s for Fixed RAG vs 14.5 s for the agent; total estimated
  cost was $0.062 vs $0.160. Fixed RAG ran first, so caching affects the timing.
- **Critic prompt:** matched 8/8 synthetic, AI-proposed labels after revision (5/8 before).
  These were prompt-development examples, not an independent test.
- **Automated tests:** 180 tests run in CI on every push, plus a Docker smoke test of both
  workflows in fixture mode. They use fake data and make no paid calls.
- **Not yet measured:** answer correctness, citation support, critic calibration, the
  agent with the hybrid retriever, human review and red-teaming.

**Monitoring**

- `GET /api/health` (alive) and `GET /api/ready` (ready, busy, mode, model, remaining budget).
- Every response includes a trace: each step, its latency, tokens, cost and stop reason.
- There is no central logging, alerting or dashboard; this is a local, single-user app.

## Known limitations

- **Answer quality is unproven.** An "answered" status and valid citation IDs do not
  mean the answer is correct or that each claim is supported.
- **Critic confidence is not calibrated.** It guides searching; it is not an accuracy score.
- **Incomplete answers** can happen at the round limit or when useful facts fall outside
  the selected passages.
- **Invalid citations.** The model sometimes cites passages it wasn't given; these are
  caught and returned as errors instead of answers.
- **Ambiguous or reworded questions** do worse. There is no clarifying-question dialogue.
- **No reliable dates** in the corpus, so "latest" and date-limited questions are weak.
- **Single user.** One query runs at a time. A timed-out query keeps its slot until it
  finishes, so later queries wait.
- **Spending caps reset on restart** and don't cover what was already spent with the provider.
- **Heavy memory use.** The full hybrid setup needs well over 8 GiB; full-corpus Docker
  operation is not verified.

## Mitigations summary

| Risk | Mitigation |
| --- | --- |
| Made-up facts | Evidence-only prompt; insufficient-evidence status; cited excerpts shown for checking |
| Fake or wrong citations | Citations must match supplied passages or the answer is rejected |
| Prompt injection in documents | Evidence treated as untrusted data; JSON separation; output schema checks |
| Data outside a user's scope | Server-owned identity; scope checked during search, generation and response |
| Runaway cost | Explicit opt-in, per-question and session caps, cost reserved before each call, no auto-retries |
| Overload | One query at a time, time limits, token and round limits |
| Leaking secrets or text in errors | Fixed error messages; key read only from the environment; `.env` ignored by Git |
| Overtrust in answers | UI shows sources and search steps; docs state quality is unproven |

## Open issues and questions

- Measure answer quality and citation support with human review and a validated judge,
  using the rubric in [Evaluation](EVALUATION.md).
- Red-team the system (prompt injection in documents, scope escapes, cost abuse) in Phase 4.
- Decide whether to add a content-moderation filter and a clarifying-question step.
- Calibrate or replace the critic's confidence threshold.
- Verify full-corpus Docker operation and measure CPU latency.
- Real identity and permission integration would be required for any real deployment.
- Run the untouched final split once all settings are frozen.

## Incident response contact

This is a local demo with no live users. Report problems (for example a
wrong or unsafe answer, a scope leak, unexpected spending or an exposed key) by opening
an issue on the project's GitHub repository, addressed to the team: Hira Mian, Jane
Choi and Alexander Xykis. If an API key may have been exposed, revoke it in the provider
console first, then report it.

**Related documents:** [Model card](model_card_hybrid_retriever.md) ·
[System](SYSTEM.md) · [Setup](WEB_APP.md) · [Evaluation](EVALUATION.md) ·
[Data card](DATA_CARD.md)

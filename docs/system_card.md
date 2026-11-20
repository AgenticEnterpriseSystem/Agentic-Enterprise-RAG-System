# System card: Agentic RAG for Enterprise Knowledge Search

**Updated:** November 16, 2026 · **API version:** 0.1.0
**Owners:** Hira Mian, Jane Choi and Alexander Xykis, CIS 5980 capstone team

This card says what the app does, what data it uses and where it can fail. For the request flow see [how the system works](SYSTEM.md). For checks and open work see [status](STATUS.md).

## System overview and purpose

The app helps a signed-in user find information in 511,958 synthetic workplace documents about the fictional company Redwood Inference. It returns a cited answer, a partial answer, a clarification question or an error. It can also look up and count imported Jira fields.

- **Fixed RAG:** one search round, then an answer.
- **Adaptive agent:** adds a critic that checks requested facts and can ask for more text or another search.

This is a local course demo. It is not validated for real company data, public hosting, SSO or a distributed service. A department, manager, folder or role label does not grant access. No verified enterprise access map has been imported.

## Architecture and components

```text
Browser → sign-in/CSRF checks → documents this user can read
        → keyword + meaning search → selected permitted evidence
        → critic/follow-up (adaptive) → answer or clarification
        → access + citation checks → source cards and trace
```

Terms (scope, permission mask, RLS) are explained in [SYSTEM.md](SYSTEM.md). The browser and the model cannot create permission claims.

| Component | Behavior |
| --- | --- |
| Search | BM25+ keywords and BGE-base embeddings, merged by weighted rank fusion |
| Router | Plans searches; can reword a query |
| Critic | Checks requested facts; proposes more reading, reformulation or clarification |
| Generator | Writes from the selected evidence; validates citation IDs |
| PostgreSQL | Documents, accounts, grants, sessions, native records, run metadata |
| Frontend | Login, answers, source previews, citations, steps, restricted Observability page |

The live profile uses `claude-haiku-4-5-20251001` at temperature 0 for routing, critique and generation. The team did not train or fine-tune any model. The [model card](model_card_hybrid_retriever.md) covers retrieval.

Profiles:

- `retrieval`: searches real permitted documents, no model-written answers.
- `live`: provider calls, enabled explicitly and capped.
- `fixture`: scripted benchmark/test profile, unavailable with local authentication.

## Data flows

1. **Question.** The browser sends the question, workflow and filters. An HttpOnly cookie identifies a server session; CSRF protects writes. The question cannot choose an identity or add access.
2. **Permissions.** PostgreSQL checks the account, session and explicit direct/group grants. Filters only narrow access.
3. **Search.** Embeddings and indexes are shared, precomputed files. Scope is checked before the question is encoded. Only selected permitted text is read.
4. **Models.** The router, critic and generator get the question and bounded permitted evidence. They do not get the whole corpus, passwords, cookies or API keys.
5. **Release.** The app rechecks access, source identity and citation membership. These checks show source identity and access, not that every claim is true.

Evidence ceiling: 8 unique documents (configurable to 5, 8 or 10) and 24,000 evidence characters. Prompt text adds more.

Live calls send questions and selected evidence to Anthropic, under its retention policy. Local deletion does not delete remote copies. Setup may download source data, dependencies and model weights.

## Data retention

| Where | What remains |
| --- | --- |
| PostgreSQL | Documents, Jira data, password hashes, token digests, grants, minimized run and count records |
| Run history | Question hash, owner/scope binding, config hash, status, timing, evidence IDs, usage. No raw questions, answers or passage text |
| Count records | Count and filter values, coverage; filters can reveal intent |
| Browser/process memory | Questions, answers, evidence and session state while in use |
| Evaluation exports | Full questions, answers, evidence and traces in separate local files |
| Diagnostics | Content-free request/stage records and allowlisted JSON logs |

The frontend stores no answers, evidence or passwords in local or session storage. Signing out revokes the session and clears the display, but not the database, backups or exports. A question hash is linkable, not anonymous. There is no automatic history purge or user-facing deletion; operators manage local files, logs and backups. Do not publish credentials, session files or private captures.

**Observability** needs a current local session and an entry in a server-side user-ID allowlist (empty by default). Document grants do not confer it. Jane's installation allows `jane` and `hira`. History is in memory: up to 500 requests, 256 events each, one hour, cleared on restart. It excludes raw questions, document text and IDs, credentials and exception messages.

## Safety measures

- **Permission checks before release.** Masks, authorized reads and RLS cover documents, native records and counts. Unavailable authority denies access.
- **Untrusted documents.** Prompts treat document text as quoted data, not instructions. This lowers risk but is not proven against every prompt injection.
- **Structured output.** Truncated or invalid JSON, missing claims and invalid citations fail closed. The critic does not retry. Diagnostics keep fixed categories, not raw provider text.
- **Fact checklist.** The critic decides sufficiency from supported requested facts. Its confidence field is a placeholder, not a probability. It checks evidence, not the finished answer.
- **Exact provenance.** Citations carry document, revision and character span. A valid citation does not prove factual support.
- **Input checks.** Unknown fields, bad dates, and blank or overlong questions are rejected. Logout invalidates displayed results.

### Limits and deployment

| Managed limit | Value | Setting |
| --- | --- | --- |
| Concurrent expensive queries | 1 | Server admission |
| Agent run time | 120 s | `RAG_MAX_SECONDS` |
| HTTP wait | 180 s | `RAG_REQUEST_TIMEOUT_SECONDS` |
| Tokens per question | 16,000 | `RAG_MAX_TOKENS` |
| Critic output | 2,048 tokens | `RAG_CRITIC_MAX_OUTPUT_TOKENS` |
| Router/generator output | 1,024 tokens | `RAG_MODEL_MAX_OUTPUT_TOKENS` |
| Search rounds | 2 | `RAG_MAX_ROUNDS` |
| Shared context | 8 documents / 24,000 characters | `RAG_EVIDENCE_DOC_LIMIT`, `RAG_MAX_CONTEXT_CHARS` |
| Text results per query | Up to 8 | `RAG_TOP_K`, `RAG_EVIDENCE_DOC_LIMIT` |
| Native row limit | 5 | `RAG_TOP_K` |
| Cost per question | $0.10 | `RAG_QUESTION_BUDGET_USD` |
| Provider cap per process | $0.50 (reviewed local setup) | `RAG_BUDGET_USD` |
| Idle transaction | 15 s (library default 5 s) | `RAG_DB_IDLE_TRANSACTION_TIMEOUT_MS` |
| SQL statement / lock wait | 5 s / at most 2 s | Database settings |

Live mode needs explicit enablement, a model key and positive prices and budgets. Tokens are counted and worst-case cost reserved before generation. SDK retries are off. Spend counters reset on restart and are not invoices. An HTTP timeout does not cancel its running worker.

The reviewed Docker setup publishes HTTP only on host loopback and no PostgreSQL port. The app runs as a non-root user with read-only data, index and model files and a restricted non-owner database role; a separate role does imports, migrations and account work. Validated resource limits: 6 GiB / 2 CPUs for the app, 384 MiB / 1 CPU for PostgreSQL. Local HTTP uses an explicit cookie exception. Public HTTPS and SSO are outside the validated deployment.

## Evaluation and monitoring

**A working flow is not a correct answer.** Acceptance is 19 of 23 gates, not an accuracy percentage. In the table, r2 and r3 are local code snapshots, not model versions.

| Evidence | Shows | Does not show |
| --- | --- | --- |
| Historical full-corpus retrieval | Hybrid Recall@10 0.722 (development), 0.767 (calibration) | Current permission-scoped answer quality |
| 30 saved fixed outcomes, 10 per retrieval arm | A linked generation/scoring handoff: 25 passage-eligible cases, 4 unsupported native-evidence cases, 1 runtime error | A strict matched comparison across worker, session and cache conditions |
| 3 valid external machine grades and citation audits | The scoring path ran | Human-validated scores; two outputs have review flags |
| Partial critic collection | 9 outcomes and 4 native decisions captured | Calibrated stopping: no human labels or accepted freeze |
| Generated-answer UI check (r2) | 2 critic rounds, read-more, generator, source/citation UI, session cleanup worked | A correct answer: the cited source says April 7, the reference says April 6 |
| Missing-information UI check (r3) | 2 critics and read-more produced a clarification; no generator needed | A full numeric answer or proof the corpus lacks it |

Between r2 and r3 only policy and database handling changed; critic and generator code did not. The r3 clarification had an ownership phrase only partly visible in a cropped source. Failures were kept, not replayed. These are single observations, not a 5-versus-8 study.

The final question split has not been used. Reference answers stay outside generation. Historical GPT-OSS self-judged results and scripted critic examples use other protocols and do not measure the current app. See [evaluation](EVALUATION.md).

Health and readiness endpoints show whether the service is alive and ready. Admin diagnostics show stage timings, counts and fixed failure categories. They are not a central alerting service.

## Known limitations

- **Right source, wrong target.** Similar announcements can differ in date, customer or incident. The April 7 answer matched its source but missed the target. The system cannot tell which announcement should win.
- **Valid citations can support false claims.** In 6 offline scripted red-team cases, a valid citation to a 10 MB limit did not stop a claim of 10,000 MB. Invented citations were rejected. This shows a missing final support check, not a live Haiku failure rate.
- **Prompt injection is not proven safe.** No content-moderation filter exists.
- **Critic calibration is unfinished.** No human labels or accepted freeze. Token, time or cost limits can still be hit.
- **Evidence is incomplete.** A useful fact can fall outside the selected or cropped passage. Unknown dates cannot prove freshness.
- **No chat memory or streaming.** Clarification needs a new complete question.
- **Permissions are local.** No enterprise ACL import or folder inheritance. Synthetic tests do not validate a real organization.
- **Operations are local.** Fresh teammate setup, load tests, public hosting, distributed admission and automatic retention are not established.

## Open work

Evaluation and delivery remain: human review of answers and machine scores, critic calibration, frozen-setting comparisons, broader adversarial checks, fresh-setup verification and release/demo. A final-answer verifier would need labeled support/contradiction examples and measurements of false accepts, false rejects, latency and cost. See [status](STATUS.md).

## Incident response contact

Report wrong answers, suspected access leaks, exposed keys or unexpected spending through the project's GitHub issues. Do not post credentials or private questions or evidence. Revoke an exposed provider key first.

<a id="november-11-routing-failure-development-diagnostic"></a>
The earlier routing failure and its repairs are research diagnostics in [evaluation](EVALUATION.md), not a retrieval-method ranking.

**Related:** [System flow](SYSTEM.md) · [Setup](WEB_APP.md) · [Evaluation](EVALUATION.md) · [Answer evaluation](ANSWER_EVALUATION.md) · [Data card](DATA_CARD.md) · [Model card](model_card_hybrid_retriever.md)

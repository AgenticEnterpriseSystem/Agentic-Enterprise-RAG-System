# System card: Agentic RAG for Enterprise Knowledge Search

**Updated:** November 16, 2026 · **API version:** 0.1.0
**Owners:** Hira Mian, Jane Choi and Alexander Xykis, CIS 5980 capstone team

This card explains what the app does, what data it uses, and where it can fail. Start with [how the system works](SYSTEM.md) for the request flow or [current status](STATUS.md) for completed checks and open work.

## System overview and purpose

The app helps a signed-in user find information in 511,958 synthetic workplace documents about the fictional company Redwood Inference. It can return a cited answer, a partial answer, a clarification question or an error. It can also use verified imported Jira fields for supported lookups and counts.

**RAG** means finding documents before asking a language model to answer. **Fixed RAG** uses one search round. The **Adaptive agent** adds an evidence critic that checks requested facts and can ask for more text or another search. A round can contain multiple queries.

This is a local research/course demo. It is not validated for real company data, public hosting, SSO or a distributed service. Local accounts and reviewed document grants are implemented. A department, manager, folder or role label does not automatically grant access. The dataset has policy prose and organizational structure, but no complete verified enterprise access map has been imported.

## Architecture and components

```text
Browser → sign-in/CSRF checks → documents this user can read
        → keyword + meaning search → selected permitted evidence
        → critic/follow-up when adaptive → answer or clarification
        → current-access/citation checks → source cards and trace
```

A **scope** is the set of documents allowed for the current user/session. A **permission mask** is a compact yes/no lookup used during ranking. **RLS**, or row-level security, is the database's check of which rows the account can read. These layers work together; the browser and model cannot create permission claims.

| Component | Current behavior |
| --- | --- |
| Search | BM25+ keywords and BGE-base embeddings, combined with weighted rank fusion |
| Router | Plans searches/native operations; can reword a query from the first round |
| Critic | Checks requested facts and proposes more reading, reformulation or clarification |
| Generator | Writes from the selected evidence; validates supplied citation IDs |
| PostgreSQL | Stores canonical documents, accounts/grants/sessions, native records and run metadata |
| Frontend | Login, answers, exact source previews, citations, steps and a separate restricted Observability page |

The managed live profile uses `claude-haiku-4-5-20251001` at temperature 0 for routing, critique and generation. The critic can output up to 2,048 tokens; routing and answer writing use up to 1,024 each. The team did not train or fine-tune these models. [The model card](model_card_hybrid_retriever.md) describes retrieval models and historical measurements.

The `retrieval` profile searches real permitted documents without model-generated answers. The `live` profile enables capped provider calls explicitly. Scripted `fixture` mode is a separate benchmark/test profile and is unavailable with local authentication. Failed local authentication never falls back to a demo identity.

## Data flows

1. **User to app.** The browser sends a question, workflow and optional filters. Its HttpOnly cookie identifies a server session; a CSRF value protects writes from another site. The question cannot choose an identity or add access.
2. **Permission lookup.** PostgreSQL checks the active account, session and explicit direct/group grants. Filters and operator restrictions only narrow access. Buffered grant pages commit before Python row conversion; a fresh transaction rechecks authority before each page is used.
3. **Search.** Document embeddings and indexes are shared, precomputed files. The app checks scope before encoding the question. Dense ranking excludes denied rows before similarity scoring; BM25 masks before candidate selection. Only selected permitted bodies are then read from PostgreSQL. Shared shards still have to be visited; this is not a separate smaller index per user.
4. **Model processing.** The router receives the question, available capabilities and relevant critic feedback. Critic/generator calls receive bounded selected permitted text/metadata or native evidence. They do not receive the whole corpus, account passwords, browser cookies or API keys as evidence.
5. **Release.** The app rechecks current access, source identity and citation membership. The UI displays exact excerpts. These checks establish source identity and access, not whether every answer claim is true or follows from its citation.

The shared model context has a default ceiling of eight unique documents, configurable to 5, 8 or 10. It may contain fewer because of text/token limits. Supported earlier passages are retained exactly, and the critic/generator use the same frozen ordered evidence. Multiple passages from one document count once. Counts are separate atomic evidence, not eight-row samples.

The default 24,000-character evidence limit counts passage bodies and serialized native items. Titles, instructions, schema and other prompt metadata add more content. Actual provider token counting and per-question budgets remain binding.

Live calls send questions and selected evidence to Anthropic. Initial setup may download source data, dependencies and model weights. Separately invoked cloud evaluation has its own data flow. Provider processing/retention follows the configured provider's policy; local deletion does not promise remote deletion.

## Data retention

| Where | What remains |
| --- | --- |
| PostgreSQL volumes | Canonical documents, imported Jira data, account password hashes, token digests, grants and minimized run/count records |
| Ordinary run history | Question hash, owner/snapshot/scope binding, configuration hash, status, timing, evidence IDs and usage; not raw questions, answers or passage bodies in the run repository |
| Count artifacts | Count/filter values, coverage and lineage; filters can still reveal query intent |
| Browser/process memory | Questions, answers, evidence and session-display/CSRF state while in use |
| Explicit evaluation exports | Full questions, answers, evidence and traces in separate controlled local files |
| Diagnostics | Bounded content-free request/stage records and allowlisted JSON logs |

The frontend does not store answers, evidence, passwords or token payloads in local/session storage. Its HttpOnly cookies have configured expiry. Signing out revokes the session and clears displayed evidence; it does not erase the corpus, database history, backups or evaluation exports. A question hash is linkable metadata, not an anonymity guarantee.

There is no automatic run-history purge or user-facing history-deletion feature. Operators must manage local files, logs and backups. Do not publish credentials, session files or private captures with the repository.

**Observability access is separate from document access.** The page requires a current local session and an explicit server-side user-ID allowlist, which is empty by default. A role label or document grant does not confer access. Jane's checked installation explicitly allows the `jane` and `hira` accounts. History is process-local: up to 500 requests, 256 events per request and one hour, cleared on restart. Raw questions, document text/IDs, credentials and arbitrary exception messages are excluded. Safe request/run IDs, counts, times and fixed error categories remain. Standard log retention is separate from in-memory expiry.

## Safety measures

- **Permission checks before evidence release.** Scope masks, authorized reads and database RLS restrict documents, native records and counts. Current account/session/policy checks surround long work and model dispatch. Unavailable authority denies access.
- **Untrusted-document handling.** Prompts treat document text as quoted data, not instructions. This reduces risk but does not prove resistance to every prompt-injection attack.
- **Structured model output.** Truncated, invalid JSON/schema, missing required claims and invalid citations fail closed. The critic does not retry a failed response. Diagnostics keep fixed categories and safe provider stop reasons, not raw provider text.
- **Fact checklist.** The native critic derives sufficiency from supported requested facts. Its confidence field is a compatibility placeholder, not an accuracy probability. It checks evidence before generation, not the finished answer.
- **Exact provenance.** Passage citations include document/revision identity and exact character spans. Typed records/counts use their own provenance. A valid citation does not establish factual support.
- **Input/lifecycle checks.** Unknown fields, bad dates and blank/overlong questions are rejected. Undated sources fail date filters. Logout invalidates displayed results; late responses cannot repopulate the signed-out UI.

### Limits and deployment

| Managed limit | Value | Setting |
| --- | --- | --- |
| Concurrent expensive queries | 1 | Server admission |
| Agent execution | 120 seconds | `RAG_MAX_SECONDS` |
| HTTP wait | 180 seconds | `RAG_REQUEST_TIMEOUT_SECONDS` |
| Total input/output tokens per question | 16,000 | `RAG_MAX_TOKENS` |
| Critic output | 2,048 tokens | `RAG_CRITIC_MAX_OUTPUT_TOKENS` |
| Router/generator output | 1,024 tokens | `RAG_MODEL_MAX_OUTPUT_TOKENS` |
| Search rounds | 2 | `RAG_MAX_ROUNDS` |
| Shared context | At most 8 unique documents / 24,000 evidence characters | `RAG_EVIDENCE_DOC_LIMIT`, `RAG_MAX_CONTEXT_CHARS` |
| Text results per query | Up to 8 managed; max of top-k and document ceiling | `RAG_TOP_K`, `RAG_EVIDENCE_DOC_LIMIT` |
| Native row limit | 5 | `RAG_TOP_K` |
| Per-question cost | $0.10 | `RAG_QUESTION_BUDGET_USD` |
| Process provider cap | $0.50 in the reviewed local setup | `RAG_BUDGET_USD` |
| Idle transaction | 15 seconds managed; 5 seconds library default | `RAG_DB_IDLE_TRANSACTION_TIMEOUT_MS` |
| SQL statement / lock wait | 5 seconds / at most 2 seconds | Database settings |

The library's HTTP timeout override is optional; when omitted, it uses the agent limit. The explicit managed 180-second wait also covers permission work before the agent starts. SDK requests are limited to at most 30 seconds and remaining agent time. An HTTP timeout does not cancel an already running synchronous worker; its busy slot remains occupied until completion.

Live mode requires explicit enablement, model/key and positive rates/budgets. Token counting and worst-case reservations precede generation. SDK retries are disabled. Process counters reset on restart; they are not account lifetime limits or provider invoices. Evaluation journals separately retain uncertain reservations and never treat a lost response as permission to replay.

The reviewed Docker setup publishes HTTP only on host loopback and does not publish PostgreSQL on a host port. The app runs as a non-root user with read-only source-data/index/model files and a restricted nonowner database role. Import/migration/account work uses a separate role. The validated local resource limits are 6 GiB/2 CPUs for the app and 384 MiB/1 CPU for PostgreSQL; they are not a universal memory or latency guarantee. Local HTTP uses an explicit cookie exception; public HTTPS/SSO is outside the validated deployment.

## Evaluation and monitoring

**A working flow is different from a correct answer.** Current acceptance is 19 of 23 gates, not an answer-accuracy percentage. [Current status](STATUS.md) records the remaining work and evidence. The main results have different purposes. Here, r2 and r3 name successive local code snapshots, not model versions:

| Evidence | What it establishes | What it does not establish |
| --- | --- | --- |
| Historical full-corpus retrieval experiments | Hybrid Recall@10 of 0.722 development and 0.767 calibration under that protocol | Current permission-scoped agent answer quality |
| 30 saved fixed outcomes, 10 per retrieval arm | A linked generation/scoring handoff; 25 passage-eligible cases, 4 unsupported native-evidence cases and 1 runtime error | A strict matched performance comparison across different worker/session/cache conditions |
| Three valid external machine grades and citation audits | The guarded scoring/data-binding path ran | Human-validated scores; two outputs have important review flags |
| Partial critic collection | Nine outcomes and four native decisions were captured | Calibrated stopping: human labels and an accepted freeze are absent |
| Actual generated-answer UI check, r2 | Two critic rounds, read-more, generator, source/citation UI and session cleanup worked | Fully correct target answer: cited source says April 7, reference expects April 6 |
| Actual missing-information UI check, final r3 | Two critics and read-more produced an appropriate clarification; no generator or citation click was needed | A complete numeric answer or complete-corpus absence proof |

The r3 changes after the generated-answer check concern policy/database handling; critic/generator code was unchanged. The r3 clarification had a partly visible ownership phrase in a cropped source. These are specific observations, not a matched 5-versus-8 study or a quality-superiority result. One original slot remained unused; failures were preserved rather than replayed.

The final question split has not been used. Reference answers remain outside generation. Historical self-judged GPT-OSS results and scripted critic examples use different protocols; they do not measure the current app. See [evaluation](EVALUATION.md) and the [model card](model_card_hybrid_retriever.md) for the research details.

Health/readiness endpoints show whether the service is alive and ready. Current-session admin diagnostics show stage timings, counts and fixed failure categories, including separate HTTP and worker completion. They are local diagnostics, not a central alerting service.

## Known limitations

- **Correct sources can still be the wrong target.** Similar announcements may refer to different dates, customers or incidents. The live date mismatch above was supported by its retrieved source but missed the benchmark target. The system did not prove which announcement should take precedence.
- **Valid citations can support false claims.** In six offline scripted red-team cases, a valid citation to a 10 MB limit did not prevent a generated claim of 10,000 MB. Invented citations were rejected. This demonstrates a missing final semantic-support check, not a live Haiku failure rate. The evidence critic runs before the answer is written.
- **Prompt-injection resistance is not proven.** Access tests and defensive prompts establish specific controls, not safety against every malicious document. No content-moderation filter is implemented.
- **Critic calibration is unfinished.** The fact checklist is useful structure, not human truth. Model decisions have no independent human calibration labels or accepted freeze. The larger output allowance and context-sharing heuristic can still hit token, time or cost limits.
- **Evidence is incomplete.** A useful fact can be outside a selected/cropped passage. Eight documents is a ceiling, not a relevance guarantee. Unknown dates cannot prove freshness or current production state.
- **Clarification has no chat memory.** The user must submit a new complete question. There is no persistent multi-turn conversation or answer streaming.
- **Permissions are locally defined.** The app does not import native enterprise ACLs or infer manager/folder inheritance. Synthetic access tests do not validate a real organization's policies.
- **Operations remain local.** Fresh teammate setup, broader load/hardware tests, public hosting, distributed admission and automatic retention/deletion are not established.

## Open work

The remaining work is evaluation and delivery, not a claim that every implemented component is correct: independent human review of answers and machine scores, critic calibration, final frozen-setting comparisons, broader adversarial checks, fresh setup verification and release/demo work. A final-answer verifier would need labeled support/contradiction examples and measurement of false acceptance, false rejection, latency and cost. Track exact acceptance in [status](STATUS.md).

## Incident response contact

Report wrong answers, suspected access leaks, exposed keys or unexpected spending through the project's GitHub issues to the team above. Do not put credentials or private raw questions/evidence in a public issue. Revoke an exposed provider key before reporting it.

<a id="november-11-routing-failure-development-diagnostic"></a>
The earlier routing failure and later repairs remain research diagnostics in [evaluation](EVALUATION.md); they are not a current retrieval-method ranking.

**Related:** [System flow](SYSTEM.md) · [Setup](WEB_APP.md) · [Evaluation](EVALUATION.md) · [Answer evaluation](ANSWER_EVALUATION.md) · [Data card](DATA_CARD.md) · [Model card](model_card_hybrid_retriever.md)

# System

## Request flow

```text
React question form → FastAPI → configured retriever
  Fixed RAG: search → generate cited answer
  Agent:     search → check facts → read more/search again → cited answer or clarification
```

The app serves the built frontend and `/api/*` from one origin. Both workflows
use the same generator and `search(request) → evidence` retrieval interface.
The backend loads one retriever at startup and shares it across requests.
[Setup](WEB_APP.md) explains runtime modes and artifact configuration.

## Components

| Location | Responsibility |
| --- | --- |
| `src/contracts.py` | Validated requests, evidence, answers and traces |
| `src/db/`, `migrations/` | PostgreSQL foundation, immutable records, bounded runtime connections and forward migrations; application adoption remains pending |
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
The agent preserves passages cited for supported facts across rounds. It chooses
the final context before assessment and reuses it for complete or partial answers.
If a matching document is clipped, `read_more` reads one adjacent window of at
most 3,000 characters from each of up to two already-retrieved documents. It uses
the existing corpus table and preserves permissions, source/date filters and
exact citation offsets. The parent hit's score is provenance, not a new score.
Selection can still omit useful facts; these heuristics do not establish quality.
The optional cross-encoder has a 512-token default input limit; the web app does
not currently enable it.

The critic assesses requested facts and their supporting passages. The adapter
validates source IDs and derives sufficiency from the fact checklist. Claim
summaries and scope decisions remain model judgments; valid IDs do not prove
claim support. Generation receives the same evidence and assessed facts.

The agent stops on sufficient evidence, clarification, no new evidence, limits or
failure. Normal search stops can produce a cited partial answer; errors and
exhausted budgets do not trigger fallback generation. Critic confidence is not
calibrated answer accuracy. Clarification requests a distinguishing detail, such
as an incident date. Conversation state and answer streaming are not implemented;
submit a new complete question after clarification.

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

## Implementation contracts

CONTRACT-01 defines opt-in interfaces for later implementation. It does **not**
enable login, PostgreSQL, structured queries or a new retrieval backend. The
current API keeps its existing response shape in the shared
[contracts module](../src/contracts.py). Future APP-01 and WIRE-01 adopt
snapshot-bound responses after their prerequisites pass.

### Versions and source ownership

- Common wire types: [contracts.py](../src/contracts.py), with snapshot-bound wire version `2`.
- Server identity/scope boundaries: `src/access/contracts.py`.
- Storage/catalog/candidate boundaries: `src/storage/contracts.py`.
- Public login/session DTOs: `src/auth/contracts.py`.
- Runtime export versus external scoring: `src/evaluation/experiment_contracts.py`.
- A synthetic mixed passage/zero-count example is in
  [contracts.json](../tests/fixtures/contracts.json), under `grounded_response`.

P owns these shared interfaces, database metadata/migrations, dependency pins,
runtime/factory wiring and auth/UI. R implements retrieval against the accepted
interfaces and fakes; R does not create alternate account or database schemas.
Schema/migration changes belong to DB-01 and later explicit P tickets, not this
contract handoff. Keep one migration head and immutable source/snapshot IDs.

### Dataset and permission binding

`SnapshotRef` contains snapshot ID, corpus-file SHA-256, ordered catalog SHA-256
and document count. A matching ID alone is insufficient when hashes differ.
DOC-01 must publish the canonical ordered-catalog digest alongside the import.
The codec is frozen in `src/storage/fingerprints.py`: prefix `rag-catalog-v2\n`,
then one compact UTF-8 JSON array per position-ordered row, followed by a newline.
Each array is `[position, doc_id, revision_id, source_type, text_length,
timestamp_or_null, metadata_origin]`. Timestamps use UTC with six fractional
digits and `Z`; text and identifiers are not Unicode-normalized. SHA-256 covers
the complete byte stream. `catalog_fingerprint(..., expected_count=...)` rejects
position gaps and incomplete/excess catalogues without buffering the iterable.
The database enforces document-ID uniqueness. `canonical_revision_id(text)`
hashes the exact canonical UTF-8 text. Golden fixture digests lock this format;
this ticket does not invent a fingerprint for existing unimported data.

`ScopeBinding` adds a server-owned scope ID, user ID, policy revision and account
session version. Revisions/counts use strict nonnegative integers, including zero;
booleans, numeric strings and fractional values are rejected. A resolved scope
never comes from a question, document, browser role field or model response.

`ResolvedIdentity` contains trusted account metadata and a session row ID, never
the raw session cookie. `RequestContext(identity, scope)` captures its binding and
rejects a mismatched owner/session version. A scope cannot change its binding
during a request. Role/department/project labels remain descriptive.

`AuthorizationScope` exposes `binding`, `permits(doc_id)` and
`mask_for(snapshot, doc_ids)`. The mask preserves input order and duplicates;
unknown IDs deny. A different snapshot/fingerprint must reject, not reuse a cached
mask. `LegacySetScope` is the immutable small-fixture/reference adapter. Large
catalog implementations may use compact bitmaps behind this interface.

`FreshnessGuard.assert_current(context)` must check current session/account
validity, expiry, policy revision and artifact binding; it fails closed on absent
authoritative state. It must not refresh or expand a scope. Guards run before
sensitive operations and after long work before release. Operation-boundary
checks do not recall information already dispatched to a provider or browser.

### Text and computed evidence

`PassageEvidence` preserves all V1 `Evidence` fields and adds `kind="passage"`,
snapshot ID and document revision ID. Canonical text is the existing
`document_text(title, content)` representation. Offsets are Unicode-character,
zero-based and end-exclusive, never byte offsets. Preserve current citation IDs
for the immutable corpus; store snapshot/revision identity with every result.

`AggregateEvidence` has `kind="aggregate"`, artifact/citation IDs, scope binding,
query template/version, unique typed filters, count, lineage digest, declared
record coverage and an aware creation timestamp. It has no fabricated `Chunk`.
Zero is a valid result; unknown/unavailable data is not converted to zero.
Coverage is explicitly verified-record or synthetic-fixture coverage, not a claim
that all enterprise source records are available.

Filters accept strict strings, integers and booleans. They are structurally
validated here; TOOL-01 must still enforce a template's supported fields,
operations and values with parameterized queries over the complete authorized
relation before pagination. Duplicate grant joins must not inflate counts.

`validate_grounded_evidence` checks freshness/binding, snapshots, permitted
passages and duplicate citations. Aggregate metadata alone cannot authorize a
result: a trusted `AggregateRegistry` must resolve its ID to the exact canonical,
owner/scope/version-bound artifact. TOOL-01 must persist the authorized-query
lineage. Passage text/revision authenticity remains the trusted DocumentStore's
responsibility; permission or a valid citation does not establish factual support.

`QueryResponse` remains the single response model. Supplying `snapshot` opts
into tagged passage/aggregate evidence, wire version `2`, and snapshot consistency
checks; every source must then carry provenance. Snapshot-bound sources or a
version without a snapshot are rejected. Existing responses omit both optional
fields and retain their original wire shape. Citations must resolve to unique
supplied sources in either form. Model validation checks structure; adapters must
also enforce session authorization through `validate_grounded_evidence`.

### Document-store interface and reference fixture

`DocumentStore` exposes `snapshot()`, bounded `iter_catalog(batch_size=1000)`,
internal `source_types()`, `read_passage(context, candidate, *, guard, filters)`
and `read_more(context, anchors, *, guard, max_chars=3000, filters)`. Metadata
iteration yields ordered batches, not document bodies or a public source list.

`CatalogEntry` identifies an immutable revision, source, position, text length,
nullable timestamp and metadata origin. `RankedCandidate` carries those matching
IDs/spans plus a finite score, method and scorer/index provenance. A trusted store
checks that metadata before fetching text. `ReadFilters` can narrow source/date
scope only; unknown timestamps fail date filters. Source and date restrictions
do not grant document access.

Reads require an explicit freshness guard and recheck before release. Expansion
validates every anchor, uses at most two already-authorized documents and returns
one adjacent window per document. The default is 3,000 characters and the hard
maximum is 6,000 per document, matching current adapter bounds. No adjacent text
returns an empty tuple; score/method remain parent provenance.

`src/storage/testing.py` provides an eager conformance fixture capped at 1,000
documents. It is not a production storage or retrieval implementation. It makes
denied pre-body reads, stale contexts, exact Unicode spans and bounded catalogue
iteration executable for R before PostgreSQL exists. No ranking algorithm or
full-corpus loader is part of this ticket.

### Experiment and external-evaluator interface

`ExperimentManifest` records an immutable arm/repetition, full source commit and
dirty-tree digest when relevant, explicit split/question IDs, snapshot/scope,
model/prompt/index/config identities and generation/evidence limits. The DTO
provenance hashes are declarations. The offline exporter now verifies actual
files, source checkout and a supplied effective runtime configuration;
the contract itself performs no source-file or database I/O.

`RuntimeCaseExport` records exact generated outcomes, trace/usage, generation
inputs and retained sanitized errors. Unknown cost remains null. A clarification
can cite assessed sources without a generator call; its generation evidence is
empty. Error answers and error traces must agree. Runtime exports forbid extra
reference/gold fields.

`ScoringReference` and `ScoringInput` are a separate scoring-only join. They bind
reference data to question identity and the canonical runtime-case digest after
generation; they are not runtime requests. The offline export library and CLI
are documented in [evaluation](EVALUATION.md#reproducible-offline-experiment-records).
They add persistence and scoring handoff, without a judge or live generation runner.

`validate_runtime_exports` distinguishes partial resumable output from a complete
declared arm. `validate_resume_compatibility` rejects changed scientific settings.
`validate_paired_arms` checks matched fixed/adaptive question sets, generator,
scope, retrieval and budgets while permitting declared workflow/round/critic
differences. Failed attempts remain outcomes. These helpers revalidate copied
models; structural validity and matched settings do not establish answer quality.

### Authentication HTTP contract — future routes

| Route | Input/result | Later behavior |
| --- | --- | --- |
| `POST /api/auth/login` | `LoginRequest` → `SessionResponse` and cookie | Provisioned account; generic failed login; renew session token |
| `GET /api/auth/me` | `SessionResponse` | Current display identity, expiry and CSRF value; no document grants or session digest |
| `POST /api/auth/logout` | CSRF-protected request | Revoke server session, expire cookie, clear UI state |
| Defined pre-login CSRF bootstrap | `CsrfResponse` | AUTH-01 implements its same-origin/login-CSRF lifecycle |

The cookie name is `rag_session`; the CSRF header is `X-CSRF-Token`. Session tokens
stay HttpOnly and server-managed; CSRF values intentionally reach JS memory.
Credentials use `SecretStr` for redacted model/repr serialization, not for password
hashing. AUTH-01 still implements Argon2id, expiry, throttling, cookie transport
flags and CSRF checks. Login input rejects extra role/permission fields.

Errors use `{detail: {code, message}}` with generic messages. Status codes:
authentication required/invalid credentials/session expired → 401;
scope denied/CSRF failed → 403; stale policy context → 409; unavailable policy
store → 503. No error should reveal restricted source existence or provider text.
Internal `SessionExpired`, `StaleContext`, `PolicyUnavailable` and `AccessDenied`
carry explicit codes so later handlers do not inspect exception-message strings.

### Acceptance and adoption

Use the common contract checks in `tests/test_generation.py`, access checks in
`tests/test_access.py`, store/fingerprint checks in `tests/test_storage.py`,
experiment checks in `tests/test_evaluation.py`, and the existing full suite.
They validate immutable/nested data, strict fields, denied and revoked contexts,
exact text spans, snapshot mismatches, canonical aggregate provenance, export
separation and V1 compatibility. They do not prove a database import, real login,
full-corpus memory fit or answer-quality improvement.

The private Lane P implementation has accepted CONTRACT-01 and the explicitly
authorized MEM-01 profiling takeover. Wave 2 remains open until database, export
and Lane R dense work pass together; interfaces and offline fixtures do not
establish full-corpus memory fit. The ignored local plans hold actual revision
and gate evidence.


## Relational foundation

The optional database profile and migration job are documented in
[Setup](WEB_APP.md#database-foundation). The current Alembic head, `0002_catalog_import`, owns15 application tables in
schema `rag`; the original14-table foundation remains frozen in
`0001_foundation`. The initial migration is self-contained;
future metadata changes do not rewrite historical revisions. Destructive
downgrade is refused. Runtime can read migration version metadata for compatible
readiness checks, but cannot change it.

Catalog metadata contains no document bodies or titles and is internal to the
server. Text/revisions, chunks, tickets, runs/steps and aggregates have default
deny RLS. Later ACL work supplies the explicit policies; a role label does not
grant access. Database service roles are separate from application accounts.
Published snapshot identity/content and stored aggregate provenance are immutable.
Derived chunks may append against a fixed revision within canonical bounds.

`Database.transaction(context)` accepts a server-resolved request context or an
empty authority, binds identity locally, and commits/rolls back before returning
the connection. A single bounded SQLAlchemy pool performs full session reset.
Do not hold a transaction across retrieval or a model call. Authentication, scope
freshness and grant decisions remain explicit AUTH/ACL responsibilities.
[SQLAlchemy connection reset](https://docs.sqlalchemy.org/en/21/core/pooling.html#custom-reset-on-return-schemes),
[PostgreSQL identities](https://www.postgresql.org/docs/17/functions-info.html),
[Psycopg connection information](https://www.psycopg.org/psycopg3/docs/api/objects.html).


### Canonical storage and import lifecycle

The current head `0002_catalog_import` adds immutable-source checkpoints to the
14-table foundation and supports exact null-containing canonical bodies. Operator
import verifies every source hash, document order/revision and indexed span in
bounded transactions. `PostgresDocumentStore` exposes lazy metadata and exact
Unicode excerpts through the bounded nonowner pool; current policy/context is
checked before and after body access, including all anchors before expansion.

Exactly one canonical body representation is present: ordinary TEXT or original
UTF-8 BYTEA for null-containing documents. Original hashes/offsets remain exact;
binary data is validated and capped at1MiB. See [import setup](WEB_APP.md#canonical-document-import)
for batch limits, rare binary physical reads and restart/verification commands.
Default-deny RLS still applies. This ticket does not add login or activate the new
store-backed web runtime.

# V2 interface handoff

CONTRACT-01 defines opt-in interfaces for later implementation. It does **not**
enable login, PostgreSQL, structured queries or a new retrieval backend. The
current API continues using [V1 contracts](../src/contracts.py). Future APP-01 and
WIRE-01 adopt these interfaces after their prerequisites pass.

## Versions and source ownership

- Common wire types: [contracts_v2.py](../src/contracts_v2.py), version `2`.
- Server identity/scope boundaries: `src/access/contracts.py`.
- Storage/catalog/candidate boundaries: `src/storage/contracts.py`.
- Public login/session DTOs: `src/auth/contracts.py`.
- Runtime export versus external scoring: `src/evaluation/experiment_contracts.py`.
- A synthetic mixed passage/zero-count example is in
  [contracts_v2.json](../tests/fixtures/contracts_v2.json).

P owns these shared interfaces, database metadata/migrations, dependency pins,
runtime/factory wiring and auth/UI. R implements retrieval against the accepted
interfaces and fakes; R does not create alternate account or database schemas.
Schema/migration changes belong to DB-01 and later explicit P tickets, not this
contract handoff. Keep one migration head and immutable source/snapshot IDs.

## Dataset and permission binding

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

## Text and computed evidence

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

`QueryResponseV2` uses a tagged passage/aggregate union, an explicit contract
version, a common snapshot, the existing answer and trace, and unique supplied
citations. Its model validates structure, not session authorization. The running
V1 `QueryResponse` deliberately does not accept a V2 response implicitly.

## Document-store interface and reference fixture

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

## Experiment and external-evaluator interface

`ExperimentManifest` records an immutable arm/repetition, full source commit and
dirty-tree digest when relevant, explicit split/question IDs, snapshot/scope,
model/prompt/index/config identities and generation/evidence limits. Provenance
hashes are declarations; EXPORT-01 must verify the actual files and runtime
configuration. The contract itself performs no source-file or database I/O.

`RuntimeCaseExport` records exact generated outcomes, trace/usage, generation
inputs and retained sanitized errors. Unknown cost remains null. A clarification
can cite assessed sources without a generator call; its generation evidence is
empty. Error answers and error traces must agree. Runtime exports forbid extra
reference/gold fields.

`ScoringReference` and `ScoringInput` are a separate scoring-only join. They bind
reference data to question identity and the canonical runtime-case digest after
generation; they are not runtime requests. No judge or export runner is added.

`validate_runtime_exports` distinguishes partial resumable output from a complete
declared arm. `validate_resume_compatibility` rejects changed scientific settings.
`validate_paired_arms` checks matched fixed/adaptive question sets, generator,
scope, retrieval and budgets while permitting declared workflow/round/critic
differences. Failed attempts remain outcomes. These helpers revalidate copied
models; structural validity and matched settings do not establish answer quality.

## Authentication HTTP contract — future routes

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

## Acceptance and adoption

Use the focused `tests/test_v2_*contracts.py` checks and the existing full suite.
They validate immutable/nested data, strict fields, denied and revoked contexts,
exact text spans, snapshot mismatches, canonical aggregate provenance, export
separation and V1 compatibility. They do not prove a database import, real login,
full-corpus memory fit or answer-quality improvement.

P completes CONTRACT-01 independently of R/MEM-01 in wave 1. Wave 2 remains closed
until MEM-01 is actually accepted; these interfaces do not substitute for its
measured resource report.

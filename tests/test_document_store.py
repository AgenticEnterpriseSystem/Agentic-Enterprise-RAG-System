"""Real runtime-role store checks on a fresh disposable PostgreSQL database.

Positive fixture policies are deliberately narrow and exist only in the random
database this module creates. They are not an ACL implementation or a production
policy change. A separate no-policy fixture verifies the current default deny.
"""

from datetime import datetime, timedelta, timezone
import hashlib
from types import SimpleNamespace

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Store checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Store checks require the psycopg driver")
pytest.importorskip("alembic", reason="Store checks require Alembic")

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, RequestContext,
    ResolvedIdentity, SnapshotMismatch, StaleContext,
)
from src.contracts import ScopeBinding, SnapshotRef
from src.db import schema
from src.db.migrate import upgrade_database
from src.db.runtime import Database, DatabaseSettings
from src.storage.contracts import CatalogEntry, DocumentStore, RankedCandidate, ReadFilters
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import disposable_database, engine_for, pg_urls, url_string


NOW = datetime(2026, 11, 3, 12, tzinfo=timezone.utc)
SNAPSHOT_ID = "document-store-fixture"
TEXTS = (
    "  Café 🧭  \n" + "naïve🙂 e\u0301\nline\r\nspaces  " * 20,
    "Second\nA different document with a timestamp. " * 3,
    "Secret\nThis body must stay hidden by fixture RLS.",
)
TITLES = ("  Café 🧭  ", "Second", "Secret")
DOC_IDS = ("doc-é", "doc-b", "doc-secret")
ENTRIES = tuple(CatalogEntry(doc_id=doc_id, revision_id=canonical_revision_id(text),
    position=index, source_type=("jira", "slack", "gmail")[index], text_length=len(text),
    timestamp=(None, NOW, NOW - timedelta(days=1))[index], metadata_origin="synthetic")
    for index, (doc_id, text) in enumerate(zip(DOC_IDS, TEXTS)))
SNAPSHOT = SnapshotRef(snapshot_id=SNAPSHOT_ID, corpus_sha256="a" * 64,
    catalog_sha256=catalog_fingerprint(ENTRIES, expected_count=3), document_count=3)


def seed_store(connection):
    connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=SNAPSHOT_ID,
        corpus_sha256=SNAPSHOT.corpus_sha256, document_count=3))
    for entry, text, title in zip(ENTRIES, TEXTS, TITLES):
        connection.execute(schema.documents.insert().values(snapshot_id=SNAPSHOT_ID,
            **entry.model_dump()))
        connection.execute(schema.document_revisions.insert().values(snapshot_id=SNAPSHOT_ID,
            doc_id=entry.doc_id, revision_id=entry.revision_id, title=title, canonical_text=text))
    connection.execute(schema.dataset_snapshots.update().where(
        schema.dataset_snapshots.c.snapshot_id == SNAPSHOT_ID).values(
        state="ready", catalog_sha256=SNAPSHOT.catalog_sha256))


def runtime_for(url):
    return Database(DatabaseSettings(dsn=url_string(url), pool_size=1))


@pytest.fixture(scope="module")
def store_db(pg_urls):
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        owner = engine_for(fresh.migration)
        with owner.begin() as connection:
            seed_store(connection)
            # A test-only policy for this exact snapshot/principal and two IDs.
            # Third-document reads remain RLS denied even for an overbroad fake.
            connection.exec_driver_sql("""
                CREATE POLICY fixture_bounded_document_reads ON rag.document_revisions
                FOR SELECT TO rag_runtime USING (
                    snapshot_id = 'document-store-fixture'
                    AND current_setting('rag.snapshot_id', true) = snapshot_id
                    AND current_setting('rag.user_id', true) = 'fixture-reader'
                    AND doc_id IN ('doc-é', 'doc-b')
                )
            """)
        database = runtime_for(fresh.runtime)
        try:
            store = PostgresDocumentStore(database, SNAPSHOT)
            yield SimpleNamespace(urls=fresh, owner=owner, database=database, store=store)
        finally:
            database.close()
            owner.dispose()


def context(allowed=DOC_IDS, *, snapshot=SNAPSHOT, principal="fixture-reader", scope=None):
    binding = ScopeBinding(scope_id="fixture-scope", user_id=principal, snapshot=snapshot,
                           policy_revision=7, session_version=2)
    return RequestContext(ResolvedIdentity(user_id=principal, session_id="fixture-session-row",
        session_version=2, expires_at=NOW + timedelta(days=1)),
        scope or LegacySetScope(binding=binding, allowed_doc_ids=frozenset(allowed)))


class Guard:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at

    def assert_current(self, request):
        self.calls += 1
        if self.calls == self.fail_at:
            raise StaleContext("fixture freshness changed")


def candidate(index=0, *, start=0, end=20, **updates):
    entry = ENTRIES[index]
    fields = dict(snapshot=SNAPSHOT, doc_id=entry.doc_id, revision_id=entry.revision_id,
        source_type=entry.source_type, start_char=start, end_char=end, score=.75,
        retrieval_method="dense", scoring_provenance="tiny-offline-index")
    fields.update(updates)
    return RankedCandidate(**fields)


@pytest.fixture
def protected_queries(store_db):
    queries = []

    def record(_connection, _cursor, statement, parameters, _context, _many):
        if "document_revisions" in statement.lower() and statement.lstrip().lower().startswith("select"):
            queries.append((statement, parameters))

    sa.event.listen(store_db.database.engine, "before_cursor_execute", record)
    try:
        yield queries
    finally:
        sa.event.remove(store_db.database.engine, "before_cursor_execute", record)


def test_runtime_default_deny_hides_body_even_with_permissive_server_fixture_scope(pg_urls):
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        owner = engine_for(fresh.migration)
        database = runtime_for(fresh.runtime)
        try:
            with owner.begin() as connection:
                seed_store(connection)
            store = PostgresDocumentStore(database, SNAPSHOT)
            assert len(tuple(entry for batch in store.iter_catalog() for entry in batch)) == 3
            with pytest.raises(AccessDenied):
                store.read_passage(context(), candidate(), guard=Guard())
            with database.transaction(context()) as connection:
                assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)).scalar_one() == 0
        finally:
            database.close()
            owner.dispose()


def test_catalog_is_metadata_only_keyset_batched_and_ordered(store_db, protected_queries):
    store = store_db.store
    assert isinstance(store, DocumentStore) and store.snapshot() == SNAPSHOT
    batches = tuple(store.iter_catalog(batch_size=2))
    assert tuple(map(len, batches)) == (2, 1)
    assert tuple(entry for batch in batches for entry in batch) == ENTRIES
    assert all(not {"text", "canonical_text", "title", "content"} & entry.model_dump().keys()
               for batch in batches for entry in batch)
    assert store.source_types() == ("gmail", "jira", "slack")
    assert not protected_queries


def test_catalog_batches_release_pool_connection_before_the_caller_continues(store_db):
    batches = store_db.store.iter_catalog(batch_size=1)
    assert next(batches) == (ENTRIES[0],)
    assert store_db.database.engine.pool.checkedout() == 0
    # A caller may perform another short transaction while holding a batch.
    with store_db.database.transaction() as connection:
        assert connection.execute(sa.select(sa.literal(1))).scalar_one() == 1
    assert tuple(batches) == ((ENTRIES[1],), (ENTRIES[2],))


@pytest.mark.parametrize("batch_size", [0, -1, 1001, True, 1.5])
def test_catalog_rejects_invalid_batch_sizes(store_db, batch_size):
    with pytest.raises(ValueError):
        tuple(store_db.store.iter_catalog(batch_size=batch_size))


def test_exact_unicode_slice_and_provenance_require_one_bounded_body_query(store_db, protected_queries):
    # Includes an astral character and a decomposed combining sequence. PostgreSQL
    # and Python offsets must both count Unicode characters, without trimming.
    start, end = 6, 34
    guard = Guard()
    result = store_db.store.read_passage(context(), candidate(start=start, end=end), guard=guard)
    assert result.chunk.text == TEXTS[0][start:end]
    assert "🧭" in result.chunk.text and "e\u0301" in result.chunk.text
    assert result.chunk.title == TITLES[0]
    assert result.citation_id == result.chunk.chunk_id == f"{DOC_IDS[0]}:{start}:{end}"
    assert result.chunk.start_char == start and result.chunk.end_char == end
    assert result.snapshot_id == SNAPSHOT_ID and result.revision_id == ENTRIES[0].revision_id
    assert result.chunk.metadata_origin == "synthetic" and result.score == .75
    assert result.retrieval_method == "dense" and guard.calls >= 2
    assert len(protected_queries) == 1
    assert any(word in protected_queries[0][0].lower() for word in ("substring", "substr"))


@pytest.mark.parametrize("allowed,filters,index", [
    ((), ReadFilters(), 0), (("doc-b",), ReadFilters(), 0),
    (DOC_IDS, ReadFilters(source_types=("gmail",)), 0),
    (DOC_IDS, ReadFilters(after=NOW), 0),
    (DOC_IDS, ReadFilters(before=NOW), 0),
])
def test_denied_or_filtered_read_never_queries_body(store_db, protected_queries, allowed, filters, index):
    with pytest.raises(AccessDenied):
        store_db.store.read_passage(context(allowed), candidate(index), guard=Guard(), filters=filters)
    assert not protected_queries


@pytest.mark.parametrize("changes", [{"revision_id": "f" * 64}, {"source_type": "gmail"},
                                      {"end_char": 100000}])
def test_metadata_mismatch_rejected_before_body(store_db, protected_queries, changes):
    with pytest.raises(EvidenceValidationError):
        store_db.store.read_passage(context(), candidate(**changes), guard=Guard())
    assert not protected_queries


def test_real_rls_rejects_overbroad_scope_and_other_principal(store_db, protected_queries):
    for request, hit in ((context(), candidate(2)), (context(principal="other-reader"), candidate())):
        with pytest.raises(AccessDenied):
            store_db.store.read_passage(request, hit, guard=Guard())
    assert len(protected_queries) == 2


def test_inclusive_dates_allow_known_timestamp(store_db):
    result = store_db.store.read_passage(context(), candidate(1), guard=Guard(),
                                       filters=ReadFilters(after=NOW, before=NOW))
    assert result.chunk.timestamp == NOW and result.chunk.doc_id == DOC_IDS[1]


def test_stale_before_read_and_after_io_never_releases_evidence(store_db, protected_queries):
    with pytest.raises(StaleContext):
        store_db.store.read_passage(context(), candidate(), guard=Guard(fail_at=1))
    assert not protected_queries
    # Fail on the release guard independent of the exact count of preflight checks.
    class ReleaseGuard:
        def assert_current(self, request):
            if protected_queries:
                raise StaleContext("revoked during body read")
    with pytest.raises(StaleContext):
        store_db.store.read_passage(context(), candidate(), guard=ReleaseGuard())
    assert len(protected_queries) == 1


def test_copied_invalid_candidate_filters_and_altered_snapshot_fail_closed(store_db, protected_queries):
    malformed = candidate().model_copy(update={"score": float("nan")})
    with pytest.raises(EvidenceValidationError):
        store_db.store.read_passage(context(), malformed, guard=Guard())
    malformed_filter = ReadFilters().model_copy(update={"source_types": ("jira",) * 10})
    with pytest.raises(EvidenceValidationError):
        store_db.store.read_passage(context(), candidate(), guard=Guard(), filters=malformed_filter)
    changed = SNAPSHOT.model_copy(update={"corpus_sha256": "f" * 64})
    for request, hit in ((context(snapshot=changed), candidate()),
                         (context(), candidate(snapshot=changed))):
        with pytest.raises(SnapshotMismatch):
            store_db.store.read_passage(request, hit, guard=Guard())
    assert not protected_queries


def test_read_more_is_exact_bounded_deduplicated_and_preserves_anchor_provenance(store_db, protected_queries):
    request = context()
    anchor = store_db.store.read_passage(request, candidate(), guard=Guard())
    protected_queries.clear()
    result = store_db.store.read_more(request, (anchor, anchor), guard=Guard(), max_chars=64)
    assert len(result) == 1 and len(protected_queries) <= 2
    item = result[0]
    assert item.chunk.text == TEXTS[0][item.chunk.start_char:item.chunk.end_char]
    assert 0 < len(item.chunk.text) <= 64 and item.chunk.end_char > anchor.chunk.end_char
    assert item.citation_id == f"{DOC_IDS[0]}:{item.chunk.start_char}:{item.chunk.end_char}"
    assert item.score == anchor.score and item.retrieval_method == anchor.retrieval_method
    assert item.revision_id == anchor.revision_id
    assert all(any(word in sql.lower() for word in ("substring", "substr")) for sql, _ in protected_queries)


def test_read_more_checks_all_permissions_before_any_body_and_filters_skip_reads(store_db, protected_queries):
    request = context()
    anchors = tuple(store_db.store.read_passage(request, candidate(i), guard=Guard()) for i in (0, 1))
    protected_queries.clear()
    with pytest.raises(AccessDenied):
        store_db.store.read_more(context((DOC_IDS[0],)), anchors, guard=Guard())
    assert not protected_queries
    assert store_db.store.read_more(request, (anchors[0],), guard=Guard(), filters=ReadFilters(after=NOW)) == ()
    assert not protected_queries


def test_read_more_rejects_tampered_text_and_metadata_and_invalid_bounds(store_db, protected_queries):
    request = context()
    anchor = store_db.store.read_passage(request, candidate(), guard=Guard())
    for updates in ({"revision_id": "f" * 64}, {"citation_id": "invented"}):
        with pytest.raises(EvidenceValidationError):
            store_db.store.read_more(request, (anchor.model_copy(update=updates),), guard=Guard())
    malformed = anchor.model_copy(update={"chunk": anchor.chunk.model_copy(update={
        "text": "?" * len(anchor.chunk.text)})})
    with pytest.raises(EvidenceValidationError):
        store_db.store.read_more(request, (malformed,), guard=Guard())
    for value in (0, -1, 6001, True, 1.5):
        with pytest.raises(ValueError):
            store_db.store.read_more(request, (anchor,), guard=Guard(), max_chars=value)
    with pytest.raises(ValueError):
        store_db.store.read_more(request, (anchor,) * 3, guard=Guard())


def test_read_more_whole_document_has_no_progress_and_two_documents_stay_bounded(store_db, protected_queries):
    request = context()
    full = store_db.store.read_passage(request, candidate(1, end=len(TEXTS[1])), guard=Guard())
    assert store_db.store.read_more(request, (full,), guard=Guard()) == ()
    anchors = tuple(store_db.store.read_passage(request, candidate(i), guard=Guard()) for i in (1, 0))
    protected_queries.clear()
    result = store_db.store.read_more(request, anchors, guard=Guard(), max_chars=40)
    assert [item.chunk.doc_id for item in result] == [DOC_IDS[1], DOC_IDS[0]]
    assert len(protected_queries) <= 2 and all(len(item.chunk.text) <= 40 for item in result)


def test_store_constructor_rejects_changed_fingerprints_and_incomplete_imports(store_db):
    for changes in ({"corpus_sha256": "f" * 64}, {"catalog_sha256": "e" * 64}, {"document_count": 4}):
        with pytest.raises((SnapshotMismatch, EvidenceValidationError, ValueError)):
            PostgresDocumentStore(store_db.database, SNAPSHOT.model_copy(update=changes))
    with store_db.owner.begin() as connection:
        connection.execute(schema.dataset_snapshots.insert().values(snapshot_id="incomplete-store-fixture",
            corpus_sha256="c" * 64, document_count=1))
    incomplete = SNAPSHOT.model_copy(update={"snapshot_id": "incomplete-store-fixture", "corpus_sha256": "c" * 64,
                                           "document_count": 1})
    with pytest.raises((SnapshotMismatch, EvidenceValidationError, ValueError)):
        PostgresDocumentStore(store_db.database, incomplete)


def test_constructor_checks_actual_catalog_digest_and_empty_catalog_is_valid(store_db, protected_queries):
    corrupt_id = "wrong-catalog-digest-fixture"
    with store_db.owner.begin() as connection:
        connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=corrupt_id,
            corpus_sha256="d" * 64, document_count=1))
        connection.execute(schema.documents.insert().values(snapshot_id=corrupt_id,
            **ENTRIES[0].model_dump()))
        connection.execute(schema.document_revisions.insert().values(snapshot_id=corrupt_id,
            doc_id=DOC_IDS[0], revision_id=ENTRIES[0].revision_id,
            title=TITLES[0], canonical_text=TEXTS[0]))
        connection.execute(schema.dataset_snapshots.update().where(
            schema.dataset_snapshots.c.snapshot_id == corrupt_id).values(state="ready", catalog_sha256="e" * 64))
        empty_digest = catalog_fingerprint((), expected_count=0)
        connection.execute(schema.dataset_snapshots.insert().values(snapshot_id="empty-store-fixture",
            corpus_sha256="f" * 64, catalog_sha256=empty_digest, document_count=0, state="ready"))
    corrupt = SnapshotRef(snapshot_id=corrupt_id, corpus_sha256="d" * 64,
                         catalog_sha256="e" * 64, document_count=1)
    with pytest.raises(SnapshotMismatch):
        PostgresDocumentStore(store_db.database, corrupt)
    empty = PostgresDocumentStore(store_db.database, SnapshotRef(snapshot_id="empty-store-fixture",
        corpus_sha256="f" * 64, catalog_sha256=empty_digest, document_count=0))
    assert tuple(empty.iter_catalog(batch_size=1)) == () and empty.source_types() == ()
    assert not protected_queries


@pytest.fixture
def binary_store_db(pg_urls):
    """The rare exact-UTF8 path, with a policy confined to another fresh DB."""
    text = "Binary\nlead🙂\x00literal \\000 e\u0301🧭 tail\x00\x10\x02 end. " * 8
    entry = CatalogEntry(doc_id="nul-document", revision_id=canonical_revision_id(text),
        position=0, source_type="jira", text_length=len(text), metadata_origin="upstream")
    snapshot = SnapshotRef(snapshot_id="binary-store-fixture", corpus_sha256="b" * 64,
        catalog_sha256=catalog_fingerprint((entry,), expected_count=1), document_count=1)
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        owner = engine_for(fresh.migration)
        database = runtime_for(fresh.runtime)
        try:
            with owner.begin() as connection:
                connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=snapshot.snapshot_id,
                    corpus_sha256=snapshot.corpus_sha256, document_count=1))
                connection.execute(schema.documents.insert().values(snapshot_id=snapshot.snapshot_id,
                    **entry.model_dump()))
                connection.execute(schema.document_revisions.insert().values(snapshot_id=snapshot.snapshot_id,
                    doc_id=entry.doc_id, revision_id=entry.revision_id, title="Binary",
                    canonical_text=None, canonical_utf8=text.encode("utf-8")))
                connection.execute(schema.dataset_snapshots.update().where(
                    schema.dataset_snapshots.c.snapshot_id == snapshot.snapshot_id).values(
                    state="ready", catalog_sha256=snapshot.catalog_sha256))
                connection.exec_driver_sql("""
                    CREATE POLICY fixture_binary_document_read ON rag.document_revisions
                    FOR SELECT TO rag_runtime USING (
                        snapshot_id = 'binary-store-fixture'
                        AND current_setting('rag.snapshot_id', true) = snapshot_id
                        AND current_setting('rag.user_id', true) = 'fixture-reader'
                        AND doc_id = 'nul-document'
                    )
                """)
            yield SimpleNamespace(snapshot=snapshot, entry=entry, text=text,
                database=database, store=PostgresDocumentStore(database, snapshot))
        finally:
            database.close()
            owner.dispose()


def test_binary_utf8_roundtrip_preserves_nul_unicode_offsets_and_adjacent_reads(binary_store_db):
    data = binary_store_db
    request = context((data.entry.doc_id,), snapshot=data.snapshot)
    start = data.text.index("\x00") - 1
    end = start + 24
    hit = RankedCandidate(snapshot=data.snapshot, doc_id=data.entry.doc_id,
        revision_id=data.entry.revision_id, source_type=data.entry.source_type,
        start_char=start, end_char=end, score=.5, retrieval_method="dense",
        scoring_provenance="tiny-binary-index")
    evidence = data.store.read_passage(request, hit, guard=Guard())
    assert evidence.chunk.text == data.text[start:end]
    assert "\x00" in evidence.chunk.text and "\\000" in evidence.chunk.text
    assert "🙂" in evidence.chunk.text and "e\u0301" in evidence.chunk.text
    assert evidence.revision_id == canonical_revision_id(data.text)
    assert evidence.citation_id == f"nul-document:{start}:{end}"
    more = data.store.read_more(request, (evidence,), guard=Guard(), max_chars=64)
    assert len(more) == 1
    assert more[0].chunk.text == data.text[more[0].chunk.start_char:more[0].chunk.end_char]
    assert 0 < len(more[0].chunk.text) <= 64 and more[0].chunk.end_char > end
    # Reconstructing the adapter creates no text cache or repaired representation.
    restarted = PostgresDocumentStore(data.database, data.snapshot)
    assert restarted.read_passage(request, hit, guard=Guard()) == evidence


@pytest.mark.parametrize("corruption", ["both", "neither", "invalid_utf8", "bridged_utf8",
                                        "surrogate_utf8", "oversized", "hash", "length"])
def test_binary_body_guards_reject_ambiguous_corrupt_or_unbounded_representations(store_db, corruption):
    text = "Binary\n🙂\x00literal \\000 e\u0301"
    raw = text.encode("utf-8")
    revision = canonical_revision_id(text)
    text_length = len(text)
    canonical_text = None
    if corruption == "both":
        canonical_text = "another representation"
    elif corruption == "neither":
        raw = None
    elif corruption == "invalid_utf8":
        raw = b"\x00\xff\xc0\x80"
    elif corruption == "bridged_utf8":
        raw = b"\xc2\x00\xa0"
    elif corruption == "surrogate_utf8":
        raw = b"\x00\xed\xa0\x80"
    elif corruption == "oversized":
        raw = b"\x00" + b"x" * (1024 * 1024)
        revision = hashlib.sha256(raw).hexdigest()
        text_length = len(raw)
    elif corruption == "hash":
        revision = "f" * 64
    else:
        text_length += 1
    if corruption in {"invalid_utf8", "bridged_utf8", "surrogate_utf8"}:
        # Keep the raw digest correct so a hash check cannot accidentally stand
        # in for actual UTF-8 validation.
        revision = hashlib.sha256(raw).hexdigest()
        text_length = len(raw)
    snapshot_id = "invalid-binary-" + corruption
    with store_db.owner.begin() as connection:
        connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=snapshot_id,
            corpus_sha256="c" * 64, document_count=1))
        connection.execute(schema.documents.insert().values(snapshot_id=snapshot_id,
            doc_id="invalid-binary", position=0, revision_id=revision, source_type="jira",
            text_length=text_length))
        with pytest.raises(sa.exc.DBAPIError):
            with connection.begin_nested():
                connection.execute(schema.document_revisions.insert().values(snapshot_id=snapshot_id,
                    doc_id="invalid-binary", revision_id=revision, title="Binary",
                    canonical_text=canonical_text, canonical_utf8=raw))
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)
            .where(schema.document_revisions.c.snapshot_id == snapshot_id)).scalar_one() == 0

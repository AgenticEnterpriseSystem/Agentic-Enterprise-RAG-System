"""Real PostgreSQL explicit-policy checks; no full-corpus grants are applied.

Each test owns a fresh disposable database and tiny canonical catalog. Actual
opaque auth sessions are used; only password hashing is replaced with a cheap
test double. Labels and document prose remain descriptive, never authority.
"""

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from uuid import uuid4

import pytest
import numpy as np
from fastapi.testclient import TestClient

sa = pytest.importorskip("sqlalchemy", reason="Policy checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Policy checks require the psycopg driver")
pytest.importorskip("alembic", reason="Policy checks require Alembic")

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, LegacySetScope, PolicyUnavailable, RequestContext, SessionExpired,
    SnapshotMismatch, StaleContext,
)
from src.access.policy import PolicyResolver
from src.access.repository import (
    GrantSpec, GroupSpec, MembershipSpec, PolicyManifest, PolicyRepository, verify_benchmark,
)
import src.access.repository as policy_module
from src.auth.accounts import create_account, disable_account, reset_password
from src.auth.sessions import AuthError, AuthService
from src.auth.settings import AuthSettings
from src.api.main import create_app
from src.api.runtime import WebSettings
from src.contracts import ScopeBinding, SnapshotRef
from src.db import schema
from src.db.migrate import upgrade_database
from src.db.runtime import Database, DatabaseSettings
from src.storage.contracts import CatalogEntry, RankedCandidate, ReadFilters
from src.storage.fingerprints import canonical_revision_id, catalog_fingerprint
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import disposable_database, engine_for, pg_urls, url_string


PASSWORD = "Synthetic ACL fixture password 🧭"
DOC_IDS = ("general", "atlas", "finance-exec", "direct", "unclassified",
           "padding-five", "padding-six", "boundary-seven", "boundary-eight", "padding-nine",
           "padding-ten", "padding-eleven", "padding-twelve", "padding-thirteen", "padding-fourteen",
           "boundary-fifteen", "last-sixteen")
TEXTS = tuple(name.title() + "\nSynthetic document text, never an inferred grant. Café 🧭."
              for name in DOC_IDS)
SOURCES = ("jira", "slack", "confluence", "gmail", "jira", "jira", "jira", "slack", "jira",
           "jira", "jira", "jira", "jira", "jira", "jira", "slack", "jira")
ENTRIES = tuple(CatalogEntry(doc_id=name, revision_id=canonical_revision_id(text),
    position=index, source_type=SOURCES[index], text_length=len(text), metadata_origin="synthetic")
    for index, (name, text) in enumerate(zip(DOC_IDS, TEXTS)))


class FixtureHasher:
    def hash(self, password):
        return "fixture-only$" + hashlib.sha256(password.encode("utf-8")).hexdigest()

    def verify(self, encoded, password):
        return encoded == self.hash(password)


def seed_documents(connection, snapshot):
    connection.execute(schema.dataset_snapshots.insert().values(snapshot_id=snapshot.snapshot_id,
        corpus_sha256=snapshot.corpus_sha256, document_count=snapshot.document_count))
    for entry, text in zip(ENTRIES, TEXTS):
        connection.execute(schema.documents.insert().values(snapshot_id=snapshot.snapshot_id,
            **entry.model_dump()))
        connection.execute(schema.document_revisions.insert().values(snapshot_id=snapshot.snapshot_id,
            doc_id=entry.doc_id, revision_id=entry.revision_id, title=entry.doc_id.title(), canonical_text=text))
        connection.execute(schema.chunks.insert().values(snapshot_id=snapshot.snapshot_id,
            chunk_id=entry.doc_id + ":0:8", doc_id=entry.doc_id, revision_id=entry.revision_id,
            start_char=0, end_char=8, chunker_id="synthetic-acl-fixture"))
        connection.execute(schema.tickets.insert().values(snapshot_id=snapshot.snapshot_id,
            ticket_id="ticket-" + entry.doc_id, doc_id=entry.doc_id, revision_id=entry.revision_id,
            coverage="synthetic_fixture"))
    connection.execute(schema.dataset_snapshots.update().where(
        schema.dataset_snapshots.c.snapshot_id == snapshot.snapshot_id).values(
        state="ready", catalog_sha256=snapshot.catalog_sha256))


@pytest.fixture
def acl_db(pg_urls):
    with disposable_database(pg_urls) as fresh:
        upgrade_database(url_string(fresh.migration))
        owner = engine_for(fresh.migration)
        database = Database(DatabaseSettings(dsn=url_string(fresh.runtime), pool_size=1))
        snapshot = SnapshotRef(snapshot_id="acl-" + uuid4().hex, corpus_sha256="a" * 64,
            catalog_sha256=catalog_fingerprint(ENTRIES, expected_count=len(ENTRIES)), document_count=len(ENTRIES))
        with owner.begin() as connection:
            seed_documents(connection, snapshot)
        hasher = FixtureHasher()
        service = AuthService(database, AuthSettings(enabled=True, allowed_origins=("http://localhost:8000",),
            secure_cookies=False, allow_insecure_loopback=True, idle_seconds=1800, absolute_seconds=3600,
            per_login_attempts=100, per_ip_attempts=100, global_attempts=1000), hasher=hasher)
        users, grants, logins = {}, {}, {}
        descriptions = {
            "intern": dict(role="intern", department="research", project_ids=("atlas",)),
            "executive": dict(role="ceo", department="finance", project_ids=("atlas",)),
            "labels_only": dict(role="ceo", department="finance", project_ids=("atlas",)),
            "split_groups": dict(role="analyst", department="finance", project_ids=()),
            "direct_user": dict(role="staff", department="sales", project_ids=()),
        }
        try:
            for name, metadata in descriptions.items():
                login = name + "_" + uuid4().hex
                logins[name] = login
                users[name] = create_account(login, PASSWORD, display_name="Synthetic " + name,
                    url=url_string(fresh.migration), hasher=hasher, **metadata)
                grants[name] = service.login(login, PASSWORD, client_ip="127.0.0.1")
            yield SimpleNamespace(urls=fresh, owner=owner, database=database, snapshot=snapshot,
                auth=service, hasher=hasher, users=users, sessions=grants, logins=logins)
        finally:
            database.close()
            owner.dispose()


def raw_token(fixture, principal):
    return fixture.sessions[principal].token.get_secret_value()


def identity(fixture, principal):
    return fixture.auth.authenticate(raw_token(fixture, principal), touch=False).identity


def revision(fixture):
    with fixture.owner.connect() as connection:
        return connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one()


def candidate(fixture, document="general"):
    entry = ENTRIES[DOC_IDS.index(document)]
    return RankedCandidate(snapshot=fixture.snapshot, doc_id=document, revision_id=entry.revision_id,
        source_type=entry.source_type, start_char=0, end_char=8, score=.5,
        retrieval_method="bm25", scoring_provenance="tiny-acl-index")


@contextmanager
def rejected_sql(connection):
    with pytest.raises(sa.exc.DBAPIError) as error:
        with connection.begin_nested():
            yield
    assert error.value.orig.sqlstate == "42501"


def fixture_manifest(fixture, *, manifest_id="tiny-reviewed-policy", **changes):
    groups = ("company-general", "atlas-team", "finance", "executive", "finance_exec")
    memberships = {"intern": ("company-general", "atlas-team"),
        "executive": ("company-general", "finance", "executive", "finance_exec"),
        "split_groups": ("finance", "executive"), "direct_user": ("company-general",)}
    fields = dict(schema_version="1", manifest_id=manifest_id, snapshot=fixture.snapshot, origin="synthetic",
        reviewed_by="fixture-operator", provenance={"purpose": "tiny explicit operator-reviewed fixture",
            "source": "synthetic assignments; no document-content inference"},
        groups=tuple(GroupSpec(group_id=name, label=name,
            requires=("finance", "executive") if name == "finance_exec" else ()) for name in groups),
        memberships=tuple(MembershipSpec(user_id=fixture.users[principal].user_id, group_id=name)
            for principal, names in memberships.items() for name in names),
        grants=tuple(GrantSpec(doc_ids=(document,), group_id=group) for document, group in (
            ("general", "company-general"), ("atlas", "atlas-team"), ("finance-exec", "finance_exec"))) + (
                GrantSpec(doc_ids=("general", "direct"), user_id=fixture.users["direct_user"].user_id),))
    fields.update(changes)
    return PolicyManifest(**fields)


def repository(fixture):
    return PolicyRepository(url=url_string(fixture.urls.migration))


def seed_explicit_grants(fixture):
    return repository(fixture).apply(fixture_manifest(fixture))


@pytest.fixture
def policy(acl_db):
    seed_explicit_grants(acl_db)
    resolver = PolicyResolver(acl_db.database, acl_db.snapshot)
    contexts = {name: resolver.resolve(identity(acl_db, name)) for name in acl_db.users}
    return SimpleNamespace(fixture=acl_db, resolver=resolver, contexts=contexts)


def allowed_rows(fixture, context, table=schema.document_revisions):
    column = table.c.doc_id
    with fixture.database.transaction(context) as connection:
        return tuple(connection.execute(sa.select(column).order_by(column)).scalars())


def seed_run_artifacts(connection, fixture, context, suffix):
    run_id = "run-" + suffix
    binding = context.binding
    fields = dict(owner_user_id=binding.user_id, snapshot_id=binding.snapshot.snapshot_id,
        scope_id=binding.scope_id, policy_revision=binding.policy_revision, session_version=binding.session_version)
    connection.execute(schema.query_runs.insert().values(run_id=run_id, **fields,
        config_sha256="c" * 64, mode="baseline", status="answered", question="Synthetic authorized run"))
    connection.execute(schema.run_steps.insert().values(run_id=run_id, step_number=1,
                                                       payload={"note": "synthetic fixture"}))
    connection.execute(schema.aggregate_results.insert().values(artifact_id="aggregate-" + suffix,
        citation_id="count:" + suffix, run_id=run_id, **fields, template_id="ticket_count",
        template_version="1", filters=[], operation="count", count=0,
        lineage_sha256="d" * 64, coverage="synthetic_fixture"))
    return run_id


@pytest.mark.parametrize("principal,expected", [("intern", {"general", "atlas"}),
    ("executive", {"general", "finance-exec"}), ("labels_only", set()),
    ("split_groups", set()), ("direct_user", {"general", "direct"})])
def test_actual_sessions_explicit_grants_and_sql_rows_match_legacy_set(policy, principal, expected):
    fixture, context = policy.fixture, policy.contexts[principal]
    policy.resolver.assert_current(context)
    legacy = LegacySetScope(binding=context.binding, allowed_doc_ids=frozenset(expected))
    orders = (DOC_IDS, tuple(reversed(DOC_IDS)), ("unknown", *DOC_IDS, "general", "unknown"), ())
    for order in orders:
        actual = context.scope.mask_for(fixture.snapshot, order)
        assert tuple(actual) == legacy.mask_for(fixture.snapshot, order)
        assert all(isinstance(value, (bool, np.bool_)) for value in actual)
    for name in (*DOC_IDS, "unknown"):
        assert context.scope.permits(name) is legacy.permits(name)
    assert set(allowed_rows(fixture, context)) == expected
    assert set(allowed_rows(fixture, context, schema.chunks)) == expected
    assert set(allowed_rows(fixture, context, schema.tickets)) == expected


def test_scope_snapshot_fingerprints_and_order_cannot_be_substituted(policy):
    context, fixture = policy.contexts["intern"], policy.fixture
    assert isinstance(context.scope.bitmap, bytes)
    for changes in ({"snapshot_id": "unknown"}, {"corpus_sha256": "f" * 64},
                    {"catalog_sha256": "e" * 64}, {"document_count": 11}):
        changed = fixture.snapshot.model_copy(update=changes)
        with pytest.raises(SnapshotMismatch):
            context.scope.mask_for(changed, DOC_IDS)
        with pytest.raises((SnapshotMismatch, ValueError)):
            PolicyResolver(fixture.database, changed).resolve(identity(fixture, "intern"))
    with pytest.raises((AttributeError, TypeError, ValueError)):
        context.scope.bitmap = b"\xff\xff"


def test_role_department_project_metadata_and_split_base_groups_do_not_confer_authority(policy):
    labels = policy.contexts["labels_only"]
    assert labels.identity.role == "ceo" and labels.identity.department == "finance"
    assert labels.identity.project_ids == ("atlas",)
    assert not any(labels.scope.mask_for(policy.fixture.snapshot, DOC_IDS))
    split = policy.contexts["split_groups"]
    assert {"finance", "executive"} <= set(split.identity.group_ids)
    assert split.scope.permits("finance-exec") is False
    assert not allowed_rows(policy.fixture, split)


def test_operator_restrictions_intersect_grants_in_python_and_actual_sql(policy):
    fixture = policy.fixture
    resolver = PolicyResolver(fixture.database, fixture.snapshot,
        source_types=("jira",), doc_ids=("general", "atlas", "unclassified"))
    context = resolver.resolve(identity(fixture, "intern"))
    assert tuple(context.scope.mask_for(fixture.snapshot, DOC_IDS)) == (True,) + (False,) * (len(DOC_IDS) - 1)
    for table in (schema.document_revisions, schema.chunks, schema.tickets):
        assert allowed_rows(fixture, context, table) == ("general",)
    empty = PolicyResolver(fixture.database, fixture.snapshot, doc_ids=()).resolve(identity(fixture, "intern"))
    assert not any(empty.scope.mask_for(fixture.snapshot, DOC_IDS))
    assert allowed_rows(fixture, empty) == ()


def test_overlapping_direct_and_group_grants_do_not_duplicate_sql_rows(policy):
    fixture = policy.fixture
    context = policy.contexts["direct_user"]
    assert allowed_rows(fixture, context) == ("direct", "general")
    with fixture.database.transaction(context) as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)).scalar_one() == 2


def test_direct_and_multiple_group_paths_collapse_to_one_permission_bit(policy):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        # general is already granted through company-general. Add a second
        # effective group path and a direct path to that same exact document.
        connection.execute(schema.document_grants.insert().values(snapshot_id=fixture.snapshot.snapshot_id,
            doc_id="general", group_id="atlas-team", origin="synthetic"))
        connection.execute(schema.document_grants.insert().values(snapshot_id=fixture.snapshot.snapshot_id,
            doc_id="general", user_id=fixture.users["intern"].user_id, origin="synthetic"))
    context = policy.resolver.resolve(identity(fixture, "intern"))
    expected = b"\x03\x00\x00"
    assert context.scope.allowed_count == 2 and context.scope.bitmap == expected
    assert set(allowed_rows(fixture, context)) == {"general", "atlas"}
    with fixture.owner.connect() as connection:
        stored = connection.execute(sa.select(schema.policy_scopes.c.allowed_positions).where(
            schema.policy_scopes.c.scope_id == context.binding.scope_id)).scalar_one()
    assert bytes(stored) == expected


def test_no_context_denies_all_six_protected_tables_even_when_grants_and_artifacts_exist(policy):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        seed_run_artifacts(connection, fixture, policy.contexts["intern"], "intern")
        seed_run_artifacts(connection, fixture, policy.contexts["executive"], "executive")
    with fixture.database.transaction() as connection:
        for name in schema.PROTECTED_TABLES:
            assert connection.execute(sa.text(f'SELECT count(*) FROM rag."{name}"')).scalar_one() == 0


def test_run_steps_and_zero_count_artifacts_are_owner_and_scope_bound(policy):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        for principal in ("intern", "executive"):
            seed_run_artifacts(connection, fixture, policy.contexts[principal], principal)
    for principal in ("intern", "executive"):
        context = policy.contexts[principal]
        with fixture.database.transaction(context) as connection:
            assert connection.execute(sa.select(schema.query_runs.c.run_id)).scalars().all() == ["run-" + principal]
            assert connection.execute(sa.select(schema.run_steps.c.run_id)).scalars().all() == ["run-" + principal]
            assert connection.execute(sa.select(schema.aggregate_results.c.artifact_id)).scalars().all() == ["aggregate-" + principal]
            assert connection.execute(sa.select(schema.aggregate_results.c.count)).scalar_one() == 0
            other = "executive" if principal == "intern" else "intern"
            with rejected_sql(connection):
                connection.execute(schema.query_runs.insert().values(run_id="forged-" + principal,
                    owner_user_id=fixture.users[other].user_id, snapshot_id=fixture.snapshot.snapshot_id,
                    scope_id=context.binding.scope_id, policy_revision=context.binding.policy_revision,
                    session_version=context.binding.session_version, config_sha256="c" * 64,
                    mode="baseline", question="Synthetic forged owner"))


def test_one_pooled_connection_does_not_retain_another_users_scope(policy):
    fixture = policy.fixture
    pid = None
    for principal, expected in (("intern", {"general", "atlas"}), (None, set()),
                                ("executive", {"general", "finance-exec"}), (None, set())):
        context = policy.contexts[principal] if principal else None
        with fixture.database.transaction(context) as connection:
            current = connection.execute(sa.text("SELECT pg_backend_pid()")).scalar_one()
            pid = current if pid is None else pid
            assert current == pid
            actual = set(connection.execute(sa.select(schema.document_revisions.c.doc_id)).scalars())
            assert actual == expected
            if principal is None:
                for key in ("user_id", "session_id", "scope_id", "snapshot_id"):
                    assert connection.execute(sa.text("SELECT current_setting(:key, true)"),
                        {"key": "rag." + key}).scalar_one() in (None, "")


def test_document_reads_and_adjacent_windows_deny_before_materialization(policy):
    fixture = policy.fixture
    store = PostgresDocumentStore(fixture.database, fixture.snapshot)
    intern = policy.contexts["intern"]
    body_queries = []

    def watch(_connection, _cursor, statement, _parameters, _context, _many):
        if "document_revisions" in statement.lower():
            body_queries.append(statement)

    sa.event.listen(fixture.database.engine, "before_cursor_execute", watch)
    try:
        with pytest.raises(AccessDenied):
            store.read_passage(intern, candidate(fixture, "finance-exec"), guard=policy.resolver)
        assert not body_queries
        anchor = store.read_passage(intern, candidate(fixture), guard=policy.resolver)
        assert anchor.chunk.text == TEXTS[0][:8]
        expanded = store.read_more(intern, (anchor,), guard=policy.resolver, max_chars=24)
        assert len(expanded) == 1 and expanded[0].chunk.doc_id == "general"
        assert expanded[0].chunk.text == TEXTS[0][expanded[0].chunk.start_char:expanded[0].chunk.end_char]
    finally:
        sa.event.remove(fixture.database.engine, "before_cursor_execute", watch)


@pytest.mark.parametrize("missing_base", ["finance", "executive"])
def test_compound_membership_requires_each_explicit_base_and_revocation_invalidates_context(policy, missing_base):
    fixture = policy.fixture
    old = policy.contexts["executive"]
    assert old.scope.permits("finance-exec") is True
    before = revision(fixture)
    with fixture.owner.begin() as connection:
        connection.execute(schema.group_memberships.delete().where(
            schema.group_memberships.c.user_id == fixture.users["executive"].user_id,
            schema.group_memberships.c.group_id == missing_base))
    assert revision(fixture) > before
    with pytest.raises(StaleContext):
        policy.resolver.assert_current(old)
    assert allowed_rows(fixture, old) == ()
    current = policy.resolver.resolve(identity(fixture, "executive"))
    assert "finance_exec" in current.identity.group_ids
    assert current.scope.permits("finance-exec") is False
    assert allowed_rows(fixture, current) == ("general",)


def test_actual_compound_membership_alone_without_bases_grants_nothing(policy):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        connection.execute(schema.group_memberships.insert().values(user_id=fixture.users["labels_only"].user_id,
            group_id="finance_exec", origin="synthetic"))
    context = policy.resolver.resolve(identity(fixture, "labels_only"))
    assert "finance_exec" in context.identity.group_ids
    assert not any(context.scope.mask_for(fixture.snapshot, DOC_IDS))
    assert allowed_rows(fixture, context) == ()


def test_direct_explicit_grant_remains_independent_of_compound_group_membership(policy):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        connection.execute(schema.document_grants.insert().values(snapshot_id=fixture.snapshot.snapshot_id,
            doc_id="finance-exec", user_id=fixture.users["split_groups"].user_id, origin="synthetic"))
    context = policy.resolver.resolve(identity(fixture, "split_groups"))
    assert context.scope.permits("finance-exec") is True
    assert allowed_rows(fixture, context) == ("finance-exec",)


def test_all_boundary_permission_subsets_match_literal_little_endian_bytes_and_sql(acl_db):
    fixture = acl_db
    selected_positions = (0, 1, 7, 8, 15, 16)
    user_id = fixture.users["labels_only"].user_id
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    for subset in range(1 << len(selected_positions)):
        expected = {DOC_IDS[position] for bit, position in enumerate(selected_positions) if subset & (1 << bit)}
        with fixture.owner.begin() as connection:
            connection.execute(schema.document_grants.delete().where(schema.document_grants.c.user_id == user_id))
            if expected:
                connection.execute(schema.document_grants.insert(), [dict(snapshot_id=fixture.snapshot.snapshot_id,
                    doc_id=name, user_id=user_id, origin="synthetic") for name in sorted(expected)])
        context = resolver.resolve(identity(fixture, "labels_only"))
        legacy = LegacySetScope(binding=context.binding, allowed_doc_ids=frozenset(expected))
        assert tuple(context.scope.mask_for(fixture.snapshot, DOC_IDS)) == legacy.mask_for(fixture.snapshot, DOC_IDS)
        literal = bytearray((len(DOC_IDS) + 7) // 8)
        for position, name in enumerate(DOC_IDS):
            if name in expected:
                literal[position // 8] |= 1 << (position % 8)
        assert context.scope.bitmap == bytes(literal)
        # Check stored/SQL decisions as well as a Python oracle; final padding
        # bits cannot manufacture access to an unclassified document.
        assert set(allowed_rows(fixture, context)) == expected
        assert context.scope.bitmap[-1] & 0b11111110 == 0


def test_repeated_authenticated_idle_touches_reuse_scope_identity_and_registry_row(policy):
    fixture = policy.fixture
    first = policy.contexts["intern"]
    with fixture.owner.connect() as connection:
        initial = connection.execute(sa.select(sa.func.count()).select_from(schema.policy_scopes)).scalar_one()
    for _ in range(3):
        authenticated = fixture.auth.authenticate(raw_token(fixture, "intern"), touch=True)
        again = policy.resolver.resolve(authenticated.identity)
        assert again.binding == first.binding
        assert again.scope.bitmap == first.scope.bitmap
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.policy_scopes)).scalar_one() == initial


@pytest.mark.parametrize("table_name", ["group_memberships", "document_grants", "group_requirements"])
def test_permission_table_truncate_is_rejected_without_changing_epoch_or_rows(policy, table_name):
    fixture = policy.fixture
    before = revision(fixture)
    with fixture.owner.begin() as connection:
        count = connection.execute(sa.text(f'SELECT count(*) FROM rag."{table_name}"')).scalar_one()
        with pytest.raises(sa.exc.DBAPIError) as error:
            with connection.begin_nested():
                connection.exec_driver_sql(f'TRUNCATE TABLE rag."{table_name}"')
        assert error.value.orig.sqlstate == "55000"
        assert connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one() == before
        assert connection.execute(sa.text(f'SELECT count(*) FROM rag."{table_name}"')).scalar_one() == count
    assert revision(fixture) == before
    policy.resolver.assert_current(policy.contexts["intern"])
    assert set(allowed_rows(fixture, policy.contexts["intern"])) == {"general", "atlas"}


def test_grant_dml_and_permission_epoch_are_rolled_back_on_operator_failure(policy):
    fixture = policy.fixture
    before = revision(fixture)
    with fixture.owner.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(schema.document_grants.insert().values(snapshot_id=fixture.snapshot.snapshot_id,
                doc_id="unclassified", user_id=fixture.users["labels_only"].user_id, origin="operator"))
            assert connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one() > before
        finally:
            transaction.rollback()
    assert revision(fixture) == before
    assert allowed_rows(fixture, policy.contexts["labels_only"]) == ()
    policy.resolver.assert_current(policy.contexts["labels_only"])


def test_reviewed_manifest_dry_run_apply_repeat_and_revoke_have_truthful_change_counts(acl_db):
    fixture = acl_db
    repo, manifest = repository(fixture), fixture_manifest(fixture)
    before = revision(fixture)
    plan = repo.dry_run(manifest)
    assert plan.dry_run is True and plan.changed_grants == 5 and plan.changed_memberships == 9
    assert revision(fixture) == before
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_grants)).scalar_one() == 0
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.group_memberships)).scalar_one() == 0
    applied = repo.apply(manifest)
    assert applied.dry_run is False and applied.changed_grants == 5 and applied.changed_memberships == 9
    assert applied.policy_revision > before
    repeated = repo.apply(manifest)
    assert repeated.changed_grants == repeated.changed_memberships == 0
    assert repeated.policy_revision == applied.policy_revision == revision(fixture)
    revoked = repo.revoke(manifest)
    assert revoked.changed_grants == 5 and revoked.changed_memberships == 9
    assert revoked.policy_revision > applied.policy_revision
    repeated_revoke = repo.revoke(manifest)
    assert repeated_revoke.changed_grants == repeated_revoke.changed_memberships == 0
    assert repeated_revoke.policy_revision == revoked.policy_revision


def test_manifest_cannot_relabel_or_revoke_preexisting_real_operator_rows(acl_db):
    fixture = acl_db
    user_id = fixture.users["intern"].user_id
    with fixture.owner.begin() as connection:
        connection.execute(schema.groups.insert().values(group_id="company-general", label="company-general"))
        connection.execute(schema.group_memberships.insert().values(user_id=user_id,
            group_id="company-general", origin="operator"))
        connection.execute(schema.document_grants.insert().values(snapshot_id=fixture.snapshot.snapshot_id,
            doc_id="general", group_id="company-general", origin="operator"))
    repo, manifest = repository(fixture), fixture_manifest(fixture)
    applied = repo.apply(manifest)
    assert applied.changed_grants == 4 and applied.changed_memberships == 8
    repo.revoke(manifest)
    with fixture.owner.connect() as connection:
        grant = connection.execute(sa.select(schema.document_grants)).mappings().one()
        membership = connection.execute(sa.select(schema.group_memberships)).mappings().one()
    for row in (grant, membership):
        assert row["origin"] == "operator" and row["policy_change_id"] is None
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    assert allowed_rows(fixture, resolver.resolve(identity(fixture, "intern"))) == ("general",)


def test_noop_second_manifest_cannot_claim_another_manifests_owned_grant(acl_db):
    fixture = acl_db
    common = dict(groups=(), memberships=(), grants=(GrantSpec(doc_ids=("general",),
        user_id=fixture.users["labels_only"].user_id),))
    first = fixture_manifest(fixture, manifest_id="first-owner", origin="operator", **common)
    second = fixture_manifest(fixture, manifest_id="second-noop", origin="synthetic", **common)
    repo = repository(fixture)
    assert repo.apply(first).changed_grants == 1
    assert repo.apply(second).changed_grants == 0
    assert repo.revoke(second).changed_grants == 0
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    assert allowed_rows(fixture, resolver.resolve(identity(fixture, "labels_only"))) == ("general",)
    assert repo.revoke(first).changed_grants == 1


@pytest.mark.parametrize("changes", [{"reviewed_by": ""}, {"provenance": {}}, {"origin": "inferred"}])
def test_operator_manifest_requires_explicit_review_provenance_and_declared_origin(acl_db, changes):
    with pytest.raises(ValueError):
        fixture_manifest(acl_db, **changes)


def test_applied_manifest_identity_cannot_be_reused_for_changed_permissions(acl_db):
    fixture = acl_db
    repo, manifest = repository(fixture), fixture_manifest(fixture)
    repo.apply(manifest)
    before = revision(fixture)
    changed = manifest.model_copy(update={"grants": (GrantSpec(doc_ids=("unclassified",),
        user_id=fixture.users["labels_only"].user_id),)})
    with pytest.raises(ValueError):
        repo.apply(changed)
    assert revision(fixture) == before


@pytest.mark.parametrize("change", ["logout", "disable", "reset", "expiry"])
def test_existing_python_context_and_sql_rows_fail_closed_after_session_invalidation(policy, change):
    fixture = policy.fixture
    context = policy.contexts["intern"]
    assert set(allowed_rows(fixture, context)) == {"general", "atlas"}
    if change == "logout":
        fixture.auth.logout(raw_token(fixture, "intern"), fixture.sessions["intern"].response.csrf_token)
    elif change == "disable":
        disable_account(fixture.logins["intern"], url=url_string(fixture.urls.migration))
    elif change == "reset":
        reset_password(fixture.logins["intern"], "New synthetic ACL password",
            url=url_string(fixture.urls.migration), hasher=fixture.hasher)
    else:
        now = datetime.now(timezone.utc)
        with fixture.owner.begin() as connection:
            connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == context.identity.session_id)
                .values(created_at=now - timedelta(hours=1), expires_at=now - timedelta(seconds=1)))
    with pytest.raises(SessionExpired):
        policy.resolver.assert_current(context)
    assert allowed_rows(fixture, context) == ()


def test_sql_expiry_is_rechecked_in_later_statement_of_same_open_transaction(policy):
    fixture = policy.fixture
    context = policy.contexts["intern"]
    now = datetime.now(timezone.utc)
    with fixture.owner.begin() as connection:
        connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == context.identity.session_id)
            .values(created_at=now - timedelta(minutes=1), expires_at=now + timedelta(seconds=1)))
    with fixture.database.transaction(context) as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)).scalar_one() == 2
        connection.execute(sa.text("SELECT pg_sleep(1.1)"))
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_revisions)).scalar_one() == 0
    with pytest.raises(SessionExpired):
        policy.resolver.assert_current(context)


def test_authoritative_database_failure_does_not_reuse_cached_permission_decision(policy, monkeypatch):
    fixture = policy.fixture
    context = policy.contexts["intern"]
    captured_identity = identity(fixture, "intern")

    @contextmanager
    def unavailable(_context=None):
        raise sa.exc.OperationalError("private-statement-marker", {}, RuntimeError("private-db-marker"))
        yield

    monkeypatch.setattr(fixture.database, "transaction", unavailable)
    for call in (lambda: policy.resolver.assert_current(context), lambda: policy.resolver.resolve(captured_identity)):
        with pytest.raises(PolicyUnavailable) as error:
            call()
        assert "private" not in str(error.value) and PASSWORD not in str(error.value)


def test_unknown_user_or_session_cannot_resolve_authority(policy):
    fixture = policy.fixture
    original = identity(fixture, "intern")
    for changes in ({"user_id": "unknown-user"}, {"session_id": "unknown-session"}, {"session_version": 77}):
        with pytest.raises(SessionExpired):
            policy.resolver.resolve(original.model_copy(update=changes))


def test_unregistered_or_overbroad_python_scope_cannot_widen_sql_operator_restrictions(policy):
    fixture = policy.fixture
    resolver = PolicyResolver(fixture.database, fixture.snapshot, doc_ids=("general",))
    original = resolver.resolve(identity(fixture, "intern"))
    overbroad = RequestContext(original.identity, LegacySetScope(binding=original.binding,
        allowed_doc_ids=frozenset(DOC_IDS)))
    assert allowed_rows(fixture, overbroad) == ("general",)
    with pytest.raises((AccessDenied, ValueError)):
        resolver.assert_current(overbroad)
    nonexistent = original.binding.model_copy(update={"scope_id": "unknown-server-scope"})
    fake = RequestContext(original.identity, LegacySetScope(binding=nonexistent,
        allowed_doc_ids=frozenset(DOC_IDS)))
    assert allowed_rows(fixture, fake) == ()


@pytest.mark.parametrize("edge", [("finance", "finance"), ("finance_exec", "finance_exec"),
                                  ("finance", "finance_exec")])
def test_database_rejects_self_or_nested_compound_requirements(policy, edge):
    fixture = policy.fixture
    with fixture.owner.begin() as connection:
        with pytest.raises(sa.exc.DBAPIError):
            with connection.begin_nested():
                connection.execute(schema.group_requirements.insert().values(group_id=edge[0],
                    required_group_id=edge[1]))


def test_requirement_change_bumps_epoch_and_rolls_back_with_definition(policy):
    fixture = policy.fixture
    before = revision(fixture)
    with fixture.owner.connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(schema.group_requirements.update().where(
                schema.group_requirements.c.group_id == "finance_exec",
                schema.group_requirements.c.required_group_id == "finance").values(required_group_id="atlas-team"))
            assert connection.execute(sa.select(schema.policy_state.c.revision)).scalar_one() > before
        finally:
            transaction.rollback()
    assert revision(fixture) == before
    policy.resolver.assert_current(policy.contexts["executive"])


def provision_benchmark(fixture, path, **changes):
    fields = dict(snapshot=fixture.snapshot, credentials_file=path, reviewed_by="fixture-operator",
        provenance={"purpose": "tiny synthetic benchmark acceptance fixture"},
        all_documents=True, origin="synthetic", hasher=fixture.hasher)
    fields.update(changes)
    return repository(fixture).provision_benchmark("benchmark_" + uuid4().hex, **fields)


def test_private_benchmark_verification_allows_tiny_snapshot_but_never_public_auth(acl_db, tmp_path, capsys):
    fixture = acl_db
    credentials_file = tmp_path / "benchmark.private.json"
    principal = provision_benchmark(fixture, credentials_file)
    assert credentials_file.stat().st_mode & 0o777 == 0o600
    private = json.loads(credentials_file.read_text())
    assert set(private) == {"format", "login", "password", "user_id", "snapshot_id"}
    assert private["format"] == "rag-benchmark-v1"
    assert private["user_id"] == principal.user_id and private["snapshot_id"] == fixture.snapshot.snapshot_id
    with fixture.owner.connect() as connection:
        row = connection.execute(sa.select(schema.users).where(schema.users.c.user_id == principal.user_id)).mappings().one()
        audit = connection.execute(sa.select(schema.policy_changes).where(
            schema.policy_changes.c.action == "benchmark")).mappings().one()
    canonical_audit = json.dumps(audit["manifest"], sort_keys=True, ensure_ascii=False,
                                separators=(",", ":"), allow_nan=False).encode("utf-8")
    assert audit["manifest"]["all_documents"] is True
    assert audit["manifest_sha256"] == hashlib.sha256(canonical_audit).hexdigest()
    assert row["account_kind"] == "benchmark" and row["password_hash"] != private["password"]
    benchmark_identity = verify_benchmark(credentials_file, fixture.database, hasher=fixture.hasher)
    resolver = PolicyResolver(fixture.database, fixture.snapshot)
    context = resolver.resolve(benchmark_identity)
    assert all(context.scope.mask_for(fixture.snapshot, DOC_IDS))
    assert set(allowed_rows(fixture, context)) == set(DOC_IDS)
    with pytest.raises(AuthError) as password_denied:
        fixture.auth.login(private["login"], private["password"], client_ip="127.0.0.1")
    assert password_denied.value.code == "invalid_credentials" and password_denied.value.status == 401
    with pytest.raises(AuthError):
        fixture.auth.authenticate(private["password"])

    # Rebind an actually valid issued human cookie to this benchmark account in
    # the disposable fixture. Its token and CSRF digest are both valid; the
    # public account-kind guard must still reject it.
    known_cookie = raw_token(fixture, "intern")
    human_session_id = fixture.auth.authenticate(known_cookie, touch=False).identity.session_id
    with fixture.owner.begin() as connection:
        connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == human_session_id)
            .values(user_id=principal.user_id, session_version=row["session_version"]))
    with pytest.raises(AuthError) as cookie_denied:
        fixture.auth.authenticate(known_cookie)
    assert cookie_denied.value.code == "session_expired" and cookie_denied.value.status == 401

    origin = "http://localhost:8000"
    app = create_app(WebSettings(mode="fixture", demo_access=True),
        auth_settings=fixture.auth.settings, auth_service=fixture.auth)
    with TestClient(app, base_url=origin) as client:
        csrf = client.get("/api/auth/csrf").json()["csrf_token"]
        response = client.post("/api/auth/login", json={"login": private["login"], "password": private["password"]},
            headers={"Origin": origin, "X-CSRF-Token": csrf})
        assert response.status_code == 401 and response.json()["detail"]["code"] == "invalid_credentials"
        for value in (private["password"], known_cookie):
            me = client.get("/api/auth/me", headers={"Origin": origin, "Cookie": "rag_session=" + value})
            assert me.status_code == 401
            assert value not in response.text + me.text
    captured = capsys.readouterr()
    assert private["password"] not in captured.out + captured.err + repr(principal)


def test_existing_private_benchmark_file_is_not_overwritten_and_creates_no_database_authority(acl_db, tmp_path):
    fixture = acl_db
    path = tmp_path / "already.private.json"
    path.write_bytes(b"existing private artifact must stay intact")
    path.chmod(0o600)
    before = revision(fixture)
    with fixture.owner.connect() as connection:
        users_before = connection.execute(sa.select(sa.func.count()).select_from(schema.users)).scalar_one()
    with pytest.raises(ValueError):
        provision_benchmark(fixture, path)
    assert path.read_bytes() == b"existing private artifact must stay intact" and path.stat().st_mode & 0o777 == 0o600
    assert revision(fixture) == before
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.users)).scalar_one() == users_before
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.document_grants)).scalar_one() == 0


@pytest.mark.parametrize("changes", [{"all_documents": False}, {"reviewed_by": ""}, {"provenance": {}}])
def test_benchmark_creation_requires_explicit_scope_review_and_provenance(acl_db, tmp_path, changes):
    fixture = acl_db
    path = tmp_path / "not-created.private.json"
    before = revision(fixture)
    with pytest.raises(ValueError):
        provision_benchmark(fixture, path, **changes)
    assert not path.exists() and revision(fixture) == before


def test_private_benchmark_verifier_rejects_insecure_permissions(acl_db, tmp_path):
    fixture = acl_db
    path = tmp_path / "benchmark.private.json"
    provision_benchmark(fixture, path)
    path.chmod(0o644)
    with pytest.raises((AccessDenied, ValueError)):
        verify_benchmark(path, fixture.database, hasher=fixture.hasher)


def test_account_kind_is_immutable_and_metadata_cannot_turn_human_into_benchmark(acl_db):
    fixture = acl_db
    with fixture.owner.begin() as connection:
        with pytest.raises(sa.exc.DBAPIError):
            with connection.begin_nested():
                connection.execute(schema.users.update().where(schema.users.c.user_id == fixture.users["intern"].user_id)
                    .values(account_kind="benchmark"))
    assert identity(fixture, "intern").user_id == fixture.users["intern"].user_id


def test_benchmark_cli_is_explicit_private_silent_and_refuses_overwrite(acl_db, tmp_path, monkeypatch, capsys):
    fixture = acl_db
    snapshot_file, credential_file = tmp_path / "snapshot.json", tmp_path / "cli.private.json"
    snapshot_file.write_text(fixture.snapshot.model_dump_json())
    monkeypatch.setenv("RAG_MIGRATION_DATABASE_URL", url_string(fixture.urls.migration))
    monkeypatch.setenv("RAG_DATABASE_URL", url_string(fixture.urls.runtime))
    # This test checks CLI privacy and PostgreSQL behavior, not Argon2 resources.
    monkeypatch.setattr(policy_module, "PasswordHasher", lambda: fixture.hasher)
    arguments = ["benchmark-create", "cli_benchmark_" + uuid4().hex,
        "--snapshot", str(snapshot_file), "--credentials-file", str(credential_file),
        "--reviewed-by", "fixture-operator", "--provenance", "Explicit synthetic fixture review",
        "--all-documents", "--synthetic"]
    policy_module.main(arguments)
    private = json.loads(credential_file.read_text())
    assert credential_file.stat().st_mode & 0o777 == 0o600
    policy_module.main(["benchmark-verify", "--credentials-file", str(credential_file)])
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    original = credential_file.read_bytes()
    before = revision(fixture)
    with pytest.raises(SystemExit) as error:
        policy_module.main(arguments)
    assert error.value.code == 2
    assert credential_file.read_bytes() == original and revision(fixture) == before
    failure = capsys.readouterr()
    assert private["password"] not in failure.out + failure.err
    assert private["user_id"] not in failure.out + failure.err


def test_failed_benchmark_database_authorization_removes_only_new_owned_file(acl_db, tmp_path):
    fixture = acl_db
    path = tmp_path / "failed.private.json"
    before = revision(fixture)
    unprivileged = PolicyRepository(url=url_string(fixture.urls.runtime))
    with pytest.raises(ValueError):
        unprivileged.provision_benchmark("forbidden_" + uuid4().hex, snapshot=fixture.snapshot,
            credentials_file=path, reviewed_by="fixture-operator", provenance={"purpose": "synthetic negative fixture"},
            all_documents=True, origin="synthetic", hasher=fixture.hasher)
    assert not path.exists() and revision(fixture) == before
    with fixture.owner.connect() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.users)
            .where(schema.users.c.account_kind == "benchmark")).scalar_one() == 0


def test_private_verifier_reads_the_same_open_file_despite_path_replacement(acl_db, tmp_path, monkeypatch):
    fixture = acl_db
    path = tmp_path / "benchmark.private.json"
    principal = provision_benchmark(fixture, path)
    alternate = tmp_path / "replacement.private.json"
    record = json.loads(path.read_text())
    record["password"] = "incorrect private fixture password"
    alternate.write_text(json.dumps(record))
    alternate.chmod(0o600)
    original_open = policy_module.os.open
    opened = []

    def replacing_open(filename, flags, *args, **kwargs):
        descriptor = original_open(filename, flags, *args, **kwargs)
        if Path(os.fsdecode(filename)) == path:
            opened.append(descriptor)
            os.replace(alternate, path)
        return descriptor

    monkeypatch.setattr(policy_module.os, "open", replacing_open)
    verified = verify_benchmark(path, fixture.database, hasher=fixture.hasher)
    assert len(opened) == 1 and verified.user_id == principal.user_id
    assert json.loads(path.read_text())["password"] == record["password"]


def test_private_verifier_bounds_growth_after_fstat_and_closes_descriptor_on_failure(acl_db, tmp_path, monkeypatch):
    fixture = acl_db
    path = tmp_path / "benchmark.private.json"
    provision_benchmark(fixture, path)
    initial = path.stat()
    original_fstat, original_open = policy_module.os.fstat, policy_module.os.open
    affected = []

    def grow_after_fstat(descriptor):
        info = original_fstat(descriptor)
        if not affected and (info.st_dev, info.st_ino) == (initial.st_dev, initial.st_ino):
            affected.append(descriptor)
            writer = original_open(path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(writer, b" " * 4097)
            finally:
                os.close(writer)
        return info

    monkeypatch.setattr(policy_module.os, "fstat", grow_after_fstat)
    with pytest.raises(ValueError) as retained_error:
        verify_benchmark(path, fixture.database, hasher=fixture.hasher)
    assert retained_error.value and len(affected) == 1
    with pytest.raises(OSError):
        original_fstat(affected[0])


def test_private_verifier_rejects_symlink_without_following_or_replacing_it(acl_db, tmp_path):
    fixture = acl_db
    original, link = tmp_path / "benchmark.private.json", tmp_path / "linked.private.json"
    provision_benchmark(fixture, original)
    before = original.read_bytes()
    link.symlink_to(original)
    with pytest.raises(ValueError):
        verify_benchmark(link, fixture.database, hasher=fixture.hasher)
    assert link.is_symlink() and original.read_bytes() == before


def test_forged_catalog_mapping_cannot_reinterpret_registered_bitmap(policy):
    fixture = policy.fixture
    context = policy.contexts["intern"]
    positions = dict(context.scope._catalog.positions)
    positions["general"], positions["finance-exec"] = positions["finance-exec"], positions["general"]
    forged_catalog = replace(context.scope._catalog, positions=MappingProxyType(positions))
    forged_scope = replace(context.scope, _catalog=forged_catalog)
    assert forged_scope.binding == context.binding and forged_scope.bitmap_sha256 == context.scope.bitmap_sha256
    assert forged_scope.permits("finance-exec") is True
    forged_context = RequestContext(context.identity, forged_scope)
    with pytest.raises(EvidenceValidationError):
        policy.resolver.assert_current(forged_context)
    assert set(allowed_rows(fixture, forged_context)) == {"general", "atlas"}


def test_manifest_rejects_duplicate_subject_document_grants_across_entries(acl_db):
    subject = acl_db.users["labels_only"].user_id
    with pytest.raises(ValueError):
        fixture_manifest(acl_db, groups=(), memberships=(), grants=(
            GrantSpec(doc_ids=("general", "atlas"), user_id=subject),
            GrantSpec(doc_ids=("atlas",), user_id=subject)))

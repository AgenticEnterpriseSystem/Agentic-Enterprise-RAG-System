"""SQL narrowing preserves the complete authorized Jira relation and live RLS.

Root executes these disposable PostgreSQL fixtures privately. Query assertions
inspect bound synthetic IDs and SQL shape, never wall-clock thresholds or real
credentials. The old RLS-only adapter is an independent scope-equivalence path.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from src.access.contracts import EvidenceValidationError, RequestContext, SnapshotMismatch, StaleContext
from src.access.policy import PolicyResolver
from src.contracts import ToolRequest
from src.db import schema
from src.retrieval.tools import JiraTool
from src.storage.fingerprints import ValidatedCatalog
from src.storage.postgres_store import PostgresDocumentStore
from tests.db_support import pg_urls
from tests.jira_support import env, ready, source


@pytest.fixture
def scoped(ready):
    proof = PostgresDocumentStore(ready.database, ready.source.snapshot, retain_catalog=True).validated_catalog()
    return SimpleNamespace(data=ready, proof=proof,
        tool=JiraTool(ready.database, ready.source.snapshot, catalog=proof),
        legacy=JiraTool(ready.database, ready.source.snapshot))


def execute(bundle, operation='count', filters=None, *, principal='intern', tool=None,
            context=None, guard=None, limit=10):
    data = bundle.data
    return (tool or bundle.tool).execute(ToolRequest(source_type='jira', operation=operation,
        filters=filters or {}, limit=limit), context=context or data.contexts[principal],
        guard=guard or data.resolver)


@contextmanager
def ticket_queries(data):
    captured = []

    def record(_connection, _cursor, statement, parameters, _context, _many):
        normalized = ' '.join(statement.lower().split())
        if 'from rag.tickets ' in normalized or normalized.endswith('from rag.tickets'):
            ids = parameters.get('authorized_jira_doc_ids') if isinstance(parameters, dict) else None
            captured.append((normalized, tuple(ids) if ids is not None else None))

    sa.event.listen(data.database.engine, 'before_cursor_execute', record)
    try:
        yield captured
    finally:
        sa.event.remove(data.database.engine, 'before_cursor_execute', record)


class MaskScope:
    """Only mask output changes; the real registered RLS binding is retained."""
    def __init__(self, original, mask, callback=None):
        self.original, self.binding, self.mask = original, original.binding, mask
        self.callback, self.seen_ids = callback, []

    def permits(self, document):
        return self.original.permits(document)

    def mask_for(self, snapshot, document_ids):
        assert snapshot == self.binding.snapshot
        self.seen_ids.append(tuple(document_ids))
        if self.callback:
            self.callback()
        return self.mask


def masked_context(data, mask, callback=None):
    original = data.contexts['intern']
    scope = MaskScope(original.scope, mask, callback)
    context = RequestContext(original.identity, scope)

    class Guard:
        def assert_current(self, supplied):
            assert supplied.identity == original.identity and supplied.binding == original.binding
            data.resolver.assert_current(original)

    # The test-only guard checks the original authentic session/registry. SQL
    # still receives that registered scope, so this cannot expand database RLS.
    return context, Guard(), scope


@pytest.mark.parametrize('violation', ['issuer', 'snapshot', 'unissued', 'lookalike', 'copied-snapshot'])
def test_catalog_fast_path_requires_exact_issued_snapshot_and_database(scoped, violation):
    data, proof, snapshot = scoped.data, scoped.proof, scoped.data.source.snapshot
    if violation == 'issuer':
        proof = ValidatedCatalog.build(snapshot, data.source.entries, issuer=object())
    elif violation == 'snapshot':
        snapshot = snapshot.model_copy(update={'snapshot_id': 'another-ready-snapshot'})
    elif violation == 'unissued':
        forged = object.__new__(ValidatedCatalog)
        for slot in ValidatedCatalog.__slots__:
            if slot != '__weakref__':
                object.__setattr__(forged, slot, getattr(proof, slot))
        proof = forged
    elif violation == 'lookalike':
        proof = SimpleNamespace(entries=proof.entries, doc_ids=proof.doc_ids,
            require=lambda *args, **kwargs: scoped.proof)
    else:
        snapshot = snapshot.model_copy(update={'document_count': True})
    with ticket_queries(data) as queries:
        with pytest.raises((EvidenceValidationError, SnapshotMismatch, ValueError)):
            JiraTool(data.database, snapshot, catalog=proof)
    assert queries == []


@pytest.mark.parametrize('mask', [(), (True,), (True,) * 5, (1, 1, 0, 0),
    (True, True, False, None), True, None])
def test_malformed_scope_masks_fail_before_native_sql(scoped, mask):
    context, guard, _scope = masked_context(scoped.data, mask)
    with ticket_queries(scoped.data) as queries:
        with pytest.raises(EvidenceValidationError):
            execute(scoped, context=context, guard=guard)
    assert queries == []


def test_mask_is_derived_over_jira_ids_only_and_rls_remains_independent(scoped):
    context, guard, scope = masked_context(scoped.data, (True, True, True, True))
    # Even an overbroad trusted-adapter mask cannot confer the denied IDs: the
    # genuine registered SQL policy still only admits the intern's two records.
    assert execute(scoped, context=context, guard=guard).count == 2
    assert scope.seen_ids and all(ids == ('one', 'two', 'hidden', 'unknown') for ids in scope.seen_ids)
    with scoped.data.database.transaction() as connection:
        assert connection.execute(sa.select(sa.func.count()).select_from(schema.tickets)).scalar_one() == 0


@pytest.mark.parametrize('principal,expected', [('intern', 2), ('executive', 4)])
def test_narrow_and_all_jira_counts_and_lineage_equal_legacy_rls_path(scoped, principal, expected):
    optimized = execute(scoped, principal=principal, limit=1)
    baseline = execute(scoped, principal=principal, tool=scoped.legacy, limit=1)
    assert optimized.count == baseline.count == expected
    assert optimized.evidence[0].lineage_sha256 == baseline.evidence[0].lineage_sha256
    assert optimized.evidence[0].filters == baseline.evidence[0].filters
    assert optimized.evidence[0].scope == baseline.evidence[0].scope
    assert optimized.evidence[0].artifact_id != baseline.evidence[0].artifact_id


def test_visible_unknown_field_is_checked_before_filters_but_hidden_unknown_does_not_poison(scoped):
    for tool in (scoped.tool, scoped.legacy):
        hidden_unknown = execute(scoped, filters={'status': 'In Progress', 'doc_id': 'one'}, tool=tool)
        assert hidden_unknown.count == 1
        visible_unknown = execute(scoped, filters={'status': 'In Progress', 'doc_id': 'one'},
            principal='executive', tool=tool)
        assert visible_unknown.coverage == 'unknown' and visible_unknown.count is None
        assert visible_unknown.evidence == ()
    resolver = PolicyResolver(scoped.data.database, scoped.data.source.snapshot,
        doc_ids=('one', 'two'), catalog=scoped.proof)
    restricted = resolver.resolve(scoped.data.identities['executive'])
    assert execute(scoped, filters={'status': 'In Progress', 'doc_id': 'one'},
        context=restricted, guard=resolver).count == 1


def test_each_native_select_has_the_same_indexable_bound_scope_array(scoped):
    data = scoped.data
    with ticket_queries(data) as queries:
        assert execute(scoped, filters={'status': 'In Progress'}).count == 1
        records = execute(scoped, 'lookup', {'doc_id': 'one'}).records
        scoped.tool.validate_records(data.contexts['intern'], records, guard=data.resolver)
    assert len(queries) >= 5  # coverage EXISTS, count, lineage, lookup, validation
    for sql, ids in queries:
        assert 'tickets.doc_id = any' in sql and 'authorized_jira_doc_ids' in sql
        assert ids == ('one', 'two')
    assert any('exists' in sql and 'status is null' in sql for sql, _ in queries)
    assert any('count(' in sql for sql, _ in queries)
    assert any('order by rag.tickets.ticket_id' in sql for sql, _ in queries)


def test_explicit_empty_scope_uses_false_not_an_unrestricted_relation(scoped):
    resolver = PolicyResolver(scoped.data.database, scoped.data.source.snapshot, doc_ids=(), catalog=scoped.proof)
    context = resolver.resolve(scoped.data.identities['intern'])
    with ticket_queries(scoped.data) as queries:
        result = execute(scoped, filters={'status': 'In Progress'}, context=context, guard=resolver)
        assert result.count == 0 and result.evidence[0].count == 0
        assert execute(scoped, 'filter', context=context, guard=resolver).records == ()
    assert queries and all('where false' in sql and ids is None for sql, ids in queries)


def test_operator_narrowing_preserves_duplicate_key_ambiguity_and_no_extra_records(scoped):
    full = execute(scoped, 'lookup', {'native_key': 'SUP-100'}, limit=1)
    assert full.ambiguous and full.truncated and len(full.records) == 1
    resolver = PolicyResolver(scoped.data.database, scoped.data.source.snapshot, doc_ids=('one',), catalog=scoped.proof)
    context = resolver.resolve(scoped.data.identities['intern'])
    narrowed = execute(scoped, 'lookup', {'native_key': 'SUP-100'}, context=context, guard=resolver, limit=1)
    assert [record.doc_id for record in narrowed.records] == ['one']
    assert not narrowed.ambiguous and not narrowed.truncated
    assert execute(scoped, context=context, guard=resolver).count == 1


def test_no_proof_legacy_path_does_not_require_a_mask_or_change_rls(scoped):
    context, guard, scope = masked_context(scoped.data, None)
    with ticket_queries(scoped.data) as queries:
        result = execute(scoped, context=context, guard=guard, tool=scoped.legacy)
    assert result.count == 2 and scope.seen_ids == []
    assert queries and all('authorized_jira_doc_ids' not in sql for sql, _ in queries)


def test_policy_revocation_while_computing_mask_stops_before_any_native_query(scoped):
    data = scoped.data
    context, guard, _scope = masked_context(data, (True, True, False, False),
        callback=lambda: data.repository.revoke(data.manifest))
    with ticket_queries(data) as queries:
        with pytest.raises(StaleContext):
            execute(scoped, context=context, guard=guard)
    assert queries == []

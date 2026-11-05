"""Ownership, ledger and collision guards without database or service access."""
from copy import deepcopy
import hashlib
import os

import pytest

from src.contracts import SnapshotRef
from tests.deployment import security_operator as fixture


def rows():
    result = []
    for label in ('alpha', 'bravo', 'charlie'):
        title = 'Distinct native incident for SEC fixture ' + label
        text = title + '\n' + ('Real canonical source content for ' + label + ' covers a distinctive customer incident. ') * 4
        result.append({'doc_id': 'document-' + label, 'title': title,
            'revision_id': hashlib.sha256(text.encode()).hexdigest(), 'source_type': 'jira',
            'canonical_text': text, 'canonical_utf8': None, 'native_key': 'SUP-' + label,
            'status': 'Open', 'priority': 'P1', 'project_id': 'customer-support'})
    return result


@pytest.fixture
def prepared():
    snapshot = SnapshotRef(snapshot_id='enterprise-bge-base-canonical', corpus_sha256='1' * 64,
        catalog_sha256='2' * 64, document_count=fixture.COUNT)
    state = fixture.make_state(fixture.PROJECT, snapshot, 'explicit local review', fixture.choose_documents(rows()))
    return snapshot, state


@pytest.mark.parametrize('alter', ['foreign_user', 'foreign_login', 'wrong_role', 'wider_intern', 'changed_manifest'])
def test_tampered_ownership_or_policy_is_rejected_even_with_recomputed_manifest_digest(prepared, alter):
    snapshot, original = prepared
    state = deepcopy(original)
    if alter == 'foreign_user':
        state['accounts']['intern']['user_id'] = 'existing-unrelated-user'
    elif alter == 'foreign_login':
        state['accounts']['intern']['login'] = 'existing-user@example.invalid'
    elif alter == 'wrong_role':
        state['accounts']['intern']['role'] = 'intern'
    elif alter == 'wider_intern':
        state['accounts']['intern']['allowed_doc_ids'].append(state['documents']['B']['doc_id'])
    else:
        state['manifests']['intern']['grants'][0]['doc_ids'].append(state['documents']['C']['doc_id'])
        state['manifest_sha256']['intern'] = fixture.digest(state['manifests']['intern'])
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.validate_state(state, fixture.PROJECT, snapshot)


def test_fixture_uses_two_independent_manifests_with_reversed_descriptive_roles(prepared):
    snapshot, state = prepared
    manifests = fixture.validate_state(state, fixture.PROJECT, snapshot)
    assert len(manifests['intern'].grants[0].doc_ids) == 1
    assert len(manifests['executive'].grants[0].doc_ids) == 2
    assert manifests['intern'].manifest_id != manifests['executive'].manifest_id
    assert state['accounts']['intern']['role'] == 'executive'
    assert state['accounts']['executive']['role'] == 'intern'
    assert state['documents']['C']['doc_id'] in state['accounts']['executive']['forbidden_markers']
    assert state['documents']['C']['text_marker'] in state['accounts']['executive']['forbidden_markers']


def test_body_markers_are_verbatim_slices_after_title_and_absent_from_other_documents(prepared):
    _, state = prepared
    source = {row['doc_id']: row['canonical_text'] for row in rows()}
    for document in state['documents'].values():
        marker, start = document['text_marker'], document['text_marker_start']
        assert 80 <= len(marker) <= 120 and start >= len(document['title'])
        assert source[document['doc_id']][start:start + len(marker)] == marker
        assert all(marker.casefold() not in text.casefold() for doc_id, text in source.items()
                   if doc_id != document['doc_id'])


def test_selected_body_marker_cannot_be_invented_while_reusing_a_valid_revision(prepared):
    _, state = prepared
    state['documents']['B']['text_marker'] = 'Invented body marker absent from canonical data. ' * 2
    operator = object.__new__(fixture.Operator)
    operator.documents = lambda ids: rows()
    with pytest.raises(fixture.SecurityOperatorError, match='body marker'):
        operator.verify_documents(state)


def test_selection_refuses_a_prior_document_body_marker_found_in_another_record():
    candidates = rows()
    selected = fixture.choose_documents(candidates)
    candidates[1]['canonical_text'] += '\n' + selected['A']['text_marker']
    candidates[1]['revision_id'] = hashlib.sha256(candidates[1]['canonical_text'].encode()).hexdigest()
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.choose_documents(candidates)


def test_selection_refuses_markers_present_in_another_selected_canonical_body():
    candidates = rows()
    candidates[1]['canonical_text'] += '\nReference: ' + candidates[0]['title']
    candidates[1]['revision_id'] = hashlib.sha256(candidates[1]['canonical_text'].encode()).hexdigest()
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.choose_documents(candidates)


def test_selection_independently_checks_canonical_revision_bytes():
    candidates = rows()
    candidates[0]['canonical_text'] += ' changed after import'
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.choose_documents(candidates)


def test_private_ledger_refuses_symlinks_and_public_permissions(tmp_path, prepared):
    _, state = prepared
    path = tmp_path / 'state.json'
    fixture.write_state(path, state, first=True)
    assert fixture.read_json(path) == state
    link = tmp_path / 'alias.json'; link.symlink_to(path)
    with pytest.raises(OSError):
        fixture.read_json(link)
    path.chmod(0o644)
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.write_state(path, state)


def test_exclusive_initial_ledger_and_kernel_lock_support_safe_retry(tmp_path, prepared):
    _, state = prepared
    path = tmp_path / 'state.json'
    with fixture.exclusive(path):
        fixture.write_state(path, state, first=True)
        with pytest.raises(BlockingIOError):
            with fixture.exclusive(path):
                pytest.fail('Concurrent fixture mutation acquired the lock')
    with fixture.exclusive(path):
        with pytest.raises(FileExistsError):
            fixture.write_state(path, state, first=True)
        state['status'] = 'seeding'; fixture.write_state(path, state)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert fixture.read_json(path)['status'] == 'seeding'


def test_duplicate_private_fields_are_not_silently_reinterpreted(tmp_path):
    path = tmp_path / 'state.json'
    path.write_text('{"project":"allowed","project":"other"}')
    path.chmod(0o600)
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.read_json(path)


def test_project_attestation_fails_before_any_database_connection(monkeypatch, prepared):
    snapshot, _ = prepared
    monkeypatch.delenv('RAG_SEC_COMPOSE_PROJECT', raising=False)
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.Operator(fixture.PROJECT, snapshot)
    monkeypatch.setenv('RAG_SEC_COMPOSE_PROJECT', fixture.PROJECT)
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.Operator('unrelated-compose-project', snapshot)


def test_database_account_collision_is_rejected_before_disable(prepared):
    _, state = prepared
    account = state['accounts']['intern']
    row = {**account, 'account_kind': 'human', 'project_ids': []}
    row['user_id'] = 'unrelated-user'
    with pytest.raises(fixture.SecurityOperatorError):
        fixture.Operator.check_account(row, account)

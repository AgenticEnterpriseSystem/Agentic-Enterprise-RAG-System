"""Real PostgreSQL checks for the one-command full-benchmark grant script.

Uses the disposable 17-document catalog from the policy tests and a small batch
size so preparation spans several patch files. Skipped without RAG_TEST_* URLs.
"""
import json
import stat

import pytest

sa = pytest.importorskip("sqlalchemy", reason="Grant checks require optional DB dependencies")
pytest.importorskip("psycopg", reason="Grant checks require the psycopg driver")

from src.access import benchmark_grants
from src.access.benchmark_grants import GrantError, execute, prepare
from src.access.repository import PolicyRepository
from src.db import schema
from tests.db_support import pg_urls, url_string  # noqa: F401  (pytest fixture)
from tests.test_access_policy import DOC_IDS, acl_db  # noqa: F401  (pytest fixture)


def grant_count(fixture, user_id):
    with fixture.owner.connect() as connection:
        return connection.execute(sa.select(sa.func.count()).select_from(schema.document_grants).where(
            schema.document_grants.c.user_id == user_id,
            schema.document_grants.c.snapshot_id == fixture.snapshot.snapshot_id)).scalar_one()


def prepared(fixture, tmp_path, login_key='intern', **overrides):
    folder = tmp_path / 'review'
    fields = dict(login=fixture.logins[login_key], reviewer='Fixture reviewer', approval='qa-test-001', batch_size=5)
    fields.update(overrides)
    result = prepare(sa.create_engine(url_string(fixture.urls.migration)), fixture.snapshot, folder, **fields)
    return folder, result


def test_prepare_preview_apply_and_revoke_cover_exactly_the_snapshot(acl_db, tmp_path):
    folder, result = prepared(acl_db, tmp_path)
    assert result['documents'] == len(DOC_IDS) and result['patches'] == 4  # 5+5+5+2
    assert stat.S_IMODE(folder.stat().st_mode) == 0o700
    user_id = acl_db.users['intern'].user_id
    repository = PolicyRepository(url=url_string(acl_db.urls.migration))

    assert grant_count(acl_db, user_id) == 0
    assert execute('dry-run', repository, acl_db.snapshot, folder)['changed_grants'] == len(DOC_IDS)
    assert grant_count(acl_db, user_id) == 0, 'dry-run must not change grants'

    assert execute('apply', repository, acl_db.snapshot, folder)['changed_grants'] == len(DOC_IDS)
    assert grant_count(acl_db, user_id) == len(DOC_IDS)
    # Rerunning an unchanged review (for example after an interrupted apply) is a no-op.
    assert execute('apply', repository, acl_db.snapshot, folder)['changed_grants'] == 0

    execute('revoke', repository, acl_db.snapshot, folder)
    assert grant_count(acl_db, user_id) == 0
    assert grant_count(acl_db, acl_db.users['executive'].user_id) == 0, 'other accounts untouched'


@pytest.mark.parametrize('overrides,message', [
    ({'login': 'nobody-here'}, 'No active human login'),
    ({'reviewer': ' '}, 'required'),
    ({'approval': ''}, 'required'),
])
def test_prepare_refuses_unknown_login_or_missing_review_fields(acl_db, tmp_path, overrides, message):
    with pytest.raises(GrantError, match=message):
        prepared(acl_db, tmp_path, **overrides)
    assert not (tmp_path / 'review' / 'review.json').exists()


def test_prepare_never_overwrites_an_existing_review(acl_db, tmp_path):
    folder, _ = prepared(acl_db, tmp_path)
    before = (folder / 'review.json').read_bytes()
    with pytest.raises(GrantError, match='already exists'):
        prepared(acl_db, tmp_path)
    assert (folder / 'review.json').read_bytes() == before


@pytest.mark.parametrize('tamper,message', [
    (lambda folder: (folder / 'grant-0001.json').write_text('{}'), 'changed after review'),
    (lambda folder: (folder / 'review.json').write_text(json.dumps({
        **json.loads((folder / 'review.json').read_text()), 'snapshot': {'snapshot_id': 'other'}})), 'snapshot changed'),
])
def test_execute_refuses_tampered_or_stale_reviews_before_any_change(acl_db, tmp_path, tamper, message):
    folder, _ = prepared(acl_db, tmp_path)
    tamper(folder)
    repository = PolicyRepository(url=url_string(acl_db.urls.migration))
    with pytest.raises(GrantError, match=message):
        execute('apply', repository, acl_db.snapshot, folder)
    assert grant_count(acl_db, acl_db.users['intern'].user_id) == 0


def test_execute_without_prepare_explains_the_next_step(acl_db, tmp_path):
    with pytest.raises(GrantError, match='run prepare first'):
        execute('apply', PolicyRepository(url=url_string(acl_db.urls.migration)), acl_db.snapshot, tmp_path)


def test_cli_reports_refusals_without_a_traceback(tmp_path, capsys):
    missing = tmp_path / 'snapshot.json'
    with pytest.raises(SystemExit) as error:
        benchmark_grants.main(['apply', '--login', 'hira', '--snapshot', str(missing), '--directory', str(tmp_path)])
    assert 'failed' in str(error.value) and 'Traceback' not in str(error.value)

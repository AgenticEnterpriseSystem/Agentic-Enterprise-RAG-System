"""Cheap, nonowner deployment checks shared by startup and live readiness.

This module does not load models, source bodies or indexes. Actual application
loaders retain their complete artifact validation. Receipt publication is a
separate explicit operator command using the same runtime database privileges.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import StrictBool

from src.access.contracts import PolicyUnavailable
from src.contracts import Contract, SnapshotRef


class DeploymentReceipt(Contract):
    format: Literal['rag-local-deployment-v1'] = 'rag-local-deployment-v1'
    schema_head: Literal['0004_jira_feed'] = '0004_jira_feed'
    snapshot: SnapshotRef
    jira_required: StrictBool


def _json_file(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError()
            value[key] = item
        return value

    def invalid(_value):
        raise ValueError()

    try:
        with Path(path).open('rb') as stream:
            payload = stream.read(65537)
        if len(payload) > 65536:
            raise ValueError()
        return json.loads(payload, object_pairs_hook=unique, parse_constant=invalid)
    except (OSError, ValueError, TypeError, UnicodeError):
        raise PolicyUnavailable('Deployment configuration is unavailable') from None


def _snapshot(value):
    try:
        if isinstance(value, SnapshotRef):
            value = value.model_dump(warnings=False)
        return SnapshotRef.model_validate(value)
    except (ValueError, TypeError):
        raise PolicyUnavailable('Deployment snapshot is unavailable') from None


def load_receipt(path, snapshot):
    try:
        receipt = DeploymentReceipt.model_validate(_json_file(path))
    except (ValueError, TypeError):
        raise PolicyUnavailable('Deployment receipt is unavailable') from None
    if receipt.snapshot != _snapshot(snapshot):
        raise PolicyUnavailable('Deployment receipt differs from the configured snapshot')
    return receipt


def _check_storage(database, snapshot, *, jira_documents=None, receipt=None):
    import sqlalchemy as sa
    from src.db import schema
    from src.db.migrate import EXPECTED_HEAD
    from src.db.runtime import Database, DatabaseRoleError

    if not isinstance(database, Database):
        raise PolicyUnavailable('The shared runtime database is required')
    snapshot = _snapshot(snapshot)
    if jira_documents is not None and (type(jira_documents) is not int or jira_documents < 0
                                       or jira_documents > snapshot.document_count):
        raise PolicyUnavailable('Deployment catalogue requirement is invalid')
    if receipt is not None:
        if not isinstance(receipt, DeploymentReceipt):
            receipt = load_receipt(receipt, snapshot)
        else:
            try:
                receipt = DeploymentReceipt.model_validate(receipt.model_dump(warnings=False))
            except ValueError:
                raise PolicyUnavailable('Deployment receipt is unavailable') from None
        if receipt.snapshot != snapshot:
            raise PolicyUnavailable('Deployment receipt differs from the configured snapshot')
        if receipt.schema_head != EXPECTED_HEAD:
            raise PolicyUnavailable('Deployment receipt requires an incompatible schema')
    try:
        # No identity is set: only catalog/deployment metadata is readable. One
        # held pool slot also works with a one-connection test/runtime pool.
        with database.transaction() as connection:
            revisions = connection.execute(sa.text('SELECT version_num FROM rag.alembic_version')).scalars().all()
            if revisions != [EXPECTED_HEAD]:
                raise PolicyUnavailable('The database schema is incompatible')
            table = schema.dataset_snapshots
            row = connection.execute(sa.select(table.c.state, table.c.corpus_sha256,
                table.c.catalog_sha256, table.c.document_count).where(
                    table.c.snapshot_id == snapshot.snapshot_id)).mappings().one_or_none()
            if (row is None or row['state'] != 'ready' or any(row[name] != getattr(snapshot, name)
                    for name in ('corpus_sha256', 'catalog_sha256', 'document_count'))):
                raise PolicyUnavailable('The configured ready snapshot is unavailable')
            if jira_documents is None:
                # Metadata only, once for prestart publication/verification.
                # A running runtime supplies its already validated catalogue
                # count, avoiding a repeated live full-catalogue query.
                jira_documents = connection.execute(sa.select(sa.func.count()).select_from(schema.documents).where(
                    schema.documents.c.snapshot_id == snapshot.snapshot_id,
                    schema.documents.c.source_type == 'jira')).scalar_one()
            if type(jira_documents) is not int or not 0 <= jira_documents <= snapshot.document_count:
                raise PolicyUnavailable('Deployment catalogue requirement is invalid')
            required = jira_documents > 0
            if receipt is not None and receipt.jira_required != required:
                raise PolicyUnavailable('Deployment source requirements differ from the catalogue')
            if required:
                feed = schema.ticket_feeds
                ready = connection.execute(sa.select(feed.c.state, feed.c.corpus_sha256,
                    feed.c.expected_records, feed.c.imported_records, feed.c.coverage).where(
                        feed.c.snapshot_id == snapshot.snapshot_id, feed.c.source_type == 'jira')).mappings().one_or_none()
                if (ready is None or ready['state'] != 'ready'
                        or ready['corpus_sha256'] != snapshot.corpus_sha256
                        or ready['coverage'] not in ('verified_records', 'synthetic_fixture')
                        or ready['expected_records'] != jira_documents
                        or ready['imported_records'] != jira_documents):
                    raise PolicyUnavailable('The required ready native feed is unavailable')
    except (sa.exc.SQLAlchemyError, DatabaseRoleError):
        raise PolicyUnavailable('Deployment metadata is unavailable') from None
    return jira_documents


def assert_storage_ready(database, snapshot, *, jira_documents=None, receipt=None):
    """Raise on any incompatible dependency; return no inventory or authority."""
    _check_storage(database, snapshot, jira_documents=jira_documents, receipt=receipt)


def publish_receipt(database, snapshot, path):
    """Validate live metadata, then publish a new receipt without replacing one."""
    snapshot = _snapshot(snapshot)
    count = _check_storage(database, snapshot)
    receipt = DeploymentReceipt(snapshot=snapshot, jira_required=count > 0)
    path = Path(path)
    payload = (json.dumps(receipt.model_dump(mode='json'), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    temporary = path.with_name('.' + path.name + '.' + uuid4().hex + '.tmp')
    try:
        if path.exists():
            if load_receipt(path, snapshot) != receipt:
                raise PolicyUnavailable('An incompatible deployment receipt already exists')
            return receipt
        # Same-directory link is an atomic, no-replace publication after fsync.
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, 'wb') as stream:
            # This receipt contains nonsecret deployment metadata. A distinct
            # nonroot app UID must be able to read its read-only state mount.
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except (OSError, ValueError, TypeError):
        raise PolicyUnavailable('Deployment receipt could not be published') from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description='Verify or publish cheap runtime deployment metadata; no model loads.')
    parser.add_argument('--snapshot', type=Path, required=True)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument('--receipt', type=Path)
    operation.add_argument('--publish-receipt', type=Path)
    args = parser.parse_args(argv)
    database = None
    ready = False
    try:
        from src.db.runtime import Database, DatabaseSettings
        snapshot = _snapshot(_json_file(args.snapshot))
        database = Database(DatabaseSettings.from_env())
        if args.publish_receipt is not None:
            publish_receipt(database, snapshot, args.publish_receipt)
        else:
            assert_storage_ready(database, snapshot, receipt=args.receipt)
        ready = True
    except Exception:
        # No DSN, database exception, global count or operator path is printed.
        pass
    finally:
        if database is not None:
            try:
                database.close()
            except Exception:
                ready = False
    print('Deployment metadata is ready.' if ready else 'Deployment is not ready.')
    return 0 if ready else 1


if __name__ == '__main__':
    raise SystemExit(main())

"""Quarantine accounts only in an explicitly named new restored database.

This is not a live-account reset. Operator packaging additionally rejects the
authoritative source volume. The database-side guards cannot be disabled by a
CLI flag: only rag_restore_* targets with the exact migrator login and accepted
schema are writable. Restored grants/data stay intact for operator review.
"""

import argparse
from datetime import datetime
import json
import re
from typing import Annotated

from pydantic import Field

from src.contracts import Contract


class RecoveryRefused(ValueError):
    """Content-free operator refusal; never includes a DSN or SQL exception."""


class QuarantineResult(Contract):
    accounts_disabled: Annotated[int, Field(strict=True, ge=0)]
    sessions_revoked: Annotated[int, Field(strict=True, ge=0)]


def _target(expected_database):
    if (type(expected_database) is not str or len(expected_database) > 63
            or not re.fullmatch(r'rag_restore_[a-z0-9][a-z0-9_]*', expected_database)):
        raise RecoveryRefused('Quarantine requires an explicit new rag_restore_ database')
    return expected_database


def _engine(url, expected_database):
    import sqlalchemy as sa
    from sqlalchemy.pool import NullPool
    from src.db.runtime import database_url, normalize_url

    target = normalize_url(url) if url is not None else database_url(migration=True)
    if target.database != expected_database:
        raise RecoveryRefused('The configured database differs from the explicit restore target')
    return sa.create_engine(target, poolclass=NullPool, hide_parameters=True,
        connect_args={'prepare_threshold': None, 'connect_timeout': 5,
                      'application_name': 'rag-restore-quarantine'})


def _validate(connection, expected_database):
    import sqlalchemy as sa
    row = connection.execute(sa.text('''
SELECT current_database() AS database_name, current_user AS current_role,
       session_user AS login_role, r.rolsuper, r.rolbypassrls, r.rolcreaterole,
       r.rolcreatedb, r.rolreplication
FROM pg_roles r WHERE r.rolname=current_user
''')).mappings().one_or_none()
    authenticated = getattr(getattr(getattr(connection.connection, 'driver_connection', None), 'info', None), 'user', None)
    if (row is None or row['database_name'] != expected_database
            or row['current_role'] != 'rag_migrator' or row['login_role'] != 'rag_migrator'
            or authenticated != 'rag_migrator'
            or any(row[name] is not False for name in ('rolsuper', 'rolbypassrls',
                'rolcreaterole', 'rolcreatedb', 'rolreplication'))):
        raise RecoveryRefused('Quarantine requires the exact dedicated migrator on the new restore target')
    owner = connection.execute(sa.text(
        "SELECT pg_get_userbyid(nspowner) FROM pg_namespace WHERE nspname='rag'")).scalar_one_or_none()
    if owner != 'rag_migrator':
        raise RecoveryRefused('The restored application schema has an incompatible owner')
    heads = connection.execute(sa.text('SELECT version_num FROM rag.alembic_version')).scalars().all()
    if heads != ['0004_jira_feed']:
        raise RecoveryRefused('Quarantine requires the accepted restored schema')


def quarantine_restored_database(*, expected_database, url=None):
    """Atomically disable restored accounts and revoke all old session tokens.

    Only previously active users have their account version incremented. Every
    unrevoked session is revoked, including expired or future-created legacy
    sessions. A second successful run changes zero rows. No policy, content,
    native record or historical run is updated.
    """
    expected_database = _target(expected_database)
    import sqlalchemy as sa
    from src.db import schema
    engine = None
    try:
        engine = _engine(url, expected_database)
        with engine.begin() as connection:
            # All refusal checks are read-only and precede the first mutation.
            _validate(connection, expected_database)
            now = connection.execute(sa.text('SELECT clock_timestamp()')).scalar_one()
            if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
                raise RecoveryRefused('The restored database clock is unavailable')
            disabled = connection.execute(schema.users.update().where(schema.users.c.is_active.is_(True)).values(
                is_active=False, session_version=schema.users.c.session_version + 1)).rowcount
            revoked = connection.execute(schema.sessions.update().where(schema.sessions.c.revoked_at.is_(None)).values(
                revoked_at=sa.func.greatest(now, schema.sessions.c.created_at))).rowcount
            if type(disabled) is not int or type(revoked) is not int or disabled < 0 or revoked < 0:
                raise RecoveryRefused('The quarantine result could not be verified')
            result = QuarantineResult(accounts_disabled=disabled, sessions_revoked=revoked)
        return result
    except RecoveryRefused:
        raise
    except Exception:
        raise RecoveryRefused('Restore quarantine could not be completed') from None
    finally:
        if engine is not None:
            try:
                engine.dispose()
            except Exception:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description='Quarantine accounts only in an explicitly named NEW restored database.')
    parser.add_argument('--expected-database', required=True)
    args = parser.parse_args(argv)
    try:
        result = quarantine_restored_database(expected_database=args.expected_database)
    except Exception:
        print('Restore quarantine refused.')
        return 1
    print(json.dumps(result.model_dump(mode='json'), sort_keys=True, separators=(',', ':')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

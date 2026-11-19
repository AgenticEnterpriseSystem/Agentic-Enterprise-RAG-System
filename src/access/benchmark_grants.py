"""Grant one human login every document in the imported fictional benchmark.

Local QA access only, not an enterprise access policy. The steps stay separate on
purpose: ``prepare`` writes reviewable, write-once permission patches without
changing the database; ``dry-run`` previews them; ``apply`` and ``revoke`` change
grants only through ``PolicyRepository``. Patches hold at most 1,000 documents so
no single database statement covers the whole corpus.

Run inside the operator container, for example:
    python -m src.access.benchmark_grants prepare --login hira --reviewer "Jane" --approval qa-001
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import sqlalchemy as sa

from src.access.repository import GrantSpec, PolicyManifest, PolicyRepository
from src.contracts import SnapshotRef
from src.db import schema
from src.db.runtime import database_url

DEFAULT_SNAPSHOT = Path('/state/snapshot.json')
DEFAULT_ROOT = Path('/operator/benchmark-grants')
BATCH_SIZE = 1000
ACTIONS = ('prepare', 'dry-run', 'apply', 'revoke')


class GrantError(ValueError):
    """A refusal that is safe to print: no secrets, document text or SQL."""


def _active_human(connection, login):
    user_id = connection.execute(sa.select(schema.users.c.user_id).where(
        schema.users.c.login == login, schema.users.c.account_kind == 'human',
        schema.users.c.is_active.is_(True))).scalar_one_or_none()
    if user_id is None:
        raise GrantError(f'No active human login named {login!r}; create it first with src.auth.accounts create')
    return user_id


def prepare(engine, snapshot: SnapshotRef, folder: Path, *, login, reviewer, approval, batch_size=BATCH_SIZE):
    if not all(value and value.strip() for value in (login, reviewer, approval)):
        raise GrantError('--login, --reviewer and --approval are all required')
    with engine.connect() as connection:
        user_id = _active_human(connection, login)
    try:
        folder.mkdir(mode=0o700, parents=True)
    except FileExistsError:
        # A prepared review is evidence of what was approved; never rewrite it.
        raise GrantError(f'{folder} already exists; keep it, or use a new --directory for a new review') from None
    prefix = hashlib.sha256((user_id + snapshot.snapshot_id + approval).encode()).hexdigest()[:24]
    after, total, files = -1, 0, {}
    while True:
        with engine.connect() as connection:
            rows = connection.execute(sa.select(schema.documents.c.position, schema.documents.c.doc_id)
                .where(schema.documents.c.snapshot_id == snapshot.snapshot_id,
                       schema.documents.c.position > after)
                .order_by(schema.documents.c.position).limit(batch_size)).all()
        if not rows:
            break
        number = len(files) + 1
        manifest = PolicyManifest(manifest_id=f'qa-{prefix}-{number:04d}', snapshot=snapshot,
            origin='operator', reviewed_by=reviewer,
            provenance={'approval': approval, 'purpose': 'Reviewed full fictional benchmark QA'},
            grants=(GrantSpec(user_id=user_id, doc_ids=tuple(row.doc_id for row in rows)),))
        raw = manifest.model_dump_json(indent=2).encode() + b'\n'
        name = f'grant-{number:04d}.json'
        with (folder / name).open('xb') as handle:
            handle.write(raw)
        files[name] = hashlib.sha256(raw).hexdigest()
        total += len(rows)
        after = rows[-1].position
    if total != snapshot.document_count:
        raise GrantError(f'Selected {total} documents but the snapshot declares {snapshot.document_count}')
    review = {'login': login, 'user_id': user_id, 'snapshot': snapshot.model_dump(mode='json'),
              'document_count': total, 'files': files}
    with (folder / 'review.json').open('x') as handle:
        json.dump(review, handle, indent=2)
    return {'action': 'prepare', 'login': login, 'documents': total, 'patches': len(files),
            'directory': str(folder)}


def execute(action, repository, snapshot: SnapshotRef, folder: Path):
    """Run dry-run/apply/revoke on exactly the files recorded at prepare time."""
    try:
        review = json.loads((folder / 'review.json').read_text())
    except FileNotFoundError:
        raise GrantError(f'No prepared review in {folder}; run prepare first') from None
    if review['snapshot'] != snapshot.model_dump(mode='json'):
        raise GrantError('The imported snapshot changed since prepare; prepare a new review')
    operation = {'dry-run': repository.dry_run, 'apply': repository.apply, 'revoke': repository.revoke}[action]
    changed = 0
    for name, digest in sorted(review['files'].items()):
        raw = (folder / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise GrantError(f'{name} changed after review')
        manifest = PolicyManifest.model_validate_json(raw)
        if manifest.snapshot != snapshot or any(
                g.user_id != review['user_id'] or len(g.doc_ids) > BATCH_SIZE for g in manifest.grants):
            raise GrantError(f'{name} no longer matches the reviewed scope')
        changed += operation(manifest).changed_grants
    return {'action': action, 'login': review['login'], 'documents': review['document_count'],
            'patches': len(review['files']), 'changed_grants': changed}


def parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('action', choices=ACTIONS)
    parser.add_argument('--login', required=True, help='Existing human login, for example hira')
    parser.add_argument('--reviewer', help='Who approved this access (prepare only)')
    parser.add_argument('--approval', help='Approval or ticket ID recorded in the patches (prepare only)')
    parser.add_argument('--directory', type=Path,
                        help='Review folder; defaults to /operator/benchmark-grants/<login>')
    parser.add_argument('--snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    return parser


def main(argv=None):
    args = parser().parse_args(argv)
    os.umask(0o077)
    folder = args.directory or DEFAULT_ROOT / args.login
    try:
        snapshot = SnapshotRef.model_validate_json(args.snapshot.read_bytes())
        if args.action == 'prepare':
            engine = sa.create_engine(database_url(migration=True), hide_parameters=True)
            try:
                result = prepare(engine, snapshot, folder, login=args.login,
                                 reviewer=args.reviewer, approval=args.approval)
            finally:
                engine.dispose()
        else:
            result = execute(args.action, PolicyRepository(), snapshot, folder)
    except GrantError as error:
        raise SystemExit(f'Benchmark grant {args.action} refused: {error}') from None
    except (ValueError, OSError, sa.exc.SQLAlchemyError):
        # Never echo database URLs or driver messages; partial files stay for review.
        raise SystemExit(f'Benchmark grant {args.action} failed. Partial review files, if any, were kept.') from None
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Offline, streaming preparation of immutable local Docker data volumes.

No model is imported or downloaded by staging. Existing published bytes are
verified, never replaced; interrupted unpublished targets require explicit
--adopt-existing after the operator investigates the failed attempt.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from uuid import uuid4

# Direct ``python scripts/local_data.py`` starts with only /app/scripts on the
# import path. Resolve this known repository layout independently of cwd or a
# caller-provided PYTHONPATH before the importer lazily loads src modules.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARTIFACT_RECEIPT = 'artifacts.json'
MODEL_RECEIPT = 'model.json'
_LEGACY_INDEX_RECEIPT = '.artifact-copy.json'
_LEGACY_CORPUS_RECEIPT = '.verified-copy.json'
_BM25_FILES = {'manifest.json', 'doc_ids.json', 'params.index.json', 'vocab.index.json',
    'data.csc.index.npy', 'indices.csc.index.npy', 'indptr.csc.index.npy', 'nonoccurrence_array.index.npy'}


class PreparationError(ValueError):
    pass


def _json(path, *, limit=4 * 1024**2):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise PreparationError('Duplicate metadata fields')
            result[key] = value
        return result
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise PreparationError('Preparation metadata must be a bounded ordinary file')
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise PreparationError('Preparation metadata grew beyond its bound')
    value = json.loads(raw, object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    return value, hashlib.sha256(raw).hexdigest()


def _signature(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _digest(path):
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(descriptor, 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PreparationError('Staging inputs must be ordinary files')
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024**2), b''):
            digest.update(block)
        if _signature(before) != _signature(os.fstat(handle.fileno())):
            raise PreparationError('Staging input changed while hashing')
    return digest.hexdigest(), before.st_size


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8') + b'\n'


def _write_same(path, value):
    raw = _canonical(value)
    if path.exists() or path.is_symlink():
        previous, _ = _json(path)
        if previous != value:
            raise PreparationError('Published preparation metadata differs; use a new data target')
        return
    temporary = path.parent / ('.publish-' + uuid4().hex)
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _json(path)[0] != value:
                raise PreparationError('Concurrent publication differs; no replacement is allowed') from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise PreparationError('Staging lock exists; inspect the prior job before removing its stale lock') from None
    identity = _signature(os.fstat(descriptor))[:2]
    try:
        yield
    finally:
        os.close(descriptor)
        if path.exists() and _signature(path.lstat())[:2] == identity:
            path.unlink()


def _inventory(directory, *, ignored=()):
    result = set()
    for path in directory.rglob('*'):
        if path.name in ignored and path.parent == directory:
            continue
        if path.is_symlink():
            raise PreparationError('Artifact targets cannot contain symbolic links')
        if path.is_file():
            result.add(path.relative_to(directory).as_posix())
        elif not path.is_dir():
            raise PreparationError('Staging targets cannot contain special files')
    return result


def _model_inventory(snapshot, cache):
    """The exact selected revision, allowing only internal ordinary-file links.

    Other model revisions may coexist in a cache; extra files inside the
    selected snapshot must never change which weights/configuration are loaded.
    """
    result = set()
    for path in snapshot.rglob('*'):
        if path.is_symlink():
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(cache.resolve()) or not resolved.is_file():
                raise PreparationError('Model target links must stay inside its cache and identify ordinary files')
        elif path.is_dir():
            continue
        elif not path.is_file():
            raise PreparationError('Model targets cannot contain special files')
        result.add(path.relative_to(snapshot).as_posix())
    return result


def _prepare_state(directory, *, forbidden=()):
    """Prepare only the managed, nonsecret state directory for app UID10001.

    Root staging must not leave a fresh state volume unwritable to subsequent
    nonroot import/receipt jobs. Existing receipt contents are checked and left
    untouched; source trees, model data and operator credentials are not chowned.
    """
    directory = Path(directory)
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise PreparationError('Preparation state must be an ordinary directory')
    resolved = directory.resolve()
    for source in map(Path, forbidden):
        protected = source.resolve()
        if (resolved == protected or resolved.is_relative_to(protected)
                or protected.is_relative_to(resolved)
                or (directory.exists() and source.exists() and directory.samefile(source))):
            raise PreparationError('Preparation state must be separate from all source/data directories')
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) | getattr(os, 'O_NOFOLLOW', 0))
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise PreparationError('Preparation state must be an ordinary directory')
        for name in (ARTIFACT_RECEIPT, MODEL_RECEIPT, 'snapshot.json', 'corpus-import.json', 'deployment.json'):
            path = directory / name
            if path.exists() or path.is_symlink():
                _json(path)  # Ordinary, bounded, unambiguous JSON; preserve bytes.
        if os.geteuid() == 0:
            os.fchown(descriptor, 10001, 10001)
        os.fchmod(descriptor, 0o755)
    finally:
        os.close(descriptor)


def _copy(source, destination, digest):
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(source_fd, 'rb') as reader:
        before = os.fstat(reader.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PreparationError('Staging input is not an ordinary file')
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        copied = hashlib.sha256()
        with os.fdopen(descriptor, 'wb') as writer:
            for block in iter(lambda: reader.read(1024**2), b''):
                writer.write(block)
                copied.update(block)
            writer.flush()
            os.fsync(writer.fileno())
        if _signature(before) != _signature(os.fstat(reader.fileno())) or copied.hexdigest() != digest:
            raise PreparationError('Source bytes changed during staging; target was not published')
    if _digest(destination)[0] != digest:
        raise PreparationError('Independent destination hash differs; target was not published')
    os.chmod(destination, 0o444)


def _seal(directory):
    for path in directory.rglob('*'):
        if not path.is_symlink():
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(directory, 0o555)


def _reserve(directory, needed, min_free_bytes):
    if shutil.disk_usage(directory).free < needed + min_free_bytes:
        raise PreparationError('Insufficient Docker-volume disk headroom for copy plus the retained reserve')


def _adopted_metadata(corpus, indexes, files, sizes):
    """Validate only the two historical copy receipts at their exact roots.

    These files are metadata, not substitutes for independently hashing every
    core source and destination. Preserve their bytes and bind their raw hashes
    in the new state receipt, so later modification/removal is not adoption.
    """
    hashes, ignored = {}, {'corpus': (), 'indexes': ()}
    index_path = indexes / _LEGACY_INDEX_RECEIPT
    if index_path.exists() or index_path.is_symlink():
        value, digest = _json(index_path, limit=256 * 1024)
        expected = {name[len('indexes/'):]: {'bytes': sizes[name], 'sha256': files[name]}
                    for name in files if name.startswith('indexes/')}
        if (type(value) is not dict or value.get('format') != 'verified-search-indexes-1'
                or value.get('complete') is not True or type(value.get('file_count')) is not int
                or value['file_count'] != len(expected) or type(value.get('bytes')) is not int
                or value['bytes'] != sum(item['bytes'] for item in expected.values())
                or type(value.get('files')) is not list or len(value['files']) != len(expected)):
            raise PreparationError('Legacy index receipt differs from the verified core inventory')
        recorded = {}
        for item in value['files']:
            if (type(item) is not dict or type(item.get('relative_path')) is not str
                    or item['relative_path'] in recorded or type(item.get('bytes')) is not int
                    or type(item.get('sha256')) is not str):
                raise PreparationError('Legacy index receipt requires distinct exact file/hash/size records')
            recorded[item['relative_path']] = {'bytes': item['bytes'], 'sha256': item['sha256']}
        if recorded != expected:
            raise PreparationError('Legacy index receipt differs from independently hashed core files')
        hashes['indexes/' + _LEGACY_INDEX_RECEIPT] = digest
        ignored['indexes'] = (_LEGACY_INDEX_RECEIPT,)
    corpus_path = corpus / _LEGACY_CORPUS_RECEIPT
    if corpus_path.exists() or corpus_path.is_symlink():
        value, digest = _json(corpus_path, limit=256 * 1024)
        if (type(value) is not dict or value.get('format') != 'verified-raw-corpus-1'
                or value.get('complete') is not True or type(value.get('bytes')) is not int
                or value['bytes'] != sizes['corpus/documents.parquet']
                or type(value.get('corpus_sha256')) is not str
                or value['corpus_sha256'] != files['corpus/documents.parquet']):
            raise PreparationError('Legacy corpus receipt differs from the independently hashed corpus')
        hashes['corpus/' + _LEGACY_CORPUS_RECEIPT] = digest
        ignored['corpus'] = (_LEGACY_CORPUS_RECEIPT,)
    return hashes, ignored


def stage_artifacts(source, corpus_output, indexes_output, state, *, adopt_existing=False, min_free_bytes=8 * 1024**3):
    source, corpus_output, indexes_output, state = map(Path, (source, corpus_output, indexes_output, state))
    dense = source / 'indexes/dense-bge-base'
    lexical = source / 'indexes/bm25'
    manifest, dense_sha = _json(dense / 'manifest.json', limit=65536)
    bm25, bm25_sha = _json(lexical / 'manifest.json', limit=65536)
    count, shards = manifest.get('documents'), manifest.get('shards')
    dense_present = _inventory(dense)
    if (manifest.get('complete') is not True or type(manifest.get('version')) is not int
            or manifest['version'] != 1 or type(count) is not int or count < 1
            or type(shards) is not int or not 1 <= shards <= count or 2 + 2 * shards != len(dense_present)
            or type(bm25.get('version')) is not int or bm25['version'] != 1
            or bm25.get('corpus_sha256') != manifest.get('corpus_sha256')
            or type(bm25.get('stats', {}).get('documents')) is not int
            or bm25.get('stats', {}).get('documents') != count):
        raise PreparationError('Complete matching canonical BM25 and dense manifests are required')
    dense_names = {'manifest.json', 'doc_ids.json'} | {
        f'shard_{index:05d}{suffix}' for index in range(shards) for suffix in ('.npy', '.parquet')}
    if dense_present != dense_names or _inventory(lexical) != _BM25_FILES:
        raise PreparationError('Index artifact inventory is incomplete or contains unexpected files')
    sources = {'corpus/documents.parquet': source / 'documents.parquet'}
    sources.update({'indexes/dense-bge-base/' + name: dense / name for name in sorted(dense_names)})
    sources.update({'indexes/bm25/' + name: lexical / name for name in sorted(_BM25_FILES)})
    files, sizes = {}, {}
    for relative, path in sources.items():
        files[relative], sizes[relative] = _digest(path)
    if (files['indexes/dense-bge-base/manifest.json'] != dense_sha
            or files['indexes/bm25/manifest.json'] != bm25_sha
            or files['corpus/documents.parquet'] != manifest['corpus_sha256']):
        raise PreparationError('Source manifest or corpus changed or differs from its declared identity')
    receipt = {'format': 'rag-local-artifacts-v1', 'files': files, 'bytes': sum(sizes.values()),
               'documents': count, 'dense_shards': shards, 'corpus_sha256': manifest['corpus_sha256']}
    indexes_output.mkdir(parents=True, exist_ok=True)
    corpus_output.mkdir(parents=True, exist_ok=True)
    if (corpus_output.samefile(indexes_output)
            or corpus_output.resolve().is_relative_to(indexes_output.resolve())
            or indexes_output.resolve().is_relative_to(corpus_output.resolve())):
        raise PreparationError('Corpus and index targets must be separate directories')
    _prepare_state(state, forbidden=(source, corpus_output, indexes_output))
    with _lock(state, 'artifacts.lock'):
        metadata, ignored = _adopted_metadata(corpus_output, indexes_output, files, sizes)
        if metadata:
            receipt['adopted_metadata_sha256'] = metadata
        published = (state / ARTIFACT_RECEIPT).exists()
        if published and _json(state / ARTIFACT_RECEIPT)[0] != receipt:
            raise PreparationError('Published artifact target differs; it will not be overwritten')
        targets = {'corpus': corpus_output, 'indexes': indexes_output}
        for prefix, directory in targets.items():
            expected = {name.split('/', 1)[1] for name in sources if name.startswith(prefix + '/')}
            existing = _inventory(directory, ignored=ignored[prefix])
            if existing and (existing != expected or (not published and not adopt_existing)):
                raise PreparationError('Nonempty unpublished target; investigate then explicitly adopt identical bytes or use a new volume')
            if published and existing != expected:
                raise PreparationError('Published artifact inventory differs; no repair is implicit')
        missing_bytes = sum(size for name, size in sizes.items()
            if not (targets[name.split('/', 1)[0]] / name.split('/', 1)[1]).exists())
        for directory in targets.values():
            _reserve(directory, missing_bytes, min_free_bytes)
        for relative, original in sources.items():
            prefix, name = relative.split('/', 1)
            destination = targets[prefix] / name
            if destination.exists():
                if destination.samefile(original):
                    raise PreparationError('Source inputs cannot also be writable staging targets')
                if _digest(destination)[0] != files[relative]:
                    raise PreparationError('Existing target bytes differ; no replacement is allowed')
            else:
                _copy(original, destination, files[relative])
        # Re-read every source so same-run replacement cannot publish a receipt.
        if any(_digest(path)[0] != files[name] for name, path in sources.items()):
            raise PreparationError('Source set changed before publication')
        if _adopted_metadata(corpus_output, indexes_output, files, sizes)[0] != metadata:
            raise PreparationError('Adopted legacy metadata changed before publication')
        for prefix, directory in targets.items():
            if _inventory(directory, ignored=ignored[prefix]) != {
                    name.split('/', 1)[1] for name in sources if name.startswith(prefix + '/')}:
                raise PreparationError('Destination inventory changed before publication')
        _write_same(state / ARTIFACT_RECEIPT, receipt)
    _seal(corpus_output)
    _seal(indexes_output)
    return receipt


def stage_models(source, dense_directory, output, state, *, adopt_existing=False, min_free_bytes=8 * 1024**3):
    source, dense_directory, output, state = map(Path, (source, dense_directory, output, state))
    manifest, _ = _json(dense_directory / 'manifest.json', limit=65536)
    model = manifest.get('model', {})
    name, revision = model.get('name', ''), model.get('revision', '')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', name) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise PreparationError('The dense manifest must pin one immutable model revision')
    relative_root = Path('models--' + name.replace('/', '--')) / 'snapshots' / revision
    snapshot = source / relative_root
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise PreparationError('The exact pinned offline model snapshot is unavailable')
    root = (source / relative_root.parts[0]).resolve()
    sources = {}
    for path in sorted(snapshot.rglob('*')):
        if path.is_dir() and not path.is_symlink():
            continue
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise PreparationError('Model links must resolve to ordinary files inside that model cache')
        sources[(relative_root / path.relative_to(snapshot)).as_posix()] = resolved
    if not sources or not {'config.json', 'modules.json', 'tokenizer.json'} <= {
            Path(name).relative_to(relative_root).as_posix() for name in sources}:
        raise PreparationError('The pinned sentence-transformer snapshot is incomplete')
    files, sizes = {}, {}
    for relative, path in sources.items():
        files[relative], sizes[relative] = _digest(path)
    receipt = {'format': 'rag-local-model-v1', 'model': name, 'revision': revision,
               'files': files, 'bytes': sum(sizes.values())}
    output.mkdir(parents=True, exist_ok=True)
    _prepare_state(state, forbidden=(source, dense_directory, output))
    with _lock(state, 'model.lock'):
        selected = output / relative_root
        expected = {Path(name).relative_to(relative_root).as_posix() for name in sources}
        present = _model_inventory(selected, output) if selected.exists() else set()
        if present - expected:
            raise PreparationError('Selected model snapshot contains unexpected files outside its source receipt')
        published = (state / MODEL_RECEIPT).exists()
        existing = any(output.iterdir())
        if published and _json(state / MODEL_RECEIPT)[0] != receipt:
            raise PreparationError('Published model target differs; it will not be overwritten')
        if published and any(not (output / name).exists() for name in sources):
            raise PreparationError('Published model inventory is incomplete; no repair is implicit')
        if existing and not published and not adopt_existing:
            raise PreparationError('Nonempty model target requires explicit identical-byte adoption')
        needed = sum(size for relative, size in sizes.items() if not (output / relative).exists())
        _reserve(output, needed, min_free_bytes)
        for relative, original in sources.items():
            destination = output / relative
            if destination.exists() or destination.is_symlink():
                resolved = destination.resolve(strict=True)
                if resolved.samefile(original):
                    raise PreparationError('Source models cannot also be writable staging targets')
                if not resolved.is_relative_to(output.resolve()) or _digest(resolved)[0] != files[relative]:
                    raise PreparationError('Existing pinned model bytes differ; no replacement is allowed')
            else:
                _copy(original, destination, files[relative])
        if any(_digest(path)[0] != files[name] for name, path in sources.items()):
            raise PreparationError('Model source changed before publication')
        if _model_inventory(selected, output) != expected:
            raise PreparationError('Selected model snapshot inventory changed before publication')
        _write_same(state / MODEL_RECEIPT, receipt)
    _seal(output)
    return receipt


def import_corpus(corpus, dense_directory, state, snapshot_id, *, verify_only=False):
    from src.data.import_catalog import import_catalog, verify_import
    operation = verify_import if verify_only else import_catalog
    report = operation(Path(corpus), Path(dense_directory), snapshot_id=snapshot_id)
    if report.state != 'ready' or report.snapshot is None:
        raise PreparationError('Only a verified ready canonical import may publish its snapshot')
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    _write_same(state / 'snapshot.json', report.snapshot.model_dump(mode='json'))
    _write_same(state / 'corpus-import.json', report.as_dict())
    return {'snapshot_id': report.snapshot.snapshot_id, 'state': report.state,
            'documents': report.document_count, 'passages': report.passage_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    artifacts = commands.add_parser('stage-artifacts')
    artifacts.add_argument('--source', type=Path, required=True)
    artifacts.add_argument('--corpus-output', type=Path, required=True)
    artifacts.add_argument('--indexes-output', type=Path, required=True)
    models = commands.add_parser('stage-models')
    models.add_argument('--source', type=Path, required=True)
    models.add_argument('--dense-directory', type=Path, required=True)
    models.add_argument('--output', type=Path, required=True)
    for command in (artifacts, models):
        command.add_argument('--state', type=Path, required=True)
        command.add_argument('--adopt-existing', action='store_true')
        command.add_argument('--min-free-gib', type=int, default=8,
            help='Disk reserve after pending copy: default8GiB; explicit operator minimum4GiB. Not a total-capacity guarantee.')
    importer = commands.add_parser('import-corpus')
    importer.add_argument('--corpus', type=Path, required=True)
    importer.add_argument('--dense-directory', type=Path, required=True)
    importer.add_argument('--state', type=Path, required=True)
    importer.add_argument('--snapshot-id', required=True)
    importer.add_argument('--verify-only', action='store_true')
    args = vars(parser.parse_args())
    name = args.pop('command').replace('-', '_')
    if 'min_free_gib' in args:
        reserve = args.pop('min_free_gib')
        if reserve < 4:
            parser.error('Explicit staging reserves must retain at least4GiB Docker-volume disk headroom')
        args['min_free_bytes'] = reserve * 1024**3
    try:
        report = globals()[name](**args)
    except PreparationError as error:
        parser.error(str(error))
    except Exception:
        parser.error('Local preparation failed; inspect unchanged inputs and private operator configuration')
    print(json.dumps({key: value for key, value in report.items() if key != 'files'}, sort_keys=True))


if __name__ == '__main__':
    main()

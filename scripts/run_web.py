"""Run the web app natively using local .env and the assembled hybrid bundle.

Use this on macOS when Docker has insufficient RAM for the full hybrid index.
Hybrid is the default. BM25 requires --retrieval bm25 explicitly.
Other shell settings override .env; Docker /data paths map to ARTIFACTS_DIR.
"""
import argparse
import json
import os
import shlex
import sys
from pathlib import Path


def build_environment(root, inherited, retrieval=None):
    env = dict(inherited)
    settings = root / '.env'
    if settings.exists():
        for line in settings.read_text().splitlines():
            key, separator, value = line.strip().partition('=')
            if not separator or not (key.startswith('RAG_') or key in (
                    'ANTHROPIC_API_KEY', 'ARTIFACTS_DIR', 'WEB_PORT')):
                continue
            tokens = shlex.split(value, comments=True)
            env.setdefault(key, ' '.join(tokens))
    if retrieval is None and env.get('RAG_RETRIEVAL', '').strip() not in ('', 'hybrid'):
        raise ValueError('Hybrid startup refused a conflicting RAG_RETRIEVAL setting. '
                         'Set RAG_RETRIEVAL=hybrid in .env and unset stale shell overrides. '
                         'For an intentional baseline, use --retrieval bm25.')
    env['RAG_RETRIEVAL'] = retrieval or 'hybrid'
    artifacts = Path(env.get('ARTIFACTS_DIR', './.cache/hybrid-artifacts'))
    if not artifacts.is_absolute():
        artifacts = root / artifacts
    for key, default in (
        ('RAG_CORPUS', 'documents.parquet'),
        ('RAG_BM25_INDEX', 'indexes/bm25'),
        ('RAG_DENSE_INDEX', 'indexes/dense-bge-base'),
    ):
        value = env.get(key, '')
        if value.startswith('/data/'):
            env[key] = str(artifacts / value.removeprefix('/data/'))
        elif not value:
            env[key] = str(artifacts / default)
    for key in ('RAG_ALLOWED_DOC_IDS_FILE',):
        if env.get(key, '').startswith('/data/'):
            env[key] = str(artifacts / env[key].removeprefix('/data/'))
    env.setdefault('RAG_DEVICE', 'cpu')
    env.setdefault('RAG_FRONTEND_DIST', str(root / 'frontend/dist'))
    if sys.platform == 'darwin' and env.get('RAG_RETRIEVAL') == 'hybrid':
        library = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/lib'
        if library.is_dir():
            env['DYLD_LIBRARY_PATH'] = str(library) + (
                ':' + env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retrieval', choices=('hybrid', 'bm25'),
                        help='Explicit retrieval override; otherwise require hybrid')
    parser.add_argument('--check', action='store_true', help='Show effective non-secret settings without starting')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        env = build_environment(root, os.environ, args.retrieval)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({'mode': env.get('RAG_MODE', 'retrieval'),
                      'retrieval': env['RAG_RETRIEVAL'], 'device': env['RAG_DEVICE'],
                      'corpus': env['RAG_CORPUS'], 'dense_index': env['RAG_DENSE_INDEX'],
                      'port': env.get('WEB_PORT', '8000')}), flush=True)
    if args.check:
        return
    os.chdir(root)
    os.execve(sys.executable, [sys.executable, '-m', 'uvicorn', 'src.api.main:app',
              '--host', '127.0.0.1', '--port', env.get('WEB_PORT', '8000'), '--workers', '1'], env)


if __name__ == '__main__':
    main()

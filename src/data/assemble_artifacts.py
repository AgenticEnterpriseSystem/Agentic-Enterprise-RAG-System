"""Assemble split downloads into one self-contained web artifact directory.

Hard links avoid duplicating large files on the same filesystem. Cross-device
copies are supported. Inputs are never moved, changed, or deleted.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.retrieval.corpus_bm25 import file_sha256


def assemble(roots, output, dense_name='dense_base_titled'):
    roots, output = [Path(r) for r in roots], Path(output)
    if output.exists():
        raise ValueError('Output already exists; choose a new directory')

    def locate(relative):
        matches = [root / relative for root in roots if (root / relative).is_file()]
        if not matches:
            raise ValueError(f'Missing artifact: {relative}')
        if len(matches) > 1:
            digest = file_sha256(matches[0])
            if any(file_sha256(p) != digest for p in matches[1:]):
                raise ValueError(f'Conflicting artifact copies: {relative}')
        return matches[0]

    corpus = locate(Path('documents.parquet'))
    dense_manifest = locate(Path(dense_name) / 'manifest.json')
    bm25_manifest = locate(Path('indexes/bm25/manifest.json'))
    dense, bm25 = [json.loads(p.read_text()) for p in (dense_manifest, bm25_manifest)]
    digest = file_sha256(corpus)
    if any(m.get('corpus_sha256') != digest for m in (dense, bm25)):
        raise ValueError('Corpus checksum differs from index manifests')
    if dense.get('version') != 1 or not dense.get('complete') or not isinstance(dense.get('shards'), int) or dense['shards'] < 1:
        raise ValueError('Dense manifest is incomplete or unsupported')
    files = {Path('documents.parquet'): corpus}
    # bm25s persistence includes sparse arrays, vocabulary, parameters and IDs.
    for name in ('manifest.json', 'doc_ids.json', 'params.index.json', 'vocab.index.json',
                 'data.csc.index.npy', 'indices.csc.index.npy', 'indptr.csc.index.npy',
                 'nonoccurrence_array.index.npy'):
        rel = Path('indexes/bm25') / name
        files[rel] = locate(rel)
    for name in ('manifest.json', 'doc_ids.json'):
        files[Path('indexes/dense-bge-base') / name] = locate(Path(dense_name) / name)
    ids = json.loads(files[Path('indexes/bm25/doc_ids.json')].read_text())
    dense_ids = json.loads(files[Path('indexes/dense-bge-base/doc_ids.json')].read_text())
    if ids != dense_ids or len(ids) != dense['documents']:
        raise ValueError('Index document IDs/order differ')
    corpus_ids = pq.read_table(corpus, columns=['doc_id']).column('doc_id').to_pylist()
    if corpus_ids != ids:
        raise ValueError('Corpus document order differs from indexes')
    dimension, passages = None, 0
    for i in range(dense['shards']):
        name = f'shard_{i:05d}'
        vectors_path = locate(Path(dense_name) / (name + '.npy'))
        spans_path = locate(Path(dense_name) / (name + '.parquet'))
        vectors = np.load(vectors_path, mmap_mode='r', allow_pickle=False)
        spans = pq.read_table(spans_path)
        if vectors.ndim != 2 or len(vectors) != len(spans):
            raise ValueError(f'Invalid shard shape: {name}')
        if dimension is not None and vectors.shape[1] != dimension:
            raise ValueError(f'Inconsistent shard dimensions: {name}')
        dimension = vectors.shape[1]
        for start in range(0, len(vectors), 10000):
            if not np.isfinite(vectors[start:start + 10000]).all():
                raise ValueError(f'Non-finite vectors: {name}')
        doc_indices = spans.column('doc_index').to_numpy()
        starts, ends = [spans.column(c).to_numpy() for c in ('start_char', 'end_char')]
        if (np.any(doc_indices < 0) or np.any(doc_indices >= len(ids))
                or np.any(starts < 0) or np.any(ends <= starts)):
            raise ValueError(f'Invalid passage metadata: {name}')
        passages += len(vectors)
        for suffix, path in (('.npy', vectors_path), ('.parquet', spans_path)):
            files[Path('indexes/dense-bge-base') / (name + suffix)] = path
    # Validate every input before creating the output directory.
    for relative, source in files.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
    report = {'documents': len(ids), 'passages': passages, 'dimension': dimension,
              'model': dense['model'], 'corpus_sha256': digest,
              'dense_index': 'indexes/dense-bge-base',
              'sources': [str(r.resolve()) for r in roots]}
    (output / 'assembly.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(args.roots, args.output), indent=2))


if __name__ == '__main__':
    main()

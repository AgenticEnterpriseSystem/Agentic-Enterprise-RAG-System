"""Split downloads must produce the same loadable indexes, without altering inputs."""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.data.assemble_artifacts import assemble
from src.retrieval.corpus_bm25 import CorpusBM25, file_sha256


def test_split_bundle_assembly_and_conflicts(tmp_path):
    first, second = tmp_path / 'first', tmp_path / 'second'
    first.mkdir()
    second.mkdir()
    corpus = first / 'documents.parquet'
    pq.write_table(pa.table({'doc_id': ['d'], 'title': ['Policy'],
                            'content': ['Retention is 90 days.'], 'source_type': ['wiki']}), corpus)
    CorpusBM25.build(corpus).save(first / 'indexes/bm25')
    dense = second / 'dense_base_titled'
    dense.mkdir()
    (dense / 'manifest.json').write_text(json.dumps({
        'version': 1, 'complete': True, 'documents': 1, 'shards': 1,
        'corpus_sha256': file_sha256(corpus), 'model': {'name': 'test'}}))
    (dense / 'doc_ids.json').write_text('["d"]')
    np.save(dense / 'shard_00000.npy', np.ones((1, 2), dtype=np.float32))
    pq.write_table(pa.table({'doc_index': [0], 'start_char': [0], 'end_char': [6]}),
                   dense / 'shard_00000.parquet')
    output = tmp_path / 'assembled'
    report = assemble([first, second], output)
    assert report['passages'] == 1 and report['dimension'] == 2
    loaded = CorpusBM25.load(output / 'indexes/bm25', corpus_path=output / 'documents.parquet')
    assert loaded.doc_ids == ['d']
    assert corpus.exists() and (dense / 'shard_00000.npy').exists()
    with pytest.raises(ValueError, match='already exists'):
        assemble([first, second], output)
    (second / 'documents.parquet').write_bytes(b'conflicting download')
    with pytest.raises(ValueError, match='Conflicting'):
        assemble([first, second], tmp_path / 'conflict')
    assert not (tmp_path / 'conflict').exists()

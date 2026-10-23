"""Web and offline evaluation must produce the same configured retrieval."""
import argparse
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.api.runtime import WebSettings, load_retriever
from src.config import RetrievalSettings
from src.contracts import SearchRequest, UserContext
from src.evaluation.run_hybrid import run_variant, search_once
from src.retrieval.corpus_bm25 import CorpusBM25
from src.retrieval.corpus_dense import BGEModel, embed_corpus
from src.retrieval.factory import add_retrieval_arguments, load_pair, settings_from_args
from tests.test_retrieval import FakeModel


@pytest.mark.parametrize('top_k', [1, 3])
def test_web_and_evaluation_use_identical_settings_and_results(tmp_path, monkeypatch, top_k):
    corpus = tmp_path / 'documents.parquet'
    pq.write_table(pa.table({
        'doc_id': ['a', 'b', 'c'], 'title': ['Policy', 'Deploy', 'Grid'],
        'content': ['canary rollback recovery', 'canary deployment', 'grid elevation'],
        'source_type': ['slack'] * 3,
    }), corpus)
    CorpusBM25.build(corpus).save(tmp_path / 'bm25')
    embed_corpus(corpus, tmp_path / 'dense', FakeModel(), shard_docs=2, window=4, overlap=1)
    monkeypatch.setattr(BGEModel, 'for_index', lambda *a, **kw: FakeModel())
    values = dict(corpus=corpus, bm25_index=tmp_path / 'bm25', dense_index=tmp_path / 'dense',
                  bm25_weight=.6, candidate_k=2, rerank_k=3, rrf_k=20.)
    for name in WebSettings.model_fields:
        monkeypatch.delenv('RAG_' + name.upper(), raising=False)
    for name, value in values.items():
        monkeypatch.setenv('RAG_' + name.upper(), str(value))
    monkeypatch.setenv('RAG_RETRIEVAL', 'hybrid')
    web = load_retriever(WebSettings.from_env())

    parser = argparse.ArgumentParser()
    add_retrieval_arguments(parser)
    args = parser.parse_args([part for name, value in values.items()
                              for part in ('--' + name.replace('_', '-'), str(value))])
    settings = settings_from_args(args, retrieval='hybrid')
    bm25, dense = load_pair(settings)
    assert bm25.corpus is dense.corpus
    questions = [dict(question_id='q', question_type='basic', source_types=['slack'],
                      question='canary rollback', expected_doc_ids=['a'])]
    user = UserContext(user_id='test', allowed_doc_ids=frozenset(web.doc_ids))
    caches, seconds, errors = search_once((bm25, dense), questions, user,
                                         depth=max(top_k, settings.candidate_k))
    rows, _, manifest = run_variant((bm25, dense), caches, seconds, errors, questions, user,
        bm25_weight=settings.bm25_weight, candidate_k=settings.candidate_k,
        rerank_k=settings.rerank_k, rrf_k=settings.rrf_k, top_k=top_k)
    hits = web.search(SearchRequest(query='canary rollback', user=user, top_k=top_k))
    assert rows[0]['retrieved_doc_ids'] == [hit.chunk.doc_id for hit in hits]
    assert rows[0]['scores'] == [round(hit.score, 4) for hit in hits]
    assert manifest['config'] == web.manifest['config']

    # Both entry points reject the same incompatible corpus before model loading.
    path = tmp_path / 'dense/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['corpus_sha256'] = 'wrong'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='same corpus'):
        load_pair(settings)
    with pytest.raises(ValueError, match='same corpus'):
        load_retriever(WebSettings.from_env())


def test_cli_defaults_and_validation_match_web():
    parser = argparse.ArgumentParser()
    add_retrieval_arguments(parser)
    cli = settings_from_args(parser.parse_args([]))
    web = WebSettings()
    assert cli == RetrievalSettings.model_validate({
        name: getattr(web, name) for name in RetrievalSettings.model_fields})
    for flag, value in (('--bm25-weight', '1.1'), ('--candidate-k', '0'), ('--rrf-k', 'nan')):
        with pytest.raises(ValueError):
            settings_from_args(parser.parse_args([flag, value]))

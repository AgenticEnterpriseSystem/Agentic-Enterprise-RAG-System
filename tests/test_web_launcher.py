"""Normal launches must not silently serve a BM25 baseline."""
import json

import pytest

from scripts import run_web
from src.api.runtime import WebSettings
from src.config import RetrievalSettings


def test_hybrid_defaults_without_local_settings(tmp_path):
    env = run_web.build_environment(tmp_path, {})
    assert env['RAG_RETRIEVAL'] == 'hybrid'
    assert env['RAG_CORPUS'] == str(tmp_path / '.cache/hybrid-artifacts/documents.parquet')
    assert env['RAG_DENSE_INDEX'].endswith('/indexes/dense-bge-base')
    assert WebSettings().retrieval == RetrievalSettings().retrieval == 'hybrid'


@pytest.mark.parametrize('shell,file_value', [({'RAG_RETRIEVAL': 'bm25'}, 'hybrid'), ({}, 'bm25')])
def test_stale_bm25_setting_refuses_normal_launch(tmp_path, shell, file_value):
    (tmp_path / '.env').write_text(f'RAG_RETRIEVAL={file_value}\n')
    with pytest.raises(ValueError, match='conflicting RAG_RETRIEVAL'):
        run_web.build_environment(tmp_path, shell)


def test_explicit_baseline_and_hybrid_overrides(tmp_path):
    (tmp_path / '.env').write_text('RAG_RETRIEVAL=hybrid\n')
    assert run_web.build_environment(tmp_path, {}, 'bm25')['RAG_RETRIEVAL'] == 'bm25'
    assert run_web.build_environment(tmp_path, {'RAG_RETRIEVAL': 'bm25'}, 'hybrid')['RAG_RETRIEVAL'] == 'hybrid'


def test_check_shows_effective_config_without_secrets_or_starting(tmp_path, monkeypatch, capsys):
    (tmp_path / '.env').write_text('RAG_MODE=live\nRAG_RETRIEVAL=hybrid\nANTHROPIC_API_KEY=do-not-print\n')
    monkeypatch.setattr(run_web, '__file__', str(tmp_path / 'scripts/run_web.py'))
    monkeypatch.setattr('sys.argv', ['run_web.py', '--check'])
    monkeypatch.setattr(run_web.os, 'environ', {})
    def no_start(*args):
        pytest.fail('Check must not start a server')
    monkeypatch.setattr(run_web.os, 'execve', no_start)
    run_web.main()
    output = capsys.readouterr().out
    assert json.loads(output)['retrieval'] == 'hybrid'
    assert 'do-not-print' not in output and 'ANTHROPIC_API_KEY' not in output


def test_check_reports_the_reranker_the_app_will_load():
    hybrid = {'RAG_RETRIEVAL': 'hybrid'}
    assert run_web.reranker_summary(hybrid) == (
        'bge-reranker-base (20 candidates, 4 windows, weight 0.7)')
    assert run_web.reranker_summary(hybrid | {'RAG_RERANKER': 'off'}) is None
    assert run_web.reranker_summary({'RAG_RETRIEVAL': 'bm25'}) is None
    assert run_web.reranker_summary(hybrid | {'RAG_RERANKER': 'bge-reranker-v2-m3', 'RAG_RERANK_K': '50'}) == (
        'bge-reranker-v2-m3 (50 candidates, 4 windows, weight 0.7)')

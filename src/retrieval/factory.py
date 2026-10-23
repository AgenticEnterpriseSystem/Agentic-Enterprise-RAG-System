"""Manifest-aware retrieval construction shared by the app and evaluation tools."""
import argparse
import json
from pathlib import Path

from src.config import RetrievalSettings


class RetrievalConfigurationError(ValueError):
    """Operator-safe setup error with no credentials or provider messages."""


def load_dense(settings, *, corpus_path=None, corpus=None, bm25_manifest=None):
    from src.retrieval.corpus_dense import BGEModel, CorpusDense, INDEX_VERSION

    directory = settings.dense_index
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest.get('version') != INDEX_VERSION or not manifest.get('complete'):
        raise RetrievalConfigurationError('Dense manifest is incomplete or unsupported. Supply a complete version-1 dense index.')
    if bm25_manifest is not None and manifest.get('corpus_sha256') != bm25_manifest.get('corpus_sha256'):
        raise RetrievalConfigurationError('BM25 and dense indexes must match the same corpus file (corpus_sha256).')
    if not (directory / 'doc_ids.json').is_file():
        raise FileNotFoundError('Dense document IDs are missing')
    if not isinstance(manifest.get('shards'), int) or manifest['shards'] < 1:
        raise RetrievalConfigurationError('Dense manifest must record a positive shard count.')
    if settings.fp16 is not None and settings.fp16 != manifest['model']['fp16']:
        raise RetrievalConfigurationError('Query precision must match the dense index manifest. Unset RAG_FP16 to use the recorded precision.')
    for i in range(manifest['shards']):
        for suffix in ('.npy', '.parquet'):
            if not (directory / f'shard_{i:05d}{suffix}').is_file():
                raise RetrievalConfigurationError('Dense index is missing a shard. Supply every shard_XXXXX.npy and .parquet file.')
    try:
        model = BGEModel.for_index(directory, settings.device, fp16=settings.fp16)
        return CorpusDense.load(directory, model=model, corpus_path=corpus_path, corpus=corpus)
    except ValueError:
        raise RetrievalConfigurationError('Dense model or index is incompatible. Check the pinned model revision, query encoding, precision, corpus order and shard dimensions.') from None


def load_pair(settings):
    from src.retrieval.corpus_bm25 import CorpusBM25
    bm25 = CorpusBM25.load(settings.bm25_index, corpus_path=settings.corpus)
    dense = load_dense(settings, corpus_path=settings.corpus, corpus=bm25.corpus,
                       bm25_manifest=bm25.manifest)
    return bm25, dense


def load_retriever(settings):
    from src.retrieval.corpus_bm25 import CorpusBM25
    from src.retrieval.hybrid import HybridRetriever
    if settings.retrieval == 'bm25':
        return CorpusBM25.load(settings.bm25_index, corpus_path=settings.corpus)
    bm25, dense = load_pair(settings)
    return HybridRetriever(bm25, dense, config=settings.hybrid_config())


def add_retrieval_arguments(parser, *, index_flag='--bm25-index', weight=True):
    """Explicit CLI settings; evaluation never reads a developer's .env implicitly."""
    defaults = RetrievalSettings()
    parser.add_argument('--corpus', type=Path, default=defaults.corpus)
    parser.add_argument(index_flag, dest='bm25_index', type=Path, default=defaults.bm25_index)
    parser.add_argument('--dense-index', type=Path, default=defaults.dense_index)
    parser.add_argument('--device', default=defaults.device)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=defaults.fp16,
                        help='Query precision; default follows the index manifest')
    if weight:
        parser.add_argument('--bm25-weight', type=float, default=defaults.bm25_weight)
    for name in ('candidate_k', 'rerank_k', 'rrf_k'):
        value = getattr(defaults, name)
        parser.add_argument('--' + name.replace('_', '-'), type=type(value), default=value)


def settings_from_args(args, **overrides):
    values = {name: getattr(args, name) for name in RetrievalSettings.model_fields
              if hasattr(args, name)}
    return RetrievalSettings(**(values | overrides))

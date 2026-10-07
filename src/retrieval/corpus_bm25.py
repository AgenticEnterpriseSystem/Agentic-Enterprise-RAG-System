"""Document-level BM25+ over the full corpus, stored as a bm25s sparse index.

The index is built with vectorized NumPy instead of bm25s' per-token Python loop so
~500k documents fit in memory; tests check the result matches `bm25s.BM25.index`.
"""

import argparse
import json
import re
import time
from collections import Counter
from hashlib import sha256
from pathlib import Path

import bm25s
import numpy as np
import pyarrow.parquet as pq
from pydantic import Field

from src.retrieval.passages import bounded_passage
from src.config import CORPUS_DIR
from src.contracts import Chunk, Contract, Evidence, SearchRequest, validate_evidence

INDEX_VERSION = 1


class BM25Config(Contract):
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    delta: float = Field(default=1, ge=0)
    tokenizer: str = "unicode-words-lower-v1"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def document_text(title: str, content: str) -> str:
    # Upstream baselines index title and content as one field.
    return f"{title}\n{content}" if title else content


def build_bm25(texts, config: BM25Config) -> tuple[bm25s.BM25, dict]:
    """Build a bm25s BM25+ model from an iterable of document texts."""
    vocab: dict[str, int] = {}
    term_ids, freqs, lengths = [], [], []
    for text in texts:
        counts = Counter(tokenize(text))
        term_ids.append(
            np.fromiter(
                (vocab.setdefault(t, len(vocab)) for t in counts),
                dtype=np.int32,
                count=len(counts),
            )
        )
        freqs.append(np.fromiter(counts.values(), dtype=np.float32, count=len(counts)))
        lengths.append(counts.total())
    if not lengths:
        raise ValueError("Cannot index an empty corpus")
    n_docs, n_vocab = len(lengths), len(vocab)
    lengths = np.asarray(lengths, dtype=np.float64)
    avg_len = lengths.mean()
    doc_idx = np.repeat(np.arange(n_docs, dtype=np.int32), [len(t) for t in term_ids])
    term_ids = np.concatenate(term_ids)
    freqs = np.concatenate(freqs)
    df = np.bincount(term_ids, minlength=n_vocab)
    idf = np.log((n_docs + 1) / np.maximum(df, 1)).astype(np.float32)
    k1, b, delta = config.k1, config.b, config.delta
    # bm25s stores tf-dependent scores and adds idf * delta at query time.
    nonoccurrence = (idf * np.float32(delta)).astype(np.float32)
    norm = (k1 * (1 - b + b * lengths / avg_len)).astype(np.float32)
    scores = np.empty(len(term_ids), dtype=np.float32)
    step = 10_000_000  # Bound temporary arrays on large corpora.
    for start in range(0, len(term_ids), step):
        part = slice(start, start + step)
        tf = freqs[part]
        scores[part] = idf[term_ids[part]] * (
            (k1 + 1) * tf / (norm[doc_idx[part]] + tf)
        )
    del freqs
    order = np.argsort(term_ids, kind="stable")
    indptr = np.zeros(n_vocab + 1, dtype=np.int64)
    np.cumsum(df, out=indptr[1:])
    postings = len(term_ids)
    del term_ids
    data = scores[order]
    del scores
    indices = doc_idx[order]
    del doc_idx, order
    model = bm25s.BM25(k1=k1, b=b, delta=delta, method="bm25+")
    model.scores = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "num_docs": n_docs,
    }
    model.vocab_dict = vocab
    model.nonoccurrence_array = nonoccurrence
    stats = {
        "documents": n_docs,
        "vocabulary": n_vocab,
        "postings": int(postings),
        "average_tokens": float(avg_len),
    }
    return model, stats


def corpus_texts(corpus_path: Path, doc_ids: list):
    """Stream document texts in corpus order, collecting IDs as a side effect."""
    for batch in pq.ParquetFile(corpus_path).iter_batches(
        batch_size=2000, columns=["doc_id", "title", "content"]
    ):
        rows = batch.to_pydict()
        doc_ids.extend(rows["doc_id"])
        yield from map(document_text, rows["title"], rows["content"])


class CorpusBM25:
    """Full-corpus document retriever; `benchmark_all_docs` skips per-document checks."""

    def __init__(self, model, doc_ids, config, corpus=None, manifest=None):
        self.model = model
        self.doc_ids = list(doc_ids)
        self.position = {d: i for i, d in enumerate(self.doc_ids)}
        self.config = config
        self.corpus = corpus
        self.manifest = manifest or {}

    def rank(self, query: str, top_k: int, mask: np.ndarray | None = None):
        """Return (doc_id, score) pairs for documents sharing at least one query term."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        ids = [
            self.model.vocab_dict[t]
            for t in tokenize(query)
            if t in self.model.vocab_dict
        ]
        if not ids:
            return []
        matched = self.model.get_scores_from_ids(ids)
        # Remove the constant delta term so nonmatching documents score zero.
        matched -= self.model.nonoccurrence_array[ids].sum()
        if mask is not None:
            matched[~mask] = 0
        candidates = np.flatnonzero(matched > 0)
        if len(candidates) > top_k:
            cutoff = np.partition(matched[candidates], len(candidates) - top_k)[len(candidates) - top_k]
            above = candidates[matched[candidates] > cutoff]
            ties = candidates[matched[candidates] == cutoff]
            chosen_ties = sorted(ties, key=lambda i: self.doc_ids[i])[:top_k - len(above)]
            candidates = np.concatenate((above, np.asarray(chosen_ties, dtype=np.int64)))
        offset = float(self.model.nonoccurrence_array[ids].sum())
        ranked = sorted(candidates, key=lambda i: (-matched[i], self.doc_ids[i]))
        return [(self.doc_ids[i], float(matched[i]) + offset) for i in ranked]

    def search(self, request: SearchRequest):
        """Contract-compatible search returning bounded query-relevant evidence."""
        if self.corpus is None:
            raise ValueError("Load the index with a corpus to return document text")
        if request.after or request.before:
            return ()  # The corpus has no timestamps; unknown dates never match.
        mask = np.zeros(len(self.doc_ids), dtype=bool)
        allowed = [
            self.position[d] for d in request.user.allowed_doc_ids if d in self.position
        ]
        if request.user.user_id is not None:
            mask[allowed] = True
        if request.source_types:
            sources = np.asarray(self.corpus.column("source_type").to_pylist())
            mask &= np.isin(sources, request.source_types)
        results = []
        for doc_id, score in self.rank(request.query, request.top_k, mask):
            chunk = self._chunk(doc_id)
            if chunk:  # The mask already applied permissions and sources.
                results.append(
                    bounded_passage(Evidence(
                        citation_id=chunk.chunk_id,
                        chunk=chunk,
                        score=score,
                        retrieval_method="bm25",
                    ), request.query)
                )
        results = tuple(results)
        validate_evidence(request.user, results)
        return results

    def _chunk(self, doc_id):
        # Slice one row without concatenating the full chunked string columns.
        row = self.corpus.slice(self.position[doc_id], 1).to_pylist()[0]
        text = document_text(row["title"], row["content"])
        if not text.strip():
            return None
        return Chunk(
            chunk_id=f"{doc_id}:0:{len(text)}",
            doc_id=doc_id,
            source_type=row["source_type"],
            title=row["title"],
            text=text,
            start_char=0,
            end_char=len(text),
        )

    @classmethod
    def build(cls, corpus_path: Path, config: BM25Config | None = None):
        config = config or BM25Config()
        doc_ids: list[str] = []
        texts = corpus_texts(corpus_path, doc_ids)
        model, stats = build_bm25(texts, config)
        manifest = {
            "version": INDEX_VERSION,
            "method": "bm25+ (bm25s)",
            "unit": "document (title + content)",
            "config": config.model_dump(),
            "corpus_sha256": file_sha256(corpus_path),
            "stats": stats,
        }
        return cls(model, doc_ids, config, manifest=manifest)

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save(directory)
        (directory / "doc_ids.json").write_text(json.dumps(self.doc_ids))
        (directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, directory: Path, corpus_path: Path | None = None, mmap=True, *, corpus=None):
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["version"] != INDEX_VERSION:
            raise ValueError("Unsupported corpus BM25 index version")
        if corpus is not None and corpus_path is None:
            raise ValueError("A shared corpus requires corpus_path for hash validation")
        if corpus_path is not None:
            if file_sha256(corpus_path) != manifest["corpus_sha256"]:
                raise ValueError("Corpus differs from the one used to build the index")
            if corpus is None:
                corpus = pq.read_table(
                    corpus_path, columns=["doc_id", "source_type", "title", "content"]
                )
        model = bm25s.BM25.load(directory, mmap=mmap)
        doc_ids = json.loads((directory / "doc_ids.json").read_text())
        if corpus is not None and corpus.column("doc_id").to_pylist() != doc_ids:
            raise ValueError("Corpus document order differs from the index")
        if len(doc_ids) != model.scores["num_docs"]:
            raise ValueError("Document ID count does not match the index")
        return cls(
            model,
            doc_ids,
            BM25Config.model_validate(manifest["config"]),
            corpus,
            manifest,
        )


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--output", type=Path, default=CORPUS_DIR / "indexes" / "bm25")
    args = parser.parse_args()
    start = time.perf_counter()
    index = CorpusBM25.build(args.corpus)
    index.manifest["build_seconds"] = round(time.perf_counter() - start, 1)
    index.save(args.output)
    print(json.dumps(index.manifest, indent=2))


if __name__ == "__main__":
    main()

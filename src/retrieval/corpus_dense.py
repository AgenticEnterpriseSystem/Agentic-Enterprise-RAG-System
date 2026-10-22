"""Full-corpus dense (BGE, GTE, Arctic, Qwen3) + FAISS document retrieval (max over passages).

Documents longer than the model limit are split into overlapping token windows.
`embed` writes resumable shards (run it on a GPU); `rank` searches all passages
with exact inner product and scores each document by its best passage.
"""

import argparse
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import CORPUS_DIR, EMBEDDING_MODELS, ROOT
from src.contracts import Chunk, Evidence, SearchRequest, validate_evidence
from src.retrieval.corpus_bm25 import document_text, file_sha256

INDEX_VERSION = 1
WINDOW_TOKENS = 500  # Models accept 512 tokens including special tokens; leave margin.
OVERLAP_TOKENS = 50
TITLE_TOKENS = 48  # Cap for the title label repeated on every passage.
PASSAGE_SCHEMA = pa.schema(
    [("doc_index", pa.int32()), ("start_char", pa.int32()), ("end_char", pa.int32())]
)


def windows(offsets, window=WINDOW_TOKENS, overlap=OVERLAP_TOKENS):
    """Character spans of overlapping token windows covering a document."""
    if not offsets:
        return []
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    spans, start, n = [], 0, len(offsets)
    while True:
        end = min(start + window, n)
        spans.append((offsets[start][0], offsets[end - 1][1]))
        if end == n:
            return spans
        start = end - overlap


def document_passages(titles, contents, offsets, window, overlap, title_every=False):
    """Yield, per document, (text to embed, start, end) with spans into document_text.

    With title_every, the content is windowed and each passage is prefixed with
    the (capped) title, so passages after the first keep the document's name.
    Spans then cover only the content part of the combined title + content text.
    """
    if not title_every:
        texts = list(map(document_text, titles, contents))
        for text, offs in zip(texts, offsets(texts)):
            yield [(text[s:e], s, e) for s, e in windows(offs, window, overlap)]
        return
    for title, content, title_offs, content_offs in zip(
        titles, contents, offsets(titles), offsets(contents)
    ):
        if not title:
            yield [(content[s:e], s, e) for s, e in windows(content_offs, window, overlap)]
            continue
        kept = title_offs[:TITLE_TOKENS]
        label = title[: kept[-1][1]] if kept else title
        shift = len(title) + 1  # document_text joins title and content with "\n".
        if not content_offs:
            yield [(label, 0, len(title))] if label.strip() else []
            continue
        content_window = max(window - len(kept), overlap + 1)
        yield [
            (f"{label}\n{content[s:e]}", s + shift, e + shift)
            for s, e in windows(content_offs, content_window, overlap)
        ]


def model_label(name):
    """Registry label for a model name, e.g. "bge-base"."""
    for label, spec in EMBEDDING_MODELS.items():
        if spec["name"] == name:
            return label
    raise ValueError(f"Unsupported embedding model: {name}")


def method_label(name):
    """Short label for manifests and result folders; unregistered names keep their last part."""
    try:
        return model_label(name)
    except ValueError:
        return name.split("/")[-1]


class EmbeddingModel:
    """A registered sentence-transformers model at its pinned revision.

    Queries (and, if the model needs it, passages) get the model's own prefix.
    Precision is fp16 on GPU unless disabled; bf16 suits newer GPUs such as L4.
    """

    def __init__(self, device="cpu", fp16=None, batch_size=128, model="bge-small", bf16=False):
        import torch
        from sentence_transformers import SentenceTransformer

        if model not in EMBEDDING_MODELS:
            raise ValueError(f"Unsupported embedding model: {model}")
        spec = EMBEDDING_MODELS[model]
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Embedding revision must be a 40-character commit hash")
        self.fp16 = (device.startswith("cuda") and not bf16) if fp16 is None else fp16
        if self.fp16 and bf16:
            raise ValueError("Choose fp16 or bf16, not both")
        self.model = SentenceTransformer(
            spec["name"],
            revision=spec["revision"],
            device=device,
            cache_folder=str(ROOT / ".cache" / "models"),
            trust_remote_code=False,
            model_kwargs={"dtype": torch.bfloat16} if bf16 else None,
        )
        if self.fp16:
            self.model.half()
        # Passages are at most WINDOW_TOKENS tokens, so every model gets the same limit.
        self.model.max_seq_length = min(self.model.max_seq_length, 512)
        self.query_prefix = spec["query_prefix"]
        self.document_prefix = spec.get("document_prefix", "")
        self.batch_size = batch_size
        self.identity = {
            "name": spec["name"],
            "revision": spec["revision"],
            "query_prefix": self.query_prefix,
            "normalize": True,
            "max_seq_length": self.model.max_seq_length,
            "fp16": self.fp16,
        }
        # Older manifests have only the keys above; add new ones only when used.
        if self.document_prefix:
            self.identity["document_prefix"] = self.document_prefix
        if bf16:
            self.identity["bf16"] = True

    @classmethod
    def for_index(cls, directory: Path, device="cpu", fp16=None, batch_size=128):
        """Query model matching an index manifest's model, revision and precision."""
        settings = json.loads((Path(directory) / "manifest.json").read_text())["model"]
        label = model_label(settings["name"])
        if settings["revision"] != EMBEDDING_MODELS[label]["revision"]:
            raise ValueError("Index was built with a different model revision")
        bf16 = settings.get("bf16", False)
        return cls(device, fp16=(settings["fp16"] if fp16 is None else fp16) and not bf16,
                   batch_size=batch_size, model=label, bf16=bf16)

    def offsets(self, texts):
        encoded = self.model.tokenizer(
            texts,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        return encoded["offset_mapping"]

    def encode(self, texts, query=False):
        prefix = self.query_prefix if query else self.document_prefix
        out = self.model.encode(
            [prefix + t for t in texts] if prefix else list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError("Model produced non-finite embeddings; try bf16 or fp32 precision")
        return out


BGEModel = EmbeddingModel  # Earlier name, kept for existing callers.


def embed_corpus(
    corpus_path: Path,
    output: Path,
    model,
    shard_docs=20000,
    window=WINDOW_TOKENS,
    overlap=OVERLAP_TOKENS,
    title_every=False,
):
    """Write one embedding shard per `shard_docs` documents; existing shards are kept."""
    output.mkdir(parents=True, exist_ok=True)
    corpus = pq.ParquetFile(corpus_path)
    manifest = {
        "version": INDEX_VERSION,
        "method": f"{method_label(model.identity['name'])} + faiss (max over passages)",
        "unit": (
            f"{window}-token windows, {overlap}-token overlap, of content; "
            f"title (max {TITLE_TOKENS} tokens) prefixed to every passage"
            if title_every
            else f"{window}-token windows, {overlap}-token overlap, of title + content"
        ),
        "title_every_passage": title_every,
        "model": model.identity,
        "corpus_sha256": file_sha256(corpus_path),
        "documents": corpus.metadata.num_rows,
        "shard_docs": shard_docs,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        previous.setdefault("title_every_passage", False)  # Older manifests.
        if {k: previous.get(k) for k in manifest} != manifest:
            raise ValueError("Existing shards were made with different settings")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    shard, start, doc_ids = 0, 0, []
    batches = corpus.iter_batches(
        batch_size=shard_docs, columns=["doc_id", "title", "content"]
    )
    for batch in batches:
        rows = batch.to_pydict()
        doc_ids.extend(rows["doc_id"])
        name = f"shard_{shard:05d}"
        if not (output / f"{name}.npy").exists():
            began = time.perf_counter()
            passages, meta = [], {"doc_index": [], "start_char": [], "end_char": []}
            per_doc = document_passages(
                rows["title"], rows["content"], model.offsets, window, overlap, title_every
            )
            for i, doc_passages in enumerate(per_doc):
                for text, s, e in doc_passages:
                    passages.append(text)
                    meta["doc_index"].append(start + i)
                    meta["start_char"].append(s)
                    meta["end_char"].append(e)
            vectors = model.encode(passages).astype(np.float16)
            pq.write_table(
                pa.table(meta, schema=PASSAGE_SCHEMA), output / f"{name}.parquet"
            )
            # Write vectors last: their presence marks the shard complete.
            np.save(output / f"{name}.tmp.npy", vectors)
            (output / f"{name}.tmp.npy").replace(output / f"{name}.npy")
            print(
                f"{name}: {len(rows['doc_id'])} docs, {len(passages)} passages, "
                f"{time.perf_counter() - began:.0f}s",
                flush=True,
            )
        start += len(rows["doc_id"])
        shard += 1
    (output / "doc_ids.json").write_text(json.dumps(doc_ids))
    manifest["complete"] = True
    manifest["shards"] = shard
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class CorpusDense:
    def __init__(self, vectors, doc_index, doc_ids, model, manifest, *, corpus=None, spans=None, index=None):
        self.index = index if index is not None else faiss.IndexFlatIP(vectors.shape[1])
        if index is None:
            self.index.add(vectors)
        if self.index.ntotal != len(doc_index):
            raise ValueError("Embedding shards are inconsistent")
        self.doc_index = doc_index
        self.doc_ids = doc_ids
        self.model = model
        self.manifest = manifest
        self.corpus, self.spans = corpus, spans
        self.position = {doc: i for i, doc in enumerate(doc_ids)}
        if len(self.position) != len(doc_ids):
            raise ValueError("Duplicate document IDs")
        if np.any(doc_index < 0) or np.any(doc_index >= len(doc_ids)):
            raise ValueError("Passage document index outside corpus")
        if corpus is not None and corpus.column("doc_id").to_pylist() != list(doc_ids):
            raise ValueError("Corpus document order differs from embeddings")
        if spans is not None and len(spans) != len(doc_index):
            raise ValueError("Passage offset count differs from embeddings")

    def rank(self, query: str, top_k: int, mask: np.ndarray | None = None):
        """Return (doc_id, score) for the top documents by best passage score."""
        return self.rank_vector(self.model.encode([query], query=True)[0], top_k, mask)

    def rank_vector(self, vector, top_k, mask=None):
        return [(self.doc_ids[d], score) for d, score, _ in self._rank_passages(vector, top_k, mask)]

    def _rank_passages(self, vector, top_k, mask=None):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        fetch = min(max(10 * top_k, 100), self.index.ntotal)
        while True:
            scores, positions = self.index.search(vector, fetch)
            best = {}
            for score, position in zip(scores[0], positions[0]):
                if position < 0:
                    continue
                doc = int(self.doc_index[position])
                if (mask is None or mask[doc]) and doc not in best:
                    best[doc] = (float(score), int(position))  # Results are sorted, so first is max.
            if len(best) >= top_k or fetch == self.index.ntotal:
                break
            fetch = min(fetch * 4, self.index.ntotal)
        ranked = sorted(
            best.items(), key=lambda item: (-item[1][0], self.doc_ids[item[0]])
        )
        return [(d, score, position) for d, (score, position) in ranked[:top_k]]

    def search(self, request: SearchRequest):
        """Return the best exact title/content passage of each authorized document."""
        if self.corpus is None or self.spans is None:
            raise ValueError("Load embeddings with corpus_path to return evidence")
        if request.user.user_id is None or not request.user.allowed_doc_ids:
            return ()
        # The upstream corpus has no timestamp column; unknown dates fail date filters.
        if request.after or request.before:
            return ()
        mask = np.zeros(len(self.doc_ids), dtype=bool)
        for doc in request.user.allowed_doc_ids:
            if doc in self.position:
                mask[self.position[doc]] = True
        if request.source_types:
            mask &= np.isin(self.corpus.column("source_type").to_pylist(), request.source_types)
        if not mask.any():
            return ()
        vector = self.model.encode([request.query], query=True)[0]
        results = []
        for doc, score, position in self._rank_passages(vector, request.top_k, mask):
            # Avoid Arrow's 32-bit string-offset overflow on the full corpus.
            row = self.corpus.slice(doc, 1).to_pylist()[0]
            text = document_text(row["title"], row["content"])
            start, end = map(int, self.spans[position])
            if not 0 <= start < end <= len(text):
                raise ValueError("Passage offsets outside document text")
            chunk = Chunk(chunk_id=f"{row['doc_id']}:{start}:{end}", doc_id=row["doc_id"],
                          source_type=row["source_type"], title=row["title"],
                          text=text[start:end], start_char=start, end_char=end)
            results.append(Evidence(citation_id=chunk.chunk_id, chunk=chunk,
                                    score=score, retrieval_method="dense"))
        results = tuple(results)
        validate_evidence(request.user, results)
        return results

    @classmethod
    def load(cls, directory: Path, model=None, corpus_path: Path | None = None, *, corpus=None):
        """Load exact FAISS search one shard at a time; optionally reuse a corpus table.

        Only one writable float32 shard exists outside FAISS at a time. Passing
        corpus_path still verifies the index hash when corpus is shared.
        """
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("version") != INDEX_VERSION or not manifest.get("complete"):
            raise ValueError("Embedding directory is incomplete or unsupported")
        if (
            corpus_path is not None
            and file_sha256(corpus_path) != manifest["corpus_sha256"]
        ):
            raise ValueError("Corpus differs from the one used for the embeddings")
        if model is not None and model.identity != manifest["model"]:
            raise ValueError("Query model settings differ from the embeddings")
        names = [f"shard_{i:05d}" for i in range(manifest["shards"])]
        index = None
        doc_parts, span_parts = [], []
        for name in names:
            # A private writable float32 copy is required for normalization;
            # mmap avoids retaining all original fp16 shards in memory.
            stored = np.load(directory / f"{name}.npy", mmap_mode="r")
            if stored.ndim != 2:
                raise ValueError("Embedding shards are inconsistent")
            table = pq.read_table(directory / f"{name}.parquet")
            if len(stored) != len(table):
                raise ValueError("Embedding shards are inconsistent")
            if index is None:
                index = faiss.IndexFlatIP(stored.shape[1])
            elif stored.shape[1] != index.d:
                raise ValueError("Embedding shard dimensions differ")
            vectors = np.array(stored, dtype=np.float32, order="C", copy=True)
            faiss.normalize_L2(vectors)  # Undo fp16 rounding of vector norms.
            index.add(vectors)
            del vectors, stored
            doc_parts.append(table.column("doc_index").to_numpy())
            span_parts.append(np.column_stack((table.column("start_char").to_numpy(),
                                              table.column("end_char").to_numpy())))
        if index is None:
            raise ValueError("Embedding directory has no shards")
        doc_index = np.concatenate(doc_parts)
        spans = np.concatenate(span_parts)
        doc_ids = json.loads((directory / "doc_ids.json").read_text())
        if len(doc_ids) != manifest["documents"]:
            raise ValueError("Embedding shards are inconsistent")
        if corpus is None and corpus_path is not None:
            corpus = pq.read_table(corpus_path, columns=["doc_id", "source_type", "title", "content"])
        return cls(None, doc_index, doc_ids, model, manifest, corpus=corpus,
                   spans=spans, index=index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_DIR / "documents.parquet")
    parser.add_argument("--output", type=Path, default=CORPUS_DIR / "indexes" / "dense")
    parser.add_argument(
        "--model", choices=sorted(EMBEDDING_MODELS), default="bge-small",
        help="Embedding model (use a new --output for each model)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--precision", choices=["auto", "fp16", "bf16", "fp32"], default="auto",
                        help="auto: fp16 on GPU, fp32 on CPU; bf16 needs an Ampere+ GPU (e.g. L4)")
    parser.add_argument("--fp32", action="store_true", help="Same as --precision fp32")
    parser.add_argument("--shard-docs", type=int, default=20000)
    parser.add_argument(
        "--title-every-passage",
        action="store_true",
        help="Prefix each document's title to every passage (use a new --output)",
    )
    args = parser.parse_args()
    precision = "fp32" if args.fp32 else args.precision
    model = EmbeddingModel(
        args.device,
        fp16={"auto": None, "fp16": True}.get(precision, False),
        batch_size=args.batch_size,
        model=args.model,
        bf16=precision == "bf16",
    )
    started = time.perf_counter()
    manifest = embed_corpus(
        args.corpus,
        args.output,
        model,
        args.shard_docs,
        title_every=args.title_every_passage,
    )
    print(
        json.dumps(
            {**manifest, "seconds": round(time.perf_counter() - started)}, indent=2
        )
    )


if __name__ == "__main__":
    main()

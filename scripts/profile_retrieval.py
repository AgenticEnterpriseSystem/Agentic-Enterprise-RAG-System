"""Measure the existing full-corpus retriever without generation or index changes.

Each invocation supervises one fresh worker. Outputs are exclusive, resumable
evidence of that attempt, including failures. File caches are never dropped:
"first" means first in this process, not a guaranteed cold filesystem.
"""

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.resource_monitor import EVENT_PREFIX, sample_memory, supervise


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_identity(root=ROOT, declared_commit=None):
    """Hash actual runtime source; externally supplied Git IDs stay labeled."""
    if declared_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", declared_commit):
        raise ValueError("Source commit must be a full lowercase Git SHA")
    files = sorted((*root.glob("src/**/*.py"), root / "scripts/profile_retrieval.py",
                    root / "scripts/benchmark_web.py", root / "requirements.txt",
                    root / "requirements.lock.txt"))
    hashes = {str(path.relative_to(root)): file_digest(path) for path in files if path.is_file()}
    actual_commit = None
    dirty = None
    try:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        pass
    if actual_commit and declared_commit and actual_commit != declared_commit:
        raise ValueError("Declared source commit differs from the checkout")
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"commit": actual_commit or declared_commit,
            "commit_verification": "git" if actual_commit else "external_declaration",
            "dirty": dirty, "runtime_files": hashes,
            "runtime_source_sha256": hashlib.sha256(encoded).hexdigest()}


def emit(event, **fields):
    print(EVENT_PREFIX + json.dumps({"event": event, "recorded_at": datetime.now(timezone.utc).isoformat(),
                                     **fields}, allow_nan=False), flush=True)


@contextmanager
def phase(name, **fields):
    began = time.monotonic()
    emit("start", phase=name, memory=sample_memory(os.getpid()), **fields)
    try:
        yield
    except BaseException as error:
        emit("end", phase=name, elapsed_seconds=time.monotonic() - began,
             success=False, error_type=type(error).__name__,
             memory=sample_memory(os.getpid()))
        raise
    else:
        emit("end", phase=name, elapsed_seconds=time.monotonic() - began,
             success=True, memory=sample_memory(os.getpid()))


def inspect_artifacts(corpus, bm25, dense, *, hash_payloads=False):
    """Inspect every shard header/footer; calculate payload separately from RSS."""
    import numpy as np
    import pyarrow.parquet as pq

    corpus, bm25, dense = map(Path, (corpus, bm25, dense))
    bm = json.loads((bm25 / "manifest.json").read_text())
    dm = json.loads((dense / "manifest.json").read_text())
    if bm.get("version") != 1 or dm.get("version") != 1 or dm.get("complete") is not True:
        raise ValueError("Complete supported BM25/dense manifests are required")
    count = dm.get("shards")
    if type(count) is not int or count < 1:
        raise ValueError("Dense shard count must be a positive integer")
    documents = pq.ParquetFile(corpus).metadata.num_rows
    if documents != dm["documents"] or documents != bm["stats"]["documents"]:
        raise ValueError("Corpus and index document counts differ")
    if bm["corpus_sha256"] != dm["corpus_sha256"]:
        raise ValueError("Indexes identify different corpus files")
    expected = {f"shard_{i:05d}{suffix}" for i in range(count) for suffix in (".npy", ".parquet")}
    actual = {p.name for p in dense.glob("shard_*") if p.suffix in {".npy", ".parquet"}}
    if actual != expected:
        raise ValueError("Missing or unexpected dense shards")
    rows, dimensions, shards = 0, None, []
    for i in range(count):
        name = f"shard_{i:05d}"
        path = dense / f"{name}.npy"
        stored = np.load(path, mmap_mode="r", allow_pickle=False)
        try:
            if stored.ndim != 2 or stored.dtype.kind != "f" or stored.shape[1] < 1:
                raise ValueError("Invalid dense vector header")
            if dimensions is not None and stored.shape[1] != dimensions:
                raise ValueError("Dense shard dimensions differ")
            dimensions = int(stored.shape[1])
            n = int(stored.shape[0])
            metadata = dense / f"{name}.parquet"
            if pq.ParquetFile(metadata).metadata.num_rows != n:
                raise ValueError("Vector and metadata shard rows differ")
            rows += n
            shards.append({"name": name, "rows": n, "dimension": dimensions,
                           "stored_dtype": str(stored.dtype),
                           "calculated_float32_bytes": n * dimensions * 4})
        finally:
            stored._mmap.close()
    files = [corpus, *sorted(bm25.iterdir()), *sorted(dense.iterdir())]
    inventory = []
    for path in files:
        if path.is_file():
            item = {"path": str(path), "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                    "sha256": file_digest(path) if hash_payloads else None}
            inventory.append(item)
    if hash_payloads:
        corpus_hash = next(item["sha256"] for item in inventory if item["path"] == str(corpus))
        if corpus_hash != bm["corpus_sha256"]:
            raise ValueError("Actual corpus hash differs from the index manifests")
    return {"documents": documents, "passages": rows, "dimension": dimensions,
            "shard_count": count, "model": dm["model"],
            "corpus_sha256": bm["corpus_sha256"],
            "payload_hashes_verified": hash_payloads, "files": inventory, "shards": shards,
            "calculated_float32_vector_bytes": rows * dimensions * 4,
            "largest_calculated_conversion_bytes": max(s["calculated_float32_bytes"] for s in shards)}


@contextmanager
def instrument_loaders(settings):
    """Observe original calls; do not substitute a ranker, model or corpus."""
    with phase("imports.bm25"):
        import bm25s
        import numpy as np
        import pyarrow.parquet as pq
        from src.retrieval import corpus_bm25, factory
    with ExitStack() as stack:
        def patch(obj, name, replacement):
            original = inspect.getattr_static(obj, name)
            setattr(obj, name, replacement)
            stack.callback(setattr, obj, name, original)

        original_read = pq.read_table
        original_hash = corpus_bm25.file_sha256
        original_bm25 = bm25s.BM25.load
        original_dense = factory.load_dense
        original_array = np.array
        state = {"shard": None}

        def read_table(path, *args, **kwargs):
            name = "corpus.arrow" if Path(path) == settings.corpus else f"dense.metadata.{Path(path).stem}"
            with phase(name):
                return original_read(path, *args, **kwargs)

        def hash_file(path):
            with phase("corpus.hash_verification"):
                return original_hash(path)

        def bm25_load(_cls, *args, **kwargs):
            with phase("bm25.index_mmap"):
                return original_bm25(*args, **kwargs)

        def array(source, *args, **kwargs):
            if isinstance(source, np.memmap) and Path(source.filename).parent == settings.dense_index:
                with phase(f"dense.convert.{Path(source.filename).stem}",
                           calculated_float32_bytes=int(source.size) * 4):
                    return original_array(source, *args, **kwargs)
            return original_array(source, *args, **kwargs)

        patch(pq, "read_table", read_table)
        patch(corpus_bm25, "file_sha256", hash_file)
        patch(bm25s.BM25, "load", classmethod(bm25_load))
        patch(np, "array", array)

        def load_dense(*args, **kwargs):
            with phase("imports.dense"):
                import faiss
                from src.retrieval import corpus_dense
            original_model = corpus_dense.BGEModel.for_index
            original_add = faiss.IndexFlatIP.add
            original_normalize = faiss.normalize_L2
            original_np_load = np.load

            def model_load(_cls, *model_args, **model_kwargs):
                with phase("model.load"):
                    model = original_model(*model_args, **model_kwargs)
                    emit("model_identity", identity=model.identity)
                    return model

            def np_load(path, *load_args, **load_kwargs):
                if isinstance(path, (str, os.PathLike)) and Path(path).parent == settings.dense_index:
                    state["shard"] = Path(path).stem
                    with phase(f"dense.map.{state['shard']}"):
                        return original_np_load(path, *load_args, **load_kwargs)
                return original_np_load(path, *load_args, **load_kwargs)

            def normalize(vectors):
                with phase(f"dense.normalize.{state['shard']}", rows=len(vectors)):
                    return original_normalize(vectors)

            def add(index, vectors, *add_args, **add_kwargs):
                with phase(f"dense.add.{state['shard']}", rows=len(vectors),
                           index_rows_before=int(index.ntotal)):
                    result = original_add(index, vectors, *add_args, **add_kwargs)
                    emit("index_growth", shard=state["shard"], index_rows=int(index.ntotal),
                         memory=sample_memory(os.getpid()))
                    return result

            patch(corpus_dense, "file_sha256", hash_file)
            patch(corpus_dense.BGEModel, "for_index", classmethod(model_load))
            patch(np, "load", np_load)
            patch(faiss, "normalize_L2", normalize)
            patch(faiss.IndexFlatIP, "add", add)
            with phase("dense.load"):
                return original_dense(*args, **kwargs)

        patch(factory, "load_dense", load_dense)
        yield factory


def query_workloads(retriever, *, repeats=3, top_k=5, narrow_size=32):
    from scripts.benchmark_web import WORKLOADS
    from src.contracts import SearchRequest, UserContext, validate_evidence

    with phase("scope.construct"):
        broad = frozenset(retriever.doc_ids)
        narrow = frozenset(sorted(broad)[:narrow_size])
        scopes = (("broad", broad), ("narrow", narrow), ("denied", frozenset()))
    for scope_name, ids in scopes:
        user = UserContext(user_id="resource-benchmark", allowed_doc_ids=ids)
        scope_hash = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
        emit("scope", name=scope_name, document_count=len(ids), sha256=scope_hash,
             purpose="synthetic_benchmark_only_not_application_login")
        for repeat in range(repeats):
            for name, question in WORKLOADS:
                request = SearchRequest(query=question, user=user, top_k=top_k)
                label = f"query.{scope_name}.{name}.{repeat}"
                with phase(label, cache_state="first_in_scope" if repeat == 0 else "warm_repeat"):
                    began = time.monotonic()
                    hits = tuple(retriever.search(request))
                    elapsed = time.monotonic() - began
                    validate_evidence(user, hits)
                    if len(hits) > top_k:
                        raise ValueError("Retriever exceeded the requested result bound")
                    emit("query_result", scope=scope_name, workload=name, repeat=repeat,
                         elapsed_seconds=elapsed, result_count=len(hits),
                         hits=[{"doc_id": item.chunk.doc_id, "citation_id": item.citation_id,
                                "start_char": item.chunk.start_char, "end_char": item.chunk.end_char,
                                "score": item.score} for item in hits])
                if repeat == 0 and hits and hasattr(retriever, "read_more"):
                    with phase(f"read_more.{scope_name}.{name}"):
                        more = tuple(retriever.read_more(request, hits[:2], max_chars=3000))
                        validate_evidence(user, more)
                        anchors = {item.chunk.doc_id for item in hits[:2]}
                        if (len(more) > 2 or any(len(item.chunk.text) > 3000
                                                or item.chunk.doc_id not in anchors for item in more)):
                            raise ValueError("Expansion exceeded its result bounds")
                if repeat == 0 and hits:
                    with phase(f"source_filter.{scope_name}.{name}"):
                        filtered = tuple(retriever.search(request.model_copy(
                            update={"source_types": (hits[0].chunk.source_type,)})))
                        validate_evidence(user, filtered)
                        if (len(filtered) > top_k or any(
                                item.chunk.source_type != hits[0].chunk.source_type for item in filtered)):
                            raise ValueError("Source filtering widened scope")


def worker(args):
    from src.config import RetrievalSettings, runtime_versions

    emit("environment", versions=runtime_versions(), cpu_threads=args.threads,
         cache_protocol="No OS cache eviction; loader hash verification reads the corpus. First query is process-first only.")
    if args.mode == "inventory":
        with phase("inventory.full_hashes"):
            inventory = inspect_artifacts(args.corpus, args.bm25_index, args.dense_index, hash_payloads=True)
            emit("artifact_inventory", **inventory)
    elif args.mode == "corpus":
        with phase("imports.arrow"):
            import pyarrow.parquet as pq
        with phase("corpus.arrow"):
            corpus = pq.read_table(args.corpus, columns=["doc_id", "source_type", "title", "content"])
            emit("corpus_loaded", documents=len(corpus), arrow_bytes=corpus.nbytes)
    elif args.mode == "model":
        with phase("imports.model"):
            from src.retrieval.corpus_dense import BGEModel
        with phase("model.load"):
            model = BGEModel.for_index(args.dense_index, device="cpu")
            emit("model_identity", identity=model.identity)
        from scripts.benchmark_web import WORKLOADS
        for repeat in range(args.repeats):
            for name, question in WORKLOADS:
                with phase(f"encode.{name}.{repeat}"):
                    vectors = model.encode([question], query=True)
                    emit("encoding_result", workload=name, repeat=repeat, shape=list(vectors.shape))
    else:
        settings = RetrievalSettings(retrieval=args.mode, corpus=args.corpus,
            bm25_index=args.bm25_index, dense_index=args.dense_index, device="cpu")
        with instrument_loaders(settings) as factory:
            with phase("retriever.startup"):
                retriever = factory.load_retriever(settings)
                dense = getattr(retriever, "dense", None)
                emit("ready", documents=len(retriever.doc_ids),
                     passages=int(dense.index.ntotal) if dense is not None else None,
                     shared_corpus=(retriever.bm25.corpus is dense.corpus) if dense is not None else None)
            query_workloads(retriever, repeats=args.repeats, top_k=args.top_k)
    emit("complete", mode=args.mode)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("inventory", "corpus", "model", "bm25", "hybrid"), required=True)
    result.add_argument("--corpus", type=Path, required=True)
    result.add_argument("--bm25-index", type=Path, required=True)
    result.add_argument("--dense-index", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--source-commit")
    result.add_argument("--threads", type=int, default=2)
    result.add_argument("--repeats", type=int, default=3)
    result.add_argument("--top-k", type=int, default=5)
    result.add_argument("--max-rss-gib", type=float, default=5.5)
    result.add_argument("--max-cgroup-gib", type=float, default=5.5)
    result.add_argument("--min-vm-available-mib", type=float, default=256)
    result.add_argument("--timeout", type=float, default=300)
    result.add_argument("--sample-seconds", type=float, default=.2)
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main():
    args = parser().parse_args()
    limits = (args.max_rss_gib, args.max_cgroup_gib, args.timeout, args.sample_seconds)
    if (args.threads < 1 or args.repeats < 1 or not 1 <= args.top_k <= 100
            or not all(math.isfinite(value) and value > 0 for value in limits)
            or not math.isfinite(args.min_vm_available_mib) or args.min_vm_available_mib < 0):
        raise SystemExit("Thread/repeat/limit settings must be positive and top-k must be 1..100")
    for name in ("corpus", "bm25_index", "dense_index", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        try:
            worker(args)
        except Exception as error:
            emit("worker_error", error_type=type(error).__name__, message=str(error))
            raise
        return
    identity = code_identity(declared_commit=args.source_commit)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    overrides = {name: str(args.threads) for name in
                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ARROW_NUM_THREADS")}
    overrides.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                      "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
                      "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                      "NUMBA_CACHE_DIR": "/tmp/rag-profile-numba"})
    environment.update(overrides)
    manifest = {"protocol": 1, "mode": args.mode, "source": identity,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "settings": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items() if key != "worker"},
                "environment_overrides": overrides,
                "cache_state": "uncontrolled; no eviction; first in process is not cold disk",
                "baseline": "existing factory/loader and manifest-selected CPU model; no algorithm changes"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    summary = supervise(command, args.output, max_rss_bytes=int(args.max_rss_gib * 1024**3),
                        max_cgroup_bytes=int(args.max_cgroup_gib * 1024**3),
                        min_vm_available_bytes=int(args.min_vm_available_mib * 1024**2),
                        max_seconds=args.timeout, sample_seconds=args.sample_seconds, env=environment)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "completed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

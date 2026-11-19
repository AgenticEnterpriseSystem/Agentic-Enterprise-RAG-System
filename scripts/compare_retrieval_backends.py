"""Offline development retrieval parity in separate, supervised processes.

encode freezes queries/vectors; reference uses independent bounded exhaustive
FAISS; candidate measures the real scoped PostgreSQL/sharded path with a resident
manifest-matched encoder; compare checks immutable artifacts and source text.
No generator/provider is imported or called. Imports for --help are stdlib only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

EVENT_PREFIX = "PROFILE_EVENT "
ATOL = RTOL = 2e-6
DEV_IDS = ("qst_0001", "qst_0431", "qst_0412", "qst_0384", "qst_0471", "qst_0482",
           "qst_0301", "qst_0451", "qst_0342", "qst_0176")


class ParityError(ValueError):
    pass


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ParityError("Duplicate JSON fields")
        result[key] = value
    return result


def _invalid(_value):
    raise ParityError("Non-finite JSON values")


def _json(path):
    try:
        return json.loads(Path(path).read_bytes(), object_pairs_hook=_unique, parse_constant=_invalid)
    except (ValueError, UnicodeError):
        raise ParityError("Invalid or partial parity JSON") from None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
            try:
                os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass
    return digest.hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ParityError(name + " must be a positive integer")


def emit(event, phase=None, **values):
    print(EVENT_PREFIX + json.dumps({"event": event, **({"phase": phase} if phase else {}), **values}, allow_nan=False), flush=True)


@contextmanager
def phase(name, **metadata):
    emit("start", name, **metadata)
    try:
        yield
    finally:
        emit("end", name, **metadata)


@contextmanager
def _timed(timings, name, *, events=False):
    """Accumulate inclusive component time; nested totals must not be summed."""
    if timings is None:
        yield
        return
    begin = time.perf_counter()
    if events:
        emit("start", name)
    try:
        yield
    finally:
        timings[name + "_seconds"] = timings.get(name + "_seconds", 0.0) + time.perf_counter() - begin
        timings[name + "_calls"] = timings.get(name + "_calls", 0) + 1
        if events:
            emit("end", name)


@dataclass(frozen=True)
class RankHit:
    doc_id: str
    score: float
    start_char: int
    end_char: int
    tie_count: int = 1
    legacy_start_char: int | None = None
    legacy_end_char: int | None = None

    def __post_init__(self):
        if (not isinstance(self.doc_id, str) or not self.doc_id.strip()
                or isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score)
                or type(self.start_char) is not int or type(self.end_char) is not int or not 0 <= self.start_char < self.end_char
                or type(self.tie_count) is not int or self.tie_count < 1):
            raise ParityError("Invalid finite ranked passage")
        if ((self.legacy_start_char is None) != (self.legacy_end_char is None)
                or (self.legacy_start_char is not None and (type(self.legacy_start_char) is not int
                    or type(self.legacy_end_char) is not int or not 0 <= self.legacy_start_char < self.legacy_end_char))):
            raise ParityError("Invalid legacy tie representative")
        object.__setattr__(self, "score", float(self.score))

    def as_dict(self):
        return asdict(self)


def _hit(value):
    try:
        return RankHit(**(asdict(value) if isinstance(value, RankHit) else value))
    except (TypeError, ValueError):
        raise ParityError("Invalid ranked hit boundary") from None


@dataclass(frozen=True)
class ReferenceResult:
    rankings: dict
    documents: int
    passages: int
    shards: int
    dimension: int


def exhaustive_faiss(directory, vectors, *, snapshot, catalog, masks, top_k=5, block_rows=8192) -> ReferenceResult:
    """All-row native FAISS oracle with globally matched bounded dispatch.

    The independent helper reproduces full-index thread partitions/BLAS tiles;
    constructing arbitrary small FlatIP indexes changes the native score kernel
    at10000 database rows and is deliberately not used as reference truth.
    Retained integer passage metadata is O(passages); scores/spans are
    O(queries*documents). Vector reads and native score tiles stay bounded.
    """
    import numpy as np
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq
    from src.contracts import SnapshotRef
    from src.storage.contracts import CatalogEntry
    from src.storage.fingerprints import catalog_fingerprint
    from src.retrieval.faiss_reference import iter_scores

    _positive(top_k, "top_k")
    _positive(block_rows, "block_rows")
    snapshot = SnapshotRef.model_validate(snapshot.model_dump())
    doc_ids, lengths = [], []
    def entries():
        for supplied in catalog:
            entry = CatalogEntry.model_validate(supplied.model_dump())
            doc_ids.append(entry.doc_id)
            lengths.append(entry.text_length)
            yield entry
    if catalog_fingerprint(entries(), expected_count=snapshot.document_count) != snapshot.catalog_sha256:
        raise ParityError("Reference catalogue differs from the snapshot")
    if len(set(doc_ids)) != len(doc_ids):
        raise ParityError("Reference document IDs are duplicated")
    directory = Path(directory)
    manifest, ids = _json(directory / "manifest.json"), _json(directory / "doc_ids.json")
    if (type(manifest.get("version")) is not int or manifest["version"] != 1 or manifest.get("complete") is not True
            or type(manifest.get("documents")) is not int or manifest["documents"] != len(doc_ids)
            or manifest.get("corpus_sha256") != snapshot.corpus_sha256 or ids != doc_ids):
        raise ParityError("Reference index identity is incomplete or mismatched")
    _positive(manifest.get("shards"), "shards")
    files = {path.name for path in directory.iterdir() if path.name.startswith("shard_")}
    if len(files) != 2 * manifest["shards"]:
        raise ParityError("Reference shard count differs")
    names = tuple(f"shard_{index:05d}" for index in range(manifest["shards"]))
    if files != {name + suffix for name in names for suffix in (".npy", ".parquet")}:
        raise ParityError("Reference shards are missing or unexpected")
    queries = np.asarray(vectors)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2 or queries.dtype.kind not in "fiu":
        raise ParityError("Reference vectors must be a real query matrix")
    queries = np.array(queries, dtype=np.float32, order="C", copy=True)
    if not len(queries) or not np.isfinite(queries).all():
        raise ParityError("Reference query vectors must be finite")
    checked_masks = {}
    for name, mask in masks.items():
        mask = np.asarray(mask)
        if mask.dtype != np.dtype(bool) or mask.shape != (len(doc_ids),):
            raise ParityError("Reference masks must be exact boolean document masks")
        checked_masks[name] = mask.copy()
    shard_rows, offsets, dimension, dtype = [], [0], None, None
    for name in names:
        stored = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
        try:
            if (not isinstance(stored, np.memmap) or stored.ndim != 2 or stored.shape[1] != queries.shape[1]
                    or stored.dtype not in (np.dtype("float16"), np.dtype("float32")) or not stored.flags.c_contiguous):
                raise ParityError("Reference vector shape/dtype differs")
            if dimension is not None and (stored.shape[1] != dimension or stored.dtype != dtype):
                raise ParityError("Reference shards have inconsistent dimensions/dtypes")
            dimension, dtype = stored.shape[1], stored.dtype
            shard_rows.append(len(stored))
            offsets.append(offsets[-1] + len(stored))
        finally:
            if isinstance(stored, np.memmap):
                stored._mmap.close()
            elif hasattr(stored, "close"):
                stored.close()
    total = offsets[-1]
    docs, starts, ends = (np.empty(total, dtype=np.int64) for _ in range(3))
    lengths = np.asarray(lengths, dtype=np.int64)
    for shard, name in enumerate(names):
        parquet = pq.ParquetFile(directory / (name + ".parquet"), memory_map=False, pre_buffer=False)
        try:
            if (parquet.metadata.num_rows != shard_rows[shard] or tuple(parquet.schema_arrow.names) != ("doc_index", "start_char", "end_char")
                    or any(not pa.types.is_signed_integer(field.type) for field in parquet.schema_arrow)):
                raise ParityError("Reference span schema/row count differs")
            position = offsets[shard]
            for batch in parquet.iter_batches(batch_size=block_rows, use_threads=False):
                if any(column.null_count for column in batch.columns):
                    raise ParityError("Reference span metadata contains nulls")
                stop = position + len(batch)
                for array, column in zip((docs, starts, ends), batch.columns):
                    array[position:stop] = column.to_numpy()
                position = stop
            if position != offsets[shard + 1]:
                raise ParityError("Reference span coverage differs")
        finally:
            parquet.close()
    if (np.any(docs < 0) or np.any(docs >= len(doc_ids)) or np.any(starts < 0)
            or np.any(ends <= starts) or np.any(ends > lengths[docs])):
        raise ParityError("Reference spans exceed canonical metadata")
    def read_rows(start, stop):
        result = np.empty((stop - start, dimension), dtype=np.float32, order="C")
        copied = 0
        for shard, name in enumerate(names):
            first, last = max(start, offsets[shard]), min(stop, offsets[shard + 1])
            if first >= last:
                continue
            stored = np.load(directory / (name + ".npy"), mmap_mode="r", allow_pickle=False)
            try:
                result[copied:copied + last - first] = stored[first - offsets[shard]:last - offsets[shard]]
            finally:
                stored._mmap.close()
            copied += last - first
        if copied != len(result) or not np.isfinite(result).all():
            raise ParityError("Reference vector source is incomplete/nonfinite")
        faiss.normalize_L2(result)
        return result
    shape = (len(queries), len(doc_ids))
    maximum = np.full(shape, -np.inf, dtype=np.float32)
    sentinel = np.iinfo(np.int64).max
    first_start, first_end = np.full(shape, sentinel, np.int64), np.full(shape, sentinel, np.int64)
    legacy_start, legacy_end = np.full(shape, -1, np.int64), np.full(shape, -1, np.int64)
    legacy_ordinal = np.full(shape, -1, np.int64)
    tie_count = np.zeros(shape, np.int64)
    cursor = 0
    from contextlib import closing
    with closing(iter_scores(queries, total_vectors=total, dimension=dimension, read_rows=read_rows)) as blocks:
        for start, scores in blocks:
            stop = start + scores.shape[1]
            if start != cursor or scores.shape[0] != len(queries):
                raise ParityError("Reference native score coverage differs")
            local_docs, local_starts, local_ends = docs[start:stop], starts[start:stop], ends[start:stop]
            touched = np.unique(local_docs)
            for query_index, values in enumerate(scores):
                previous = maximum[query_index, touched].copy()
                np.maximum.at(maximum[query_index], local_docs, values)
                improved = touched[maximum[query_index, touched] > previous]
                first_start[query_index, improved] = first_end[query_index, improved] = sentinel
                legacy_ordinal[query_index, improved] = -1
                tie_count[query_index, improved] = 0
                tied = values == maximum[query_index, local_docs]
                np.add.at(tie_count[query_index], local_docs[tied], 1)
                previous_starts = first_start[query_index, touched].copy()
                np.minimum.at(first_start[query_index], local_docs[tied], local_starts[tied])
                first_end[query_index, touched[first_start[query_index, touched] < previous_starts]] = sentinel
                tied_first = tied & (local_starts == first_start[query_index, local_docs])
                np.minimum.at(first_end[query_index], local_docs[tied_first], local_ends[tied_first])
                ordinals = np.arange(start, stop, dtype=np.int64)
                np.maximum.at(legacy_ordinal[query_index], local_docs[tied], ordinals[tied])
                latest = tied & (ordinals == legacy_ordinal[query_index, local_docs])
                legacy_start[query_index, local_docs[latest]] = local_starts[latest]
                legacy_end[query_index, local_docs[latest]] = local_ends[latest]
            cursor = stop
    if cursor != total:
        raise ParityError("Reference native scoring omitted passage rows")
    rankings = {}
    for name, mask in checked_masks.items():
        per_query = []
        for query_index in range(len(queries)):
            positions = heapq.nsmallest(top_k, np.flatnonzero(mask & np.isfinite(maximum[query_index])),
                key=lambda position: (-float(maximum[query_index, position]), doc_ids[position]))
            per_query.append(tuple(RankHit(doc_ids[position], float(maximum[query_index, position]),
                int(first_start[query_index, position]), int(first_end[query_index, position]),
                int(tie_count[query_index, position]), int(legacy_start[query_index, position]),
                int(legacy_end[query_index, position])) for position in positions))
        rankings[name] = tuple(per_query)
    return ReferenceResult(rankings, len(doc_ids), total, len(names), dimension)


def compare_hits(reference, candidate, *, atol=ATOL, rtol=RTOL):
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 for value in (atol, rtol)):
        raise ParityError("Rank tolerances must be finite and nonnegative")
    left, right = tuple(map(_hit, reference)), tuple(map(_hit, candidate))
    if len({hit.doc_id for hit in left}) != len(left) or len({hit.doc_id for hit in right}) != len(right):
        raise ParityError("Ranked documents must be unique")
    differences, passed, strict = [], True, True
    for rank in range(max(len(left), len(right))):
        expected, actual = left[rank] if rank < len(left) else None, right[rank] if rank < len(right) else None
        common = {"rank": rank + 1, "reference": expected.as_dict() if expected else None,
                  "candidate": actual.as_dict() if actual else None}
        if expected is None or actual is None or expected.doc_id != actual.doc_id:
            differences.append({"kind": "rank_order", **common})
            passed, strict = False, False
            continue
        delta = actual.score - expected.score
        within = abs(delta) <= atol + rtol * abs(expected.score)
        if delta:
            differences.append({"kind": "score", **common, "delta": delta, "within_tolerance": within})
            strict = False
            passed &= within
        if (expected.start_char, expected.end_char) != (actual.start_char, actual.end_char):
            genuine = expected.tie_count > 1 and (actual.start_char, actual.end_char) == (expected.legacy_start_char, expected.legacy_end_char)
            differences.append({"kind": "span", **common, "genuine_exact_tie": genuine})
            passed, strict = False, False
    return {"strict_equal": strict, "passed": bool(passed), "differences": differences}


def verify_source_evidence(corpus, evidence, *, snapshot, catalog):
    """Check each selected exact excerpt against original canonical source rows."""
    import pyarrow.parquet as pq
    from src.contracts import PassageEvidence
    from src.storage.contracts import CatalogEntry
    passages = tuple(PassageEvidence.model_validate(value.model_dump() if isinstance(value, PassageEvidence) else value) for value in evidence)
    entries = {entry.doc_id: CatalogEntry.model_validate(entry.model_dump()) for entry in catalog}
    wanted = {}
    for passage in passages:
        entry = entries.get(passage.chunk.doc_id)
        if entry is None or passage.snapshot_id != snapshot.snapshot_id or passage.revision_id != entry.revision_id:
            raise ParityError("Evidence has no matching source catalogue revision")
        wanted.setdefault(entry.position, []).append(passage)
    parquet = pq.ParquetFile(corpus, memory_map=False, pre_buffer=False)
    checked, offset = 0, 0
    try:
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            if any(offset <= position < offset + count for position in wanted):
                local = offset
                for batch in parquet.iter_batches(batch_size=64, row_groups=[group],
                        columns=["doc_id", "source_type", "title", "content"], use_threads=False):
                    for row in batch.to_pylist():
                        for passage in wanted.get(local, ()):
                            entry, chunk = entries[passage.chunk.doc_id], passage.chunk
                            text = row["title"] + "\n" + row["content"] if row["title"] else row["content"]
                            if (row["doc_id"] != entry.doc_id or row["source_type"] != entry.source_type
                                    or chunk.source_type != entry.source_type or chunk.timestamp != entry.timestamp
                                    or chunk.metadata_origin != entry.metadata_origin or chunk.chunk_id != passage.citation_id
                                    or row["title"] != chunk.title or _sha(text.encode("utf-8")) != entry.revision_id
                                    or len(text) != entry.text_length or chunk.end_char > len(text)
                                    or text[chunk.start_char:chunk.end_char] != chunk.text
                                    or passage.citation_id != f"{entry.doc_id}:{chunk.start_char}:{chunk.end_char}"):
                                raise ParityError("Evidence differs from exact canonical source/offsets")
                            checked += 1
                        local += 1
            offset += count
    finally:
        parquet.close()
    if checked != len(passages):
        raise ParityError("Source verification missed selected evidence")
    return {"checked": checked}


def latency_gate(startup_seconds, records, *, startup_limit=120, first_limit=30, warm_limit=10):
    limits = (startup_limit, first_limit, warm_limit)
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in limits):
        raise ParityError("Latency limits must be finite and positive")
    if isinstance(startup_seconds, bool) or not isinstance(startup_seconds, (int, float)) or not math.isfinite(startup_seconds) or startup_seconds < 0:
        raise ParityError("Startup timing must be finite and nonnegative")
    violations = []
    if startup_seconds > startup_limit:
        violations.append({"kind": "startup", "seconds": startup_seconds, "limit": startup_limit})
    for request_index, record in enumerate(records):
        elapsed = record["timings"]["total_seconds"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ParityError("Request timings must be finite and nonnegative")
        if type(record["iteration"]) is not int or record["iteration"] < 0:
            raise ParityError("Request iteration must be a nonnegative integer")
        if type(record.get("request_index")) is not int or record["request_index"] != request_index:
            raise ParityError("Request sequence must start at zero and remain contiguous within its process")
        limit = first_limit if request_index == 0 else warm_limit
        if elapsed > limit:
            violations.append({"kind": "first" if request_index == 0 else "warm",
                "question_id": record["question_id"], "case": record["case"], "iteration": record["iteration"],
                "request_index": request_index,
                "seconds": elapsed, "limit": limit})
    return {"passed": not violations, "violations": violations, "limits_seconds": {
        "startup": startup_limit, "first": first_limit, "warm": warm_limit}}


def resource_gate(report):
    """Consume concurrent measured resource evidence, never sum isolated peaks."""
    required = {"complete", "peak_aggregate_bytes", "min_vm_available_bytes", "oom_events_delta", "guard_status",
        "scope", "candidate_manifest_sha256", "measurement_kind", "sample_count", "raw_sampler_sha256", "app_supervisor_sha256"}
    if not isinstance(report, dict) or not required <= report.keys():
        return {"passed": False, "status": "unverified", "violations": ["Missing concurrent resource evidence"]}
    if any(report[field] is None or report[field] == "unknown" for field in
           ("peak_aggregate_bytes", "min_vm_available_bytes", "oom_events_delta")):
        return {"passed": False, "status": "unverified", "violations": ["Resource counters are unknown"], "measurement": report}
    failures = []
    if report["complete"] is not True or report["guard_status"] != "not_triggered":
        failures.append("Resource run did not complete without a guard")
    if report["measurement_kind"] != "sampled_sequential_app_db_cgroup_1s":
        failures.append("Resource measurement does not describe paired app/database samples")
    if type(report["sample_count"]) is not int or report["sample_count"] < 1:
        failures.append("Resource samples are unavailable")
    for field in ("candidate_manifest_sha256", "raw_sampler_sha256", "app_supervisor_sha256"):
        value = report[field]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ParityError("Resource evidence requires actual SHA256 digests")
    for name in ("peak_aggregate_bytes", "min_vm_available_bytes", "oom_events_delta"):
        if type(report[name]) is not int or report[name] < 0:
            raise ParityError("Resource counters must be measured nonnegative integers")
    if report["peak_aggregate_bytes"] > 6.5 * 1024**3:
        failures.append("Concurrent app+database peak exceeded6.5GiB")
    if report["peak_aggregate_bytes"] == 0:
        failures.append("Concurrent memory measurement is unavailable")
    if report["min_vm_available_bytes"] < 256 * 1024**2:
        failures.append("VM available memory fell below256MiB")
    if report["oom_events_delta"]:
        failures.append("Observed OOM events increased")
    return {"passed": not failures, "status": "measured", "violations": failures,
            "measurement": report, "peak_kind": "observed concurrent samples, not an unavailable kernel peak"}


def _new(path, payload):
    with Path(path).open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _output(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700)
    return path


def _write_artifact(path, manifest, records=()):
    records = tuple(records)
    payload = b"".join(_canonical({"case": record, "sha256": _sha(_canonical(record))}) + b"\n" for record in records)
    _new(path / "results.jsonl", payload)
    _new(path / "manifest.json", _canonical({**manifest, "schema_version": "1", "complete": True,
        "records": len(records), "results_sha256": _sha(payload)}) + b"\n")


def load_artifact(path):
    path = Path(path)
    manifest = _json(path / "manifest.json")
    if manifest.get("schema_version") != "1" or manifest.get("complete") is not True:
        raise ParityError("Parity artifact is incomplete or unsupported")
    raw = (path / "results.jsonl").read_bytes()
    if _sha(raw) != manifest.get("results_sha256"):
        raise ParityError("Parity case journal differs from its digest")
    records, keys = [], set()
    for line in raw.splitlines():
        envelope = json.loads(line, object_pairs_hook=_unique, parse_constant=_invalid)
        if not isinstance(envelope, dict) or set(envelope) != {"case", "sha256"} or _sha(_canonical(envelope["case"])) != envelope["sha256"]:
            raise ParityError("Parity case digest differs")
        record = envelope["case"]
        key = (record["question_id"], record["case"], record.get("iteration", 0))
        if key in keys:
            raise ParityError("Duplicate parity cases")
        keys.add(key)
        for field in ("dense_hits", "hybrid_hits"):
            tuple(map(_hit, record.get(field, ())))
        records.append(record)
    if len(records) != manifest.get("records"):
        raise ParityError("Parity case count differs")
    if manifest.get("stage") == "encode":
        import numpy as np
        vectors = np.load(path / "query_vectors.npy", allow_pickle=False)
        questions = manifest["questions"]
        if (_sha(_canonical(manifest["matrix"])) != manifest.get("matrix_sha256")
                or tuple(question["question_id"] for question in questions) != tuple(manifest["matrix"]["question_ids"])
                or any(set(question) != {"question_id", "question", "question_sha256"} for question in questions)
                or file_sha256(path / "query_vectors.npy") != manifest.get("vectors_sha256")
                or vectors.shape != tuple(manifest["vector_shape"]) or len(vectors) != len(questions)
                or vectors.dtype != np.dtype("float32") or not np.isfinite(vectors).all()
                or len({question["question_id"] for question in questions}) != len(questions)
                or any(_sha(question["question"].encode("utf-8")) != question["question_sha256"] for question in questions)):
            raise ParityError("Frozen query vectors or question identities differ")
    return manifest, tuple(records)


def _packages():
    result = {}
    for name in ("numpy", "faiss-cpu", "pyarrow", "torch", "transformers", "sentence-transformers", "bm25s", "SQLAlchemy", "psycopg"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "unknown"
    return result


def _code():
    root = Path(__file__).resolve().parents[1]
    names = ("scripts/compare_retrieval_backends.py", "src/retrieval/sharded_dense.py", "src/retrieval/store.py",
        "src/retrieval/corpus_dense.py", "src/retrieval/corpus_bm25.py", "src/storage/postgres_store.py", "src/access/policy.py",
        "src/retrieval/faiss_reference.py", "src/retrieval/passages.py", "src/retrieval/hybrid.py", "src/retrieval/fusion.py",
        "src/contracts.py", "src/storage/contracts.py", "src/storage/fingerprints.py", "src/access/contracts.py", "src/db/runtime.py")
    return {name: file_sha256(root / name) for name in names}


def _artifact_hashes(corpus, dense, bm25):
    result = {"corpus": file_sha256(corpus)}
    for role, directory in (("dense", dense), ("bm25", bm25)):
        if directory is None:
            raise ParityError("Full hybrid parity requires the BM25 artifact directory")
        directory = Path(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                result[role + "/" + path.relative_to(directory).as_posix()] = file_sha256(path)
    return result


def _snapshot(path):
    from src.contracts import SnapshotRef
    value = _json(path)
    return SnapshotRef.model_validate(value.get("snapshot", value))


def _make_matrix(questions_path, splits_path, dense):
    splits = _json(splits_path)["splits"]
    if not set(DEV_IDS) <= set(splits["development"]):
        raise ParityError("The predeclared ten IDs are not all development questions")
    selected = {}
    with Path(questions_path).open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if value.get("question_id") in DEV_IDS:
                if value["question_id"] in selected:
                    raise ParityError("Duplicate selected development question")
                selected[value["question_id"]] = value
    if set(selected) != set(DEV_IDS):
        raise ParityError("A predeclared development question is missing")
    expected = {doc for question in selected.values() for doc in question.get("expected_doc_ids", ())}
    ids = _json(Path(dense) / "doc_ids.json")
    if len(expected) != 14 or not expected <= set(ids):
        raise ParityError("The frozen development reference-document union differs")
    narrow = sorted(expected | set(sorted(set(ids) - expected)[:18]))
    if len(narrow) != 32:
        raise ParityError("The narrow workload must contain exactly32 frozen documents")
    sources = {"qst_0001": ["github"], "qst_0431": ["confluence"], "qst_0412": ["confluence", "google_drive"],
        "qst_0384": ["confluence"], "qst_0471": [], "qst_0482": [], "qst_0301": ["linear"],
        "qst_0451": ["slack"], "qst_0342": ["confluence", "gmail"], "qst_0176": ["slack"]}
    matrix = {"question_ids": list(DEV_IDS), "narrow_doc_ids": narrow, "sources_by_query": sources,
        "unknown_after": "2020-01-01T00:00:00Z", "candidate_k": 50, "rrf_k": 60,
        "bm25_weight": .7, "dense_weight": 1 - .7, "rank_prefixes": [5, 10, 20], "timed_top_k": 5,
        "warm_repeats": 3, "atol": ATOL, "rtol": RTOL, "reranker": None}
    questions = [{"question_id": key, "question": selected[key]["question"],
                  "question_sha256": _sha(selected[key]["question"].encode("utf-8"))} for key in DEV_IDS]
    return matrix, questions


def _encode(args):
    started = time.perf_counter()
    output = _output(args.output)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    with phase("input_verification"):
        snapshot = _snapshot(args.snapshot)
        matrix, questions = _make_matrix(args.questions, args.splits, args.dense_directory)
        files = _artifact_hashes(args.corpus, args.dense_directory, args.bm25_directory)
        if files["corpus"] != snapshot.corpus_sha256:
            raise ParityError("Actual corpus differs from the snapshot")
    with phase("model_load"):
        from src.retrieval.corpus_dense import EmbeddingModel
        import numpy as np
        model = EmbeddingModel.for_index(Path(args.dense_directory), device="cpu", batch_size=1)
    vectors, timings = [], []
    for question in questions:
        with phase("query_encode", question_id=question["question_id"]):
            begin = time.perf_counter()
            vector = model.encode([question["question"]], query=True)[0]
            timings.append(time.perf_counter() - begin)
            vectors.append(vector)
    vectors = np.asarray(vectors, dtype=np.float32)
    if not np.isfinite(vectors).all():
        raise ParityError("Encoder produced non-finite vectors")
    with (output / "query_vectors.npy").open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    _write_artifact(output, {"stage": "encode", "snapshot": snapshot.model_dump(mode="json"),
        "questions": questions, "matrix": matrix, "matrix_sha256": _sha(_canonical(matrix)),
        "question_file_sha256": file_sha256(args.questions), "splits_sha256": file_sha256(args.splits),
        "artifact_hashes": files, "vectors_sha256": file_sha256(output / "query_vectors.npy"),
        "vector_shape": list(vectors.shape), "model_identity": model.identity, "encode_seconds": timings,
        "elapsed_seconds": time.perf_counter() - started, "packages": _packages(), "source_hashes": _code(),
        "source_revision_attestation": args.source_revision, "offline": True})


def _shared(args):
    import numpy as np
    from src.contracts import SnapshotRef
    shared, _ = load_artifact(args.shared)
    if shared.get("stage") != "encode" or _sha(_canonical(shared["matrix"])) != shared.get("matrix_sha256"):
        raise ParityError("Frozen shared protocol differs")
    if _artifact_hashes(args.corpus, args.dense_directory, args.bm25_directory) != shared["artifact_hashes"]:
        raise ParityError("Actual retrieval files differ from the frozen shared inputs")
    return shared, SnapshotRef.model_validate(shared["snapshot"]), np.load(Path(args.shared) / "query_vectors.npy", allow_pickle=False)


def _runtime(args, snapshot, *, scope, timings=None):
    from src.db.runtime import Database, DatabaseSettings
    from src.access.repository import verify_benchmark
    from src.access.policy import PolicyResolver
    from src.storage.postgres_store import PostgresDocumentStore
    with _timed(timings, "database", events=True):
        database = Database(DatabaseSettings.from_env())
    try:
        with _timed(timings, "identity", events=True):
            identity = verify_benchmark(args.credentials_file, database)
        with _timed(timings, "store", events=True):
            store = PostgresDocumentStore(database, snapshot, retain_catalog=True)
        shared, _ = load_artifact(args.shared)
        restricted = None if scope == "full" else shared["matrix"]["narrow_doc_ids"] if scope == "narrow" else ()
        with _timed(timings, "policy", events=True):
            resolver = PolicyResolver(database, snapshot, doc_ids=restricted, catalog=store.validated_catalog())
        return database, identity, store, resolver
    except BaseException:
        database.close()
        raise


def _filters(shared, question, case):
    from src.storage.contracts import ReadFilters
    if case == "source":
        return ReadFilters(source_types=tuple(shared["matrix"]["sources_by_query"][question["question_id"]]))
    if case == "unknown_date":
        return ReadFilters(after=datetime.fromisoformat(shared["matrix"]["unknown_after"].replace("Z", "+00:00")))
    return ReadFilters()


def _case_name(scope, case):
    return scope if case == "plain" else scope + ":" + case


def _fuse(lexical, dense, matrix, limit=20):
    scores, representatives = {}, {}
    for ranking, weight, is_dense in ((lexical, matrix["bm25_weight"], False), (dense, matrix["dense_weight"], True)):
        seen = set()
        for rank, hit in enumerate(ranking, 1):
            hit = _hit(hit)
            if hit.doc_id in seen:
                raise ParityError("Independent RRF received duplicate documents")
            seen.add(hit.doc_id)
            if weight:
                scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (matrix["rrf_k"] + rank)
                if is_dense or hit.doc_id not in representatives:
                    representatives[hit.doc_id] = hit
    return tuple(RankHit(doc, scores[doc], representatives[doc].start_char, representatives[doc].end_char)
        for doc in sorted(scores, key=lambda doc: (-scores[doc], doc))[:limit])


def _candidate(args):
    started = getattr(args, "process_started", time.perf_counter())
    output = _output(args.output)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    database = None
    startup_timings, request_timings = {}, {}
    with phase("startup", scope=args.scope):
        with _timed(startup_timings, "input_verification", events=True):
            shared, snapshot, vectors = _shared(args)
        database, identity, store, resolver = _runtime(args, snapshot, scope=args.scope, timings=startup_timings)
        with _timed(startup_timings, "imports", events=True):
            from src.retrieval.corpus_dense import EmbeddingModel
            from src.retrieval.corpus_bm25 import CorpusBM25
            from src.retrieval.sharded_dense import ShardedDense
            from src.retrieval.store import StoreBM25Retriever, StoreDenseRetriever, StoreHybridRetriever, reciprocal_rank_candidates
            from src.retrieval.hybrid import HybridConfig
            from src.storage.fingerprints import ValidatedCatalog
            import numpy as np
        with _timed(startup_timings, "catalog", events=True):
            catalog_handle = ValidatedCatalog.require(store.validated_catalog(), snapshot)
        with _timed(startup_timings, "dense_load", events=True):
            dense = ShardedDense.load(Path(args.dense_directory), snapshot=snapshot, catalog=catalog_handle,
                block_rows=args.block_rows, file_cache_policy=args.file_cache_policy)
        catalog = dense.catalog
        with _timed(startup_timings, "bm25_load", events=True):
            lexical = CorpusBM25.load(Path(args.bm25_directory), corpus_path=None, mmap=True)
        with _timed(startup_timings, "model_load", events=True):
            model = EmbeddingModel.for_index(Path(args.dense_directory), device="cpu", batch_size=1)
        if model.identity != shared["model_identity"]:
            raise ParityError("Actual query model differs from frozen vectors")
        class MeasuredEncoder:
            identity = model.identity
            current = 0
            seconds = 0.0
            calls = 0
            def encode(self, texts, query=False):
                begin = time.perf_counter()
                fresh = model.encode(texts, query=query)
                self.seconds += time.perf_counter() - begin
                self.calls += 1
                if not np.array_equal(fresh, vectors[self.current:self.current + 1]):
                    raise ParityError("Fresh per-request encoding differs from frozen query vector bytes")
                return vectors[self.current:self.current + 1].copy()
        class CaptureIndex:
            def __init__(self, actual):
                self.actual, self.last = actual, ()
            def __getattr__(self, key):
                return getattr(self.actual, key)
            def rank_candidates(self, *arguments, **keywords):
                with _timed(request_timings, "rank_bm25", events=True):
                    self.last = self.actual.rank_candidates(*arguments, **keywords)
                return self.last
            def candidates_vector(self, *arguments, **keywords):
                with _timed(request_timings, "rank_dense", events=True):
                    self.last = self.actual.candidates_vector(*arguments, **keywords)
                return self.last
        class TimedGuard:
            def assert_current(self, context):
                with _timed(request_timings, "guard"):
                    resolver.assert_current(context)
        class TimedBM25(StoreBM25Retriever):
            def _materialize(self, *arguments, **keywords):
                with _timed(request_timings, "materialize", events=True):
                    return super()._materialize(*arguments, **keywords)
        encoder, lexical_capture, dense_capture = MeasuredEncoder(), CaptureIndex(lexical), CaptureIndex(dense)
        config = HybridConfig(candidate_k=50, rerank_k=50, rrf_k=60, bm25_weight=.7, dense_weight=1 - .7)
        guard = TimedGuard()
        with _timed(startup_timings, "adapters", events=True):
            hybrid = StoreHybridRetriever(TimedBM25(lexical_capture, store, catalog=catalog_handle),
                StoreDenseRetriever(dense_capture, encoder, store), config=config)
        startup = time.perf_counter() - started
    emit("measurement", "startup", scope=args.scope, seconds=startup, components=startup_timings,
         file_cache_policy=dense.file_cache_policy)
    records = []
    try:
        cases = ("plain", "source", "unknown_date") if args.scope == "full" else ("plain",)
        for query_index, question in enumerate(shared["questions"]):
            for case in cases:
                filters = _filters(shared, question, case)
                for iteration in range(shared["matrix"]["warm_repeats"] + 1):
                    label = _case_name(args.scope, case)
                    request_index = len(records)
                    with phase("request", scope=args.scope, case=label, question_id=question["question_id"], iteration=iteration,
                               request_index=request_index):
                        begin = time.perf_counter()
                        request_timings.clear()
                        encoder.current, encoder.seconds, encoder.calls = query_index, 0.0, 0
                        lexical_capture.last = dense_capture.last = ()
                        resolve_begin = time.perf_counter()
                        context = resolver.resolve(identity)
                        resolve_seconds = time.perf_counter() - resolve_begin
                        evidence = hybrid.search(context, question["question"], guard=guard,
                            top_k=shared["matrix"]["timed_top_k"], filters=filters)
                        with _timed(request_timings, "read_more", events=True):
                            expansion = hybrid.read_more(context, evidence[:2], guard=guard, max_chars=3000, filters=filters)
                        guard.assert_current(context)
                        elapsed = time.perf_counter() - begin
                    emit("measurement", "request", scope=args.scope, case=label, question_id=question["question_id"],
                         iteration=iteration, request_index=request_index, seconds=elapsed, resolve_seconds=resolve_seconds,
                         encode_seconds=encoder.seconds, encode_calls=encoder.calls, components=dict(request_timings))
                    fused = reciprocal_rank_candidates((lexical_capture.last, dense_capture.last), k=60,
                                                       weights=(.7, 1 - .7), limit=20)
                    selected = {item.chunk.doc_id for item in (*evidence, *expansion)}
                    entries = [catalog[hybrid.bm25._positions[doc]] for doc in sorted(selected)]
                    records.append({"question_id": question["question_id"], "question_sha256": question["question_sha256"],
                        "case": label, "iteration": iteration, "request_index": request_index, "filters": filters.model_dump(mode="json"),
                        "scope_id": context.binding.scope_id, "allowed_documents": context.scope.allowed_count,
                        "policy_revision": context.binding.policy_revision, "permission_bitmap_sha256": context.scope.bitmap_sha256,
                        "dense_hits": [RankHit(item.doc_id, item.score, item.start_char, item.end_char).as_dict() for item in dense_capture.last],
                        "hybrid_hits": [RankHit(item.doc_id, item.score, item.start_char, item.end_char).as_dict() for item in fused],
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                        "read_more": [item.model_dump(mode="json") for item in expansion],
                        "catalog": [entry.model_dump(mode="json") for entry in entries],
                        "timings": {"total_seconds": elapsed, "resolve_seconds": resolve_seconds, "encode_seconds": encoder.seconds,
                                    "encode_calls": encoder.calls, **request_timings}, "vector_attestation":
                            "fresh encoding byte-equal to frozen input" if encoder.calls else "authorized empty/filter fast path skipped encoding"})
        manifest = {"stage": "candidate", "scope": args.scope, "shared_manifest_sha256": file_sha256(Path(args.shared) / "manifest.json"),
            "snapshot": snapshot.model_dump(mode="json"), "matrix_sha256": shared["matrix_sha256"], "vectors_sha256": shared["vectors_sha256"],
            "startup_seconds": startup, "startup_components": startup_timings,
            "component_timing_policy": "inclusive nested durations; guard time overlaps materialize/read_more; do not sum",
            "file_cache_policy": dense.file_cache_policy,
            "latency_gate": latency_gate(startup, records), "source_hashes": _code(), "packages": _packages(),
            "startup_boundary": "Linux process start; driver entry fallback on systems without procfs",
            "first_request_policy": "request_index0 only; every later request is warm regardless of question/case/iteration",
            "coverage": {"documents": len(catalog), "passages": dense.total_vectors, "shards": dense.manifest["shards"]},
            "catalog_instances": 1, "catalog_reuse": "one issued store metadata capsule shared by policy/dense/BM25",
            "policy_resolvers": 1, "component": "full hybrid, BM25 mmap + bounded dense + resident BGE + PostgreSQL policy/store",
            "cache_condition": "first process request and warm repeats; no cold-disk claim", "block_rows": args.block_rows}
        _write_artifact(output, manifest, records)
    finally:
        if database is not None:
            database.close()


def _reference(args):
    """Independent dense scores/native BM25/RRF, with authorized reference reads."""
    started = time.perf_counter()
    output = _output(args.output)
    shared, snapshot, vectors = _shared(args)
    database = None
    try:
        with phase("reference_setup"):
            import numpy as np
            from src.retrieval.corpus_bm25 import CorpusBM25
            from src.retrieval.passages import bounded_passage
            from src.storage.contracts import RankedCandidate
            from src.access.policy import PolicyResolver
            from src.retrieval.faiss_reference import reference_configuration
            from src.storage.fingerprints import ValidatedCatalog
            database, identity, store, full = _runtime(args, snapshot, scope="full")
            catalog_handle = ValidatedCatalog.require(store.validated_catalog(), snapshot)
            catalog, positions = catalog_handle.entries, catalog_handle.positions
            resolvers = {"full": full, "narrow": PolicyResolver(database, snapshot,
                             doc_ids=shared["matrix"]["narrow_doc_ids"], catalog=catalog_handle),
                         "empty": PolicyResolver(database, snapshot, doc_ids=(), catalog=catalog_handle)}
            contexts = {name: resolver.resolve(identity) for name, resolver in resolvers.items()}
            base_masks = {name: np.array(context.scope.mask_for(snapshot, catalog_handle.doc_ids), dtype=bool) for name, context in contexts.items()}
            masks = dict(base_masks)
            for question in shared["questions"]:
                for case in ("source", "unknown_date"):
                    filters = _filters(shared, question, case)
                    masks[question["question_id"] + ":" + case] = base_masks["full"] & np.array([filters.matches(entry) for entry in catalog], dtype=bool)
        with phase("reference_exhaustive_faiss"):
            reference = exhaustive_faiss(args.dense_directory, vectors, snapshot=snapshot, catalog=catalog,
                masks=masks, top_k=shared["matrix"]["candidate_k"], block_rows=args.block_rows)
        with phase("reference_bm25_load"):
            lexical = CorpusBM25.load(Path(args.bm25_directory), corpus_path=None, mmap=True)
            if tuple(lexical.doc_ids) != catalog_handle.doc_ids or lexical.manifest["corpus_sha256"] != snapshot.corpus_sha256:
                raise ParityError("Reference BM25 identity differs from the canonical snapshot")
        records = []
        for query_index, question in enumerate(shared["questions"]):
            for scope in ("full", "narrow", "empty"):
                for case in (("plain", "source", "unknown_date") if scope == "full" else ("plain",)):
                    label = _case_name(scope, case)
                    mask_key = scope if case == "plain" else question["question_id"] + ":" + case
                    dense_hits = reference.rankings[mask_key][query_index]
                    context, resolver, filters = contexts[scope], resolvers[scope], _filters(shared, question, case)
                    with phase("reference_case", question_id=question["question_id"], case=label):
                        resolver.assert_current(context)
                        lexical_hits = tuple(RankHit(doc, score, 0, catalog[positions[doc]].text_length)
                            for doc, score in lexical.rank(question["question"], shared["matrix"]["candidate_k"], masks[mask_key])
                            if catalog[positions[doc]].text_length > 0)
                        hybrid = _fuse(lexical_hits, dense_hits, shared["matrix"])
                        evidence, legacy, tie_primary = [], [], []
                        for hit in hybrid[:shared["matrix"]["timed_top_k"]]:
                            entry = catalog[positions[hit.doc_id]]
                            candidate = RankedCandidate(snapshot=snapshot, doc_id=entry.doc_id, revision_id=entry.revision_id,
                                source_type=entry.source_type, start_char=hit.start_char, end_char=hit.end_char,
                                score=hit.score, retrieval_method="hybrid", scoring_provenance="parity-independent-rrf")
                            passage = store.read_passage(context, candidate, guard=resolver, filters=filters)
                            evidence.append(bounded_passage(passage, question["question"]))
                        for hit in dense_hits:
                            if (hit.legacy_start_char, hit.legacy_end_char) != (hit.start_char, hit.end_char):
                                entry = catalog[positions[hit.doc_id]]
                                candidate = RankedCandidate(snapshot=snapshot, doc_id=entry.doc_id, revision_id=entry.revision_id,
                                    source_type=entry.source_type, start_char=hit.legacy_start_char, end_char=hit.legacy_end_char,
                                    score=hit.score, retrieval_method="dense", scoring_provenance="parity-exact-max-legacy-tie")
                                legacy.append(store.read_passage(context, candidate, guard=resolver, filters=filters))
                                primary = candidate.model_copy(update={"start_char": hit.start_char, "end_char": hit.end_char})
                                tie_primary.append(store.read_passage(context, primary, guard=resolver, filters=filters))
                        expansion = store.read_more(context, tuple(evidence[:2]), guard=resolver, max_chars=3000, filters=filters)
                        resolver.assert_current(context)
                    selected = {item.chunk.doc_id for item in (*evidence, *expansion, *legacy, *tie_primary)}
                    records.append({"question_id": question["question_id"], "question_sha256": question["question_sha256"],
                        "case": label, "iteration": 0, "filters": filters.model_dump(mode="json"),
                        "scope_id": context.binding.scope_id, "allowed_documents": context.scope.allowed_count,
                        "policy_revision": context.binding.policy_revision, "permission_bitmap_sha256": context.scope.bitmap_sha256,
                        "dense_hits": [hit.as_dict() for hit in dense_hits], "hybrid_hits": [hit.as_dict() for hit in hybrid],
                        "evidence": [item.model_dump(mode="json") for item in evidence],
                        "read_more": [item.model_dump(mode="json") for item in expansion],
                        "legacy_tie_evidence": [item.model_dump(mode="json") for item in legacy],
                        "tie_primary_evidence": [item.model_dump(mode="json") for item in tie_primary],
                        "catalog": [catalog[positions[doc]].model_dump(mode="json") for doc in sorted(selected)]})
        _write_artifact(output, {"stage": "reference", "shared_manifest_sha256": file_sha256(Path(args.shared) / "manifest.json"),
            "snapshot": snapshot.model_dump(mode="json"), "matrix_sha256": shared["matrix_sha256"], "vectors_sha256": shared["vectors_sha256"],
            "coverage": {"documents": reference.documents, "passages": reference.passages, "shards": reference.shards},
            "elapsed_seconds": time.perf_counter() - started, "source_hashes": _code(), "packages": _packages(),
            "native_reference": reference_configuration(total_vectors=reference.passages, dimension=reference.dimension),
            "component": "independent global-kernel FAISS + native BM25 + independent RRF; no model resident",
            "catalog_instances": 1, "policy_resolvers": 3,
            "catalog_reuse": "three resolver-local wrappers over one issued store metadata capsule",
            "setup_is_harness_overhead_not_app_readiness": True}, records)
    finally:
        if database is not None:
            database.close()


def _compare(args):
    from src.contracts import SnapshotRef
    from src.storage.contracts import CatalogEntry
    output = _output(args.output)
    shared, _ = load_artifact(args.shared)
    reference, expected = load_artifact(args.reference)
    expected_keys = {(question["question_id"], case) for question in shared["questions"]
        for case in ("full", "full:source", "full:unknown_date", "narrow", "empty")}
    if (reference.get("stage") != "reference" or len(expected) != len(expected_keys)
            or any(type(row["iteration"]) is not int or row["iteration"] != 0 for row in expected)
            or {(row["question_id"], row["case"]) for row in expected} != expected_keys):
        raise ParityError("Reference lacks the full predeclared question/scope matrix")
    shared_sha = file_sha256(Path(args.shared) / "manifest.json")
    snapshot = SnapshotRef.model_validate(shared["snapshot"])
    if file_sha256(args.corpus) != snapshot.corpus_sha256:
        raise ParityError("Canonical source corpus differs from the frozen snapshot")
    baseline = {(row["question_id"], row["case"]): row for row in expected}
    comparisons, observed, latencies, passed, candidate_digests = [], set(), [], True, {}
    cache_policies, block_sizes = set(), set()
    source_catalog, source_passages = {}, {}
    def collect_source(record):
        for raw in record["catalog"]:
            entry = CatalogEntry.model_validate(raw)
            if entry.doc_id in source_catalog and source_catalog[entry.doc_id] != entry:
                raise ParityError("Source-check catalogue metadata conflicts across cases")
            source_catalog[entry.doc_id] = entry
        for passage in (*record["evidence"], *record["read_more"], *record.get("legacy_tie_evidence", ()),
                        *record.get("tie_primary_evidence", ())):
            source_passages[_sha(_canonical(passage))] = passage
    def tie_proofs(record):
        primary = {value["citation_id"]: value for value in record.get("tie_primary_evidence", ())}
        legacy = {value["citation_id"]: value for value in record.get("legacy_tie_evidence", ())}
        proofs = []
        for raw in record["dense_hits"]:
            hit = _hit(raw)
            if hit.legacy_start_char is None or (hit.start_char, hit.end_char) == (hit.legacy_start_char, hit.legacy_end_char):
                continue
            first_key = f"{hit.doc_id}:{hit.start_char}:{hit.end_char}"
            legacy_key = f"{hit.doc_id}:{hit.legacy_start_char}:{hit.legacy_end_char}"
            if (hit.tie_count < 2 or first_key not in primary or legacy_key not in legacy
                    or primary[first_key]["score"] != hit.score or legacy[legacy_key]["score"] != hit.score):
                raise ParityError("Legacy tie compatibility lacks exact-maximum and both-excerpt proof")
            proofs.append({**raw, "canonical_citation_id": first_key, "legacy_citation_id": legacy_key,
                           "score_proof": "both native scores strictly equal the independently computed document maximum"})
        return proofs
    def provenance(manifest):
        if (manifest["shared_manifest_sha256"] != shared_sha or manifest["matrix_sha256"] != shared["matrix_sha256"]
                or manifest["vectors_sha256"] != shared["vectors_sha256"] or manifest["snapshot"] != shared["snapshot"]
                or manifest["coverage"] != reference["coverage"]):
            raise ParityError("Reference/candidate provenance or full corpus coverage differs")
    provenance(reference)
    for path in args.candidate:
        manifest, records = load_artifact(path)
        if manifest.get("stage") != "candidate":
            raise ParityError("Expected a completed candidate artifact")
        if manifest["scope"] not in {"full", "narrow", "empty"} or manifest["scope"] in candidate_digests:
            raise ParityError("Candidate scope is unknown or duplicated")
        cache_policy = manifest.get("file_cache_policy", "discard")
        if not isinstance(cache_policy, str) or cache_policy not in {"discard", "retain"}:
            raise ParityError("Candidate file cache policy is invalid")
        cache_policies.add(cache_policy)
        if len(cache_policies) != 1:
            raise ParityError("Candidate file cache policies differ across scope processes")
        block_rows = manifest.get("block_rows")
        _positive(block_rows, "Candidate block_rows")
        block_sizes.add(block_rows)
        if len(block_sizes) != 1:
            raise ParityError("Candidate block_rows differ across scope processes")
        candidate_digests[manifest["scope"]] = file_sha256(path / "manifest.json")
        provenance(manifest)
        if manifest["packages"] != reference["packages"]:
            raise ParityError("Candidate/reference native package versions differ")
        if manifest["source_hashes"] != reference["source_hashes"]:
            raise ParityError("Candidate/reference executed source files differ")
        if manifest["source_hashes"]["src/retrieval/corpus_dense.py"] != shared["source_hashes"]["src/retrieval/corpus_dense.py"]:
            raise ParityError("Query-encoder implementation differs from the frozen input stage")
        latency = latency_gate(manifest["startup_seconds"], records)
        latencies.append({"scope": manifest["scope"], **latency})
        passed &= latency["passed"]
        for row in records:
            allowed_cases = {"full", "full:source", "full:unknown_date"} if manifest["scope"] == "full" else {manifest["scope"]}
            if row["case"] not in allowed_cases:
                raise ParityError("Candidate records do not match their declared scope process")
            key = row["question_id"], row["case"]
            if key not in baseline or row["question_sha256"] != baseline[key]["question_sha256"]:
                raise ParityError("Candidate question/case identity differs")
            record_key = (*key, row["iteration"])
            if record_key in observed:
                raise ParityError("Candidate matrix reuses a completed case")
            observed.add(record_key)
            oracle = baseline[key]
            if row["filters"] != oracle["filters"] or row["allowed_documents"] != oracle["allowed_documents"]:
                raise ParityError("Candidate filter or permission scope differs from the declared workload")
            if (row["policy_revision"] != oracle["policy_revision"]
                    or row["permission_bitmap_sha256"] != oracle["permission_bitmap_sha256"]):
                raise ParityError("Candidate/reference policy epoch or actual permission bitmap differs")
            checks = {}
            for field in ("dense_hits", "hybrid_hits"):
                checks[field] = {}
                for prefix in shared["matrix"]["rank_prefixes"]:
                    comparison = compare_hits(oracle[field][:prefix], row[field][:prefix])
                    checks[field][str(prefix)] = comparison
                    passed &= comparison["passed"]
            for case_record in (oracle, row):
                collect_source(case_record)
            legacy_proofs = tie_proofs(oracle)
            # Compare exact materialized windows and adjacent reads independently
            # of rounded retrieval scores; native score deltas are retained above.
            def text_identity(values):
                return [{key: value[key] for key in ("citation_id", "snapshot_id", "revision_id", "retrieval_method", "chunk")} for value in values]
            exact_text = text_identity(oracle["evidence"]) == text_identity(row["evidence"])
            exact_more = text_identity(oracle["read_more"]) == text_identity(row["read_more"])
            passed &= exact_text and exact_more
            comparisons.append({"question_id": key[0], "case": key[1], "iteration": row["iteration"],
                "checks": checks, "exact_materialized_windows": exact_text, "exact_read_more": exact_more,
                "legacy_tie_differences": legacy_proofs,
                "legacy_policy": "intentional exact-maximum tie representative difference; both excerpts source checked"})
    with phase("independent_source_validation"):
        source_validation = verify_source_evidence(args.corpus, tuple(source_passages.values()),
                                                  snapshot=snapshot, catalog=tuple(source_catalog.values()))
    complete = observed == {(*key, iteration) for key in expected_keys for iteration in range(shared["matrix"]["warm_repeats"] + 1)}
    resources, resource_scopes = [], set()
    for path in args.resources:
        report = _json(path)
        scope = report.get("scope")
        if scope not in candidate_digests or scope in resource_scopes:
            raise ParityError("Resource evidence is duplicated or belongs to another scope")
        if report.get("candidate_manifest_sha256") != candidate_digests[scope]:
            raise ParityError("Resource evidence belongs to another candidate execution")
        resource_scopes.add(scope)
        resources.append(resource_gate(report))
    resource_verified = resource_scopes == {"full", "narrow", "empty"} and all(value["passed"] for value in resources)
    passed &= complete and resource_verified
    _write_artifact(output, {"stage": "compare", "passed": bool(passed), "matrix_complete": complete,
        "file_cache_policy": next(iter(cache_policies), None),
        "block_rows": next(iter(block_sizes), None),
        "rank_policy": "exact score maximum; lexical document ID; earliest exact-tied span; no near-tie buckets",
        "latency_gates": latencies, "resource_gates": resources, "resources_verified": resource_verified,
        "source_hashes": _code(), "source_validation": source_validation, "shared_manifest_sha256": shared_sha}, comparisons)
    return bool(passed)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    encode = commands.add_parser("encode", help="Freeze ten development questions and offline manifest-matched query vectors")
    encode.add_argument("--questions", required=True, type=Path)
    encode.add_argument("--splits", required=True, type=Path)
    encode.add_argument("--snapshot", required=True, type=Path)
    encode.add_argument("--source-revision")
    for command in (encode, commands.add_parser("reference"), commands.add_parser("candidate")):
        command.add_argument("--corpus", required=True, type=Path)
        command.add_argument("--dense-directory", required=True, type=Path)
        command.add_argument("--bm25-directory", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if command is not encode:
            command.add_argument("--shared", required=True, type=Path)
            command.add_argument("--credentials-file", required=True, type=Path)
            command.add_argument("--block-rows", type=int, default=8192)
            if command.prog.endswith(" candidate"):
                command.add_argument("--scope", choices=("full", "narrow", "empty"), required=True)
                command.add_argument("--file-cache-policy", choices=("discard", "retain"), default="discard",
                    help="discard consumed dense file cache (default), or retain it for a separately measured memory/latency tradeoff")
    compare = commands.add_parser("compare")
    compare.add_argument("--shared", required=True, type=Path)
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--candidate", required=True, action="append", type=Path)
    compare.add_argument("--corpus", required=True, type=Path)
    compare.add_argument("--resources", action="append", type=Path, default=[])
    compare.add_argument("--output", required=True, type=Path)
    return result


def main(argv=None):
    process_started = time.perf_counter()
    try:
        fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19]) / os.sysconf("SC_CLK_TCK")
        uptime = time.clock_gettime(time.CLOCK_BOOTTIME)
        process_started -= max(0, uptime - ticks)
    except (OSError, ValueError, AttributeError, IndexError):
        pass
    command = parser()
    args = command.parse_args(argv)
    args.process_started = process_started
    try:
        operation = globals()["_" + args.command]
        passed = operation(args)
        emit("complete", stage=args.command)
        if passed is False:
            return 2
        return 0
    except Exception as error:
        message = str(error) if isinstance(error, ParityError) else "Parity stage failed safely; retain this run and inspect its phase/resource evidence"
        emit("failure", stage=args.command, error_type=type(error).__name__, message=message)
        command.exit(2, message + "\n")


if __name__ == "__main__":
    raise SystemExit(main())

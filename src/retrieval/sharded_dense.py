"""Bounded exact dense ranking over the existing read-only embedding shards.

Stored fp16/fp32 rows are copied into bounded writable float32 blocks and
renormalized with the legacy FAISS primitive. No FAISS index, model or document
body table is constructed. The retained catalogue and reduction arrays are
O(documents); Arrow batches, vector copies and sorting scratch are O(block_rows).
An mmap spans one shard virtually. Page/cache release is best-effort advice,
not a process/cgroup memory guarantee; full-corpus fit and latency need separate
measurement. The default discards consumed file-cache pages; opt-in retain
keeps the OS file cache eligible for reuse while still releasing mmap pages.
This module resolves no identity or fresh authorization policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import heapq
import json
import mmap
import os
from pathlib import Path
import stat

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.access.contracts import (
    AccessDenied, AuthorizationScope, EvidenceValidationError, SnapshotMismatch,
    StaleContext,
)
from src.contracts import ScopeBinding, SnapshotRef
from src.storage.contracts import CatalogEntry, RankedCandidate, ReadFilters
from src.storage.fingerprints import ValidatedCatalog, catalog_fingerprint

_COLUMNS = ("doc_index", "start_char", "end_char")
_DTYPES = (np.dtype("float16"), np.dtype("float32"))


def _positive(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(label + " must be a positive integer")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate index JSON fields")
        value[key] = item
    return value


def _invalid_constant(_value):
    raise ValueError("Non-finite index JSON values")


def _read_json(path):
    try:
        raw = path.read_bytes()
        return json.loads(raw, object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant), raw
    except (UnicodeError, ValueError):
        raise ValueError("Invalid dense index JSON") from None


def _signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Dense index artifacts must be regular files")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


@contextmanager
def _mapped(path):
    stored = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(stored, np.memmap):
        if hasattr(stored, "close"):
            stored.close()
        raise ValueError("Dense vectors must be a memory-mappable NPY array")
    try:
        yield stored
    finally:
        stored._mmap.close()


def _release_pages(stored, start, stop):
    """Discard consumed mapped pages where supported; never claim eviction."""
    mapping = stored._mmap
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        return
    relative = stored.offset % mmap.ALLOCATIONGRANULARITY
    first = relative + start * stored.strides[0]
    last = relative + stop * stored.strides[0]
    first = first // mmap.PAGESIZE * mmap.PAGESIZE
    last = min(len(mapping), (last + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE)
    if last > first:
        try:
            mapping.madvise(mmap.MADV_DONTNEED, first, last - first)
        except (OSError, ValueError):
            pass


def _file_cache_hint(descriptor, offset=0, length=0):
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            os.posix_fadvise(descriptor, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def _normalized_block(stored, start, stop):
    """One bounded, private C float32 copy; preserve FAISS zero-row behavior."""
    vectors = np.array(stored[start:stop], dtype=np.float32, order="C", copy=True)
    if not np.isfinite(vectors).all():
        raise ValueError("Dense embeddings contain non-finite values")
    faiss.normalize_L2(vectors)
    if not np.isfinite(vectors).all():
        raise ValueError("Dense normalization produced non-finite values")
    return vectors


def _top_candidates(best, top_k):
    """Prune below the exact kth score, retaining every cutoff tie for the heap."""
    positions = np.flatnonzero(np.isfinite(best))
    if len(positions) > top_k:
        scores = best[positions]
        offset = len(scores) - top_k
        cutoff = np.partition(scores, offset)[offset]
        positions = positions[scores >= cutoff]
    return positions


@dataclass(frozen=True)
class _Shard:
    name: str
    rows: int
    dimension: int
    dtype: np.dtype
    vectors_signature: tuple
    metadata_signature: tuple


class ShardedDense:
    """Metadata-only exact maximum-over-passages search for one frozen snapshot.

    load validates all stored vectors/spans in bounded batches. Queries recheck
    artifact signatures, stream every shard, and never keep a mapping open after
    returning or raising. rank_vector is a trusted offline primitive, not a
    permission API. candidates_vector requires a trusted server scope and emits
    only permitted metadata; a store must still check freshness before text reads.
    """

    @classmethod
    def load(cls, directory: str | Path, *, snapshot: SnapshotRef,
             catalog: Iterable[CatalogEntry] | ValidatedCatalog, block_rows: int = 8192,
             file_cache_policy: str = "discard") -> ShardedDense:
        _positive(block_rows, "block_rows")
        if not isinstance(file_cache_policy, str) or file_cache_policy not in {"discard", "retain"}:
            raise ValueError("Dense file cache policy must be discard or retain")
        if not isinstance(snapshot, SnapshotRef):
            raise ValueError("Dense loading requires a snapshot")
        snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        directory = Path(directory).resolve()
        manifest_signature = _signature(directory / "manifest.json")
        ids_signature = _signature(directory / "doc_ids.json")
        manifest, manifest_bytes = _read_json(directory / "manifest.json")
        if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
                or manifest["version"] != 1 or manifest.get("complete") is not True):
            raise ValueError("Dense directory is incomplete or unsupported")
        if type(manifest.get("documents")) is not int or manifest["documents"] != snapshot.document_count:
            raise SnapshotMismatch("Dense document count differs from the snapshot")
        if manifest.get("corpus_sha256") != snapshot.corpus_sha256:
            raise SnapshotMismatch("Dense corpus differs from the snapshot")
        for key in ("snapshot_id", "catalog_sha256"):
            if key in manifest and manifest[key] != getattr(snapshot, key):
                raise SnapshotMismatch("Dense manifest differs from the snapshot")
        if "snapshot" in manifest and SnapshotRef.model_validate(manifest["snapshot"]) != snapshot:
            raise SnapshotMismatch("Dense manifest differs from the snapshot")
        if not isinstance(manifest.get("model"), dict) or manifest["model"].get("normalize") is not True:
            raise ValueError("Dense manifest must declare normalized embeddings")
        _positive(manifest.get("shards"), "shards")
        actual = {path.name for path in directory.iterdir() if path.name.startswith("shard_")}
        if len(actual) != 2 * manifest["shards"]:
            raise ValueError("Dense directory does not contain exactly the declared shards")
        names = tuple(f"shard_{index:05d}" for index in range(manifest["shards"]))
        expected = {name + suffix for name in names for suffix in (".npy", ".parquet")}
        if actual != expected:
            raise ValueError("Dense directory does not contain exactly the declared shards")
        ids, _ = _read_json(directory / "doc_ids.json")
        if (not isinstance(ids, list) or len(ids) != snapshot.document_count
                or any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("Dense document IDs are invalid or duplicated")
        catalog_handle = None
        if isinstance(catalog, ValidatedCatalog):
            catalog_handle = ValidatedCatalog.require(catalog, snapshot)
            if tuple(ids) != catalog_handle.doc_ids:
                raise SnapshotMismatch("Dense document order differs from the catalogue")
            entries, doc_ids = catalog_handle.entries, catalog_handle.doc_ids
        else:
            entries = []

            def checked_catalog():
                for position, supplied in enumerate(catalog):
                    if not isinstance(supplied, CatalogEntry):
                        raise ValueError("Dense catalogue requires CatalogEntry records")
                    entry = CatalogEntry.model_validate(supplied.model_dump(warnings=False))
                    if position >= len(ids) or entry.doc_id != ids[position]:
                        raise SnapshotMismatch("Dense document order differs from the catalogue")
                    entries.append(entry)
                    yield entry

            fingerprint = catalog_fingerprint(checked_catalog(), expected_count=snapshot.document_count)
            if fingerprint != snapshot.catalog_sha256:
                raise SnapshotMismatch("Dense catalogue differs from the snapshot")
            entries = tuple(entries)
            doc_ids = tuple(entry.doc_id for entry in entries)
        result = cls()
        result._directory, result._snapshot = directory, snapshot
        result._manifest = deepcopy(manifest)
        result._catalog_handle = catalog_handle
        result._catalog = entries
        result._doc_ids = doc_ids
        result._lengths = np.fromiter((entry.text_length for entry in entries), dtype=np.int64,
                                     count=len(entries))
        result._block_rows = block_rows
        result._file_cache_policy = file_cache_policy
        result._manifest_signature, result._ids_signature = manifest_signature, ids_signature
        result._provenance = "sharded-dense-v1:" + hashlib.sha256(manifest_bytes).hexdigest()
        shards = tuple(result._inspect_shard(name) for name in names)
        if len({(shard.dimension, shard.dtype.str) for shard in shards}) != 1:
            raise ValueError("Dense shard dimensions or dtypes differ")
        result._shards = shards
        result._dimension = shards[0].dimension
        result._total_vectors = sum(shard.rows for shard in shards)
        for shard in shards:
            with closing(result._blocks(shard)) as blocks:
                for stored, start, stop, _docs, _starts, _ends in blocks:
                    if not np.isfinite(stored[start:stop]).all():
                        raise ValueError("Dense embeddings contain non-finite values")
        result._check_identity()
        return result

    @property
    def snapshot(self) -> SnapshotRef:
        return self._snapshot

    @property
    def catalog(self) -> tuple[CatalogEntry, ...]:
        return self._catalog

    @property
    def catalog_handle(self) -> ValidatedCatalog | None:
        return self._catalog_handle

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return self._doc_ids

    @property
    def manifest(self) -> dict:
        return deepcopy(self._manifest)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def total_vectors(self) -> int:
        return self._total_vectors

    @property
    def block_rows(self) -> int:
        return self._block_rows

    @property
    def file_cache_policy(self) -> str:
        return self._file_cache_policy

    @property
    def scoring_provenance(self) -> str:
        return self._provenance

    def _inspect_shard(self, name):
        vectors, metadata = self._directory / (name + ".npy"), self._directory / (name + ".parquet")
        vector_signature, metadata_signature = _signature(vectors), _signature(metadata)
        with _mapped(vectors) as stored:
            if stored.ndim != 2 or stored.shape[1] < 1 or stored.dtype not in _DTYPES or not stored.flags.c_contiguous:
                raise ValueError("Dense vectors require C-order fp16/fp32 rows and a positive dimension")
            rows, dimension, dtype = len(stored), stored.shape[1], stored.dtype
        parquet = pq.ParquetFile(metadata, memory_map=False, pre_buffer=False)
        try:
            schema = parquet.schema_arrow
            if (tuple(schema.names) != _COLUMNS
                    or any(not pa.types.is_signed_integer(schema.field(column).type) for column in _COLUMNS)
                    or parquet.metadata.num_rows != rows):
                raise ValueError("Dense passage metadata schema or row count differs")
        finally:
            parquet.close()
        return _Shard(name, rows, dimension, dtype, vector_signature, metadata_signature)

    def _check_identity(self):
        if (_signature(self._directory / "manifest.json") != self._manifest_signature
                or _signature(self._directory / "doc_ids.json") != self._ids_signature):
            raise ValueError("Dense index identity changed after loading")
        for shard in self._shards:
            if (_signature(self._directory / (shard.name + ".npy")) != shard.vectors_signature
                    or _signature(self._directory / (shard.name + ".parquet")) != shard.metadata_signature):
                raise ValueError("Dense shard changed after loading")

    def _blocks(self, shard):
        """Yield bounded integer metadata; finally release every owned resource."""
        vector_path = self._directory / (shard.name + ".npy")
        metadata_path = self._directory / (shard.name + ".parquet")
        descriptor = os.open(vector_path, os.O_RDONLY)
        parquet = None
        try:
            with _mapped(vector_path) as stored:
                if stored.shape != (shard.rows, shard.dimension) or stored.dtype != shard.dtype:
                    raise ValueError("Dense vector header changed")
                parquet = pq.ParquetFile(metadata_path, memory_map=False, pre_buffer=False)
                start = 0
                for batch in parquet.iter_batches(batch_size=self.block_rows, columns=list(_COLUMNS), use_threads=False):
                    stop = start + len(batch)
                    try:
                        if len(batch) > self.block_rows or any(column.null_count for column in batch.columns):
                            raise ValueError("Dense metadata batch is invalid")
                        docs, starts, ends = (batch.column(index).to_numpy(zero_copy_only=False) for index in range(3))
                        if (np.any(docs < 0) or np.any(docs >= len(self._catalog)) or np.any(starts < 0)
                                or np.any(ends <= starts) or np.any(ends > self._lengths[docs])):
                            raise ValueError("Dense passage metadata is outside the catalogue")
                        yield stored, start, stop, docs, starts, ends
                    finally:
                        _release_pages(stored, start, stop)
                        if self.file_cache_policy == "discard":
                            _file_cache_hint(descriptor, stored.offset + start * stored.strides[0],
                                             (stop - start) * stored.strides[0])
                    start = stop
                if start != shard.rows:
                    raise ValueError("Dense metadata row count differs")
        finally:
            try:
                if parquet is not None:
                    parquet.close()
            finally:
                if self.file_cache_policy == "discard":
                    _file_cache_hint(descriptor)
                os.close(descriptor)

    def _vector(self, vector):
        raw = np.asarray(vector)
        if raw.shape not in ((self.dimension,), (1, self.dimension)) or raw.dtype.kind not in "fiu":
            raise ValueError("Query vector must be real numeric with the index dimension")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.array(raw, dtype=np.float32, order="C", copy=True).reshape(self.dimension)
        if not np.isfinite(result).all():
            raise ValueError("Query vector must be finite float32")
        return result

    def _mask(self, mask):
        if mask is None:
            return None
        result = np.asarray(mask)
        if result.dtype != np.dtype(bool) or result.shape != (len(self.doc_ids),):
            raise ValueError("Document mask must be a boolean vector matching document order")
        return result.copy()

    def _rank_passages(self, vector, top_k, mask):
        if mask is not None and not mask.any():
            return []
        self._check_identity()
        count = len(self.doc_ids)
        best = np.full(count, -np.inf, dtype=np.float32)
        starts_best = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
        ends_best = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
        for shard in self._shards:
            with closing(self._blocks(shard)) as blocks:
                for stored, start, stop, docs, starts, ends in blocks:
                    eligible = None if mask is None else mask[docs]
                    if eligible is not None and not eligible.any():
                        continue
                    if eligible is None or eligible.all():
                        vectors = _normalized_block(stored, start, stop)
                    else:
                        selected = stored[start:stop][eligible]
                        vectors = _normalized_block(selected, 0, len(selected))
                        del selected
                        docs, starts, ends = docs[eligible], starts[eligible], ends[eligible]
                    scores = vectors @ vector
                    del vectors
                    if not np.isfinite(scores).all():
                        raise ValueError("Dense scores must be finite")
                    order = np.lexsort((ends, starts, -scores, docs))
                    ordered_docs = docs[order]
                    first = np.r_[True, ordered_docs[1:] != ordered_docs[:-1]]
                    winners = order[first]
                    docs, scores, starts, ends = docs[winners], scores[winners], starts[winners], ends[winners]
                    improve = ((scores > best[docs]) | ((scores == best[docs]) &
                        ((starts < starts_best[docs]) | ((starts == starts_best[docs]) & (ends < ends_best[docs])))))
                    docs, scores, starts, ends = docs[improve], scores[improve], starts[improve], ends[improve]
                    best[docs], starts_best[docs], ends_best[docs] = scores, starts, ends
        self._check_identity()
        positions = heapq.nsmallest(top_k, _top_candidates(best, top_k),
            key=lambda index: (-float(best[index]), self.doc_ids[index], int(starts_best[index]), int(ends_best[index])))
        return [(index, float(best[index]), int(starts_best[index]), int(ends_best[index])) for index in positions]

    def rank_vector(self, vector, top_k: int, mask=None) -> list[tuple[str, float]]:
        """Trusted offline ranking; query scale/zero semantics match CorpusDense."""
        _positive(top_k, "top_k")
        vector, mask = self._vector(vector), self._mask(mask)
        return [(self.doc_ids[index], score) for index, score, _start, _end in self._rank_passages(vector, top_k, mask)]

    def candidates_vector(self, vector, top_k: int, *, scope: AuthorizationScope,
                          filters: ReadFilters = ReadFilters()) -> tuple[RankedCandidate, ...]:
        """Return authorized spans only; empty scope avoids vector work entirely.

        Exact score ties use lexical document ID. Within a document, tied best
        passages use start_char then end_char, independent of block/shard order.
        Floating point tolerance for a parity test never changes this tie rule.
        """
        _positive(top_k, "top_k")
        if not isinstance(scope, AuthorizationScope) or not isinstance(scope.binding, ScopeBinding):
            raise EvidenceValidationError("A trusted authorization scope is required")
        binding = ScopeBinding.model_validate(scope.binding.model_dump(warnings=False))
        if binding.snapshot != self.snapshot:
            raise SnapshotMismatch("Authorization scope differs from the dense snapshot")
        if not isinstance(filters, ReadFilters):
            raise EvidenceValidationError("Dense restrictions require ReadFilters")
        filters = ReadFilters.model_validate(filters.model_dump(warnings=False))
        mask = self._mask(scope.mask_for(self.snapshot, self.doc_ids))
        if mask is None:
            raise EvidenceValidationError("Authorization scope must return an explicit mask")
        if scope.binding != binding:
            raise StaleContext("Authorization scope changed during ranking")
        if filters.source_types or filters.after or filters.before:
            for index in np.flatnonzero(mask):
                if not filters.matches(self._catalog[index]):
                    mask[index] = False
        if not mask.any():
            return ()
        ranked = self._rank_passages(self._vector(vector), top_k, mask)
        if scope.binding != binding:
            raise StaleContext("Authorization scope changed during ranking")
        candidates = []
        for index, score, start, end in ranked:
            entry = self._catalog[index]
            permitted = scope.permits(entry.doc_id)
            if type(permitted) not in (bool, np.bool_) or not permitted:
                raise AccessDenied("Authorization scope denied a selected candidate")
            candidate = RankedCandidate(snapshot=self.snapshot, doc_id=entry.doc_id,
                revision_id=entry.revision_id, source_type=entry.source_type,
                start_char=start, end_char=end, score=score, retrieval_method="dense",
                scoring_provenance=self.scoring_provenance)
            candidate.validate_catalog(self.snapshot, entry)
            candidates.append(candidate)
        if scope.binding != binding:
            raise StaleContext("Authorization scope changed during ranking")
        return tuple(candidates)

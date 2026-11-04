"""Reference-only bounded reproduction of FAISS1.15.1 single-query IP scores.

The global IndexFlatIP dispatch depends on the *whole* database size. Searching
arbitrary small indexes therefore changes its numerical kernel. Reproduce the
global database-parallel partitions and SGEMM tile shapes, then recover every
physical row's native score; never truncate passages or reuse candidate ranking.

Pinned algorithm: https://github.com/facebookresearch/faiss/blob/v1.15.1/faiss/utils/distances.cpp
(`should_use_db_parallel`, `knn_db_parallel_impl`, `Run_search_inner_product`).
Each input query represents a separate legacy nq=1 search, never a batched nq>1
baseline. Run in the isolated reference process: FAISS dispatch cvars are global.
BLAS threading is left exactly as configured by that process, never overridden.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import threading


class UnsupportedReferenceConfiguration(ValueError):
    """The pinned native dispatch cannot be reproduced under this configuration."""


_DISPATCH_LOCK = threading.RLock()
_CVARS = ("distance_compute_blas_threshold", "distance_compute_blas_query_bs",
          "distance_compute_blas_database_bs", "distance_compute_min_k_reservoir")


@dataclass(frozen=True)
class _Configuration:
    version: str
    threads: int
    threshold: int
    query_bs: int
    database_bs: int
    min_k_reservoir: int
    database_parallel: bool
    blas: bool


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive Python integer")


def _configuration(faiss, total_vectors, dimension, threads):
    if type(total_vectors) is not int or total_vectors < 0:
        raise ValueError("total_vectors must be a nonnegative Python integer")
    _positive(dimension, "dimension")
    if getattr(faiss, "__version__", None) != "1.15.1":
        raise UnsupportedReferenceConfiguration("Reference dispatch is pinned to FAISS1.15.1")
    actual_threads = faiss.omp_get_max_threads()
    _positive(actual_threads, "configured FAISS threads")
    if threads is not None:
        _positive(threads, "threads")
        if threads != actual_threads:
            raise UnsupportedReferenceConfiguration("Requested threads differ from the actual legacy baseline")
    if os.environ.get("OMP_DYNAMIC", "false").lower() not in {"false", "0", ""}:
        raise UnsupportedReferenceConfiguration("Dynamic OpenMP teams are not a supported legacy baseline")
    values = tuple(getattr(faiss.cvar, name) for name in _CVARS)
    threshold, query_bs, database_bs, reservoir = values
    if (any(type(value) is not int for value in values) or threshold < 0
            or query_bs < 1 or database_bs != 1024 or reservoir < 1):
        raise UnsupportedReferenceConfiguration("Unsupported FAISS dispatch tile or threshold configuration")
    parallel = actual_threads > 1 and total_vectors >= max(10000, actual_threads * database_bs)
    return _Configuration("1.15.1", actual_threads, threshold, query_bs, database_bs, reservoir,
                          parallel, parallel or dimension >= threshold)


def reference_configuration(*, total_vectors, dimension, threads=None):
    """JSON-safe native provenance; unknown BLAS libraries stay undiscovered.

    This does not change a thread setting. Observed libraries and explicit
    environment controls are retained so a different BLAS profile is visible.
    """
    import faiss
    configuration = _configuration(faiss, total_vectors, dimension, threads)
    try:
        from threadpoolctl import threadpool_info
        libraries = [{key: value.get(key) for key in
            ("user_api", "internal_api", "prefix", "version", "num_threads", "architecture")}
            for value in threadpool_info()]
    except ImportError:
        libraries = None
    return {"faiss_version": configuration.version, "faiss_threads": configuration.threads,
        "query_batch_size": 1, "database_parallel": configuration.database_parallel,
        "score_kernel": "sgemm" if configuration.blas else "faiss_simd_ip",
        "database_block_rows": configuration.database_bs,
        "global_blas_threshold": configuration.threshold, "global_blas_query_bs": configuration.query_bs,
        "global_min_k_reservoir": configuration.min_k_reservoir,
        "blas_thread_override": False, "observed_thread_libraries": libraries,
        "thread_environment": {key: os.environ.get(key) for key in
            ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "MKL_DYNAMIC", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}}


def _ranges(total_vectors, configuration):
    partitions = configuration.threads if configuration.database_parallel else 1
    for partition in range(partitions):
        begin, end = partition * total_vectors // partitions, (partition + 1) * total_vectors // partitions
        for start in range(begin, end, configuration.database_bs):
            yield start, min(start + configuration.database_bs, end)


@contextmanager
def _native_dispatch(faiss, configuration):
    """Restore before yielding a tile, including on an injected native failure."""
    with _DISPATCH_LOCK:
        current = tuple(getattr(faiss.cvar, name) for name in _CVARS)
        expected = (configuration.threshold, configuration.query_bs, configuration.database_bs,
                    configuration.min_k_reservoir)
        actual_threads = faiss.omp_get_max_threads()
        if current != expected or actual_threads != configuration.threads:
            raise UnsupportedReferenceConfiguration("Native settings changed during the reference stream")
        try:
            # Local tiles are below the database-parallel threshold. Force the
            # same SGEMM shape used by each global thread's database tile.
            if configuration.blas:
                faiss.cvar.distance_compute_blas_threshold = 0
            yield
        finally:
            for name, value in zip(_CVARS, current):
                setattr(faiss.cvar, name, value)
            faiss.omp_set_num_threads(actual_threads)


def iter_scores(query_vectors, *, total_vectors: int, dimension: int, read_rows, threads=None):
    """Yield all global row scores as `(start, float32[nqueries,nrows])`.

    read_rows(start,stop) returns exactly those already legacy-normalized rows as
    a C-contiguous native float32 matrix. It may stitch physical shards. At most
    1024 rows are requested, with global thread-partition boundaries preserved.
    Input queries are finite real vectors and are not normalized or mutated.
    Supported: pinned FAISS1.15.1, static teams, no selector, independent nq=1 IP.
    Different versions/tiles/teams fail explicitly; no approximate fallback.
    """
    import faiss
    import numpy as np

    configuration = _configuration(faiss, total_vectors, dimension, threads)
    if not callable(read_rows):
        raise TypeError("read_rows must be a bounded normalized-row callback")
    queries = np.asarray(query_vectors)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if queries.ndim != 2 or queries.shape[1] != dimension or not len(queries) or queries.dtype.kind not in "fiu":
        raise ValueError("Reference queries must be a nonempty real matrix matching dimension")
    with np.errstate(over="ignore", invalid="ignore"):
        queries = np.array(queries, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(queries).all():
        raise ValueError("Reference queries must remain finite in float32")
    for start, stop in _ranges(total_vectors, configuration):
        rows = read_rows(start, stop)
        if (not isinstance(rows, np.ndarray) or rows.dtype != np.dtype("float32")
                or not rows.dtype.isnative or rows.shape != (stop - start, dimension)
                or not rows.flags.c_contiguous or not np.isfinite(rows).all()):
            raise ValueError("Reference row callback must return exact finite normalized C float32 rows")
        result = np.empty((len(queries), len(rows)), dtype=np.float32)
        with _native_dispatch(faiss, configuration):
            index = faiss.IndexFlatIP(dimension)
            index.add(rows)
            for query_index, query in enumerate(queries):
                scores, positions = index.search(query.reshape(1, -1), len(rows))
                ids = positions[0]
                if (ids.shape != (len(rows),) or np.any(ids < 0) or np.any(ids >= len(rows))
                        or len(np.unique(ids)) != len(rows) or not np.isfinite(scores).all()):
                    raise ValueError("Native exhaustive reference returned incomplete or non-finite row scores")
                result[query_index, ids] = scores[0]
        del rows, index
        yield start, result

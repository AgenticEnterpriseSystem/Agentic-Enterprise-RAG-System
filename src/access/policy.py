"""Trusted grant resolution with compact immutable scopes and fresh checks.

Role, department, project labels and document prose confer no permission. A
direct grant or an effective explicitly joined group grants access; operator
restrictions only narrow it. Scope bitmaps are LSB-first, matching PostgreSQL
get_bit(bytea, position), and are registered before any protected SQL can use
their opaque ID. One catalogue map is shared by scopes, never copied per user.
"""

from collections import OrderedDict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
from itertools import chain, islice
import json
import time
from types import MappingProxyType
from threading import Lock

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.access.contracts import (
    AccessDenied, EvidenceValidationError, PolicyUnavailable, RequestContext,
    ResolvedIdentity, SessionExpired, SnapshotMismatch, StaleContext,
)
from src.contracts import ScopeBinding, SnapshotRef
from src.db import schema
from src.db.clock import session_time
from src.db.runtime import Database, DatabaseRoleError
from src.observability import emit, span
from src.storage.contracts import CatalogEntry
from src.storage.fingerprints import ValidatedCatalog, catalog_fingerprint

_CATALOG_FIELDS = tuple(CatalogEntry.model_fields)
# Shared immutable byte decoder, not a per-principal expanded mask cache.
_BYTE_BOOLEAN_BITS = tuple(tuple(bool(value & (1 << bit)) for bit in range(8)) for value in range(256))
_GRANT_PAGE_SIZE = 8192
_SAFE_DATABASE_SQLSTATES = frozenset({"25P03", "57014", "57P01", "08003", "08006", "53300", "55P03", "40001", "40P01"})
_GRANT_DOCUMENTS_SQL = """
WITH effective_groups AS MATERIALIZED (
 SELECT membership.group_id
 FROM rag.group_memberships membership
 WHERE membership.user_id=:user_id
 AND NOT EXISTS (
  SELECT 1 FROM rag.group_requirements requirement
  WHERE requirement.group_id=membership.group_id
  AND NOT EXISTS (
   SELECT 1 FROM rag.group_memberships required_membership
   WHERE required_membership.user_id=:user_id
   AND required_membership.group_id=requirement.required_group_id
  )
 )
), authorized_documents AS (
 SELECT grant_row.doc_id FROM rag.document_grants grant_row
 WHERE grant_row.user_id=:user_id AND grant_row.snapshot_id=:snapshot
 UNION ALL
 SELECT grant_row.doc_id FROM effective_groups effective
 JOIN rag.document_grants grant_row ON grant_row.group_id=effective.group_id
 WHERE grant_row.group_id IS NOT NULL AND grant_row.snapshot_id=:snapshot
)
SELECT authorized.doc_id FROM authorized_documents authorized
WHERE (CAST(:after AS text) IS NULL OR authorized.doc_id > CAST(:after AS text))
ORDER BY authorized.doc_id
LIMIT :page_size
"""


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class _Catalog:
    snapshot: SnapshotRef
    doc_ids: tuple[str, ...]
    sources: tuple[str, ...]
    positions: object = field(repr=False)


@dataclass(frozen=True)
class BitmapScope:
    """One frozen decision; bytes use bit(position%8) in byte(position//8)."""

    binding: ScopeBinding
    _catalog: _Catalog = field(repr=False)
    bitmap: bytes = field(repr=False)
    restriction_sha256: str

    def __post_init__(self):
        binding = ScopeBinding.model_validate(self.binding.model_dump(warnings=False))
        if binding.snapshot != self._catalog.snapshot:
            raise SnapshotMismatch("Scope catalogue belongs to another snapshot")
        count = binding.snapshot.document_count
        if type(self.bitmap) is not bytes or len(self.bitmap) != (count + 7) // 8:
            raise EvidenceValidationError("Invalid scope bitmap size or storage")
        if count % 8 and self.bitmap[-1] >> (count % 8):
            raise EvidenceValidationError("Scope bitmap padding must deny")
        if not isinstance(self.restriction_sha256, str) or len(self.restriction_sha256) != 64:
            raise EvidenceValidationError("Invalid operator restriction identity")
        object.__setattr__(self, "binding", binding)

    @property
    def bitmap_sha256(self):
        return hashlib.sha256(self.bitmap).hexdigest()

    @property
    def allowed_count(self):
        return sum(byte.bit_count() for byte in self.bitmap)

    def permits(self, doc_id: str) -> bool:
        position = self._catalog.positions.get(doc_id) if isinstance(doc_id, str) else None
        return position is not None and bool(self.bitmap[position // 8] & (1 << (position % 8)))

    def mask_for(self, snapshot: SnapshotRef, doc_ids: Sequence[str]) -> tuple[bool, ...]:
        if (not isinstance(snapshot, SnapshotRef)
                or SnapshotRef.model_validate(snapshot.model_dump(warnings=False)) != self.binding.snapshot):
            raise SnapshotMismatch("Permission mask belongs to another snapshot")
        if type(self) is BitmapScope and doc_ids is self._catalog.doc_ids:
            # The resolver's immutable catalogue already fixes bit positions.
            # Preserve LSB order, real bool elements and the existing tuple API;
            # arbitrary ID collections and subclasses keep permits semantics.
            decoded = chain.from_iterable(_BYTE_BOOLEAN_BITS[value] for value in self.bitmap)
            return tuple(islice(decoded, self.binding.snapshot.document_count))
        return tuple(self.permits(doc_id) for doc_id in doc_ids)


class PolicyResolver:
    """A runtime-pool resolver and FreshnessGuard for one ready snapshot.

    doc_ids=None means unrestricted; an explicit empty collection denies all.
    A bounded LRU may reuse only an immutable bitmap decision for the exact
    user/session/version, live epoch, snapshot and operator restriction. Every
    resolve checks live authority before lookup and validates the registry and
    live authority again before release. Identities, request contexts and guard
    checks are never cached. Moving idle expiry does not identify a bitmap;
    each returned context carries the current validated incoming identity.
    """

    def __init__(self, database: Database, snapshot: SnapshotRef, *, source_types=(), doc_ids=None, clock=None,
                 cache_size=32, catalog: ValidatedCatalog | None = None):
        if not isinstance(database, Database) or not isinstance(snapshot, SnapshotRef):
            raise TypeError("Policy resolution requires Database and SnapshotRef")
        if type(cache_size) is not int or not 0 <= cache_size <= 32:
            raise ValueError("Policy bitmap cache size must be an integer from 0 through 32")
        self.database = database
        self.snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        self._default_clock = lambda: datetime.now(timezone.utc)
        self.clock = self._default_clock if clock is None else clock
        self._cache_size = cache_size
        self._scope_cache = OrderedDict()
        self._cache_lock = Lock()
        if catalog is None:
            self._catalog = self._load_catalog()
        else:
            proof = ValidatedCatalog.require(catalog, self.snapshot, issuer=database)
            with self._transaction() as connection:
                self._ready(connection)
            # Keep a resolver-local identity wrapper so a scope created by a
            # different resolver cannot replace this one's validated catalogue.
            self._catalog = _Catalog(self.snapshot, proof.doc_ids, proof.sources, proof.positions)
        if isinstance(source_types, str) or isinstance(doc_ids, str):
            raise ValueError("Operator restrictions must be document/source collections")
        sources = frozenset(source_types)
        documents = None if doc_ids is None else frozenset(doc_ids)
        if not sources <= frozenset(self._catalog.sources):
            raise ValueError("Operator restrictions contain unknown source types")
        if documents is not None and not documents <= self._catalog.positions.keys():
            raise ValueError("Operator restrictions contain unknown document IDs")
        bits = bytearray((self.snapshot.document_count + 7) // 8)
        for position, doc_id in enumerate(self._catalog.doc_ids):
            if (not sources or self._catalog.sources[position] in sources) and (documents is None or doc_id in documents):
                bits[position // 8] |= 1 << (position % 8)
        self._operator_bitmap = bytes(bits)
        self._restriction_sha256 = hashlib.sha256(self._operator_bitmap).hexdigest()

    def _cache_key(self, identity, epoch):
        return (identity.user_id, identity.session_id, identity.session_version, epoch,
                self.snapshot.snapshot_id, self.snapshot.corpus_sha256, self.snapshot.catalog_sha256,
                self.snapshot.document_count, self._restriction_sha256)

    def _cached_scope(self, key):
        if not self._cache_size:
            return None
        with self._cache_lock:
            scope = self._scope_cache.get(key)
            if scope is not None:
                self._scope_cache.move_to_end(key)
            return scope

    def _remember_scope(self, key, scope):
        if not self._cache_size:
            return
        with self._cache_lock:
            self._scope_cache[key] = scope
            self._scope_cache.move_to_end(key)
            while len(self._scope_cache) > self._cache_size:
                self._scope_cache.popitem(last=False)

    def _now(self, connection):
        return session_time(connection, clock=None if self.clock is self._default_clock else self.clock)

    @contextmanager
    def _transaction(self, context=None):
        try:
            with self.database.transaction(context) as connection:
                yield connection
        except (sa.exc.SQLAlchemyError, DatabaseRoleError) as error:
            raise self._database_unavailable(error) from None

    @staticmethod
    def _database_unavailable(error):
        # Never preserve driver messages, connection strings, SQL or parameters.
        failure = PolicyUnavailable("Authoritative policy is unavailable")
        state = getattr(getattr(error, "orig", error), "sqlstate", None)
        failure.database_sqlstate = state if type(state) is str and state in _SAFE_DATABASE_SQLSTATES else None
        return failure

    def _ready(self, connection):
        row = connection.execute(sa.select(schema.dataset_snapshots).where(
            schema.dataset_snapshots.c.snapshot_id == self.snapshot.snapshot_id)).mappings().one_or_none()
        if (row is None or row["state"] != "ready" or any(row[key] != getattr(self.snapshot, key)
                for key in ("corpus_sha256", "catalog_sha256", "document_count"))):
            raise SnapshotMismatch("Policy snapshot is unavailable or differs from the catalogue")

    def _load_catalog(self):
        ids, sources, canonical_sources, position = [], [], {}, -1
        table = schema.documents
        def entries():
            nonlocal position
            while True:
                with self._transaction() as connection:
                    self._ready(connection)
                    rows = connection.execute(sa.select(*(table.c[name] for name in _CATALOG_FIELDS)).where(
                        table.c.snapshot_id == self.snapshot.snapshot_id, table.c.position > position)
                        .order_by(table.c.position).limit(1000)).mappings().all()
                if not rows:
                    return
                for row in rows:
                    entry = CatalogEntry.model_validate(dict(row))
                    ids.append(entry.doc_id)
                    sources.append(canonical_sources.setdefault(entry.source_type, entry.source_type))
                    position = entry.position
                    yield entry
        digest = catalog_fingerprint(entries(), expected_count=self.snapshot.document_count)
        if digest != self.snapshot.catalog_sha256 or len(set(ids)) != len(ids):
            raise SnapshotMismatch("Policy catalogue fingerprint or document IDs differ")
        return _Catalog(self.snapshot, tuple(ids), tuple(sources), MappingProxyType({value: index for index, value in enumerate(ids)}))

    def _identity(self, connection, identity, *, epoch=None):
        self._ready(connection)
        sessions, users = schema.sessions, schema.users
        row = connection.execute(sa.select(sessions.c.created_at, sessions.c.expires_at, sessions.c.revoked_at,
            sessions.c.session_version.label("issued_version"), users.c.is_active,
            users.c.session_version.label("account_version")).select_from(sessions.join(users,
            sessions.c.user_id == users.c.user_id)).where(sessions.c.session_id == identity.session_id,
            sessions.c.user_id == identity.user_id)).mappings().one_or_none()
        now = self._now(connection)
        if (row is None or not row["is_active"] or row["revoked_at"] is not None
                or row["created_at"] > now or row["expires_at"] <= now or identity.expires_at <= now
                or row["issued_version"] != identity.session_version or row["account_version"] != identity.session_version):
            raise SessionExpired("Account or session is unavailable")
        revision = connection.execute(sa.select(schema.policy_state.c.revision).where(
            schema.policy_state.c.singleton_id == 1)).scalar_one_or_none()
        if revision is None:
            raise PolicyUnavailable("Authoritative policy revision is unavailable")
        now = self._now(connection)
        if row["expires_at"] <= now or identity.expires_at <= now:
            raise SessionExpired("Account or session expired while checking policy")
        if epoch is not None and revision != epoch:
            raise StaleContext("Document policy changed; resolve a fresh scope")
        return revision

    def resolve(self, identity: ResolvedIdentity) -> RequestContext:
        with span("policy.resolve"):
            return self._resolve(identity)

    def _resolve(self, identity: ResolvedIdentity) -> RequestContext:
        if not isinstance(identity, ResolvedIdentity):
            raise EvidenceValidationError("Policy resolution requires a trusted identity")
        identity = ResolvedIdentity.model_validate(identity.model_dump(warnings=False))
        with self._transaction() as connection:
            epoch = self._identity(connection, identity)
            cache_key = self._cache_key(identity, epoch)
            cached_scope = self._cached_scope(cache_key)
        if cached_scope is not None:
            context = RequestContext(identity, cached_scope)
            self.assert_current(context)
            return context
        # No database transaction remains open during local bitmap work. Each
        # bounded page rechecks the same live authority epoch; registration and
        # release below recheck it again. Duplicate grants are harmless: skipping
        # the rest of an equal-doc_id boundary cannot omit a different document.
        bitmap = self._resolve_bitmap(identity, epoch)
        metadata = {"user_id": identity.user_id, "session_id": identity.session_id,
            "session_version": identity.session_version, "policy_revision": epoch,
            "snapshot": self.snapshot.model_dump(mode="json"), "restriction_sha256": self._restriction_sha256}
        scope_id = "scope:" + hashlib.sha256(b"rag-policy-scope-v1\n" + _json(metadata) + b"\n" + bitmap).hexdigest()
        binding = ScopeBinding(scope_id=scope_id, user_id=identity.user_id,
            session_version=identity.session_version, policy_revision=epoch, snapshot=self.snapshot)
        scope = BitmapScope(binding, self._catalog, bitmap, self._restriction_sha256)
        context = RequestContext(identity, scope)
        values = {"scope_id": scope_id, "user_id": identity.user_id, "session_id": identity.session_id,
            "session_version": identity.session_version, "policy_revision": epoch,
            **self.snapshot.model_dump(), "allowed_positions": bitmap, "bitmap_sha256": scope.bitmap_sha256,
            "restriction_sha256": self._restriction_sha256}
        with span("policy.register"):
            with self._transaction(context) as connection:
                self._identity(connection, identity, epoch=epoch)
                connection.execute(pg_insert(schema.policy_scopes).values(**values).on_conflict_do_nothing(index_elements=["scope_id"]))
                stored = connection.execute(sa.select(schema.policy_scopes).where(schema.policy_scopes.c.scope_id == scope_id)).mappings().one_or_none()
                if stored is None or any((bytes(stored[key]) if key == "allowed_positions" else stored[key]) != value for key, value in values.items()):
                    raise EvidenceValidationError("Registered scope differs from its frozen decision")
                self._identity(connection, identity, epoch=epoch)
        self.assert_current(context)
        self._remember_scope(cache_key, scope)
        return context

    def _resolve_bitmap(self, identity, epoch):
        # Keep a constant number of events even for a full-corpus scope. Timings
        # sum active DB/page-fetch work separately from local bitmap CPU work;
        # neither observer I/O nor bitmap processing holds a DB transaction.
        elapsed = {"policy.grants": 0.0, "policy.bitmap": 0.0}
        pages, rows = 0, 0
        phase, failure = "policy.bitmap", None
        for stage in elapsed:
            emit(stage, "start", status="in_progress", duration_is_cumulative=True)
        try:
            began = time.monotonic()
            try:
                bits = bytearray(len(self._operator_bitmap))
            finally:
                elapsed[phase] += (time.monotonic() - began) * 1000
            pending = iter(self._grant_pages(identity, epoch))
            while True:
                phase, began = "policy.grants", time.monotonic()
                try:
                    documents = next(pending)
                except StopIteration:
                    break
                finally:
                    elapsed[phase] += (time.monotonic() - began) * 1000
                pages += 1
                rows += len(documents)
                phase, began = "policy.bitmap", time.monotonic()
                try:
                    self._apply_grant_page(bits, documents)
                finally:
                    elapsed[phase] += (time.monotonic() - began) * 1000
            phase, began = "policy.bitmap", time.monotonic()
            try:
                return bytes(bits)
            finally:
                elapsed[phase] += (time.monotonic() - began) * 1000
        except BaseException as error:
            failure = error
            raise
        finally:
            for stage, milliseconds in elapsed.items():
                active_failure = failure is not None and phase == stage
                emit(stage, "error" if active_failure else "end",
                    status=("error" if active_failure else "cancelled") if failure is not None else "ok",
                    duration_ms=max(0, milliseconds), duration_is_cumulative=True,
                    grant_page_count=pages, grant_row_count=rows,
                    **({"error": failure, "database_phase": getattr(failure, "database_phase", None),
                        "database_sqlstate": getattr(failure, "database_sqlstate", None)} if active_failure else {}))

    def _grant_pages(self, identity, epoch):
        after = None
        while True:
            result, phase = None, "execute"
            try:
                with self._transaction() as connection:
                    self._identity(connection, identity, epoch=epoch)
                    # psycopg's ordinary cursor buffers the server result at
                    # execute. Keep Python row conversion/GC outside this TX.
                    result = connection.execution_options(stream_results=False).execute(sa.text(_GRANT_DOCUMENTS_SQL), {
                        "snapshot": self.snapshot.snapshot_id, "user_id": identity.user_id,
                        "after": after, "page_size": _GRANT_PAGE_SIZE,
                    })
                phase = "materialize"
                documents = self._materialize_grant_page(result)
                phase = "revalidate"
                # Fresh authority after materialization is mandatory even for
                # the final/empty page; only then may any page enter the bitmap.
                with self._transaction() as connection:
                    self._identity(connection, identity, epoch=epoch)
            except PolicyUnavailable as error:
                error.database_phase = phase
                raise
            finally:
                if result is not None:
                    # Also release a buffered cursor if commit/revalidation fails.
                    try:
                        result.close()
                    except Exception:
                        pass
            if not documents:
                return
            yield documents
            if len(documents) < _GRANT_PAGE_SIZE:
                return
            after = documents[-1]

    def _materialize_grant_page(self, result):
        try:
            with result:
                return tuple(result.scalars().all())
        except (sa.exc.SQLAlchemyError, DatabaseRoleError) as error:
            raise self._database_unavailable(error) from None

    def _apply_grant_page(self, bits, documents):
        for doc_id in documents:
            position = self._catalog.positions.get(doc_id)
            if position is None or not 0 <= position < self.snapshot.document_count:
                raise EvidenceValidationError("Grant differs from the canonical catalogue")
            bit = 1 << (position % 8)
            if self._operator_bitmap[position // 8] & bit:
                bits[position // 8] |= bit

    def assert_current(self, context: RequestContext) -> None:
        if not isinstance(context, RequestContext) or not isinstance(context.scope, BitmapScope):
            raise EvidenceValidationError("A registered server policy context is required")
        context.assert_bound()
        if context.binding.snapshot != self.snapshot:
            raise SnapshotMismatch("Request scope belongs to another snapshot")
        if context.scope._catalog is not self._catalog:
            raise EvidenceValidationError("Scope does not use this resolver's validated catalogue")
        if context.scope.restriction_sha256 != self._restriction_sha256:
            raise StaleContext("Operator restrictions differ from this request scope")
        with self._transaction() as connection:
            self._identity(connection, context.identity, epoch=context.binding.policy_revision)
            row = connection.execute(sa.select(schema.policy_scopes).where(
                schema.policy_scopes.c.scope_id == context.binding.scope_id)).mappings().one_or_none()
            if (row is None or row["user_id"] != context.identity.user_id or row["session_id"] != context.identity.session_id
                    or row["session_version"] != context.binding.session_version
                    or row["policy_revision"] != context.binding.policy_revision
                    or row["bitmap_sha256"] != context.scope.bitmap_sha256
                    or row["restriction_sha256"] != self._restriction_sha256
                    or any(row[key] != getattr(self.snapshot, key) for key in
                           ("snapshot_id", "corpus_sha256", "catalog_sha256", "document_count"))):
                raise AccessDenied("Registered scope is unavailable for this request")
            self._identity(connection, context.identity, epoch=context.binding.policy_revision)
        context.assert_bound()

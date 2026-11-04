"""Reviewed additive policy patches and isolated operator benchmark identities.

This operator interface uses separate migrator credentials. Applying a manifest
never overwrites existing policy; revocation removes only grants/memberships
created by that exact manifest. Group definitions and prerequisites remain in
place, so revoking a patch cannot remove an AND prerequisite and widen others.
Benchmark credentials are exclusive0600 files, never public login/cookie input.
"""

import argparse
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Literal
from uuid import uuid4

from pydantic import Field, model_validator
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.pool import NullPool

from src.access.contracts import ResolvedIdentity
from src.auth.accounts import PasswordHasher, normalize_login
from src.contracts import Contract, Identifier, SnapshotRef
from src.db import schema
from src.db.clock import session_time
from src.db.migrate import EXPECTED_HEAD
from src.db.runtime import Database, database_url, normalize_url


class PolicyError(ValueError):
    """Sanitized policy/operator failure."""


class GroupSpec(Contract):
    group_id: Identifier
    label: Identifier
    requires: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def flat_requirements(self):
        if self.requires and (len(self.requires) < 2 or len(set(self.requires)) != len(self.requires) or self.group_id in self.requires):
            raise ValueError("Compound groups require at least two distinct base groups")
        return self


class MembershipSpec(Contract):
    user_id: Identifier
    group_id: Identifier


class GrantSpec(Contract):
    doc_ids: tuple[Identifier, ...] = Field(min_length=1)
    user_id: Identifier | None = None
    group_id: Identifier | None = None

    @model_validator(mode="after")
    def one_subject(self):
        if (self.user_id is None) == (self.group_id is None) or len(self.doc_ids) != len(set(self.doc_ids)):
            raise ValueError("A grant requires one subject and distinct explicit documents")
        return self


class PolicyManifest(Contract):
    schema_version: Literal["1"] = "1"
    manifest_id: Identifier
    snapshot: SnapshotRef
    origin: Literal["operator", "synthetic"]
    reviewed_by: Identifier
    provenance: dict[str, str]
    groups: tuple[GroupSpec, ...] = ()
    memberships: tuple[MembershipSpec, ...] = ()
    grants: tuple[GrantSpec, ...] = ()

    @model_validator(mode="after")
    def explicit_review(self):
        if not self.provenance or any(not key.strip() or not value.strip() for key, value in self.provenance.items()):
            raise ValueError("Reviewed policy provenance must be explicit")
        if len({group.group_id for group in self.groups}) != len(self.groups):
            raise ValueError("Duplicate group definitions")
        if len({(member.user_id, member.group_id) for member in self.memberships}) != len(self.memberships):
            raise ValueError("Duplicate memberships")
        keys = [(grant.user_id, grant.group_id, doc_id) for grant in self.grants for doc_id in grant.doc_ids]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate document grants")
        defined = {group.group_id: group for group in self.groups}
        if any(required in defined and defined[required].requires for group in self.groups for required in group.requires):
            raise ValueError("Nested compound groups are not supported")
        return self


@dataclass(frozen=True)
class PolicyResult:
    manifest_id: str
    action: str
    dry_run: bool
    changed_grants: int
    changed_memberships: int
    policy_revision: int


@dataclass(frozen=True)
class BenchmarkPrincipal:
    user_id: str
    login: str
    snapshot: SnapshotRef
    policy_revision: int


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _checked(manifest):
    try:
        raw = manifest.model_dump(warnings=False) if isinstance(manifest, PolicyManifest) else manifest
        return PolicyManifest.model_validate(raw)
    except (ValueError, TypeError):
        raise PolicyError("Invalid reviewed policy manifest") from None


def _ready(connection, snapshot):
    row = connection.execute(sa.select(schema.dataset_snapshots).where(
        schema.dataset_snapshots.c.snapshot_id == snapshot.snapshot_id)).mappings().one_or_none()
    if (row is None or row["state"] != "ready" or any(row[key] != getattr(snapshot, key)
            for key in ("corpus_sha256", "catalog_sha256", "document_count"))):
        raise PolicyError("Reviewed snapshot is unavailable or differs from the database")


def _epoch(connection):
    result = connection.execute(sa.select(schema.policy_state.c.revision).where(schema.policy_state.c.singleton_id == 1)).scalar_one_or_none()
    if result is None:
        raise PolicyError("Policy revision is unavailable")
    return result


class PolicyRepository:
    def __init__(self, url=None):
        self._url = normalize_url(url) if url is not None else database_url(migration=True)

    @contextmanager
    def _operator(self, *, readonly=False):
        engine = sa.create_engine(self._url, poolclass=NullPool, hide_parameters=True,
            connect_args={"prepare_threshold": None, "connect_timeout": 5, "application_name": "rag-policy-admin"})
        try:
            with engine.begin() as connection:
                if readonly:
                    connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                current, session = connection.execute(sa.text("SELECT current_user,session_user")).one()
                if current != "rag_migrator" or session != current or connection.connection.driver_connection.info.user != current:
                    raise PolicyError("Policy changes require the dedicated migrator login")
                if connection.execute(sa.text("SELECT version_num FROM rag.alembic_version")).scalar_one() != EXPECTED_HEAD:
                    raise PolicyError("Database migration head does not support document policies")
                connection.execute(sa.text("SELECT pg_advisory_xact_lock(59800003)"))
                yield connection
        except sa.exc.SQLAlchemyError:
            raise PolicyError("Policy database operation failed") from None
        finally:
            engine.dispose()

    def _validate(self, connection, manifest):
        _ready(connection, manifest.snapshot)
        definitions = {group.group_id: group for group in manifest.groups}
        existing = {row["group_id"]: row["label"] for row in connection.execute(sa.select(schema.groups)).mappings()}
        requirements = {}
        for row in connection.execute(sa.select(schema.group_requirements)).mappings():
            requirements.setdefault(row["group_id"], set()).add(row["required_group_id"])
        for group in manifest.groups:
            if group.group_id in existing and (existing[group.group_id] != group.label
                    or requirements.get(group.group_id, set()) != set(group.requires)):
                raise PolicyError("Manifest would overwrite an existing group definition")
            if any(required not in existing and required not in definitions for required in group.requires):
                raise PolicyError("Compound group refers to an unknown base group")
            if any(requirements.get(required) or (required in definitions and definitions[required].requires) for required in group.requires):
                raise PolicyError("Nested compound groups are not supported")
            if group.requires and any(group.group_id in values for values in requirements.values()):
                raise PolicyError("A required base group cannot become compound")
        groups = existing.keys() | definitions.keys()
        users = {member.user_id for member in manifest.memberships} | {grant.user_id for grant in manifest.grants if grant.user_id is not None}
        if users:
            actual = set(connection.execute(sa.select(schema.users.c.user_id).where(schema.users.c.user_id.in_(users))).scalars())
            if actual != users:
                raise PolicyError("Manifest refers to unknown users")
        if any(member.group_id not in groups for member in manifest.memberships) or any(
                grant.group_id is not None and grant.group_id not in groups for grant in manifest.grants):
            raise PolicyError("Manifest refers to unknown groups")
        documents = {doc_id for grant in manifest.grants for doc_id in grant.doc_ids}
        if documents:
            actual = set(connection.execute(sa.select(schema.documents.c.doc_id).where(
                schema.documents.c.snapshot_id == manifest.snapshot.snapshot_id, schema.documents.c.doc_id.in_(documents))).scalars())
            if actual != documents:
                raise PolicyError("Manifest refers to unknown documents")

    @staticmethod
    def _audit(connection, manifest, action):
        body = manifest.model_dump(mode="json")
        if action == "benchmark":
            body["all_documents"] = True
        digest = _sha(_canonical(body))
        change_id = _sha((action + ":" + manifest.manifest_id + ":" + digest).encode("utf-8"))
        row = connection.execute(sa.select(schema.policy_changes).where(
            schema.policy_changes.c.manifest_id == manifest.manifest_id,
            schema.policy_changes.c.action == action)).mappings().one_or_none()
        if row is not None and row["manifest_sha256"] != digest:
            raise PolicyError("Manifest ID already identifies a different reviewed policy")
        return body, digest, change_id, row

    def dry_run(self, manifest) -> PolicyResult:
        manifest = _checked(manifest)
        with self._operator(readonly=True) as connection:
            self._validate(connection, manifest)
            self._audit(connection, manifest, "apply")
            membership_count = sum(not connection.execute(sa.select(sa.literal(True)).select_from(schema.group_memberships).where(
                schema.group_memberships.c.user_id == member.user_id, schema.group_memberships.c.group_id == member.group_id)).scalar()
                for member in manifest.memberships)
            grant_count = 0
            for grant in manifest.grants:
                subject = schema.document_grants.c.user_id == grant.user_id if grant.user_id else schema.document_grants.c.group_id == grant.group_id
                existing = set(connection.execute(sa.select(schema.document_grants.c.doc_id).where(
                    schema.document_grants.c.snapshot_id == manifest.snapshot.snapshot_id, subject,
                    schema.document_grants.c.doc_id.in_(grant.doc_ids))).scalars())
                grant_count += len(set(grant.doc_ids) - existing)
            return PolicyResult(manifest.manifest_id, "apply", True, grant_count, membership_count, _epoch(connection))

    def apply(self, manifest) -> PolicyResult:
        manifest = _checked(manifest)
        with self._operator() as connection:
            self._validate(connection, manifest)
            body, digest, change_id, prior = self._audit(connection, manifest, "apply")
            revoked = connection.execute(sa.select(schema.policy_changes.c.change_id).where(
                schema.policy_changes.c.manifest_id == manifest.manifest_id, schema.policy_changes.c.action == "revoke")).scalar_one_or_none()
            if revoked is not None:
                raise PolicyError("A revoked manifest requires a new review and manifest ID")
            if prior is not None:
                return PolicyResult(manifest.manifest_id, "apply", False, 0, 0, _epoch(connection))
            connection.execute(schema.policy_changes.insert().values(change_id=change_id, manifest_id=manifest.manifest_id,
                manifest_sha256=digest, action="apply", snapshot_id=manifest.snapshot.snapshot_id,
                origin=manifest.origin, reviewed_by=manifest.reviewed_by, manifest=body))
            for group in manifest.groups:
                connection.execute(pg_insert(schema.groups).values(group_id=group.group_id, label=group.label).on_conflict_do_nothing())
            for group in manifest.groups:
                if group.requires:
                    connection.execute(pg_insert(schema.group_requirements).values([
                        {"group_id": group.group_id, "required_group_id": required} for required in group.requires]).on_conflict_do_nothing())
            changed_memberships = 0
            if manifest.memberships:
                changed_memberships = len(connection.execute(pg_insert(schema.group_memberships).values([
                    {"user_id": member.user_id, "group_id": member.group_id, "origin": manifest.origin,
                     "policy_change_id": change_id} for member in manifest.memberships]).on_conflict_do_nothing().returning(
                     schema.group_memberships.c.user_id)).all())
            rows = [{"snapshot_id": manifest.snapshot.snapshot_id, "doc_id": doc_id,
                     "user_id": grant.user_id, "group_id": grant.group_id, "origin": manifest.origin,
                     "policy_change_id": change_id} for grant in manifest.grants for doc_id in grant.doc_ids]
            changed_grants = len(connection.execute(pg_insert(schema.document_grants).values(rows).on_conflict_do_nothing().returning(
                schema.document_grants.c.grant_id)).all()) if rows else 0
            return PolicyResult(manifest.manifest_id, "apply", False, changed_grants, changed_memberships, _epoch(connection))

    def revoke(self, manifest) -> PolicyResult:
        manifest = _checked(manifest)
        with self._operator() as connection:
            _ready(connection, manifest.snapshot)
            _body, _digest, _change, applied = self._audit(connection, manifest, "apply")
            body, digest, change_id, prior = self._audit(connection, manifest, "revoke")
            if applied is None:
                raise PolicyError("Manifest has no applied policy to revoke")
            if prior is not None:
                return PolicyResult(manifest.manifest_id, "revoke", False, 0, 0, _epoch(connection))
            changed_grants = connection.execute(schema.document_grants.delete().where(
                schema.document_grants.c.policy_change_id == applied["change_id"])).rowcount
            changed_memberships = connection.execute(schema.group_memberships.delete().where(
                schema.group_memberships.c.policy_change_id == applied["change_id"])).rowcount
            connection.execute(schema.policy_changes.insert().values(change_id=change_id, manifest_id=manifest.manifest_id,
                manifest_sha256=digest, action="revoke", snapshot_id=manifest.snapshot.snapshot_id,
                origin=manifest.origin, reviewed_by=manifest.reviewed_by, manifest=body))
            return PolicyResult(manifest.manifest_id, "revoke", False, changed_grants, changed_memberships, _epoch(connection))

    def provision_benchmark(self, login, *, snapshot: SnapshotRef, credentials_file,
                            reviewed_by, provenance, all_documents=False, origin="operator", hasher=None) -> BenchmarkPrincipal:
        if all_documents is not True:
            raise PolicyError("Benchmark provisioning requires explicit all_documents=True")
        login = normalize_login(login)
        snapshot = SnapshotRef.model_validate(snapshot.model_dump(warnings=False))
        review = PolicyManifest(manifest_id="benchmark:" + login + ":" + snapshot.snapshot_id,
            snapshot=snapshot, reviewed_by=reviewed_by, provenance=provenance, origin=origin)
        path = Path(credentials_file)
        if path.exists() or path.is_symlink():
            raise PolicyError("Benchmark credentials output already exists")
        password, user_id = secrets.token_urlsafe(32), uuid4().hex
        encoded = (hasher or PasswordHasher()).hash(password)
        credentials = {"format": "rag-benchmark-v1", "login": login, "password": password,
                       "user_id": user_id, "snapshot_id": snapshot.snapshot_id}
        descriptor = None
        owned = None
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            owned = os.fstat(descriptor)
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                handle.write(_canonical(credentials) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            with self._operator() as connection:
                _ready(connection, snapshot)
                body, digest, change_id, prior = self._audit(connection, review, "benchmark")
                if prior is not None:
                    raise PolicyError("Benchmark principal already has a reviewed provisioning record")
                connection.execute(schema.policy_changes.insert().values(change_id=change_id, manifest_id=review.manifest_id,
                    manifest_sha256=digest, action="benchmark", snapshot_id=snapshot.snapshot_id,
                    origin=origin, reviewed_by=reviewed_by, manifest=body))
                connection.execute(schema.users.insert().values(user_id=user_id, login=login,
                    password_hash=encoded, display_name="Isolated retrieval benchmark", account_kind="benchmark"))
                columns = ("snapshot_id", "doc_id", "user_id", "origin", "policy_change_id")
                selection = sa.select(schema.documents.c.snapshot_id, schema.documents.c.doc_id,
                    sa.literal(user_id), sa.literal(origin), sa.literal(change_id)).where(
                    schema.documents.c.snapshot_id == snapshot.snapshot_id)
                connection.execute(schema.document_grants.insert().from_select(columns, selection))
                epoch = _epoch(connection)
            return BenchmarkPrincipal(user_id, login, snapshot, epoch)
        except (OSError, ValueError, sa.exc.SQLAlchemyError):
            if owned is not None:
                with suppress(OSError):
                    current = path.lstat()
                    if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                        path.unlink()
            raise PolicyError("Benchmark provisioning failed without replacing existing policy or credentials") from None
        finally:
            if descriptor is not None:
                os.close(descriptor)


def verify_benchmark(credentials_file, database: Database, *, clock=None, hasher=None, ttl_seconds=3600) -> ResolvedIdentity:
    if not isinstance(database, Database) or type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 3600:
        raise PolicyError("Benchmark verification requires the runtime pool and bounded lifetime")
    path = Path(credentials_file)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600 or info.st_size > 4096:
                raise PolicyError("Benchmark credentials must be an ordinary0600 file")
            raw = handle.read(4097)
            if len(raw) > 4096:
                raise PolicyError("Benchmark credentials exceed the bounded file size")
        record = json.loads(raw)
        if (not isinstance(record, dict) or set(record) != {"format", "login", "password", "user_id", "snapshot_id"}
                or record["format"] != "rag-benchmark-v1" or any(not isinstance(value, str) for value in record.values())):
            raise PolicyError("Invalid private benchmark credentials")
        login = normalize_login(record["login"])
        with database.transaction() as connection:
            observed = connection.execute(sa.select(schema.users).where(schema.users.c.login == login,
                schema.users.c.user_id == record["user_id"])).mappings().one_or_none()
            observed = dict(observed) if observed is not None else None
        if (observed is None or observed["account_kind"] != "benchmark" or not observed["is_active"]
                or not (hasher or PasswordHasher()).verify(observed["password_hash"], record["password"])):
            raise PolicyError("Private benchmark credentials are unavailable")
        session_id = uuid4().hex
        users, sessions = schema.users, schema.sessions
        with database.transaction() as connection:
            now = session_time(connection, clock=clock)
            values = {"session_id": session_id, "token_hash": _sha(secrets.token_bytes(32)),
                      "csrf_hash": _sha(secrets.token_bytes(32)), "created_at": now,
                      "expires_at": now + timedelta(seconds=ttl_seconds)}
            selection = sa.select(*(sa.literal(value) for value in values.values()), users.c.user_id, users.c.session_version).where(
                users.c.user_id == observed["user_id"], users.c.account_kind == "benchmark", users.c.is_active.is_(True),
                users.c.session_version == observed["session_version"], users.c.password_hash == observed["password_hash"])
            inserted = connection.execute(sessions.insert().from_select(tuple(values) + ("user_id", "session_version"), selection)
                .returning(sessions.c.session_id)).scalar_one_or_none()
            if inserted is None:
                raise PolicyError("Private benchmark credentials changed during verification")
        return ResolvedIdentity(user_id=observed["user_id"], session_id=session_id,
            session_version=observed["session_version"], expires_at=values["expires_at"])
    except (OSError, ValueError, sa.exc.SQLAlchemyError):
        raise PolicyError("Private benchmark authentication failed") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for action in ("dry-run", "apply", "revoke"):
        command = commands.add_parser(action)
        command.add_argument("--manifest", required=True, type=Path)
    benchmark = commands.add_parser("benchmark-create")
    benchmark.add_argument("login")
    benchmark.add_argument("--snapshot", required=True, type=Path)
    benchmark.add_argument("--credentials-file", required=True, type=Path)
    benchmark.add_argument("--reviewed-by", required=True)
    benchmark.add_argument("--provenance", required=True)
    benchmark.add_argument("--all-documents", action="store_true")
    benchmark.add_argument("--synthetic", action="store_true")
    verify = commands.add_parser("benchmark-verify")
    verify.add_argument("--credentials-file", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "benchmark-verify":
            from src.db.runtime import DatabaseSettings
            database = Database(DatabaseSettings.from_env())
            try:
                verify_benchmark(args.credentials_file, database)
            finally:
                database.close()
        else:
            repository = PolicyRepository()
            if args.command == "benchmark-create":
                repository.provision_benchmark(args.login, snapshot=SnapshotRef.model_validate_json(args.snapshot.read_bytes()),
                    credentials_file=args.credentials_file, reviewed_by=args.reviewed_by,
                    provenance={"review": args.provenance}, all_documents=args.all_documents,
                    origin="synthetic" if args.synthetic else "operator")
            else:
                manifest = PolicyManifest.model_validate_json(args.manifest.read_bytes())
                result = getattr(repository, args.command.replace("-", "_"))(manifest)
                print(json.dumps(result.__dict__, sort_keys=True))
    except (ValueError, OSError, sa.exc.SQLAlchemyError):
        parser.error("Reviewed policy operation failed; check operator configuration and inputs")


if __name__ == "__main__":
    main()

"""Opaque local sessions, bounded login attempts and operation-boundary checks.

Only digests enter PostgreSQL. Password verification occurs after the initial
user read transaction has closed. Runtime credentials never update or row-lock
users; conditional writes and a fresh read reject intervening account changes.
Every later operation rechecks the account and issued version. This does not
promise an HTTP response cannot overlap a later administrative revocation.
"""

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import math
import re
import secrets
import threading
from contextlib import contextmanager
from uuid import uuid4

from pydantic import SecretStr
import sqlalchemy as sa

from src.access.contracts import ResolvedIdentity
from src.auth.accounts import AccountError, PasswordHasher, normalize_login, password_value
from src.auth.contracts import CsrfResponse, CurrentUser, SessionResponse
from src.auth.settings import AuthSettings
from src.db import schema
from src.db.runtime import Database, DatabaseRoleError

_TOKEN = re.compile(r"[A-Za-z0-9_-]{43}\Z")
_BOOTSTRAP = re.compile(r"v1\.([0-9]{1,12})\.([A-Za-z0-9_-]{43})\.([A-Za-z0-9_-]{43})\Z")


class AuthError(ValueError):
    def __init__(self, code, status, message, *, retry_after=None):
        self.code, self.status, self.message = code, status, message
        self.retry_after = retry_after
        super().__init__(message)


def _invalid_credentials():
    return AuthError("invalid_credentials", 401, "Login or password is invalid.")


def _expired():
    return AuthError("session_expired", 401, "This session is unavailable. Sign in again.")


def _unavailable():
    return AuthError("policy_unavailable", 503, "Authentication is temporarily unavailable.")


@dataclass(frozen=True)
class SessionGrant:
    token: SecretStr = field(repr=False)
    response: SessionResponse


@dataclass(frozen=True)
class AuthenticatedSession:
    identity: ResolvedIdentity
    response: SessionResponse


@dataclass(frozen=True)
class BootstrapGrant:
    token: SecretStr = field(repr=False)
    response: CsrfResponse = field(repr=False)


def _b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _digest(value):
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _csrf(token):
    return _b64(hashlib.sha256(b"rag-session-csrf-v1\0" + token.encode("ascii")).digest())


class _AttemptLimiter:
    """Fixed-window counters, two bounded LRU maps and a non-evictable cap."""

    def __init__(self, settings):
        self.settings, self.logins, self.ips = settings, OrderedDict(), OrderedDict()
        self.global_bucket = None
        self.lock = threading.Lock()

    def _bucket(self, previous, now):
        return (now, 0) if previous is None or now - previous[0] >= self.settings.throttle_window_seconds else previous

    def attempt(self, login, ip, now):
        ip = ip if isinstance(ip, str) and 0 < len(ip) <= 128 else "unknown"
        with self.lock:
            global_bucket = self._bucket(self.global_bucket, now)
            login_bucket = self._bucket(self.logins.get(login), now)
            ip_bucket = self._bucket(self.ips.get(ip), now)
            if (global_bucket[1] >= self.settings.global_attempts or login_bucket[1] >= self.settings.per_login_attempts
                    or ip_bucket[1] >= self.settings.per_ip_attempts):
                retry = max(1, math.ceil(self.settings.throttle_window_seconds - (now - global_bucket[0])))
                raise AuthError("rate_limited", 429, "Too many sign-in attempts. Try again later.", retry_after=retry)
            self.global_bucket = global_bucket[0], global_bucket[1] + 1
            for mapping, key, bucket in ((self.logins, login, login_bucket), (self.ips, ip, ip_bucket)):
                mapping[key] = bucket[0], bucket[1] + 1
                mapping.move_to_end(key)
                while len(mapping) > self.settings.throttle_capacity:
                    mapping.popitem(last=False)


class AuthService:
    def __init__(self, database: Database, settings: AuthSettings, *, clock=None, hasher=None, csrf_secret=None):
        if not isinstance(database, Database) or not isinstance(settings, AuthSettings):
            raise TypeError("Authentication requires the runtime Database and AuthSettings")
        self.database = database
        self.settings = AuthSettings.model_validate(settings.model_dump(warnings=False))
        if not self.settings.enabled:
            raise ValueError("Local authentication must be explicitly enabled")
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.hasher = hasher or PasswordHasher()
        self._dummy_hash = self.hasher.hash(secrets.token_urlsafe(32))
        self._csrf_secret = secrets.token_bytes(32) if csrf_secret is None else csrf_secret
        if not isinstance(self._csrf_secret, bytes) or len(self._csrf_secret) < 32:
            raise ValueError("Bootstrap signing requires at least32 secret bytes")
        self.limiter = _AttemptLimiter(self.settings)

    def _now(self):
        now = self.clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Authentication clock must return an aware datetime")
        return now.astimezone(timezone.utc)

    @contextmanager
    def _transaction(self):
        try:
            with self.database.transaction() as connection:
                yield connection
        except (sa.exc.SQLAlchemyError, DatabaseRoleError):
            raise _unavailable() from None

    @staticmethod
    def _raw_token(token):
        token = token.get_secret_value() if isinstance(token, SecretStr) else token
        if token is None or token == "":
            raise AuthError("authentication_required", 401, "Sign in to continue.")
        if not isinstance(token, str) or not _TOKEN.fullmatch(token):
            raise _expired()
        return token

    def issue_bootstrap(self) -> BootstrapGrant:
        payload = "v1." + str(int(self._now().timestamp())) + "." + secrets.token_urlsafe(32)
        token = payload + "." + _b64(hmac.new(self._csrf_secret, payload.encode("ascii"), hashlib.sha256).digest())
        return BootstrapGrant(SecretStr(token), CsrfResponse(csrf_token=token))

    def validate_bootstrap(self, cookie, header) -> None:
        cookie = cookie.get_secret_value() if isinstance(cookie, SecretStr) else cookie
        match = _BOOTSTRAP.fullmatch(cookie) if isinstance(cookie, str) else None
        if (match is None or not isinstance(header, str) or not _BOOTSTRAP.fullmatch(header)
                or not hmac.compare_digest(cookie, header)):
            raise AuthError("csrf_failed", 403, "Refresh the sign-in page and try again.")
        issued = int(match.group(1))
        now = self._now().timestamp()
        payload, signature = cookie.rsplit(".", 1)
        expected = _b64(hmac.new(self._csrf_secret, payload.encode("ascii"), hashlib.sha256).digest())
        if issued > now or now - issued >= self.settings.bootstrap_seconds or not hmac.compare_digest(signature, expected):
            raise AuthError("csrf_failed", 403, "Refresh the sign-in page and try again.")

    def login(self, login, password, *, client_ip, previous_token=None) -> SessionGrant:
        try:
            login, password = normalize_login(login), password_value(password)
        except AccountError:
            raise _invalid_credentials() from None
        self.limiter.attempt(login, client_ip, self._now().timestamp())
        with self._transaction() as connection:
            observed = connection.execute(sa.select(schema.users).where(schema.users.c.login == login)).mappings().one_or_none()
            observed = dict(observed) if observed is not None else None
        # No database connection/transaction is held during password verification.
        try:
            verified = self.hasher.verify(observed["password_hash"] if observed else self._dummy_hash, password)
        except Exception:
            raise _unavailable() from None
        if not verified or observed is None or not observed["is_active"] or observed["account_kind"] != "human":
            raise _invalid_credentials()
        raw, session_id, now = secrets.token_urlsafe(32), uuid4().hex, self._now()
        csrf = _csrf(raw)
        expires = now + timedelta(seconds=self.settings.idle_seconds)
        users, sessions = schema.users, schema.sessions
        values = {"session_id": session_id, "token_hash": _digest(raw), "csrf_hash": _digest(csrf),
                  "created_at": now, "expires_at": expires}
        columns = tuple(values) + ("user_id", "session_version")
        selection = sa.select(*(sa.literal(value) for value in values.values()), users.c.user_id, users.c.session_version).where(
            users.c.user_id == observed["user_id"], users.c.is_active.is_(True), users.c.account_kind == "human",
            users.c.session_version == observed["session_version"], users.c.password_hash == observed["password_hash"])
        with self._transaction() as connection:
            inserted = connection.execute(sessions.insert().from_select(columns, selection).returning(sessions.c.session_id)).scalar_one_or_none()
            if inserted is None:
                raise _invalid_credentials()
            previous_token = previous_token.get_secret_value() if isinstance(previous_token, SecretStr) else previous_token
            if isinstance(previous_token, str) and _TOKEN.fullmatch(previous_token):
                connection.execute(sessions.update().where(sessions.c.token_hash == _digest(previous_token),
                    sessions.c.revoked_at.is_(None)).values(revoked_at=now))
        try:
            authenticated = self.authenticate(raw, touch=False)
        except AuthError as error:
            if error.status == 401:
                raise _invalid_credentials() from None
            raise
        return SessionGrant(SecretStr(raw), authenticated.response)

    def _session(self, connection, raw):
        sessions, users = schema.sessions, schema.users
        return connection.execute(sa.select(sessions.c.session_id, sessions.c.user_id, sessions.c.csrf_hash,
            sessions.c.created_at, sessions.c.expires_at, sessions.c.revoked_at,
            sessions.c.session_version.label("issued_version"), users.c.session_version.label("account_version"),
            users.c.is_active, users.c.password_hash, users.c.account_kind).select_from(sessions.join(users, sessions.c.user_id == users.c.user_id))
            .where(sessions.c.token_hash == _digest(raw)).with_for_update(of=sessions)).mappings().one_or_none()

    def _valid(self, row, now, raw):
        if (row is None or row["revoked_at"] is not None or not row["is_active"] or row["account_kind"] != "human"
                or row["issued_version"] != row["account_version"] or row["created_at"] > now
                or row["expires_at"] <= now
                or row["created_at"] + timedelta(seconds=self.settings.absolute_seconds) <= now
                or not hmac.compare_digest(row["csrf_hash"], _digest(_csrf(raw)))):
            raise _expired()

    def _account(self, connection, row):
        users = schema.users
        account = connection.execute(sa.select(users).where(users.c.user_id == row["user_id"], users.c.is_active.is_(True),
            users.c.account_kind == "human",
            users.c.session_version == row["issued_version"], users.c.password_hash == row["password_hash"])).mappings().one_or_none()
        if account is None:
            raise _expired()
        return account

    def authenticate(self, token, *, touch=True) -> AuthenticatedSession:
        raw = self._raw_token(token)
        sessions, users = schema.sessions, schema.users
        with self._transaction() as connection:
            row = self._session(connection, raw)
            now = self._now()
            self._valid(row, now, raw)
            expires = row["expires_at"]
            if touch:
                expires = min(now + timedelta(seconds=self.settings.idle_seconds),
                              row["created_at"] + timedelta(seconds=self.settings.absolute_seconds))
                account_current = sa.exists(sa.select(users.c.user_id).where(users.c.user_id == row["user_id"],
                    users.c.is_active.is_(True), users.c.account_kind == "human", users.c.session_version == row["issued_version"],
                    users.c.password_hash == row["password_hash"]))
                changed = connection.execute(sessions.update().where(sessions.c.session_id == row["session_id"],
                    sessions.c.revoked_at.is_(None), sessions.c.expires_at > now, account_current)
                    .values(expires_at=expires).returning(sessions.c.session_id)).scalar_one_or_none()
                if changed is None:
                    raise _expired()
            account = self._account(connection, row)
            groups = tuple(connection.execute(sa.select(schema.group_memberships.c.group_id).where(
                schema.group_memberships.c.user_id == row["user_id"]).order_by(schema.group_memberships.c.group_id)).scalars())
            public = CurrentUser.model_validate({name: account[name] for name in CurrentUser.model_fields})
            identity = ResolvedIdentity(user_id=row["user_id"], session_id=row["session_id"],
                session_version=row["issued_version"], expires_at=expires, role=public.role,
                department=public.department, project_ids=public.project_ids, group_ids=groups)
            # Recheck after membership I/O before releasing the authenticated DTO.
            self._account(connection, row)
            self._valid({**row, "expires_at": expires}, self._now(), raw)
        return AuthenticatedSession(identity, SessionResponse(user=public, expires_at=expires, csrf_token=_csrf(raw)))

    def logout(self, token, csrf_token) -> None:
        raw = self._raw_token(token)
        with self._transaction() as connection:
            row = self._session(connection, raw)
            now = self._now()
            self._valid(row, now, raw)
            if not isinstance(csrf_token, str) or not _TOKEN.fullmatch(csrf_token) or not hmac.compare_digest(row["csrf_hash"], _digest(csrf_token)):
                raise AuthError("csrf_failed", 403, "Refresh your session and try again.")
            self._account(connection, row)
            now = self._now()
            self._valid(row, now, raw)
            connection.execute(schema.sessions.update().where(schema.sessions.c.session_id == row["session_id"],
                schema.sessions.c.revoked_at.is_(None)).values(revoked_at=now))

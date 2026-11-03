"""A bounded, unprivileged PostgreSQL pool with transaction-local authority.

Only server-resolved RequestContext objects bind identity. This is connection
hygiene, not authentication or grant resolution; AUTH/ACL implement those.
Keep transactions short and close them before retrieval or provider calls.
"""

from contextlib import contextmanager
import os
from typing import Annotated, Mapping

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.pool import QueuePool

from src.access.contracts import RequestContext


class DatabaseConfigurationError(ValueError):
    pass


class DatabaseRoleError(RuntimeError):
    pass


def normalize_url(value: str | SecretStr | URL) -> URL:
    try:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        url = value if isinstance(value, URL) else make_url(value)
        if url.drivername == "postgresql":
            url = url.set(drivername="postgresql+psycopg")
        if url.drivername != "postgresql+psycopg" or not url.database:
            raise ValueError()
        # SQLAlchemy masks userinfo passwords, but not libpq secret query values.
        if {"password", "sslpassword"} & {key.lower() for key in url.query}:
            raise ValueError()
        return url
    except (ArgumentError, ValueError, TypeError):
        raise DatabaseConfigurationError("A PostgreSQL psycopg database URL is required") from None


def database_url(*, migration=False, environ: Mapping[str, str] | None = None) -> URL:
    """Read an explicit secret DSN or build a safely escaped structured URL."""
    values = os.environ if environ is None else environ
    direct = values.get("RAG_MIGRATION_DATABASE_URL" if migration else "RAG_DATABASE_URL")
    if direct:
        return normalize_url(direct)
    host = values.get("RAG_DB_HOST")
    password = values.get("RAG_DB_PASSWORD")
    if not host or not password:
        raise DatabaseConfigurationError("Database host and password must be configured")
    try:
        port = int(values.get("RAG_DB_PORT", "5432"))
        if not 1 <= port <= 65535:
            raise ValueError()
        return URL.create("postgresql+psycopg", host=host, port=port,
            username=values.get("RAG_DB_USER", "rag_migrator" if migration else "rag_runtime"),
            password=password, database=values.get("RAG_DB_NAME", "rag"))
    except (TypeError, ValueError):
        raise DatabaseConfigurationError("Database connection settings are invalid") from None


class DatabaseSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False,
                              hide_input_in_errors=True)
    dsn: SecretStr
    pool_size: Annotated[int, Field(strict=True, ge=1, le=4)] = 2
    pool_timeout_seconds: Annotated[float, Field(gt=0, le=30)] = 5
    connect_timeout_seconds: Annotated[int, Field(strict=True, ge=1, le=30)] = 5
    statement_timeout_ms: Annotated[int, Field(strict=True, ge=1, le=60000)] = 5000
    idle_transaction_timeout_ms: Annotated[int, Field(strict=True, ge=1, le=60000)] = 5000

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None):
        url = database_url(environ=environ)
        return cls(dsn=url.render_as_string(hide_password=False))


_ROLE_CHECK = """
SELECT r.rolsuper, r.rolbypassrls, r.rolcreaterole, r.rolcreatedb, r.rolreplication,
       current_user IS DISTINCT FROM session_user
       OR current_user IS DISTINCT FROM %s,
       EXISTS (
         SELECT 1 FROM pg_namespace n
         WHERE n.nspname = 'rag' AND pg_has_role(current_user, n.nspowner, 'MEMBER')
       ) OR EXISTS (
         SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'rag' AND pg_has_role(current_user, c.relowner, 'MEMBER')
       ) OR EXISTS (
         SELECT 1 FROM pg_database d WHERE d.datname = current_database()
         AND pg_has_role(current_user, d.datdba, 'MEMBER')
       )
FROM pg_roles r WHERE r.rolname = current_user
"""

_AUTHORITY_KEYS = ("user_id", "session_id", "session_version",
                   "policy_revision", "snapshot_id", "scope_id")


def _check_role(dbapi_connection):
    """Reject privileged credentials on every checkout, including pooled reuse."""
    try:
        with dbapi_connection.cursor() as cursor:
            # libpq retains the authenticated startup user even if server-side
            # role/session-authorization options mask current_user/session_user.
            cursor.execute(_ROLE_CHECK, (dbapi_connection.info.user,))
            row = cursor.fetchone()
        dbapi_connection.rollback()
    except BaseException:
        dbapi_connection.rollback()
        raise
    if row is None or any(row):
        raise DatabaseRoleError("Runtime database credentials must be unprivileged and nonowner")


class Database:
    def __init__(self, settings: DatabaseSettings):
        self.settings = settings
        self.engine = create_engine(normalize_url(settings.dsn),
            poolclass=QueuePool, pool_size=settings.pool_size, max_overflow=0,
            pool_timeout=settings.pool_timeout_seconds, pool_pre_ping=True,
            pool_reset_on_return=None, hide_parameters=True,
            connect_args={"prepare_threshold": None,
                          "connect_timeout": settings.connect_timeout_seconds,
                          "application_name": "rag-runtime"})

        @event.listens_for(self.engine, "checkout")
        def validate_role(connection, record, _proxy):
            try:
                _check_role(connection)
            except BaseException as error:
                record.invalidate(error)
                raise

        @event.listens_for(self.engine, "reset")
        def reset_session(connection, _record, state):
            if state.terminate_only:
                return
            connection.rollback()
            previous = connection.autocommit
            connection.autocommit = True
            try:
                with connection.cursor() as cursor:
                    cursor.execute("DISCARD ALL")
            finally:
                connection.autocommit = previous

    @contextmanager
    def transaction(self, context: RequestContext | None = None):
        if context is not None:
            if not isinstance(context, RequestContext):
                raise TypeError("Database identity must be a server-resolved request context")
            context.assert_bound()
            identity = {
                "user_id": context.identity.user_id,
                "session_id": context.identity.session_id,
                "session_version": str(context.binding.session_version),
                "policy_revision": str(context.binding.policy_revision),
                "snapshot_id": context.binding.snapshot.snapshot_id,
                "scope_id": context.binding.scope_id,
            }
        else:
            identity = {key: "" for key in _AUTHORITY_KEYS}
        with self.engine.connect() as connection, connection.begin():
            # Parameters are values to set_config, never SQL identifier fragments.
            settings = {
                "statement_timeout": str(self.settings.statement_timeout_ms),
                "idle_in_transaction_session_timeout": str(self.settings.idle_transaction_timeout_ms),
                "lock_timeout": str(min(2000, self.settings.statement_timeout_ms)),
                **{"rag." + key: value for key, value in identity.items()},
            }
            for key, value in settings.items():
                connection.execute(text("SELECT set_config(:key, :value, true)"),
                                   {"key": key, "value": value})
            yield connection
            if context is not None:
                context.assert_bound()

    def close(self):
        self.engine.dispose()

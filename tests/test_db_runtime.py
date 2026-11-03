"""Configuration boundaries that protect credentials and bound database work."""

import pytest
from pydantic import ValidationError

from src.db.runtime import (
    Database, DatabaseConfigurationError, DatabaseSettings, database_url, normalize_url,
)


def test_secret_url_components_round_trip_without_entering_repr():
    secret = "synthetic-'\\$;@password"
    url = database_url(environ={"RAG_DB_HOST": "db", "RAG_DB_USER": "rag_runtime",
        "RAG_DB_PASSWORD": secret, "RAG_DB_NAME": "rag"})
    assert url.password == secret
    settings = DatabaseSettings(dsn=url.render_as_string(hide_password=False))
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()
    assert normalize_url(settings.dsn).password == secret


@pytest.mark.parametrize("url", [
    "sqlite:///secret-value", "mysql://u:secret-value@db/app",
    "not-a-url-secret-value", "postgresql://u:secret-value@db",
])
def test_invalid_dialects_and_missing_database_do_not_echo_credentials(url):
    with pytest.raises(DatabaseConfigurationError) as error:
        normalize_url(url)
    assert "secret-value" not in str(error.value)


@pytest.mark.parametrize("name", ["password", "sslpassword", "PASSWORD"])
def test_secret_query_parameters_are_rejected_before_engine_repr_can_expose_them(name):
    with pytest.raises(DatabaseConfigurationError) as error:
        Database(DatabaseSettings(
            dsn=f"postgresql://runtime@db/rag?{name}=synthetic-secret-value"))
    assert "synthetic-secret-value" not in str(error.value)


@pytest.mark.parametrize("change", [
    {"pool_size": 0}, {"pool_size": 5}, {"pool_size": True},
    {"pool_timeout_seconds": float("nan")}, {"pool_timeout_seconds": float("inf")},
    {"statement_timeout_ms": 0}, {"idle_transaction_timeout_ms": 60001},
])
def test_connection_and_transaction_limits_are_bounded(change):
    with pytest.raises(ValidationError):
        DatabaseSettings(dsn="postgresql://runtime:synthetic@db/rag", **change)


def test_explicit_migration_credentials_are_separate_from_runtime_credentials():
    values = {
        "RAG_DATABASE_URL": "postgresql://rag_runtime:synthetic-runtime@db/rag",
        "RAG_MIGRATION_DATABASE_URL": "postgresql://rag_migrator:synthetic-owner@db/rag",
    }
    assert database_url(environ=values).username == "rag_runtime"
    assert database_url(migration=True, environ=values).username == "rag_migrator"
    with pytest.raises(DatabaseConfigurationError):
        database_url(environ={})


def test_runtime_rejects_client_identity_objects_before_a_connection_is_requested():
    db = Database(DatabaseSettings(dsn="postgresql://runtime:synthetic@invalid/rag"))
    try:
        with pytest.raises(TypeError, match="server-resolved"):
            with db.transaction({"user_id": "executive"}):
                pytest.fail("Untrusted identity was accepted")
    finally:
        db.close()

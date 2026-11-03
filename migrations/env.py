"""Alembic environment: one PostgreSQL schema and forward-only owned revisions."""

from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from src.db.runtime import database_url
from src.db.schema import metadata

config = context.config


def configure(connection=None, url=None):
    def include_name(name, kind, parent_names):
        if kind == "schema":
            return name == "rag"
        return parent_names.get("schema_name") == "rag"

    options = dict(target_metadata=metadata, version_table_schema="rag",
                   include_schemas=True, include_name=include_name,
                   compare_type=True, compare_server_default=True)
    if connection is not None:
        context.configure(connection=connection, **options)
    else:
        context.configure(url=url, literal_binds=True, dialect_opts={"paramstyle": "named"}, **options)


def online():
    existing = config.attributes.get("connection")
    if existing is not None:
        configure(connection=existing)
        with context.begin_transaction():
            context.run_migrations()
        return
    engine = create_engine(database_url(migration=True), poolclass=NullPool, hide_parameters=True)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql("SELECT pg_advisory_xact_lock(59800001)")
            connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS rag")
            configure(connection=connection)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    configure(url=database_url(migration=True))
    context.execute("CREATE SCHEMA IF NOT EXISTS rag")
    with context.begin_transaction():
        context.run_migrations()
else:
    online()

"""Forward migration entry points. Migration credentials never enter the pool."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

from src.db.runtime import database_url, normalize_url

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HEAD = "0002_catalog_import"


def migration_config():
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    return config


def migration_heads() -> tuple[str, ...]:
    return tuple(ScriptDirectory.from_config(migration_config()).get_heads())


def upgrade_database(url=None):
    engine = create_engine(normalize_url(url) if url is not None else database_url(migration=True),
                           poolclass=NullPool, hide_parameters=True)
    try:
        with engine.begin() as connection:
            # Serialize migrations in this database and let version-table creation
            # happen in the same transaction as the schema revision.
            connection.exec_driver_sql("SELECT pg_advisory_xact_lock(59800001)")
            connection.exec_driver_sql("CREATE SCHEMA IF NOT EXISTS rag")
            config = migration_config()
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    finally:
        engine.dispose()


def current_revision(url) -> str | None:
    engine = create_engine(normalize_url(url), poolclass=NullPool, hide_parameters=True)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(connection, opts={"version_table_schema": "rag"})
            return context.get_current_revision()
    finally:
        engine.dispose()


if __name__ == "__main__":
    upgrade_database()

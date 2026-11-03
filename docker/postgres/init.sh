#!/usr/bin/env bash
# Official-image init hook: NEW VOLUMES ONLY. Never a password-rotation tool.
# https://hub.docker.com/_/postgres#initialization-scripts
# psql literal/identifier interpolation and gexec:
# https://www.postgresql.org/docs/17/app-psql.html#APP-PSQL-INTERPOLATION
set +x
set -euo pipefail

: "${POSTGRES_USER:?POSTGRES_USER must be nonempty}"
: "${POSTGRES_DB:?POSTGRES_DB must be nonempty}"
: "${RAG_DB_MIGRATOR_PASSWORD:?RAG_DB_MIGRATOR_PASSWORD must be nonempty}"
: "${RAG_DB_RUNTIME_PASSWORD:?RAG_DB_RUNTIME_PASSWORD must be nonempty}"

# getenv avoids exposing passwords in process arguments. The quoted heredoc
# prevents shell expansion; :'name' and format(%I/%L) quote each SQL value.
psql --no-psqlrc --no-password --quiet \
    --username="$POSTGRES_USER" --dbname="$POSTGRES_DB" \
    --set=ON_ERROR_STOP=1 --set=VERBOSITY=terse --set=ECHO=none <<'SQL'
\getenv bootstrap_user POSTGRES_USER
\getenv database_name POSTGRES_DB
\getenv migrator_password RAG_DB_MIGRATOR_PASSWORD
\getenv runtime_password RAG_DB_RUNTIME_PASSWORD
BEGIN;
SET LOCAL search_path = pg_catalog;
-- Keep credential-bearing initialization statements out of session SQL logs.
SET LOCAL log_statement = 'none';
SET LOCAL log_min_error_statement = 'panic';
SET LOCAL log_min_duration_statement = -1;
SET LOCAL log_min_duration_sample = -1;
SET LOCAL log_duration = off;

SELECT (
    current_user = :'bootstrap_user'
    AND current_database() = :'database_name'
    AND current_database() NOT IN ('postgres', 'template0', 'template1')
    AND EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = current_user AND rolsuper)
    AND EXISTS (
        SELECT 1 FROM pg_catalog.pg_database d JOIN pg_catalog.pg_roles r ON r.oid = d.datdba
        WHERE d.datname = current_database() AND r.rolname = current_user AND d.datacl IS NULL
    )
    AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname IN ('rag_migrator', 'rag_runtime'))
    AND NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_namespace
        WHERE nspname NOT IN ('public', 'pg_catalog', 'information_schema') AND nspname !~ '^pg_'
    )
    AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_class WHERE relnamespace = 'public'::regnamespace)
    AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_proc WHERE pronamespace = 'public'::regnamespace)
    AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_type WHERE typnamespace = 'public'::regnamespace)
    AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_extension WHERE extname <> 'plpgsql')
) AS bootstrap_allowed
\gset
\if :bootstrap_allowed
\else
    DO $block$ BEGIN
        RAISE EXCEPTION 'Role bootstrap requires a pristine dedicated database and absent application roles';
    END $block$;
\endif

SELECT pg_catalog.format(
    'CREATE ROLE %I WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD %L',
    'rag_migrator', :'migrator_password')
\gexec
SELECT pg_catalog.format(
    'CREATE ROLE %I WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD %L',
    'rag_runtime', :'runtime_password')
\gexec
\unset migrator_password
\unset runtime_password

SELECT pg_catalog.format('ALTER DATABASE %I OWNER TO %I', :'database_name', 'rag_migrator')
\gexec
SELECT pg_catalog.format('REVOKE ALL PRIVILEGES ON DATABASE %I FROM PUBLIC', :'database_name')
\gexec
SELECT pg_catalog.format('GRANT CONNECT ON DATABASE %I TO %I, %I',
    :'database_name', 'rag_migrator', 'rag_runtime')
\gexec
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
ALTER ROLE rag_runtime SET search_path = pg_catalog, rag;
ALTER ROLE rag_runtime SET statement_timeout = '15s';
ALTER ROLE rag_runtime SET idle_in_transaction_session_timeout = '15s';
COMMIT;
SQL

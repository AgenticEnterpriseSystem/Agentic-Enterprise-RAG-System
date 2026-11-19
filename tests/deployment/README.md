<a id="tiny-real-auth-deployment-smoke"></a>

# Database and Docker tests

These checks use a tiny dataset to test installation and account behavior. They
are for contributors. You don't need them to follow the
[quick start](../../docs/QUICKSTART.md).

A **fixture** is prepared test data. This one has five fake documents, real
PostgreSQL, real password logins and the app's permission checks. Model
responses are scripted, so there are no paid calls.

## What CI checks automatically

**CI** means the automated GitHub checks. The complete sequence is in
[`.github/workflows/ci.yml`](../../.github/workflows/ci.yml):

1. Build the app and a separate setup container.
2. Create a disposable database and import the five documents and fake Jira data.
3. Confirm the app refuses to start serving before required setup is complete.
4. Create test accounts with different document access and check login/search.
5. Stop the database and confirm requests stop safely while the app stays alive.
6. Restore the database, restart services, and check that accounts, grants and
   sessions still behave correctly.
7. Remove only the generated disposable test installation.

The test project has a generated `rag-ci-fixture-*` name, separate from a
working installation. Its cleanup deletes data, so never use that command to
shut down a working app.

## Check the fixture configuration locally

You need Git, the Python environment from [Contributing](../../CONTRIBUTING.md),
and Docker with Compose v2.10 or later. Run from the repository root:

```sh
python tests/deployment/fixture.py prepare --directory .cache/deployment-smoke
docker compose --env-file .cache/deployment-smoke/compose.env \
  -f compose.yaml -f tests/deployment/compose.yaml \
  --profile ci-fixture --profile operator config --quiet
```

The first command makes a new directory and never overwrites one. The second
checks the configuration. **They don't run the full deployment test**; the CI
workflow does. Keep the root Compose file first, because the override builds on it.

The generated files contain test credentials and cookies. Don't commit or share
that directory.

## Real-account security acceptance

`test_access_end_to_end.py` checks two real accounts against a disposable
PostgreSQL database with scripted model responses. It covers permitted and
forbidden evidence, counts, document expansion, permission changes during a
request, and database failure and recovery.

Set the three URLs CI uses. All must point to the **same database**, whose name
starts with **`rag_test_`**:

| Setting | Role |
| --- | --- |
| `RAG_TEST_ADMIN_URL` | Create and inspect the disposable test database |
| `RAG_TEST_MIGRATION_URL` | Create schema and provision test data/accounts |
| `RAG_TEST_RUNTIME_URL` | Run the app using an unprivileged login |

Then run, from the repository root:

```sh
python -m pytest tests/test_access_end_to_end.py tests/test_security_operator.py
```

Without these settings the database tests are skipped. The CI workflow shows
the exact roles. Never point these tests at a working app's database.

## Optional checks on the full dataset

These advanced operator tools work only on the prepared
`rag-ops01-full-20261104` installation. They are not a teammate setup shortcut.
Don't change their project guards to run them elsewhere.

| Helper | Purpose |
| --- | --- |
| `security_operator.py` | Create test humans with small explicit grants, inspect them, revoke a grant and clean up only those test accounts |
| `security_http.py` | Check HTTP permissions before/after revocation, run ownership and logout |
| `frontend/tests/live-security.mjs` | Check the real browser using the same private test-account record |

The helper's private **ledger** records the accounts, grants and sessions it
owns. Cleanup touches only those. The browser helper can coordinate an
operator-controlled database interruption but doesn't write to the database. For
options, read the helper source files.

## Limits of these results

The fixture uses BM25 keyword search. Its small dummy embedding arrays only test
the import format; no embedding model is downloaded.

Passing does not show performance on all 511,958 documents, backup and restore
success, answer accuracy or production security. See
[Current status](../../docs/STATUS.md) for the full-corpus checks and remaining
work.

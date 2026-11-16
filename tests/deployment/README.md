<a id="tiny-real-auth-deployment-smoke"></a>

# Database and Docker tests

These checks use a tiny dataset to test installation and account behavior. They
are for contributors; you do not need to run them before following the ordinary
[app setup guide](../../docs/WEB_APP.md).

A **fixture** is prepared test data. This fixture has five fake documents, real
PostgreSQL storage, real password-based logins and the same permission checks as
the app. Model responses are scripted locally. There are no paid model calls.

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

The test project has a generated `rag-ci-fixture-*` name. It is separate from a
normal working installation. Its cleanup removes disposable data, so never copy
that cleanup command into a working app's shutdown routine.

## Check the fixture configuration locally

Use Git, the Python environment in [Contributing](../../CONTRIBUTING.md), Docker
and Compose v2.10 or later. Git is needed because the fixture creates a small
test repository. Run from the repository root:

```sh
python tests/deployment/fixture.py prepare --directory .cache/deployment-smoke
docker compose --env-file .cache/deployment-smoke/compose.env \
  -f compose.yaml -f tests/deployment/compose.yaml \
  --profile ci-fixture --profile operator config --quiet
```

The first command creates a new directory; it does not overwrite an existing
fixture. The second checks configuration. **These commands alone do not run the
full deployment test.** Use the CI workflow for the complete build/import/start/
restart sequence. Keep the root Compose file first because the override uses it
as its base.

The generated environment and operator files contain private test credentials
and cookies. Do not commit or share that directory.

## Real-account security acceptance

`test_access_end_to_end.py` checks two real local accounts against a disposable
PostgreSQL database, with scripted model responses. It covers permitted and
forbidden evidence, counts, document expansion, permission changes during a
request, and database failure/recovery.

Set the three disposable-database URLs used by CI. They must all point to the
**same database**, and its name must start with **`rag_test_`**:

| Setting | Role |
| --- | --- |
| `RAG_TEST_ADMIN_URL` | Create and inspect the disposable test database |
| `RAG_TEST_MIGRATION_URL` | Create schema and provision test data/accounts |
| `RAG_TEST_RUNTIME_URL` | Run the app using an unprivileged login |

Then run, from the repository root:

```sh
python -m pytest tests/test_access_end_to_end.py tests/test_security_operator.py
```

Without the required settings, database-dependent tests are skipped. Use the CI
workflow as the exact connection/role example; never point these tests at the
working application's database.

## Optional checks on the full dataset

The helpers below are advanced operator tools for the already-prepared
`rag-ops01-full-20261104` installation. They are not a generic teammate setup
shortcut. Do not change their project guards to make them run elsewhere.

| Helper | Purpose |
| --- | --- |
| `security_operator.py` | Create test humans with small explicit grants, inspect them, revoke a grant and clean up only those test accounts |
| `security_http.py` | Check HTTP permissions before/after revocation, run ownership and logout |
| `frontend/tests/live-security.mjs` | Check the real browser using the same private test-account record |

The private record is the helper's **ledger**: it records the accounts, grants
and sessions the helper owns. Cleanup acts on those recorded items, not unrelated
users or corpus data. The browser helper can coordinate an operator-controlled
database interruption; it does not write to the database itself. CLI options and
that advanced sequence are documented in the helper source files.

## Limits of these results

The five-document deployment uses BM25 keyword search. Its small dummy embedding
arrays exist only to test the import format; they are not a real searchable
embedding model. No full model is downloaded for this fixture.

Passing these checks does not establish performance on all 511,958 documents,
backup/restore success, answer accuracy or production security. Use
[Current status](../../docs/STATUS.md) for the separately recorded full-corpus
checks and remaining work.

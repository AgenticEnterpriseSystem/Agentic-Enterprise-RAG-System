# Contributing

Use this guide when changing code or documentation. To run the app first, follow
[Setup](docs/WEB_APP.md). Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Workflow

1. Start from the version your task is meant to change. For teammate testing,
   use the testing branch or exact commit shared by the team. For ordinary team
   integration, the existing flow is a personal branch into `staging`, then a
   separate reviewed change from `staging` into `main`.
2. Create your own branch, such as `hira/fix-login-message`. A branch keeps your
   changes separate until the team reviews them.
3. Make one focused change. Update the relevant guide if behavior changes.
4. Run checks that cover your change. See the commands below.
5. Commit the files you intend to share, push your branch, and open a **pull
   request (PR)**: a request for teammates to review and merge your changes.
6. In the PR, explain what changed, why, and which checks passed. Get a teammate's
   review and passing automated checks before merging.

Do not overwrite someone else's branch. A new feature or bug fix should include
useful tests; a wording-only change usually needs a link and accuracy check.

## Check Python changes

From the repository root, use Python 3.11. The commands below use a macOS/Linux
shell, matching the documented app setup. A native Windows app installation has
not been validated.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Tests use local fake model responses. A normal test run does not require a paid
API key. Database tests need a separate disposable PostgreSQL database and three
`RAG_TEST_*_URL` settings; all three must point to the same database, whose name
must start with `rag_test_`. Without them, those tests are skipped. Passing a run
with skipped database tests is not the same as passing the database checks.
The [CI workflow](.github/workflows/ci.yml) shows the disposable test setup.
Never use your working app's database as the test database.

## Check frontend changes

Use Node.js **22.12 or later**. From the repository root:

```sh
cd frontend
npm ci
npm run build
npm run test:browser
```

If the browser is missing, run `npm run test:browser:install`, then run the tests
again. These browser tests use fake API responses and do not call a model.
See the [browser test guide](frontend/tests/README.md) for the separate
Observability checks.

## Choose files for a commit

Include source files, relevant tests, setup files and documentation. New files
must be added too; committing only already-tracked changes can leave a teammate
with missing modules.

Keep these local: real `.env` files, API keys, passwords, session cookies, model
caches, indexes, downloaded data, `local-v2/`, `node_modules/` and generated build
files. `.env.example` is the shareable template. Check `git status` and the staged
diff before committing; editor settings and result spreadsheets may be untracked
without being ignored.

## Issues

For a bug, include the question or action that triggered it, expected behavior,
actual behavior and reproduction steps. Use a request ID or sanitized error code
when available. Do not attach passwords, API keys, cookies or private excerpts.

Use these labels to help the team find work:

| Label | Meaning |
| --- | --- |
| `bug` | Something does not work as expected |
| `enhancement` | A new feature or improvement |
| `documentation` | A guide or explanation needs updating |
| `question` | A clarification is needed |
| `good first issue` | A small task suitable for a new contributor |
| `help wanted` | Another contributor's help would be useful |

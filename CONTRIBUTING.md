# Contributing

To run the app first, follow the [quick start](docs/QUICKSTART.md). Please follow
the [Code of Conduct](CODE_OF_CONDUCT.md).

## Workflow

1. Start from the version your task changes. For teammate testing, use the
   branch or commit the team shared. Otherwise, the flow is a personal branch
   into `staging`, then a separately reviewed change from `staging` into `main`.
2. Create your own branch, such as `hira/fix-login-message`. Don't overwrite
   someone else's.
3. Make one focused change. Update the relevant guide if behavior changes.
4. Run the checks below that cover your change.
5. Commit the files you mean to share, push, and open a **pull request (PR)**: a
   request for teammates to review and merge your changes.
6. In the PR, say what changed, why, and which checks passed. Merge after a
   teammate's review and passing automated checks.

A feature or bug fix should include tests. A wording-only change needs a link
and accuracy check.

## Check Python changes

Use Python 3.11 on macOS or Linux. A native Windows installation hasn't been
validated.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

Tests use fake model responses, so they need no paid API key.

Database tests need a separate disposable PostgreSQL database and three
`RAG_TEST_*_URL` settings. All three must point to the same database, and its
name must start with `rag_test_`. Without them these tests are **skipped**, and
a run with skipped tests has not passed the database checks. See the
[CI workflow](.github/workflows/ci.yml) for the setup. Never use your working
app's database for tests.

## Check frontend changes

Use Node.js **22.12 or later**.

```sh
cd frontend
npm ci
npm run build
npm run test:browser
```

If the browser is missing, run `npm run test:browser:install` and try again.
These tests use fake API responses and call no model. See the
[browser test guide](frontend/tests/README.md).

## Choose files for a commit

Include source, tests, setup files and docs. Add new files too, or teammates will
get missing modules.

Keep these local: real `.env` files, API keys, passwords, cookies, model caches,
indexes, downloaded data, `local-v2/`, `node_modules/` and build output.
`.env.example` is the shareable template. Check `git status` and the staged diff
first. Editor settings and result spreadsheets can be untracked but not ignored.

## Issues

For a bug, give the question or action, expected and actual behavior, and steps
to reproduce. Add a request ID or sanitized error code if you have one. Never
attach passwords, API keys, cookies or private excerpts.

| Label | Meaning |
| --- | --- |
| `bug` | Something doesn't work as expected |
| `enhancement` | A new feature or improvement |
| `documentation` | A guide needs updating |
| `question` | A clarification is needed |
| `good first issue` | A small task for a new contributor |
| `help wanted` | Extra help would be useful |

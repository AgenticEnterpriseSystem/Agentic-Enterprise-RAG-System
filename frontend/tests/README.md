<a id="browser-regression-fixtures"></a>

# Browser tests

These tests check the React interface in a real browser using **fake API
responses**. They do not connect to the working app, PostgreSQL or a language
model. You do not need a paid API key or the full dataset.

For an actual app test, follow [Setup](../../docs/WEB_APP.md). Read
[Current status](../../docs/STATUS.md) for the latest real-browser results.

## Run the main browser tests

You need Node.js **22.12 or later**. From the repository root:

```sh
cd frontend
npm ci
npm run test:browser
```

On macOS, the runner uses installed Google Chrome when available. Otherwise it
uses Playwright's Chromium browser. If no browser is installed, run:

```sh
npm run test:browser:install
npm run test:browser
```

`PLAYWRIGHT_CHANNEL` can override the browser choice. The runner starts its own
local fake server; leave the working Docker app separate. Results, temporary
profiles and failure screenshots go in `.cache/ui-01/` at the repository root.

## Run the Observability tests

This is a separate test file. It currently requires installed **Google Chrome**,
including on systems where the main suite uses Chromium. From `frontend/`:

```sh
node --test tests/observability.browser.mjs
```

It starts its own fake server on port 4387. It checks admin access, safe error
information, request details, logout and account changes. It does not read real
request history or submit a model question.

## What the tests check

- Signing in, signing out, account changes and expired sessions.
- A late response cannot put old evidence back on screen after logout.
- Two browser tabs respond correctly when their shared session changes.
- The interface displays passages, Jira records and counts, including zero
  counts and missing fields.
- Citation clicks focus the correct source card.
- Sensitive evidence is not saved in browser storage.

The fake server uses real browser cookies and lets tests delay responses. These
controls exist only in test code, not the production interface. Fake users and
passwords here are not logins for the working app.

## What passing does not prove

A passing result shows that the tested interface behavior works with those fake
responses. It does not verify real database permissions, full-corpus performance,
model access or answer accuracy.

For database and Docker checks, see
[Deployment tests](../../tests/deployment/README.md). For a full live check, use
real permitted accounts and a separately agreed model budget.

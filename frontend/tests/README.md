<a id="browser-regression-fixtures"></a>

# Browser tests

These tests check the React interface in a real browser using **fake API
responses**. They don't touch the working app, PostgreSQL or a language model,
and need no API key or dataset.

To run the real app, follow the [quick start](../../docs/QUICKSTART.md). For the
latest real-browser results, see [Current status](../../docs/STATUS.md).

## Run the main browser tests

You need Node.js **22.12 or later**. From the repository root:

```sh
cd frontend
npm ci
npm run test:browser
```

The runner uses Google Chrome on macOS if installed, otherwise Playwright's
Chromium. If neither is installed, run:

```sh
npm run test:browser:install
npm run test:browser
```

`PLAYWRIGHT_CHANNEL` overrides the browser choice. The runner starts its own fake
server, separate from your Docker app. Results and failure screenshots go in
`.cache/ui-01/` at the repository root.

## Run the Observability tests

This separate test file needs **Google Chrome** installed. From `frontend/`:

```sh
node --test tests/observability.browser.mjs
```

It starts its own fake server on port 4387 and checks admin access, safe error
information, request details, logout and account changes. It reads no real
history and asks no model question.

## What the tests check

- Signing in, signing out, account changes and expired sessions.
- A late response cannot put old evidence back on screen after logout.
- Two browser tabs respond correctly when their shared session changes.
- The interface displays passages, Jira records and counts, including zero
  counts and missing fields.
- Citation clicks focus the correct source card.
- Sensitive evidence is not saved in browser storage.

The fake server uses real browser cookies and can delay responses. These
controls exist only in tests. Its fake users are not logins for the working app.

## What passing does not prove

Passing shows the interface works with those fake responses. It does not verify
real database permissions, full-corpus performance, model access or answer
accuracy. For database and Docker checks, see
[Deployment tests](../../tests/deployment/README.md). For a live check, use real
permitted accounts and an agreed model budget.

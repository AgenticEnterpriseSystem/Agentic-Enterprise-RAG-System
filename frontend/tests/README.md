# Browser regression fixtures

`npm run test:browser` runs the real frontend in headless Chrome when it is
already installed on macOS; otherwise it uses Playwright's Chromium.
`npm run test:browser:install` installs only Chromium into `../.cache/ui-01/browsers`.
An explicit `PLAYWRIGHT_CHANNEL` overrides the installed-browser choice.

The localhost-only mock server is test code. It uses synthetic users, passwords,
passages and real HttpOnly browser cookies; it never connects to a database or
model. Its `/__tests/*` controls are not included in the production frontend.
Results, temporary browser profiles and failure screenshots stay in `.cache/ui-01`.

Tests hold genuine HTTP responses across logout, account changes and cookie
rotation. One fault injection deliberately ignores query cancellation so a late
response and its `finally` handler must be rejected by the application's session
generation checks. Two pages in one browser context exercise shared cookies and
cross-tab notifications. Browser local/session storage, IndexedDB and CacheStorage
must remain empty after authenticated evidence rendering.

Native-evidence cases use version-3 record and mixed responses plus version-2
count responses. They cover zero counts with explicit source-record units,
verified/missing/invalid fields, duplicate-key ambiguity, unknown coverage,
all-kind citation focus, and mixed responses arriving after an account change.
Native fixtures have no fabricated `chunk`, text excerpt or character offsets.

The runner uses pinned Playwright1.64.1. Configuration and installed Chrome channel
follow the [official test configuration](https://playwright.dev/docs/test-configuration)
and [browser guidance](https://playwright.dev/docs/browsers).

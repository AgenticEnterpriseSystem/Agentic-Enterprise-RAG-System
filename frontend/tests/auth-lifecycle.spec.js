import { test, expect } from "@playwright/test";

const PASSWORD = "SYNTHETIC_PASSWORD_DO_NOT_STORE";
const INTERN = "INTERN_PRIVATE_EVIDENCE_TEST_ONLY";
const EXECUTIVE = "EXECUTIVE_PRIVATE_EVIDENCE_TEST_ONLY";
const names = { intern: "Alex Intern", executive: "Robin Executive" };

test.beforeEach(async ({ request }) => {
  expect((await request.post("/__tests/reset")).ok()).toBeTruthy();
});

async function settings(request, value) {
  expect((await request.post("/__tests/settings", { data: value })).ok()).toBeTruthy();
}
async function status(request) {
  return (await request.get("/__tests/status")).json();
}
async function signIn(page, login = "intern") {
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await page.getByLabel("Login", { exact: true }).fill(login);
  await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByText(names[login], { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
}
async function ask(page, question) {
  const field = page.getByLabel("What would you like to know?", { exact: true });
  await expect(field).toBeEnabled();
  await field.fill(question);
  await field.press("Enter");
}
async function pending(request, login) {
  await expect.poll(async () => (await status(request)).pending.filter((entry) => entry.login === login).length).toBe(1);
  return (await status(request)).pending.find((entry) => entry.login === login).id;
}
async function release(page, request, id, question) {
  await Promise.all([
    page.waitForResponse((response) => response.url().endsWith("/api/query")
      && response.request().postDataJSON()?.question === question).then((response) => response.finished()),
    request.post("/__tests/release", { data: { id } }),
  ]);
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}
async function ignoreQueryAbort(page) {
  // Cancellation is best effort. Keep the old real HTTP response alive to
  // prove generation checks also protect result/catch/finally after switching.
  await page.addInitScript(() => {
    const original = window.fetch;
    window.fetch = (input, options = {}) => original(input,
      typeof input === "string" && input === "/api/query" ? { ...options, signal: undefined } : options);
  });
}
async function noSensitiveStorage(page) {
  const value = await page.evaluate(async () => ({
    local: Object.entries(localStorage), session: Object.entries(sessionStorage),
    databases: await indexedDB.databases(), caches: await caches.keys(), visibleCookies: document.cookie,
  }));
  expect(value).toEqual({ local: [], session: [], databases: [], caches: [], visibleCookies: "" });
}

test("keyboard sign-in errors, fixed RAG, citation focus and private memory-only state", async ({ page, request, context }) => {
  // Sign-in can finish before API startup. Hold real readiness HTTP responses
  // so the test verifies autofocus only after submitting becomes available.
  let releaseReadiness;
  const readiness = new Promise((resolve) => { releaseReadiness = resolve; });
  await page.route("**/api/ready", async (route) => {
    await readiness;
    await route.fulfill({ json: { status: "ready", busy: false, mode: "live", retrieval: "hybrid", generation_enabled: true } });
  });
  await page.goto("/");
  await expect(page.getByLabel("Login", { exact: true })).toBeFocused();
  await page.getByLabel("Login", { exact: true }).fill("intern");
  await page.getByLabel("Password", { exact: true }).fill("wrong-test-password");
  await page.getByLabel("Password", { exact: true }).press("Enter");
  await expect(page.getByRole("alert")).toContainText("incorrect");
  await expect(page.getByRole("alert")).toBeFocused();
  await page.getByLabel("Login", { exact: true }).fill("intern");
  await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
  await page.getByLabel("Password", { exact: true }).press("Enter");
  const field = page.getByLabel("What would you like to know?", { exact: true });
  await expect(page.getByText(names.intern, { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Ask question", exact: false })).toBeDisabled();
  await expect(field).not.toBeFocused();
  releaseReadiness();
  await expect(field).toBeFocused();
  await page.getByRole("radio", { name: "Fixed RAG", exact: true }).check();
  await field.fill("First synthetic line");
  await field.press("Shift+Enter");
  await field.pressSequentially("Second synthetic line");
  await expect(field).toHaveValue("First synthetic line\nSecond synthetic line");
  expect((await status(request)).queries).toHaveLength(0);
  await field.press("Enter");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  const queries = (await status(request)).queries;
  expect(queries).toHaveLength(1);
  expect(queries[0]).toMatchObject({ login: "intern", mode: "baseline", question: "First synthetic line\nSecond synthetic line" });
  await page.getByRole("button", { name: "View Source 1", exact: true }).click();
  await expect(page.getByRole("article").filter({ hasText: "intern synthetic source" })).toBeFocused();
  await page.locator("details.steps > summary").click();
  await expect(page.locator("details.steps")).toContainText("One search completed");
  const cookie = (await context.cookies()).find((item) => item.name === "rag_session");
  expect(cookie).toMatchObject({ httpOnly: true, sameSite: "Strict", path: "/" });
  await noSensitiveStorage(page);
});

test("API readiness changes do not steal anonymous password focus", async ({ page }) => {
  let releaseReadiness;
  const readiness = new Promise((resolve) => { releaseReadiness = resolve; });
  await page.route("**/api/ready", async (route) => {
    await readiness;
    await route.fulfill({ json: { status: "ready", busy: false, mode: "live", retrieval: "hybrid", generation_enabled: true } });
  });
  await page.goto("/");
  await expect(page.getByLabel("Login", { exact: true })).toBeFocused();
  await page.getByLabel("Login", { exact: true }).fill("intern");
  const password = page.getByLabel("Password", { exact: true });
  await password.fill(PASSWORD);
  await expect(password).toBeFocused();
  const readyResponse = page.waitForResponse((response) => response.url().endsWith("/api/ready"));
  releaseReadiness();
  await (await readyResponse).finished();
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await expect(password).toBeFocused();
  await expect(page.getByRole("button", { name: "Sign in", exact: true })).toBeEnabled();
});

test("late previous-user result and finally cannot overwrite the next user's pending query", async ({ page, request }) => {
  await ignoreQueryAbort(page);
  await page.goto("/");
  await signIn(page, "executive");
  await settings(request, { delayNext: 1 });
  await ask(page, "Executive old request");
  const oldId = await pending(request, "executive");
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await signIn(page, "intern");
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
  await settings(request, { delayNext: 1 });
  await ask(page, "Intern new request");
  const newId = await pending(request, "intern");
  await release(page, request, oldId, "Executive old request");
  await expect(page.getByRole("button", { name: /Searching/ })).toBeDisabled();
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toBeDisabled();
  await expect(page.getByText(EXECUTIVE, { exact: false })).toHaveCount(0);
  await release(page, request, newId, "Intern new request");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  await expect(page.getByText(EXECUTIVE, { exact: false })).toHaveCount(0);
  await noSensitiveStorage(page);
});

test("same-account session rotation invalidates an old query despite unchanged user ID", async ({ page, request, context }) => {
  await ignoreQueryAbort(page);
  await page.goto("/");
  await signIn(page);
  await settings(request, { delayNext: 1 });
  await ask(page, "Old same-user session");
  const id = await pending(request, "intern");
  const oldCookie = (await context.cookies()).find((item) => item.name === "rag_session").value;
  const bootstrap = await context.request.get("/api/auth/csrf");
  const csrf = (await bootstrap.json()).csrf_token;
  const rotated = await context.request.post("/api/auth/login", { data: { login: "intern", password: PASSWORD },
    headers: { Origin: new URL(page.url()).origin, "X-CSRF-Token": csrf } });
  expect(rotated.status()).toBe(200);
  expect((await context.cookies()).find((item) => item.name === "rag_session").value).not.toBe(oldCookie);
  const me = page.waitForResponse((response) => response.url().endsWith("/api/auth/me"));
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await me;
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
  await release(page, request, id, "Old same-user session");
  await expect(page.locator(".answer-text")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Start with a question" })).toBeVisible();
});

test("two tabs share cookies and close old account evidence on remote logout/login", async ({ page, context }) => {
  await page.goto("/");
  await signIn(page, "executive");
  await ask(page, "Executive visible result");
  await expect(page.locator(".answer-text")).toContainText(EXECUTIVE);
  const other = await context.newPage();
  await other.goto("/");
  await expect(other.getByText(names.executive, { exact: true }).first()).toBeVisible();
  await other.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await expect(page.getByText(EXECUTIVE, { exact: false })).toHaveCount(0);
  await signIn(other, "intern");
  await page.bringToFront();
  await expect(page.getByText(names.intern, { exact: true }).first()).toBeVisible();
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
  await ask(page, "Intern after remote cookie change");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  await expect(page.getByText(EXECUTIVE, { exact: false })).toHaveCount(0);
  await noSensitiveStorage(page);
});

test("expired401 clears prior answer, question and source details before sign-in", async ({ page, request }) => {
  await page.goto("/");
  await signIn(page);
  await ask(page, "Visible before expiry");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  await request.post("/__tests/expire");
  await ask(page, "Must not survive session expiry");
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await expect(page.locator(".answer-text, article.source-card")).toHaveCount(0);
  await expect(page.getByText(INTERN, { exact: false })).toHaveCount(0);
  await signIn(page, "executive");
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
});

test("unconfirmed logout stays closed even while the backend cookie still authenticates", async ({ page, request, context }) => {
  await page.goto("/");
  await signIn(page, "executive");
  await ask(page, "Private result before failed sign out");
  await expect(page.locator(".answer-text")).toContainText(EXECUTIVE);
  await settings(request, { logoutFailure: true });
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByRole("button", { name: "Retry sign out", exact: true })).toBeVisible();
  expect((await context.request.get("/api/auth/me")).status()).toBe(200);
  await page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await page.waitForLoadState("networkidle");
  await expect(page.getByRole("button", { name: "Retry sign out", exact: true })).toBeVisible();
  await expect(page.locator(".answer-text, article.source-card")).toHaveCount(0);
  await expect(page.getByText(EXECUTIVE, { exact: false })).toHaveCount(0);
  await settings(request, { logoutFailure: false });
  await page.getByRole("button", { name: "Retry sign out", exact: true }).click();
  await signIn(page, "intern");
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
});

test("unknown auth profile fails closed and never becomes the benchmark demo", async ({ page, request }) => {
  await settings(request, { enabled: null });
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Retry sign-in status", exact: true })).toBeVisible();
  await expect(page.getByText("Benchmark demo identity", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Ask question", exact: false })).toBeDisabled();
  expect((await status(request)).queries).toHaveLength(0);
});

test("explicit benchmark profile restores usable controls after back-forward page lifecycle", async ({ page, request }) => {
  await settings(request, { enabled: false });
  await page.goto("/");
  await expect(page.getByText("Benchmark demo identity", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Login", { exact: true })).toHaveCount(0);
  await ask(page, "Benchmark before navigation");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })));
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toBeDisabled();
  await page.evaluate(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })));
  await expect(page.getByText("Benchmark demo identity", { exact: true })).toBeVisible();
  await expect(page.getByLabel("What would you like to know?", { exact: true })).toBeEnabled();
  await expect(page.locator(".answer-text")).toHaveCount(0);
  await ask(page, "Benchmark after navigation");
  await expect(page.locator(".answer-text")).toContainText(INTERN);
  expect((await status(request)).queries).toHaveLength(2);
  await noSensitiveStorage(page);
});

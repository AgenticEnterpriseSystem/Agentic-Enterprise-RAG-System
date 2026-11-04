import { test, expect } from "@playwright/test";

const PASSWORD = "SYNTHETIC_PASSWORD_DO_NOT_STORE";

test.beforeEach(async ({ request }) => {
  expect((await request.post("/__tests/reset")).ok()).toBeTruthy();
});

async function configure(request, value) {
  expect((await request.post("/__tests/settings", { data: value })).ok()).toBeTruthy();
}
async function signIn(page, login = "intern") {
  await page.getByLabel("Login", { exact: true }).fill(login);
  await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  await expect(page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
}
async function ask(page, text) {
  const field = page.getByLabel("What would you like to know?", { exact: true });
  await field.fill(text);
  await expect(page.getByRole("button", { name: "Ask question", exact: false })).toBeEnabled();
  await field.press("Enter");
}
async function privateStorageEmpty(page) {
  expect(await page.evaluate(async () => ({ local: Object.entries(localStorage), session: Object.entries(sessionStorage),
    databases: await indexedDB.databases(), caches: await caches.keys(), cookies: document.cookie })))
    .toEqual({ local: [], session: [], databases: [], caches: [], cookies: "" });
}

test("zero count has scoped record units, exact filters and navigable aggregate provenance", async ({ page, request }) => {
  await configure(request, { responseKind: "count" });
  await page.goto("/");
  await signIn(page);
  const response = page.waitForResponse((value) => value.url().endsWith("/api/query"));
  await ask(page, "Count this exact status");
  const payload = await (await response).json();
  expect(payload.contract_version).toBe("2");
  expect(payload.sources[0]).not.toHaveProperty("chunk");
  const card = page.locator("article.aggregate-card");
  await expect(card.locator(".record-count strong")).toHaveText("0");
  await expect(card).toContainText("canonical Jira source records");
  await expect(card).toContainText("Within your permitted records");
  await expect(card).toContainText("Exact absent status");
  await expect(card).toContainText("not a count of unique issues");
  await expect(card.locator("blockquote")).toHaveCount(0);
  await expect(card).not.toContainText("Characters");
  await page.getByRole("button", { name: "View Source 1", exact: true }).click();
  await expect(card).toBeFocused();
  await card.getByText("Count provenance", { exact: true }).click();
  await expect(card).toContainText("8".repeat(64));
  await page.locator("details.steps > summary").click();
  await expect(page.locator("details.steps")).toContainText("0 canonical Jira source records");
  await expect(page.locator("details.steps .native-warning")).toHaveCount(0);
  await privateStorageEmpty(page);
});

test("native record fields keep verified, missing and invalid provenance without fabricated excerpts", async ({ page, request }) => {
  await configure(request, { responseKind: "record" });
  await page.goto("/");
  await signIn(page);
  const response = page.waitForResponse((value) => value.url().endsWith("/api/query"));
  await ask(page, "Look up a canonical record");
  const payload = await (await response).json();
  expect(payload.contract_version).toBe("3");
  expect(payload.sources[0]).not.toHaveProperty("chunk");
  const card = page.locator("article.record-card");
  await expect(card.getByRole("heading", { name: "INTERN-NATIVE-101" })).toBeVisible();
  await expect(card).toContainText("In Progress");
  await expect(card).toContainText("Verified from source");
  await expect(card).toContainText("Missing in source");
  await expect(card).toContainText("Invalid in source");
  await expect(card.locator("blockquote")).toHaveCount(0);
  await expect(card).not.toContainText("Characters");
  await page.getByRole("button", { name: "View Source 1", exact: true }).click();
  await expect(card).toBeFocused();
  await card.getByText("Record provenance", { exact: true }).click();
  await expect(card).toContainText("generated_data/sources/jira/test/intern-record.json");
  await expect(card).toContainText("/status · verified");
  await privateStorageEmpty(page);
});

test("mixed citations focus each kind and logout removes every kind before another account signs in", async ({ page, request }) => {
  await configure(request, { responseKind: "mixed" });
  await page.goto("/");
  await signIn(page);
  await ask(page, "Combine text, native fields and count");
  await expect(page.locator("article.source-card")).toHaveCount(3);
  for (const [index, kind] of [[1, "passage"], [2, "record"], [3, "aggregate"]]) {
    await page.getByRole("button", { name: `View Source ${index}`, exact: true }).click();
    await expect(page.locator(`article.${kind}-card`)).toBeFocused();
  }
  await expect(page.locator("article.passage-card blockquote")).toContainText("INTERN_PRIVATE_EVIDENCE_TEST_ONLY");
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await expect(page.locator("article.source-card")).toHaveCount(0);
  await expect(page.locator("details.steps")).toHaveCount(0);
  await signIn(page, "executive");
  await ask(page, "New account mixed results");
  await expect(page.locator("article.record-card")).toContainText("EXECUTIVE-NATIVE-101");
  await expect(page.locator(".results")).not.toContainText("INTERN-NATIVE-101");
  await expect(page.locator(".results")).not.toContainText("INTERN_PRIVATE_EVIDENCE_TEST_ONLY");
  await privateStorageEmpty(page);
});

test("ambiguous native keys and unavailable coverage remain clarification states, never global absence", async ({ page, request }) => {
  await configure(request, { responseKind: "ambiguous" });
  await page.goto("/");
  await signIn(page);
  await ask(page, "Which duplicate key record?");
  await expect(page.getByRole("heading", { name: "Clarification needed", exact: true })).toBeVisible();
  await expect(page.locator("article.record-card")).toHaveCount(2);
  await page.locator("details.steps > summary").click();
  await expect(page.locator("details.steps")).toContainText("matches multiple permitted source records");
  await expect(page.locator("details.steps")).toContainText("Only part of the matching records is shown");
  await configure(request, { responseKind: "unknown" });
  await ask(page, "Count when field coverage is unavailable");
  await expect(page.locator("article.source-card")).toHaveCount(0);
  await page.locator("details.steps > summary").click();
  await expect(page.locator("details.steps")).toContainText("does not establish a zero count");
  await expect(page.locator(".record-count")).toHaveCount(0);
});

test("a late mixed response cannot restore old native records or aggregate counts after account change", async ({ page, request }) => {
  await page.addInitScript(() => {
    const original = window.fetch;
    window.fetch = (input, options = {}) => original(input,
      input === "/api/query" ? { ...options, signal: undefined } : options);
  });
  await configure(request, { responseKind: "mixed", delayNext: 1 });
  await page.goto("/");
  await signIn(page);
  await ask(page, "Old mixed response");
  await expect.poll(async () => (await (await request.get("/__tests/status")).json()).pending.length).toBe(1);
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await signIn(page, "executive");
  const oldResponse = page.waitForResponse((value) => value.url().endsWith("/api/query"));
  await request.post("/__tests/release", { data: {} });
  await (await oldResponse).finished();
  await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  await expect(page.locator("article.source-card")).toHaveCount(0);
  await expect(page.locator(".answer-text")).toHaveCount(0);
  await ask(page, "Current account mixed response");
  await expect(page.locator("article.record-card")).toContainText("EXECUTIVE-NATIVE-101");
  await expect(page.locator(".results")).not.toContainText("INTERN-NATIVE-101");
  await privateStorageEmpty(page);
});

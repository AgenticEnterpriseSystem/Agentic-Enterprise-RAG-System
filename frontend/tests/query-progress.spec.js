import { test, expect } from "@playwright/test";

const PASSWORD = "SYNTHETIC_PASSWORD_DO_NOT_STORE";

test.beforeEach(async ({ request }) => {
  expect((await request.post("/__tests/reset")).ok()).toBeTruthy();
});

async function configure(request, value) {
  expect((await request.post("/__tests/settings", { data: value })).ok()).toBeTruthy();
}
async function signInAndAsk(page, text) {
  await page.goto("/");
  await page.getByLabel("Login", { exact: true }).fill("intern");
  await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
  await page.getByRole("button", { name: "Sign in", exact: true }).click();
  const field = page.getByLabel("What would you like to know?", { exact: true });
  await field.fill(text);
  await expect(page.getByRole("button", { name: "Ask question", exact: false })).toBeEnabled();
  await field.press("Enter");
}

test("rows appear while running, a running row completes with a duration and the outcome is shown", async ({ page, request }) => {
  await configure(request, { scriptedProgress: {} });
  await signInAndAsk(page, "Show progress");
  const live = page.locator(".loading .query-progress");
  const search = live.locator("li", { hasText: "Search documents" });
  await expect(live.locator("li", { hasText: "Plan search" })).toContainText("Done");
  await expect(search).toContainText("Running…");
  await expect(search).toContainText("Round 1");
  await expect(search).toContainText("Done");
  await expect(search).toContainText("1.2s");
  await expect(page.locator(".loading [aria-live='polite']")).toHaveCount(1);
  await expect(page.locator(".loading ol")).not.toHaveAttribute("aria-live", /.+/);
  const details = page.locator("details.progress-details");
  await expect(details).toBeVisible();
  await expect(details).not.toHaveAttribute("open", "");
  await details.locator("summary").click();
  await expect(details.locator("li", { hasText: "Write answer" })).toContainText("Done");
  await expect(details).toContainText("Evidence judged sufficient");
  await expect(details.locator("li", { hasText: "Check document access" })).toHaveCount(1);
  await expect(details.locator("li", { hasText: /^agent/ })).toHaveCount(0);
  await expect(page.locator(".loading")).toHaveCount(0);
});

test("a failed stage shows Failed with its error code", async ({ page, request }) => {
  await configure(request, { scriptedProgress: { failStage: "critic" } });
  await signInAndAsk(page, "Show a failure");
  const details = page.locator("details.progress-details");
  await expect(details).toBeVisible({ timeout: 10_000 });
  await details.locator("summary").click();
  await expect(details.locator("li", { hasText: "Check evidence" })).toContainText("Failed (provider_error)");
  await expect(details).toContainText("Outcome:");
});

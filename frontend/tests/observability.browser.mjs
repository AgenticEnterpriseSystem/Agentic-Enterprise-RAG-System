/** Mock-only browser integration. Never connects to the real RAG application. */
import test, { before, after } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { existsSync } from "node:fs";
const frontend = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const root = path.dirname(frontend), port = 4387, origin = `http://127.0.0.1:${port}`;
const packagePaths = [path.join(frontend, "node_modules/@playwright/test/index.mjs"),
  path.join(root, "local-v2/workspaces/integration/frontend/node_modules/@playwright/test/index.mjs")];
const { chromium, expect } = await import(pathToFileURL(packagePaths.find(existsSync)).href);
const ID = "22222222-2222-4222-8222-222222222222", SECOND = "33333333-3333-4333-8333-333333333333";
let server, browser, serverErrors = "";
before(async () => {
  server = spawn(process.execPath, ["tests/mock-server.mjs"], { cwd: frontend,
    env: { ...process.env, UI_TEST_PORT: String(port) }, stdio: ["ignore", "ignore", "pipe"] });
  server.stderr.on("data", b => { serverErrors = (serverErrors + b).slice(-3000); });
  let ready = false;
  for (let i = 0; i < 80; i++) {
    if (server.exitCode != null) throw new Error("Mock fixture server failed: " + serverErrors);
    try { ready = (await fetch(origin + "/__tests/health")).ok; } catch {}
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  assert.ok(ready, "mock-only fixture must be ready");
  browser = await chromium.launch({ channel: "chrome", headless: true, args: ["--disable-background-networking"] });
});
after(async () => { await browser?.close(); server?.kill("SIGTERM"); });
function summary(id = ID) {
 return { request_id: id, run_id: null, started_at: "2026-11-16T08:00:00Z", completed_at: "2026-11-16T08:00:01Z",
  status: "error", http_status_code: 503, http_finished: true, worker_status: "not_started", duration_ms: 1234,
  http_duration_ms: 1234, worker_duration_ms: null, rounds: 0, event_count: 2, dropped_event_count: 0,
  error_type: "PolicyUnavailable", error_code: "policy_unavailable", usage: null, counts: { selected_documents: 8 } };
}
async function setup(handler) {
 const context = await browser.newContext({ baseURL: origin, serviceWorkers: "block" });
 const page = await context.newPage();
 await context.route("**/*", async route => {
  const url = new URL(route.request().url());
  if (url.origin !== origin) return route.abort();
  if (url.pathname.startsWith("/api/observability") || url.pathname === "/api/query") return handler(route, url);
  return route.continue();
 });
 return { context, page };
}
const json = (route, body, status = 200, headers = {}) => route.fulfill({ status, contentType: "application/json",
 headers: { "Cache-Control": "no-store", ...headers }, body: JSON.stringify(body) });
async function signIn(page, login = "intern") {
 await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
 await page.getByLabel("Login", { exact: true }).fill(login);
 await page.getByLabel("Password", { exact: true }).fill("SYNTHETIC_PASSWORD_DO_NOT_STORE");
 await page.getByRole("button", { name: "Sign in", exact: true }).click();
 await expect(page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
}
async function storageEmpty(page) {
 assert.deepEqual(await page.evaluate(async () => ({ local: Object.entries(localStorage), session: Object.entries(sessionStorage),
   databases: await indexedDB.databases(), caches: await caches.keys(), cookies: document.cookie })),
  { local: [], session: [], databases: [], caches: [], cookies: "" });
}

test("direct page sign-in then explicit denial never loads request data even for CEO role", async () => {
 let access = 0, protectedCalls = 0, queries = 0;
 const { context, page } = await setup((route, url) => {
  if (url.pathname === "/api/query") { queries++; return route.abort(); }
  if (url.pathname.endsWith("/access")) { access++; return json(route, { detail: { code: "scope_denied" } }, 403); }
  protectedCalls++; return route.abort();
 });
 try {
  await page.goto(origin + "/observability");
  await expect(page.getByRole("heading", { name: "Observability", exact: true })).toBeVisible();
  assert.equal(access, 0); await signIn(page, "executive");
  await expect(page.getByText("This account does not have access to observability. Search access is unchanged.")).toBeVisible();
  assert.equal(access, 1); assert.equal(protectedCalls, 0); assert.equal(queries, 0); await storageEmpty(page);
 } finally { await context.close(); }
});

test("authorized account sees bounded summaries and safe stage details with unknown cost", async () => {
 const calls = [];
 const { context, page } = await setup((route, url) => {
  calls.push(url.pathname + url.search);
  if (url.pathname.endsWith("/access")) return json(route, { allowed: true });
  if (url.pathname.endsWith(ID)) return json(route, { request: summary(), stages: [
   { sequence: 1, stage: "context.selection", event: "end", timestamp: "2026-11-16T08:00:01Z", status: "ok",
    duration_ms: 25, round_number: 1, selected_documents: 8, selected_passages: 10, reason_code: "char_budget",
    error_code: "truncated_output", provider_stop_reason: "max_tokens",
    exclusion_reasons: { char_budget: 2 }, raw_query: "SHOULD_NEVER_RENDER", document_text: "PRIVATE_SHOULD_NEVER_RENDER" },
  ] });
  return json(route, { requests: [summary()], next_offset: null, total: 1, retention_seconds: 3600, max_requests: 500 });
 });
 try {
  await page.goto(origin + "/observability"); await signIn(page);
  await expect(page.getByRole("button", { name: ID, exact: true })).toBeVisible();
  assert.ok(calls.indexOf("/api/observability/access") < calls.indexOf("/api/observability/requests?limit=50&offset=0"));
  await expect(page.locator("table")).toContainText("policy unavailable");
  await expect(page.locator("tbody tr").first().locator("td").nth(8)).toHaveText("Unavailable");
  await page.getByRole("button", { name: ID, exact: true }).click();
  await expect(page.getByRole("heading", { name: "Request details" })).toBeVisible();
  await expect(page.locator(".observation-stages")).toContainText("context.selection");
  await expect(page.locator(".observation-stages")).toContainText("Selected passages");
  await expect(page.locator(".observation-stages")).toContainText("Excluded: char budget");
  await expect(page.locator(".observation-stages")).toContainText("truncated output");
  await expect(page.locator(".observation-stages")).toContainText("Provider stop: max tokens");
  await expect(page.getByText("Not assigned", { exact: true })).toBeVisible();
  await expect(page.getByText(/SHOULD_NEVER_RENDER/)).toHaveCount(0); await storageEmpty(page);
 } finally { await context.close(); }
});

test("logout clears diagnostics and a late detail cannot leak into a new denied session", async () => {
 let allowed = true, release, requested;
 const pending = new Promise(resolve => { requested = resolve; });
 const hold = new Promise(resolve => { release = resolve; });
 const { context, page } = await setup(async (route, url) => {
  if (url.pathname.endsWith("/access")) return json(route, { allowed: true }, allowed ? 200 : 403);
  if (url.pathname.endsWith(ID)) { requested(); await hold; try { await json(route, { request: summary(), stages: [] }); } catch {} return; }
  return json(route, { requests: [summary()], next_offset: null, total: 1 });
 });
 try {
  await page.goto(origin + "/observability"); await signIn(page);
  await page.getByRole("button", { name: ID, exact: true }).click(); await pending;
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await expect(page.getByText(ID, { exact: true })).toHaveCount(0);
  allowed = false; release(); await signIn(page, "executive");
  await expect(page.getByText("This account does not have access to observability. Search access is unchanged.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Request details" })).toHaveCount(0); await storageEmpty(page);
 } finally { release(); await context.close(); }
});

test("pagination stays bounded, expired details are explicit and revoked access clears rows", async () => {
 let revoked = false;
 const { context, page } = await setup((route, url) => {
  if (url.pathname.endsWith("/access")) return json(route, { allowed: true }, revoked ? 403 : 200);
  if (url.pathname.endsWith(SECOND)) return json(route, { detail: { code: "not_found" } }, 404);
  assert.equal(url.searchParams.get("limit"), "50");
  const next = url.searchParams.get("offset") === "50";
  return json(route, { requests: [summary(next ? SECOND : ID)], next_offset: next ? null : 50, total: 51 });
 });
 try {
  await page.goto(origin + "/observability"); await signIn(page);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("button", { name: SECOND, exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Next", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: SECOND, exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("no longer retained");
  revoked = true; await page.getByRole("button", { name: "Reload requests" }).click();
  await expect(page.getByText(SECOND, { exact: true })).toHaveCount(0);
  await expect(page.getByText("This account does not have access to observability. Search access is unchanged.")).toBeVisible();
 } finally { await context.close(); }
});

test("expired session closes diagnostics and returns to sign-in", async () => {
 let expired = false;
 const { context, page } = await setup((route, url) => {
  if (expired) return json(route, { detail: { code: "session_expired" } }, 401);
  if (url.pathname.endsWith("/access")) return json(route, { allowed: true });
  return json(route, { requests: [summary()], next_offset: null, total: 1 });
 });
 try {
  await page.goto(origin + "/observability"); await signIn(page);
  await expect(page.getByRole("button", { name: ID, exact: true })).toBeVisible();
  await context.request.post(origin + "/__tests/expire"); expired = true;
  await page.getByRole("button", { name: "Reload requests" }).click();
  await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
  await expect(page.getByText(ID, { exact: true })).toHaveCount(0); await storageEmpty(page);
 } finally { await context.close(); }
});

function answer() {
 const sources = Array.from({ length: 10 }, (_, i) => {
  const doc = `synthetic-doc-${i % 8}`, text = `Synthetic evidence passage ${i}`, start = i < 8 ? 0 : 100, id = `${doc}:${start}:${start + text.length}`;
  return { kind: "passage", citation_id: id, snapshot_id: "test", revision_id: "c".repeat(64), score: 1, retrieval_method: "hybrid",
   chunk: { chunk_id: id, doc_id: doc, title: `Synthetic source ${i + 1}`, source_type: "confluence", text, start_char: start, end_char: start + text.length } };
 });
 const usage = { input_tokens: 12, output_tokens: 6, api_calls: 2, estimated_cost_usd: .000042 };
 return { sources, answer: { status: "answered", text: "Synthetic cited answer " + sources.map(s => `[${s.citation_id}]`).join(" "), citation_ids: sources.map(s => s.citation_id) },
  trace: { run_id: ID, stop_reason: "sufficient", total_latency_ms: 1200, usage, steps: [], context_selection: [{ round_number: 1, document_limit: 8,
   selected_documents: 8, selected_passages: 10, selected_native_items: 0, excluded_documents: 2, excluded_passages: 3, cropped_passages: 1 }] } };
}
test("Search preserves all ten passage citations and shows eight-document context plus request ID", async () => {
 let queries = 0;
 const { context, page } = await setup((route, url) => {
  assert.equal(url.pathname, "/api/query"); queries++; return json(route, answer(), 200, { "X-Request-ID": ID });
 });
 try {
  await page.goto(origin); await signIn(page);
  await page.getByLabel("What would you like to know?", { exact: true }).fill("Synthetic question");
  await page.getByRole("button", { name: "Ask question", exact: false }).click();
  await expect(page.locator("article.source-card")).toHaveCount(10);
  await expect(page.locator(".request-reference")).toContainText(ID);
  await page.getByText("Evidence context", { exact: true }).click();
  await expect(page.locator(".evidence-context")).toContainText("8 documents (limit 8), 10 passages");
  await expect(page.locator(".evidence-context")).toContainText("1 passages shortened");
  await page.getByRole("button", { name: "Source 10 ↗", exact: true }).click();
  await expect(page.locator("article.source-card").nth(9)).toBeFocused(); assert.equal(queries, 1);
 } finally { await context.close(); }
});
test("failed Search displays server request ID without inventing a result", async () => {
 const { context, page } = await setup(route => json(route, { detail: { code: "policy_unavailable", message: "Document access is temporarily unavailable." } }, 503, { "X-Request-ID": ID }));
 try {
  await page.goto(origin); await signIn(page); await page.getByLabel("What would you like to know?", { exact: true }).fill("Synthetic failed question");
  await page.getByRole("button", { name: "Ask question", exact: false }).click();
  await expect(page.getByRole("alert")).toContainText("Document access is temporarily unavailable.");
  await expect(page.locator(".request-reference")).toContainText(ID); await expect(page.locator(".answer-text")).toHaveCount(0);
 } finally { await context.close(); }
});

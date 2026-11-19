/** Explicit SEC-01 Chrome smoke against real local HTTP; no mock or DB control.
 * Usage: node frontend/tests/live-security.mjs --state PRIVATE_600_JSON
 *   --outdir NEW_PRIVATE_DIRECTORY [--control PRIVATE_600_JSON]
 * Optional control handshake: read outdir/status.json, stop the owned DB only
 * after phase=awaiting_db_outage, write {"phase":"outage"}; after the script
 * reports awaiting_db_recovery, restore the DB and write {"phase":"recovered"}.
 * Raw responses, passwords, cookies and screenshots are never written.
 */
import { constants, existsSync } from "node:fs";
import { open, mkdir, rename } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createHash } from "node:crypto";

class CheckFailure extends Error {}
const need = (condition, message) => { if (!condition) throw new CheckFailure(message); };
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const arguments_ = process.argv.slice(2), options = {};
for (let index = 0; index < arguments_.length; index += 2) {
  need(arguments_[index].startsWith("--") && arguments_[index + 1], "Expected named CLI option and value");
  options[arguments_[index].slice(2)] = arguments_[index + 1];
}
need(options.state && options.outdir, "--state and --outdir are required");
const output = path.resolve(options.outdir), control = options.control && path.resolve(options.control);
const timeout = Number(options["control-timeout-ms"] || 300_000);
need(Number.isInteger(timeout) && timeout >= 10_000 && timeout <= 900_000, "Invalid control timeout");
process.umask(0o077);
await mkdir(path.dirname(output), { recursive: true, mode: 0o700 });
await mkdir(output, { mode: 0o700 }); // Refuse to overwrite an earlier report.

async function privateJson(filename, bound = 1_048_576) {
  const handle = await open(filename, constants.O_RDONLY | constants.O_NOFOLLOW);
  try {
    const info = await handle.stat();
    need(info.isFile() && (info.mode & 0o777) === 0o600 && info.size <= bound,
      "Expected a bounded regular private600 JSON file");
    const raw = await handle.readFile("utf8");
    need(Buffer.byteLength(raw) <= bound, "Private file grew beyond its bound");
    return JSON.parse(raw);
  } finally { await handle.close(); }
}
async function writeJson(filename, value) {
  const temporary = `${filename}.new`;
  const handle = await open(temporary, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, 0o600);
  try { await handle.writeFile(JSON.stringify(value, null, 2) + "\n"); }
  finally { await handle.close(); }
  await rename(temporary, filename);
}
const checks = [], statusFile = path.join(output, "status.json");
async function status(phase) {
  await writeJson(statusFile, { format: "rag-sec-browser-status-v1", phase,
    checked_at_utc: new Date().toISOString(), completed_checks: checks.length });
}
async function awaitControl(phase) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    try { if ((await privateJson(control, 4096)).phase === phase) return; }
    catch (error) { if (error.code !== "ENOENT") throw error; }
    await pause(250);
  }
  throw new CheckFailure("Timed out waiting for the operator's DB phase");
}
function stable(value) {
  if (Array.isArray(value)) return value.map(stable);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map(key => [key, stable(value[key])]));
  return value;
}
const equal = (first, second) => JSON.stringify(stable(first)) === JSON.stringify(stable(second));
const digest = value => createHash("sha256").update(JSON.stringify(stable(value))).digest("hex");
function containsMarker(value, marker) {
  if (typeof value === "string") return value.includes(marker);
  if (Array.isArray(value)) return value.some(item => containsMarker(item, marker));
  if (value && typeof value === "object") return Object.entries(value).some(([key, item]) =>
    key.includes(marker) || containsMarker(item, marker));
  return false;
}
let browser;
try {
  const state = await privateJson(path.resolve(options.state));
  need(state.format === "rag-sec-human-v1", "Unexpected live-security state format");
  const base = state.base || state.base_url;
  need(base === "http://127.0.0.1:18081", "Live security target must be the explicit local Docker URL");
  need(state.snapshot && typeof state.snapshot.snapshot_id === "string", "Exact snapshot is required");
  need(typeof state.question === "string" && state.question.trim(), "Generic question is required");
  for (const role of ["intern", "executive"]) {
    const account = state.accounts?.[role];
    need(account && ["login", "password", "user_id"].every(key => typeof account[key] === "string" && account[key]),
      "Both actual human accounts are required");
    need(Array.isArray(account.allowed_doc_ids) && account.allowed_doc_ids.length > 0
      && account.allowed_doc_ids.every(value => typeof value === "string" && value), "Explicit nonempty document grants are required");
    need(account.forbidden_markers === undefined || (Array.isArray(account.forbidden_markers)
      && account.forbidden_markers.every(value => typeof value === "string" && value)), "Invalid forbidden markers");
  }
  const playwrightModule = options["playwright-module"] || [
    path.join(root, "frontend/node_modules/@playwright/test/index.mjs"),
    path.join(root, "local-v2/workspaces/integration/frontend/node_modules/@playwright/test/index.mjs"),
  ].find(candidate => existsSync(candidate));
  need(playwrightModule, "Install frontend Playwright dependencies or provide --playwright-module");
  const { chromium, expect } = await import(pathToFileURL(path.resolve(playwrightModule)).href);
  const chrome = process.env.PLAYWRIGHT_CHANNEL || "chrome";
  browser = await chromium.launch({ channel: chrome, headless: true });
  const contexts = [], views = {};
  const noStore = response => need(/\bno-store\b/.test(response.headers()["cache-control"] || ""),
    "A private HTTP response is missing Cache-Control no-store");
  const record = (name, detail = {}) => checks.push({ name, passed: true, ...detail });
  async function noStorage(page) {
    const value = await page.evaluate(async () => ({ local: Object.entries(localStorage),
      session: Object.entries(sessionStorage), databases: await indexedDB.databases(),
      caches: await caches.keys(), cookies: document.cookie }));
    need(equal(value, { local: [], session: [], databases: [], caches: [], cookies: "" }),
      "Private browser storage or a script-visible cookie was retained");
  }
  async function forbidden(page, role, payload) {
    const account = state.accounts[role];
    const rendered = await page.locator("body").textContent();
    for (const marker of account.forbidden_markers || []) {
      need(!rendered.includes(marker) && !containsMarker(payload, marker), "Denied text, title or ID reached the caller");
    }
  }
  async function session(view, role) {
    const response = await view.context.request.get(`${base}/api/auth/me`);
    noStore(response);
    need(response.status() === 200, "Actual session could not be confirmed");
    const value = await response.json();
    need(value.user?.user_id === state.accounts[role].user_id, "The browser session belongs to the wrong actual account");
    const cookies = await view.context.cookies();
    const cookie = cookies.find(item => item.name === "rag_session");
    need(cookie?.httpOnly && cookie.sameSite === "Strict" && cookie.path === "/"
      && cookie.domain === "127.0.0.1" && !cookie.secure && cookie.value, "Real loopback session-cookie flags differ from the reviewed profile");
    await noStorage(view.page);
    return cookie.value;
  }
  async function cleared(page) {
    await expect(page.locator(".results, details.steps, article.source-card")).toHaveCount(0);
    await expect(page.getByLabel("What would you like to know?", { exact: true })).toHaveValue("");
    await noStorage(page);
  }
  async function login(view, role) {
    const account = state.accounts[role], page = view.page;
    await expect(page.getByLabel("Login", { exact: true })).toBeVisible();
    await page.getByLabel("Login", { exact: true }).fill(account.login);
    await page.getByLabel("Password", { exact: true }).fill(account.password);
    const response = page.waitForResponse(value => value.url() === `${base}/api/auth/login`);
    await page.getByRole("button", { name: "Sign in", exact: true }).click();
    const result = await response;
    noStore(result);
    need(result.status() === 200, "Real account sign-in failed");
    await expect(page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
    await session(view, role);
    await cleared(page);
  }
  async function logout(view) {
    const response = view.page.waitForResponse(value => value.url() === `${base}/api/auth/logout`);
    await view.page.getByRole("button", { name: "Sign out", exact: true }).click();
    const result = await response;
    noStore(result);
    need(result.status() === 204, "Real account sign-out did not complete");
    await expect(view.page.getByLabel("Login", { exact: true })).toBeVisible();
    await cleared(view.page);
  }
  async function query(view, role, mode, question = state.question) {
    const page = view.page;
    await page.getByRole("radio", { name: mode === "baseline" ? "Fixed RAG" : "Adaptive agent", exact: true }).check();
    const field = page.getByLabel("What would you like to know?", { exact: true });
    await expect(field).toBeEnabled();
    await field.fill(question);
    await expect(page.getByRole("button", { name: "Ask question", exact: false })).toBeEnabled({ timeout: 30_000 });
    const pending = page.waitForResponse(value => value.url() === `${base}/api/query`, { timeout: 60_000 });
    await field.press("Enter");
    const response = await pending;
    noStore(response);
    need(response.status() === 200, "Actual scoped query did not complete successfully");
    const payload = await response.json();
    need(equal(payload.snapshot, state.snapshot), "Actual query used an unexpected snapshot");
    need(payload.trace?.usage?.api_calls === 0 && payload.trace?.usage?.input_tokens === 0
      && payload.trace?.usage?.output_tokens === 0, "Retrieval-only validation dispatched a model");
    need(Array.isArray(payload.sources) && payload.sources.length > 0, "Actual isolation query returned no evidence");
    const allowed = new Set(state.accounts[role].allowed_doc_ids), ids = [];
    for (const item of payload.sources) {
      const doc = item.kind === "passage" ? item.chunk?.doc_id : item.kind === "record" ? item.doc_id : undefined;
      if (doc !== undefined) {
        need(allowed.has(doc), "A source outside the actual user's explicit grant reached the caller");
        ids.push(doc);
      }
      if (item.kind !== "passage") {
        need(item.scope?.user_id === state.accounts[role].user_id && equal(item.scope?.snapshot, state.snapshot),
          "Native evidence carries a different account or snapshot scope");
      }
    }
    await expect(page.locator("article.source-card")).toHaveCount(payload.sources.length);
    await forbidden(page, role, payload);
    await noStorage(page);
    record(`real_${role}_${mode}_query`, { source_documents: new Set(ids).size,
      source_ids_sha256: digest([...new Set(ids)].sort()), snapshot_sha256: digest(payload.snapshot), api_calls: 0 });
    return new Set(ids);
  }
  async function createView() {
    const context = await browser.newContext({ baseURL: base, serviceWorkers: "block", acceptDownloads: false });
    contexts.push(context);
    const page = await context.newPage();
    page.setDefaultTimeout(15_000);
    await page.goto(base, { waitUntil: "domcontentloaded" });
    return { context, page };
  }
  await status("running_browser_isolation");
  for (const role of ["intern", "executive"]) {
    const view = await createView();
    views[role] = view;
    await login(view, role);
    const observed = new Set();
    for (const mode of ["baseline", "agentic"]) {
      for (const doc of await query(view, role, mode)) observed.add(doc);
    }
    if (role === "executive") {
      const exclusive = state.accounts.executive.allowed_doc_ids.filter(doc => !state.accounts.intern.allowed_doc_ids.includes(doc));
      need(exclusive.length > 0 && exclusive.some(doc => observed.has(doc)),
        "No executive-only granted document was observed; isolation proof is inconclusive");
    }
  }
  need(await session(views.intern, "intern") !== await session(views.executive, "executive"),
    "Separate actual accounts share a session token");
  record("separate_real_cookie_contexts");

  const switched = views.executive;
  await logout(switched);
  await login(switched, "intern");
  await forbidden(switched.page, "intern");
  await query(switched, "intern", "baseline");
  record("same_context_executive_to_intern_cleared");

  const other = { context: switched.context, page: await switched.context.newPage() };
  other.page.setDefaultTimeout(15_000);
  await other.page.goto(base, { waitUntil: "domcontentloaded" });
  await expect(other.page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
  await logout(other);
  await expect(switched.page.getByLabel("Login", { exact: true })).toBeVisible();
  await cleared(switched.page);
  await login(other, "executive");
  await switched.page.bringToFront();
  await switched.page.evaluate(() => window.dispatchEvent(new Event("focus")));
  await expect(switched.page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
  await session(switched, "executive");
  await cleared(switched.page);
  await forbidden(switched.page, "executive");
  await query(switched, "executive", "baseline");
  record("shared_cookie_tabs_logout_and_account_switch_cleared");
  await other.page.close();

  if (control) {
    await status("awaiting_db_outage");
    await awaitControl("outage");
    const ready = await switched.context.request.get(`${base}/api/ready`);
    noStore(ready);
    need(ready.status() === 503 && (await ready.json()).status === "not_ready", "Real DB outage did not close readiness");
    const me = await switched.context.request.get(`${base}/api/auth/me`);
    noStore(me);
    const denied = await me.json();
    need(me.status() === 503 && denied.detail?.code === "policy_unavailable" && !denied.sources,
      "Real DB outage did not close session verification");
    const health = await switched.context.request.get(`${base}/api/health`);
    need(health.status() === 200 && (await health.json()).status === "alive", "Health stopped being useful during DB outage");
    await switched.page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await expect(switched.page.getByRole("button", { name: "Retry sign-in status", exact: true })).toBeVisible();
    await cleared(switched.page);
    await expect(switched.page.getByRole("button", { name: "Ask question", exact: false })).toBeDisabled();
    await forbidden(switched.page, "executive");
    record("real_db_outage_closed_readiness_session_and_browser");
    await status("awaiting_db_recovery");
    await awaitControl("recovered");
    const deadline = Date.now() + 60_000;
    let recovered = false;
    while (Date.now() < deadline) {
      const response = await switched.context.request.get(`${base}/api/ready`);
      noStore(response);
      if (response.status() === 200) { recovered = true; break; }
      await pause(500);
    }
    need(recovered, "Actual readiness did not recover after DB restoration");
    await switched.page.getByRole("button", { name: "Retry sign-in status", exact: true }).click();
    await expect(switched.page.getByRole("button", { name: "Sign out", exact: true })).toBeVisible();
    await session(switched, "executive");
    await cleared(switched.page);
    await query(switched, "executive", "baseline");
    record("real_db_recovery_requires_fresh_session_and_query");
  }
  // Revoke browser-issued sessions before closing their cookie jars.
  await logout(views.intern);
  await logout(switched);
  record("all_browser_issued_sessions_logged_out");
  await writeJson(path.join(output, "summary.json"), { format: "rag-sec-browser-result-v1", passed: true,
    base, snapshot_sha256: digest(state.snapshot), checks, db_outage_checked: Boolean(control),
    paid_model_calls: 0, checked_at_utc: new Date().toISOString() });
  await status("complete");
  console.log(JSON.stringify({ passed: true, checks: checks.length, db_outage_checked: Boolean(control), report: path.join(output, "summary.json") }));
} catch (error) {
  await writeJson(path.join(output, "summary.json"), { format: "rag-sec-browser-result-v1", passed: false,
    checks, failure: error instanceof CheckFailure ? error.message : "Browser or transport check failed; raw details withheld",
    error_kind: error.name, checked_at_utc: new Date().toISOString() });
  await status("failed");
  console.error(JSON.stringify({ passed: false, report: path.join(output, "summary.json") }));
  process.exitCode = 1;
} finally { await browser?.close(); }

/** TEST ONLY: localhost mock API + the real Vite frontend. No DB/model calls. */
import http from "node:http";
import { createServer as createViteServer } from "vite";

const host = "127.0.0.1";
const port = Number(process.env.UI_TEST_PORT || 4173);
const origin = `http://${host}:${port}`;
const PASSWORD = "SYNTHETIC_PASSWORD_DO_NOT_STORE";
let state;
function reset() {
  for (const pending of state?.pending || []) {
    if (!pending.response.writableEnded) pending.response.end();
  }
  state = { counter: 0, sessions: new Map(), pending: [], queries: [], meCalls: 0,
    delayNext: 0, logoutFailure: false, queryFailure: null, meFailure: false,
    readyFailure: false, queryCsrfFailure: false, scriptedProgress: null, progress: new Map(), enabled: true, responseKind: "passage" };
}
reset();
const users = {
  intern: { user_id: "ui-test-intern", display_name: "Alex Intern", role: "intern", department: "research", project_ids: [] },
  executive: { user_id: "ui-test-executive", display_name: "Robin Executive", role: "ceo", department: "finance", project_ids: [] },
};
function cookies(request) {
  return Object.fromEntries((request.headers.cookie || "").split(";").filter(Boolean).map((item) => {
    const index = item.indexOf("=");
    return [item.slice(0, index).trim(), item.slice(index + 1)];
  }));
}
function json(response, status, value, headers = {}) {
  if (response.destroyed || response.writableEnded) return;
  response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store", ...headers });
  response.end(JSON.stringify(value));
}
function failure(response, status, code, message) {
  json(response, status, { detail: { code, message } });
}
async function body(request) {
  let text = "";
  for await (const chunk of request) {
    text += chunk;
    if (text.length > 32_768) throw new Error("Test request exceeded bound");
  }
  return text ? JSON.parse(text) : {};
}
function session(request) {
  const token = cookies(request).rag_session;
  const current = state.sessions.get(token);
  return current?.active ? current : null;
}
function sessionBody(current) {
  return { user: users[current.login], csrf_token: current.csrf,
    expires_at: new Date(Date.now() + 15 * 60_000).toISOString() };
}
// Stage timeline (ms after the query arrives) served by the progress endpoint.
const progressTimeline = [
  // The real backend nests these sub-steps and the whole-run agent span; the UI must not repeat them as rows.
  { stage: "policy.resolve", round: null, start: 0, end: 0 },
  { stage: "policy.grants", round: null, start: 0, end: 0 },
  { stage: "policy.register", round: null, start: 0, end: 0 },
  { stage: "agent", round: null, start: 0, end: 2600 },
  { stage: "route", round: 1, start: 0, end: 300 },
  { stage: "text_search", round: 1, start: 300, end: 1500 },
  { stage: "critic", round: 1, start: 1500, end: 2200 },
  { stage: "generate", round: null, start: 2200, end: 2600 },
];
const progressTotalMs = 2600;
function progressBody(entry) {
  const elapsed = Date.now() - entry.startedAt, stages = [];
  let sequence = 1;
  for (const item of entry.script ? progressTimeline : []) {
    const base = { stage: item.stage, ...(item.round ? { round_number: item.round } : {}) };
    if (elapsed >= item.start) stages.push({ sequence: sequence++, event: "start", status: "in_progress", ...base });
    if (elapsed < item.end) continue;
    const failed = entry.script.failStage === item.stage;
    stages.push(failed
      ? { sequence: sequence++, event: "error", status: "error", error_code: "provider_error", error_type: "ProviderError",
          duration_ms: item.end - item.start, ...base }
      : { sequence: sequence++, event: "end", status: "ok", duration_ms: item.end - item.start, ...base });
  }
  const done = elapsed >= progressTotalMs || !entry.script;
  return { request: { status: done ? "ok" : "in_progress", http_status_code: done ? 200 : null,
      worker_status: done ? "ok" : "running", started_at: new Date(entry.startedAt).toISOString(),
      completed_at: done ? new Date().toISOString() : null, duration_ms: elapsed, rounds: 1, error_code: null,
      error_type: null, usage: { input_tokens: null, output_tokens: null, api_calls: null, estimated_cost_usd: null } },
    stages };
}
function answer(login, request) {
  const text = login === "executive" ? "EXECUTIVE_PRIVATE_EVIDENCE_TEST_ONLY" : "INTERN_PRIVATE_EVIDENCE_TEST_ONLY";
  const doc = `${login}-note`, citation = `${doc}:0:${text.length}`;
  const usage = { input_tokens: 0, output_tokens: 0, api_calls: 0, estimated_cost_usd: 0 };
  return { contract_version: "2", snapshot: { snapshot_id: "ui-fixture", corpus_sha256: "a".repeat(64), catalog_sha256: "b".repeat(64), document_count: 2 },
    answer: { status: "answered", text: `${text} [${citation}]`, citation_ids: [citation] },
    sources: [{ kind: "passage", citation_id: citation, snapshot_id: "ui-fixture", revision_id: "c".repeat(64), score: 1,
      retrieval_method: "hybrid", chunk: { chunk_id: citation, doc_id: doc, title: `${login} synthetic source`,
        source_type: "confluence", text, start_char: 0, end_char: text.length, timestamp: null, metadata_origin: "synthetic" } }],
    trace: { run_id: `ui-query-${state.counter}`, total_latency_ms: 1, usage,
      stop_reason: request.mode === "baseline" ? "single_pass" : "sufficient", steps: [{ round_number: 1,
        query: request.question, tool: "text_search", source_types: [], evidence_ids: [citation], elapsed_ms: 1, usage }] } };
}

function nativeAnswer(login, request) {
  const response = answer(login, request), snapshot = response.snapshot;
  const scope = { scope_id: `test-scope-${login}`, user_id: users[login].user_id, snapshot,
    policy_revision: 1, session_version: 0 };
  const feed = "d".repeat(64), recordId = `${login}-record`;
  const record = (doc, key) => ({ kind: "record", scope, citation_id: `jira-record:${feed}:${doc}`,
    record_id: doc, doc_id: doc, revision_id: "c".repeat(64), native_key: key,
    status: "In Progress", priority: null, project_id: null,
    source: { feed_id: feed, repository: "https://example.invalid/synthetic-jira.git", git_commit: "e".repeat(40),
      git_blob: "f".repeat(40), path: `generated_data/sources/jira/test/${doc}.json`,
      raw_sha256: "9".repeat(64), parser_version: "synthetic-browser-fixture" },
    field_provenance: [{ field: "native_key", json_pointer: "/key", state: "verified" },
      { field: "status", json_pointer: "/status", state: "verified" },
      { field: "priority", json_pointer: "/priority", state: "invalid" },
      { field: "project_id", json_pointer: "/project", state: "missing" }] });
  const native = record(recordId, `${login.toUpperCase()}-NATIVE-101`);
  const aggregate = { kind: "aggregate", citation_id: `aggregate:test-${login}-${state.counter}`,
    artifact_id: `test-${login}-${state.counter}`, scope, template_id: "jira-canonical-record-count", template_version: "1",
    filters: [{ field: "status", value: state.responseKind === "count" ? "Exact absent status" : "In Progress" }],
    operation: "count", count: state.responseKind === "count" ? 0 : 2, lineage_sha256: "8".repeat(64),
    coverage: "verified_records", created_at: "2026-11-04T12:00:00Z" };
  const ambiguous = state.responseKind === "ambiguous", unknown = state.responseKind === "unknown";
  response.sources = state.responseKind === "count" ? [aggregate]
    : state.responseKind === "record" ? [native]
    : ambiguous ? [record(`${recordId}-one`, "DUP-100"), record(`${recordId}-two`, "DUP-100")]
    : unknown ? [] : [response.sources[0], native, aggregate];
  response.contract_version = response.sources.some((value) => value.kind === "record") ? "3" : "2";
  response.answer = { status: ambiguous || unknown ? "insufficient_evidence" : "answered",
    text: ambiguous ? "This Jira key matches multiple permitted source records. Choose a canonical record ID."
      : unknown ? "Verified field coverage is unavailable for this permitted scope."
      : state.responseKind === "count" ? "0 matching canonical Jira source records in your permitted scope."
      : "Results from permitted sources.",
    citation_ids: response.sources.map((value) => value.citation_id) };
  response.answer.text += response.answer.citation_ids.map((id) => ` [${id}]`).join("");
  response.trace.stop_reason = ambiguous ? "clarification" : unknown ? "no_new_evidence" : "single_pass";
  const nativeStep = { round_number: 1, tool: "jira", source_types: ["jira"],
    evidence_ids: response.sources.filter((value) => value.kind !== "passage").map((value) => value.citation_id),
    elapsed_ms: 1, usage: response.trace.usage,
    query: JSON.stringify({ request: { source_type: "jira", operation: state.responseKind === "count" ? "count" : "lookup",
      filters: ambiguous ? { native_key: "DUP-100" } : state.responseKind === "count" ? { status: "Exact absent status" } : { doc_id: recordId }, limit: 10 },
      result: { count: state.responseKind === "count" ? 0 : null, coverage: unknown ? "unknown" : "verified_records",
        limitation: "Counts canonical Jira source records in this pinned benchmark; native keys may repeat.",
        ambiguous, truncated: ambiguous } }) };
  response.trace.steps = state.responseKind === "mixed" ? [response.trace.steps[0], nativeStep] : [nativeStep];
  return response;
}

async function api(request, response) {
  const path = new URL(request.url, origin).pathname;
  if (path === "/__tests/health") return json(response, 200, { test_only: true });
  if (path === "/__tests/reset") { reset(); return json(response, 200, { test_only: true }); }
  if (path === "/__tests/status") return json(response, 200, { test_only: true, meCalls: state.meCalls,
    queries: state.queries, pending: state.pending.map(({ id, login }) => ({ id, login })) });
  if (path === "/__tests/settings") {
    Object.assign(state, await body(request));
    return json(response, 200, { test_only: true });
  }
  if (path === "/__tests/expire") {
    for (const value of state.sessions.values()) value.active = false;
    return json(response, 200, { test_only: true });
  }
  if (path === "/__tests/release") {
    const command = await body(request);
    for (const pending of state.pending.filter((item) => !command.id || item.id === command.id)) {
      json(pending.response, 200, pending.payload);
      state.pending = state.pending.filter((item) => item !== pending);
    }
    return json(response, 200, { test_only: true });
  }
  if (path === "/api/auth/profile") return json(response, 200, { enabled: state.enabled });
  if (path === "/api/ready") {
    if (state.readyFailure) return json(response, 503, { status: "not_ready", message: "Synthetic dependency unavailable." });
    return json(response, 200, { status: "ready", busy: false, mode: "live", retrieval: "hybrid", generation_enabled: true,
      ...(!state.enabled ? { allowed_documents: 2, source_types: ["confluence"] } : {}) });
  }
  if (path === "/api/auth/csrf") {
    const value = `test-bootstrap-${++state.counter}`;
    return json(response, 200, { csrf_token: value }, { "Set-Cookie": `rag_login_csrf=${value}; HttpOnly; SameSite=Strict; Path=/api/auth` });
  }
  if (path === "/api/auth/login") {
    const value = await body(request), previous = cookies(request).rag_session;
    if (request.headers.origin !== origin || request.headers["x-csrf-token"] !== cookies(request).rag_login_csrf)
      return failure(response, 403, "csrf_failed", "Request verification failed.");
    if (!users[value.login] || value.password !== PASSWORD)
      return failure(response, 401, "invalid_credentials", "Login or password is incorrect.");
    if (state.sessions.has(previous)) state.sessions.get(previous).active = false;
    const token = `test-session-${++state.counter}-${value.login}`;
    const current = { login: value.login, csrf: `test-csrf-${state.counter}`, active: true };
    state.sessions.set(token, current);
    return json(response, 200, sessionBody(current), { "Set-Cookie": [
      `rag_session=${token}; HttpOnly; SameSite=Strict; Path=/`,
      "rag_login_csrf=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/api/auth",
    ] });
  }
  if (path === "/api/auth/me") {
    state.meCalls++;
    if (state.meFailure) return failure(response, 503, "policy_unavailable", "Synthetic session verification unavailable.");
    const current = session(request);
    return current ? json(response, 200, sessionBody(current)) : failure(response, 401,
      cookies(request).rag_session ? "session_expired" : "authentication_required", "Sign in to continue.");
  }
  if (path === "/api/auth/logout") {
    if (state.logoutFailure) return failure(response, 503, "policy_unavailable", "Test sign out could not be confirmed.");
    const current = session(request);
    if (current && request.headers["x-csrf-token"] !== current.csrf)
      return failure(response, 403, "csrf_failed", "Request verification failed.");
    if (current) current.active = false;
    response.writeHead(204, { "Cache-Control": "no-store", "Set-Cookie": "rag_session=; Max-Age=0; HttpOnly; SameSite=Strict; Path=/" });
    return response.end();
  }
  if (path.startsWith("/api/query/progress/")) {
    const current = state.enabled === false ? { login: "intern" } : session(request);
    if (!current) return failure(response, 401, "authentication_required", "Sign in to continue.");
    const entry = state.progress.get(path.split("/").pop());
    if (!entry || entry.login !== current.login) return failure(response, 404, "not_found", "Progress is unavailable or has expired.");
    return json(response, 200, progressBody(entry));
  }
  if (path === "/api/query") {
    const current = state.enabled === false ? { login: "intern" } : session(request);
    if (!current) return failure(response, 401, "session_expired", "Session expired. Sign in again.");
    if (state.enabled !== false && request.headers["x-csrf-token"] !== current.csrf)
      return failure(response, 403, "csrf_failed", "Request verification failed.");
    if (state.queryCsrfFailure) {
      state.queryCsrfFailure = false;
      return failure(response, 403, "csrf_failed", "Request verification failed.");
    }
    const value = await body(request), id = ++state.counter;
    state.queries.push({ id, login: current.login, question: value.question, mode: value.mode });
    const progressId = request.headers["x-query-progress-id"];
    if (progressId) state.progress.set(progressId, { login: current.login, startedAt: Date.now(), script: state.scriptedProgress });
    if (state.queryFailure) return failure(response, 503, "policy_unavailable", "Synthetic test search is unavailable.");
    const payload = state.responseKind === "passage" ? answer(current.login, value) : nativeAnswer(current.login, value);
    if (state.delayNext > 0) {
      state.delayNext--;
      state.pending.push({ id, login: current.login, response, payload });
      return;
    }
    if (state.scriptedProgress) {
      await new Promise((resolve) => setTimeout(resolve, progressTotalMs));
    }
    return json(response, 200, payload);
  }
  return failure(response, 404, "test_route_missing", "Unknown test API route.");
}

const vite = await createViteServer({ server: { middlewareMode: true, hmr: false }, appType: "spa" });
const server = http.createServer((request, response) => {
  if (request.url.startsWith("/api/") || request.url.startsWith("/__tests/")) {
    api(request, response).catch(() => failure(response, 500, "test_mock_failed", "Test mock request failed."));
  } else vite.middlewares(request, response);
});
server.listen(port, host, () => console.log(`TEST ONLY UI fixture listening on ${origin}`));
async function close() { reset(); server.close(); await vite.close(); }
process.on("SIGTERM", close);
process.on("SIGINT", close);

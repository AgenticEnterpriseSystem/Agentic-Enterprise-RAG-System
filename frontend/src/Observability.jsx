import React, { useEffect, useRef, useState } from "react";

const uuid = value => typeof value === "string" && /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i.test(value);
const number = value => typeof value === "number" && Number.isFinite(value) && value >= 0;
const label = value => typeof value === "string" && /^[A-Za-z][A-Za-z0-9_.:-]{0,95}$/.test(value) ? value.replaceAll("_", " ") : "Unavailable";
const duration = value => number(value) ? `${(value / 1000).toFixed(2)} s` : "Unavailable";
const count = value => number(value) ? value.toLocaleString() : "Unavailable";
const cost = value => number(value) ? `$${value.toFixed(6)}` : "Unavailable";
const timestamp = value => typeof value === "string" && Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString() : "Unavailable";
const metrics = {
  bm25_candidate_count: "BM25 candidates", dense_candidate_count: "Dense candidates", fused_count: "Fused candidates", returned_count: "Returned passages",
  unique_document_count: "Unique documents", selected_document_count: "Selected documents", selected_passage_count: "Selected passages",
  excluded_document_count: "Excluded documents", excluded_passage_count: "Excluded passages", retained_document_count: "Retained documents", retained_passage_count: "Retained passages", native_items: "Native items",
  candidate_count: "Candidates", bm25_candidates: "BM25 candidates", dense_candidates: "Dense candidates",
  fused_candidates: "Fused candidates", reranker_candidates: "Reranker candidates", result_count: "Results",
  pool_documents: "Pool documents", pool_passages: "Pool passages", selected_documents: "Selected documents",
  selected_passages: "Selected passages", selected_native_items: "Selected native items", retained_documents: "Retained documents",
  excluded_documents: "Excluded documents", excluded_passages: "Excluded passages", cropped_passages: "Shortened passages",
  document_limit: "Document limit", selected_chars: "Selected characters", char_budget: "Character budget",
  input_tokens: "Input tokens", output_tokens: "Output tokens", api_calls: "SDK requests",
  missing_facts: "Missing facts", supported_facts: "Supported facts", conflicting_facts: "Conflicting facts",
  ambiguous_facts: "Ambiguous facts", retry_queries: "Follow-up queries", read_more_documents: "Read-more documents",
};

function SafeCounts({ value }) {
  const items = Object.entries(metrics).filter(([key]) => number(value?.[key]));
  return items.length ? <dl className="observation-counts">{items.map(([key, text]) =>
    <React.Fragment key={key}><dt>{text}</dt><dd>{count(value[key])}</dd></React.Fragment>)}</dl> : null;
}
const reasons = new Set(["document_limit", "char_budget", "retained_document_limit", "retained_char_budget", "frozen_context_mismatch",
  "doc_limit", "char_limit", "token_limit", "context_reduced", "sufficient", "missing_evidence", "read_more", "reformulation",
  "round_limit", "no_new_evidence", "budget", "below_relevance_threshold", "none"]);
function Exclusions({ value }) {
  const rows = value && typeof value === "object" ? Object.entries(value).filter(([key, n]) => reasons.has(key) && number(n)) : [];
  return rows.length ? <dl className="observation-counts">{rows.map(([key, n]) =>
    <React.Fragment key={key}><dt>Excluded: {label(key)}</dt><dd>{count(n)}</dd></React.Fragment>)}</dl> : null;
}
function Usage({ value }) {
  return <dl className="observation-counts">
    <dt>Input tokens</dt><dd>{count(value?.input_tokens)}</dd>
    <dt>Output tokens</dt><dd>{count(value?.output_tokens)}</dd>
    <dt>SDK requests</dt><dd>{count(value?.api_calls)}</dd>
    <dt>Estimated cost</dt><dd>{cost(value?.estimated_cost_usd)}</dd>
  </dl>;
}

export default function Observability({ auth, authClient }) {
  const [view, setView] = useState({ session: null, status: "idle", requests: [], detail: null });
  const work = useRef({ sequence: 0, controller: null });
  const signedIn = auth.enabled === true && auth.status === "authenticated" && authClient?.snapshot === auth;
  const visible = signedIn && view.session === auth;
  const active = () => signedIn && authClient?.snapshot === auth;

  useEffect(() => {
    work.current.controller?.abort();
    work.current.sequence += 1;
    setView({ session: auth, status: "idle", requests: [], detail: null });
    if (auth.enabled === true && auth.status === "authenticated" && authClient?.snapshot === auth) load(0, true);
    return () => { work.current.sequence += 1; work.current.controller?.abort(); };
  }, [auth, authClient]);

  async function load(offset = 0, checkAccess = false, requestId = null) {
    if (!active() || !Number.isInteger(offset) || offset < 0 || offset > 500 || (requestId && !uuid(requestId))) return;
    work.current.controller?.abort();
    const controller = new AbortController(), sequence = ++work.current.sequence;
    work.current.controller = controller;
    const current = () => active() && sequence === work.current.sequence && !controller.signal.aborted;
    setView(old => ({ ...(old.session === auth ? old : { requests: [], detail: null }), session: auth,
      status: checkAccess ? "checking" : "loading", error: "" }));
    async function get(path) {
      const response = await fetch(path, { credentials: "same-origin", cache: "no-store", signal: controller.signal });
      if (!current()) return null;
      if (response.status === 401 || response.status === 403) {
        setView({ session: auth, status: response.status === 403 ? "denied" : "signed_out", requests: [], detail: null });
        if (response.status === 401) authClient.start();
        return null;
      }
      if (response.status === 404 && requestId) {
        setView(old => ({ ...old, status: "ready", detail: null, error: "This request is no longer retained. Reload the list." }));
        return null;
      }
      if (!response.ok) throw new Error("Diagnostics unavailable");
      const body = await response.json();
      return current() ? body : null;
    }
    try {
      if (checkAccess) {
        const access = await get("/api/observability/access");
        if (!access) return;
        if (access.allowed !== true) throw new Error("Access not confirmed");
      }
      if (requestId) {
        const detail = await get(`/api/observability/requests/${encodeURIComponent(requestId)}`);
        if (!detail) return;
        if (!detail.request || detail.request.request_id !== requestId || !Array.isArray(detail.stages) || detail.stages.length > 256)
          throw new Error("Invalid diagnostics response");
        setView(old => ({ ...old, status: "ready", detail }));
      } else {
        const body = await get(`/api/observability/requests?limit=50&offset=${offset}`);
        if (!body) return;
        if (!Array.isArray(body.requests) || body.requests.length > 50 || body.requests.some(item => !uuid(item.request_id))
          || !(body.next_offset === null || (Number.isInteger(body.next_offset) && body.next_offset > offset && body.next_offset <= 500)))
          throw new Error("Invalid diagnostics response");
        setView({ session: auth, status: "ready", requests: body.requests, detail: null, offset, nextOffset: body.next_offset,
          total: body.total, retention: body.retention_seconds, maxRequests: body.max_requests, error: "" });
      }
    } catch {
      if (current()) setView({ session: auth, status: "unavailable", requests: [], detail: null,
        error: "Diagnostics could not be loaded. Your access must be confirmed before they are shown." });
    }
  }

  return <section className="observability" aria-labelledby="observability-heading">
    <p className="eyebrow">Operator diagnostics</p>
    <h1 id="observability-heading">Observability</h1>
    <p className="observability-intro">Request timing, evidence counts, usage and failures. Questions, document contents and credentials are excluded.</p>
    {!signedIn && <p className="notice" role="status">Sign in with an explicitly authorized operator account to view diagnostics.</p>}
    {visible && view.status === "denied" && <p className="notice" role="status">This account does not have access to observability. Search access is unchanged.</p>}
    {visible && view.error && <p className="notice error" role="alert">{view.error}</p>}
    {signedIn && (!visible || ["idle", "checking"].includes(view.status)) && <p role="status">Checking diagnostics access…</p>}
    {visible && !["denied", "signed_out", "checking", "idle"].includes(view.status) && <>
      <div className="observability-toolbar">
        <button type="button" onClick={() => load(0, true)} disabled={view.status === "loading"}>Reload requests</button>
        <span>Current process only · bounded recent history · cleared on restart</span>
      </div>
      {view.status === "loading" && <p role="status">Loading diagnostics…</p>}
      {view.status !== "unavailable" && <>
        <p className="fine-print">Showing {view.requests.length} of {count(view.total)} retained requests.
          {number(view.retention) ? ` Retention: ${Math.round(view.retention / 60)} minutes.` : ""}
          {number(view.maxRequests) ? ` Up to ${view.maxRequests} requests.` : ""} Unknown usage is shown as unavailable.</p>
        <div className="observability-table-wrap">
          <table className="observability-table"><caption>Recent requests</caption>
            <thead><tr><th scope="col">Started (local time)</th><th scope="col">Request ID</th><th scope="col">HTTP / outcome</th>
              <th scope="col">Duration</th><th scope="col">Worker</th><th scope="col">Evidence docs</th><th scope="col">Rounds</th><th scope="col">Tokens in / out</th><th scope="col">Estimated cost</th><th scope="col">Error</th></tr></thead>
            <tbody>{view.requests.map(item => <tr key={item.request_id}>
              <td>{timestamp(item.started_at)}</td><td><button className="text-button request-link" type="button"
                disabled={view.status === "loading"} onClick={() => load(view.offset, false, item.request_id)}>{item.request_id}</button></td>
              <td>{item.http_status_code ?? "Pending"} · {label(item.status)}</td><td>{duration(item.duration_ms)}</td>
              <td>{label(item.worker_status)}</td><td>{count(item.counts?.selected_documents ?? item.counts?.selected_document_count)}</td><td>{count(item.rounds)}</td>
              <td>{count(item.usage?.input_tokens)} / {count(item.usage?.output_tokens)}</td>
              <td>{cost(item.usage?.estimated_cost_usd)}</td><td>{item.error_code ? label(item.error_code) : item.error_type ? label(item.error_type) : "—"}</td>
            </tr>)}</tbody>
          </table>
        </div>
        {!view.requests.length && <p>No requests are currently retained.</p>}
        <div className="observability-pagination">
          <button type="button" disabled={!view.offset || view.status === "loading"} onClick={() => load(Math.max(0, view.offset - 50))}>Previous</button>
          <button type="button" disabled={view.nextOffset == null || view.status === "loading"} onClick={() => load(view.nextOffset)}>Next</button>
        </div>
        {view.detail && <section className="observation-detail" aria-labelledby="request-detail-heading">
          <h2 id="request-detail-heading">Request details</h2><p className="request-reference"><code>{view.detail.request.request_id}</code></p>
          <dl className="observation-counts">
            <dt>Run ID</dt><dd>{uuid(view.detail.request.run_id) ? view.detail.request.run_id : "Not assigned"}</dd>
            <dt>HTTP duration</dt><dd>{duration(view.detail.request.http_duration_ms)}</dd>
            <dt>Worker duration</dt><dd>{duration(view.detail.request.worker_duration_ms)}</dd>
            <dt>Worker status</dt><dd>{label(view.detail.request.worker_status)}</dd>
            <dt>Retained events</dt><dd>{count(view.detail.request.event_count)}</dd>
            <dt>Dropped events</dt><dd>{count(view.detail.request.dropped_event_count)}</dd>
          </dl>
          <Usage value={view.detail.request.usage} /><SafeCounts value={view.detail.request.counts} />
          <p className="fine-print">HTTP completion and worker completion are separate. A timed-out HTTP request can leave its worker running. Stage durations can include nested work and should not be added together.</p>
          <ol className="observation-stages">{view.detail.stages.map((event, index) => <li key={event.sequence ?? index}>
            <div><strong>{label(event.stage)}</strong><span>{label(event.event)} · {label(event.status)} · {duration(event.duration_ms)}</span></div>
            <p className="fine-print">{timestamp(event.timestamp)}{number(event.round_number) ? ` · Round ${event.round_number}` : ""}
              {event.error_type ? ` · ${label(event.error_type)}` : ""}
              {event.error_code ? ` · ${label(event.error_code)}` : ""}
              {event.provider_stop_reason ? ` · Provider stop: ${label(event.provider_stop_reason)}` : ""}
              {/^[0-9A-Z]{5}$/.test(event.database_sqlstate || "") ? ` · Database: ${event.database_sqlstate}` : ""}
              {event.database_phase ? ` · ${label(event.database_phase)}` : ""}
              {event.reason_code ? ` · ${label(event.reason_code)}` : ""}</p>
            <SafeCounts value={event.counts} /><SafeCounts value={event} /><Exclusions value={event.exclusion_reasons} />
            {typeof event.sufficient === "boolean" && <p>Evidence sufficient: {event.sufficient ? "Yes" : "No"}</p>}
          </li>)}</ol>
        </section>}
      </>}
    </>}
  </section>;
}

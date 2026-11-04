import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { createAuthLifecycle } from "./auth.js";
import "./style.css";

const stops = {
  single_pass: "One search completed",
  clarification: "More detail needed",
  sufficient: "Evidence judged sufficient",
  round_limit: "Search round limit reached",
  no_new_evidence: "No new evidence found",
  budget: "Request budget reached",
  error: "Workflow failed",
};
const money = (value) =>
  value == null ? "Unavailable" : `$${value.toFixed(4)}`;
const platforms = ["slack", "gmail", "linear", "google_drive", "hubspot", "fireflies", "github", "jira", "confluence"];
const nativeFields = { native_key: "Jira key", status: "Status", priority: "Priority", project_id: "Project" };
const coverageLabel = (coverage) => coverage === "verified_records" ? "Verified source records"
  : coverage === "synthetic_fixture" ? "Synthetic fixture" : "Coverage unavailable";

function FilterList({ filters = [] }) {
  return filters.length ? <dl className="native-filters">{filters.map(({ field, value }) =>
    <React.Fragment key={field}><dt>{nativeFields[field] || (field === "doc_id" ? "Record ID" : field)}</dt>
      <dd>{String(value)}</dd></React.Fragment>)}</dl> : <p className="fine-print">No field filters.</p>;
}

function structuredStep(step) {
  try {
    const value = JSON.parse(step.query);
    return value && value.request?.source_type === "jira" && value.result
      && typeof value.result === "object" ? value : null;
  } catch { return null; }
}

function StructuredStep({ value }) {
  const { request, result } = value;
  return <div className="native-step">
    <p>Jira {request.operation === "count" ? "record count" : request.operation === "lookup" ? "record lookup" : "record filter"}</p>
    <FilterList filters={Object.entries(request.filters || {}).map(([field, value]) => ({ field, value }))} />
    <p className="fine-print">{coverageLabel(result.coverage)}
      {typeof result.count === "number" ? ` · ${result.count.toLocaleString()} canonical Jira source records` : ""}</p>
    {result.ambiguous && <p className="native-warning">This Jira key matches multiple permitted source records. Choose a canonical record ID to narrow the lookup.</p>}
    {result.truncated && <p className="native-warning">Only part of the matching records is shown. Narrow the filters for a complete lookup.</p>}
    {result.coverage === "unknown" && <p className="native-warning">Verified field coverage is unavailable. This result does not establish a zero count.</p>}
    {result.limitation && <p className="fine-print">{result.limitation}</p>}
  </div>;
}

function NativeRecord({ item }) {
  const states = Object.fromEntries((item.field_provenance || []).map((field) => [field.field, field.state]));
  return <>
    <h3>{item.native_key || "Jira source record"}</h3>
    <dl className="record-fields">{Object.entries(nativeFields).map(([field, label]) =>
      <React.Fragment key={field}><dt>{label}</dt><dd>
        <span>{item[field] ?? "Not available"}</span>
        <small>{states[field] === "verified" ? "Verified from source" : states[field] === "missing" ? "Missing in source" : "Invalid in source"}</small>
      </dd></React.Fragment>)}</dl>
    <p className="fine-print">Each card is a canonical source record. Jira keys may repeat.</p>
    <details className="source-id"><summary>Record provenance</summary>
      <p>Record ID: {item.record_id}</p><p>Revision: {item.revision_id}</p>
      <p>Source: {item.source?.path}</p><p>Repository: {item.source?.repository}</p>
      <p>Git revision: {item.source?.git_commit}</p><p>Source file hash: {item.source?.raw_sha256}</p>
      <p>Parser: {item.source?.parser_version}</p>
      <ul>{(item.field_provenance || []).map((field) => <li key={field.field}>
        {nativeFields[field.field]}: {field.json_pointer} · {field.state}
      </li>)}</ul><p>Citation: {item.citation_id}</p>
    </details>
  </>;
}

function AggregateCount({ item }) {
  return <>
    <h3>Matching Jira source records</h3>
    <div className="record-count"><strong>{item.count.toLocaleString()}</strong><span>canonical Jira source records</span></div>
    <p>Within your permitted records and the filters below.</p>
    <FilterList filters={item.filters} />
    <p className="fine-print">{coverageLabel(item.coverage)}. Jira keys may repeat; this is not a count of unique issues.</p>
    <details className="source-id"><summary>Count provenance</summary>
      <p>Template: {item.template_id} · version {item.template_version}</p>
      <p>Lineage: {item.lineage_sha256}</p><p>Artifact: {item.artifact_id}</p>
      <p>Snapshot: {item.scope?.snapshot?.snapshot_id}</p><p>Computed: {item.created_at}</p>
      <p>Citation: {item.citation_id}</p>
    </details>
  </>;
}

function citedText(text, ids, openCitation) {
  return text.split(/(\[[^\]\n]+\])/g).map((part, index) => {
    const id = part.slice(1, -1),
      number = ids.indexOf(id) + 1;
    return part.startsWith("[") && number > 0 ? (
      <button
        className="inline-cite"
        type="button"
        key={index}
        aria-label={`View Source ${number}`}
        onClick={() => openCitation(id)}
      >
        [{number}]
      </button>
    ) : (
      part
    );
  });
}

function App() {
  const [ready, setReady] = useState(null),
    [connection, setConnection] = useState("");
  const [question, setQuestion] = useState(""),
    [mode, setMode] = useState("agentic"),
    [source, setSource] = useState("");
  const [loading, setLoading] = useState(false),
    [result, setResult] = useState(null),
    [error, setError] = useState("");
  const [selected, setSelected] = useState("");
  const [auth, setAuth] = useState({ status: "checking", enabled: null, error: "" });
  const [login, setLogin] = useState(""), [password, setPassword] = useState("");
  const cards = useRef({}), authClient = useRef(null), queryController = useRef(null);
  const querySequence = useRef(0), readySequence = useRef(0);
  const loginInput = useRef(null), authError = useRef(null), questionInput = useRef(null);

  async function refresh() {
    const client = authClient.current, epoch = client?.generation, sequence = ++readySequence.current;
    const current = () => client?.isCurrent(epoch) && sequence === readySequence.current;
    try {
      const response = await fetch("/api/ready", { credentials: "same-origin", cache: "no-store" });
      const body = await response.json();
      if (!current()) return;
      setReady(body);
      setConnection("");
    } catch {
      if (!current()) return;
      setConnection(
        "Cannot connect to the API. Check that the backend is running, then retry.",
      );
    }
  }
  useEffect(() => {
    const client = createAuthLifecycle({ onChange: setAuth, onInvalidate: () => {
      querySequence.current += 1;
      readySequence.current += 1;
      queryController.current?.abort();
      queryController.current = null;
      setQuestion(""); setSource(""); setResult(null); setSelected(""); setError("");
      setLoading(false); setLogin(""); setPassword(""); cards.current = {};
    } });
    authClient.current = client;
    client.start();
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => { clearInterval(timer); client.dispose(); queryController.current?.abort(); };
  }, []);
  useEffect(() => {
    if (auth.error) authError.current?.focus();
    else if (auth.status === "anonymous") loginInput.current?.focus();
  }, [auth.status, auth.error]);
  useEffect(() => {
    if (auth.status === "authenticated" && !auth.error && ready?.status === "ready" && !connection && !loading)
      questionInput.current?.focus();
  }, [auth.status, auth.error, ready?.status, connection]);
  useEffect(() => {
    if (["authenticated", "benchmark"].includes(auth.status)) refresh();
  }, [auth.status]);

  async function submit(event) {
    event.preventDefault();
    const client = authClient.current, session = client?.snapshot;
    if (loading || !isReady || ready?.busy || !question.trim()
      || auth !== session
      || !["authenticated", "benchmark"].includes(session?.status)) return;
    const epoch = client.generation, sequence = ++querySequence.current;
    const current = () => client.isCurrent(epoch) && sequence === querySequence.current;
    const controller = new AbortController();
    queryController.current = controller;
    setLoading(true);
    setError("");
    setResult(null);
    setSelected("");
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        credentials: "same-origin", cache: "no-store", signal: controller.signal,
        headers: { "Content-Type": "application/json",
          ...(session.enabled ? { "X-CSRF-Token": session.csrf_token } : {}) },
        body: JSON.stringify({
          question,
          mode,
          source_types: source ? [source] : [],
        }),
      });
      let body = {};
      try { body = await response.json(); } catch { /* generic failure below */ }
      if (!current()) return;
      if (response.status === 401) { client.start(); return; }
      if (response.status === 403 && body.detail?.code === "csrf_failed") { client.revalidate(); return; }
      if (body.answer && body.trace && Array.isArray(body.sources)) {
        setResult(body);
        if (!response.ok && body.answer.status !== "error")
          setError(
            "The request reached its limit. Review the result and search steps below.",
          );
      } else {
        const detail = body.detail;
        setError(
          typeof detail === "string"
            ? detail
            : detail?.message ||
                (response.status === 422
                  ? "Check the question and filters, then try again."
                  : "The request failed. Please retry."),
        );
      }
    } catch {
      if (!current()) return;
      setError(
        "The request could not finish. Check the connection. The server may still be working; wait before retrying.",
      );
    } finally {
      if (current()) {
        if (queryController.current === controller) queryController.current = null;
        setLoading(false);
        refresh();
      }
    }
  }
  function openCitation(id) {
    if (auth !== authClient.current?.snapshot) return;
    setSelected(id);
    cards.current[id]?.scrollIntoView({ behavior: "smooth", block: "center" });
    cards.current[id]?.focus({ preventScroll: true });
  }
  const canAccess = ["authenticated", "benchmark"].includes(auth.status);
  const sources = result && canAccess
    ? [...result.sources].sort(
        (a, b) =>
          Number(result.answer.citation_ids.includes(b.citation_id)) -
          Number(result.answer.citation_ids.includes(a.citation_id)),
      )
    : [];
  const isReady = ready?.status === "ready" && !connection && canAccess;
  const authBusy = ["checking", "signing_in", "signing_out"].includes(auth.status);
  const sourceOptions = auth.enabled ? platforms : ready?.source_types || [];
  const fixture = ready?.mode === "fixture",
    retrievalOnly = ready?.mode === "retrieval";
  const example = fixture
    ? "How long are artifacts retained, how quickly can I access them, and who pays the storage fee?"
    : "What is the artifact escrow retention policy and access turnaround?";

  return (
    <div className="app">
      <header>
        <a className="brand" href="/" aria-label="Redwood home">
          <span className="brand-mark">R</span> Redwood{" "}
          <span className="brand-sub">/ Knowledge search</span>
        </a>
        <span className={`connection ${isReady ? "online" : ""}`}>
          <i />
          {isReady ? "Ready" : "Not ready"}
        </span>
      </header>
      <main>
        <section className="intro">
          <div>
            <p className="eyebrow">Enterprise knowledge assistant</p>
            <h1>
              Answers with evidence.
              <br />
              <span>A search you can follow.</span>
            </h1>
            <p>
              Ask across your workspace. Read the sources and see how the answer
              was reached.
            </p>
          </div>
          <div className="mode-note">
            <span className="tag">
              {fixture
                ? "Synthetic fixture mode"
                : retrievalOnly
                  ? "Retrieval only"
                  : ready?.mode === "live"
                    ? "Live generation"
                    : "Connecting"}
            </span>
            <p>
              {fixture
                ? "Synthetic responses for UI testing. No real retrieval or model calls."
                : retrievalOnly
                  ? "Real document search. Answer generation and critique are disabled."
                  : "Server-configured retrieval and generation. Model calls may incur cost."}
            </p>
            {!fixture && isReady && (
              <small className="fine-print">
                Retriever: {ready.retrieval} {ready.model && `· ${ready.model}`}
              </small>
            )}
          </div>
        </section>
        {auth.status === "benchmark" && <div className="scope">
          <strong>Benchmark demo identity</strong>
          <span>No real sign-in. {typeof ready?.allowed_documents === "number"
            ? `${ready.allowed_documents.toLocaleString()} documents permitted by server configuration. ` : ""}
            Use only for the benchmark demonstration.</span>
        </div>}
        {auth.status === "authenticated" && <div className="scope account">
          <div><strong>{auth.user.display_name || auth.user.user_id}</strong>
            <span>Search only documents available to your account.</span></div>
          <button type="button" onClick={() => { if (auth === authClient.current?.snapshot) authClient.current.logout(); }}>Sign out</button>
        </div>}
        {auth.enabled && auth.status !== "authenticated" && <form className="auth-card" onSubmit={event => {
          event.preventDefault();
          if (auth !== authClient.current?.snapshot) return;
          const submittedLogin = login, submittedPassword = password;
          setPassword(""); authClient.current.login(submittedLogin, submittedPassword);
        }} aria-label="Sign in">
          <h2>{auth.status === "logout_pending" ? "Sign-out is not confirmed" : "Sign in"}</h2>
          <p>Use your provisioned account to search its permitted documents.</p>
          <label htmlFor="login">Login</label>
          <input ref={loginInput} id="login" name="login" value={login} onChange={event => setLogin(event.target.value)}
            autoComplete="username" maxLength={254} required disabled={authBusy} />
          <label htmlFor="password">Password</label>
          <input id="password" name="password" type="password" value={password} onChange={event => setPassword(event.target.value)}
            autoComplete="current-password" maxLength={1024} required disabled={authBusy} />
          <button type="submit" disabled={authBusy}>{auth.status === "signing_in" ? "Signing in…" : "Sign in"}</button>
          {auth.status === "logout_pending" && <button type="button" className="secondary"
            onClick={() => { if (auth === authClient.current?.snapshot) authClient.current.logout(); }}>Retry sign out</button>}
        </form>}
        {auth.status === "checking" && <div className="notice" role="status">Checking sign-in status…</div>}
        {auth.error && <div className="notice" role="alert" tabIndex={-1} ref={authError}>{auth.error}{" "}
          {auth.status === "unavailable" && <button type="button" className="text-button"
            onClick={() => authClient.current.start()}>Retry sign-in status</button>}</div>}
        {(connection || ready?.status === "not_ready") && (
          <div className="notice" role="status">
            {connection || ready.message}{" "}
            <button type="button" className="text-button" onClick={refresh}>
              Retry connection
            </button>
          </div>
        )}
        <form className="question-card" onSubmit={submit}>
          <label htmlFor="question">What would you like to know?</label>
          <textarea
            ref={questionInput}
            id="question"
            placeholder="Ask a specific question about your documents…"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
                e.preventDefault();
                if (!e.repeat) e.currentTarget.form.requestSubmit();
              }
            }}
            aria-describedby="question-keyboard-help"
            maxLength={8000}
            required
            rows={3}
            wrap="soft"
            disabled={loading || !canAccess}
          />
          <small id="question-keyboard-help">Text wraps automatically · Enter to ask</small>
          <div className="form-controls">
            <fieldset disabled={loading || !canAccess}>
              <legend className="sr-only">Search workflow</legend>
              {[
                ["baseline", "Fixed RAG"],
                ["agentic", "Adaptive agent"],
              ].map(([value, label]) => (
                <label
                  className={`choice ${mode === value ? "active" : ""}`}
                  key={value}
                >
                  <input
                    type="radio"
                    name="workflow"
                    value={value}
                    checked={mode === value}
                    onChange={() => setMode(value)}
                  />
                  {label}
                </label>
              ))}
            </fieldset>
            <label className="source-select">
              <span className="sr-only">Source platform</span>
              <select
                value={source}
                onChange={(e) => setSource(e.target.value)}
                disabled={loading || !canAccess}
              >
                <option value="">All permitted platforms</option>
                {sourceOptions.map((s) => (
                  <option key={s} value={s}>
                    {s}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="primary"
              type="submit"
              disabled={loading || !isReady || ready?.busy || !question.trim()}
            >
              {loading ? "Searching…" : "Ask question"}{" "}
              <span aria-hidden="true">↗</span>
            </button>
          </div>
          <div className="form-help">
            <span>
              {ready?.busy && !loading
                ? "Another query is running on the server. Please wait."
                : retrievalOnly
                  ? "Both workflows currently return retrieval results only."
                  : mode === "agentic"
                    ? "Search → check evidence → follow up if needed → answer"
                    : "One search → generate an answer"}
            </span>
            <button
              className="text-button"
              type="button"
              onClick={() => setQuestion(example)}
              disabled={loading}
            >
              Try an example
            </button>
          </div>
        </form>
        {loading && (
          <div className="loading" role="status">
            <span className="spinner" /> Searching the permitted sources…
            <small>
              The agent may check evidence and search again. Please keep this
              page open.
            </small>
          </div>
        )}
        {error && (
          <div className="notice error" role="alert">
            {error}
          </div>
        )}
        {result && canAccess && (
          <div className="results">
            <div className="answer-column">
              <section className="answer-card" aria-labelledby="answer-heading">
                <div className="section-top">
                  <h2 id="answer-heading">
                    {retrievalOnly
                      ? "Retrieved evidence"
                      : result.answer.status === "error"
                        ? "Unable to answer"
                        : result.trace.stop_reason === "clarification"
                          ? "Clarification needed"
                        : result.answer.status === "insufficient_evidence"
                          ? "Partial answer / missing information"
                          : "Answer"}
                  </h2>
                  <span
                    className={`tag ${result.answer.status === "answered" ? "green" : ""}`}
                  >
                    {retrievalOnly
                      ? "Search only"
                      : result.answer.status === "answered"
                        ? "Cited response"
                        : result.answer.status === "error"
                          ? "Failed"
                          : "Incomplete"}
                  </span>
                </div>
                <p className="answer-text">
                  {citedText(
                    result.answer.text,
                    result.answer.citation_ids,
                    openCitation,
                  )}
                </p>
                {!!result.answer.citation_ids.length && (
                  <div className="citations">
                    <span>Supporting sources</span>
                    {result.answer.citation_ids.map((id, i) => (
                      <button
                        type="button"
                        key={id}
                        onClick={() => openCitation(id)}
                      >
                        Source {i + 1} ↗
                      </button>
                    ))}
                  </div>
                )}
                <div className="metrics">
                  <div>
                    <span>Latency</span>
                    <strong>
                      {(result.trace.total_latency_ms / 1000).toFixed(2)} s
                    </strong>
                  </div>
                  <div>
                    <span>Estimated cost</span>
                    <strong>
                      {money(result.trace.usage.estimated_cost_usd)}
                    </strong>
                  </div>
                  <div>
                    <span>Provider calls</span>
                    <strong>{result.trace.usage.api_calls}</strong>
                  </div>
                </div>
                {ready?.mode === "live" && (
                  <p className="fine-print">
                    Cost is an estimate for this request. Server reservations
                    may be higher. A cited response can still be incomplete or
                    incorrect.
                  </p>
                )}
              </section>
              <details className="steps">
                <summary>
                  Search steps <span>{result.trace.steps.length} steps</span>
                </summary>
                <div className="stop">
                  <strong>Stopped:</strong>{" "}
                  {stops[result.trace.stop_reason] || result.trace.stop_reason}
                </div>
                <ol>
                  {result.trace.steps.map((step, i) => (
                    <li key={i}>
                      <div>
                        <strong>
                          {step.tool === "text_search"
                            ? "Search"
                            : step.tool === "read_more"
                              ? "Read more from source"
                            : step.tool === "critic"
                              ? "Evidence check"
                              : step.tool === "generate"
                                ? "Generate answer"
                                : step.tool === "route"
                                  ? "Plan search"
                                  : step.tool}
                        </strong>
                        <span>
                          Round {step.round_number} ·{" "}
                          {(step.elapsed_ms / 1000).toFixed(2)} s
                        </span>
                      </div>
                      {structuredStep(step) ? <StructuredStep value={structuredStep(step)} /> : <p>{step.query}</p>}
                      {step.tool === "text_search" && (
                        <small>
                          {step.evidence_ids.length} passages ·{" "}
                          {step.source_types.length
                            ? step.source_types.join(", ")
                            : "permitted platforms"}
                        </small>
                      )}
                      {step.critic && (
                        <>
                          <p>{step.critic.reason}</p>
                          {step.critic.facts?.map((fact, index) => (
                            <p key={index}>
                              {fact.status === "supported" ? "Supported" : fact.status === "ambiguous"
                                ? "Needs clarification" : fact.status === "conflicting" ? "Conflicting" : "Missing"}
                              : {fact.fact}
                            </p>
                          ))}
                          {!!step.critic.missing_facts.length && (
                            <p className="missing">
                              <strong>Missing:</strong>{" "}
                              {step.critic.missing_facts.join("; ")}
                            </p>
                          )}
                        </>
                      )}
                    </li>
                  ))}
                </ol>
                <p className="fine-print">
                  Evidence checks guide the next search. They are not
                  measurements of answer accuracy.
                </p>
              </details>
            </div>
            <aside className="evidence" aria-label="Supporting evidence">
              <div className="section-top">
                <h2>Evidence</h2>
                <span>{sources.length} {sources.length === 1 ? "item" : "items"}</span>
              </div>
              <p className="fine-print">
                {sources.some((item) => item.kind === "record" || item.kind === "aggregate")
                  ? "Source excerpts, native records, and counts from your permitted sources."
                  : "Exact retrieved excerpts, with document character offsets."}
              </p>
              {!sources.length && (
                <p className="empty">No supporting evidence was returned.</p>
              )}
              {sources.map((item) => {
                const cited = result.answer.citation_ids.indexOf(
                  item.citation_id,
                );
                return (
                  <article
                    tabIndex={-1}
                    ref={(el) => { if (el) cards.current[item.citation_id] = el; else delete cards.current[item.citation_id]; }}
                    key={item.citation_id}
                    className={`source-card ${item.kind || "passage"}-card ${selected === item.citation_id ? "selected" : ""}`}
                  >
                    <div className="source-meta">
                      <span>{item.kind === "record" ? "Native Jira record" : item.kind === "aggregate" ? "Scoped Jira count" : item.chunk?.source_type || "Source"}</span>
                      <span>
                        {cited >= 0
                          ? `Cited · Source ${cited + 1}`
                          : "Retrieved"}
                      </span>
                    </div>
                    {item.kind === "record" ? <NativeRecord item={item} />
                      : item.kind === "aggregate" ? <AggregateCount item={item} />
                      : item.chunk ? <><h3>{item.chunk.title || "Untitled document"}</h3>
                    <blockquote>{item.chunk.text}</blockquote>
                    <details className="source-id">
                      <summary>Document details</summary>
                      <p>{item.chunk.doc_id}</p>
                      <p>
                        Characters {item.chunk.start_char}–{item.chunk.end_char}{" "}
                        · {item.retrieval_method}
                      </p>
                      <p>Citation: {item.citation_id}</p>
                    </details></> : <p>Source details are unavailable.</p>}
                  </article>
                );
              })}
            </aside>
          </div>
        )}
        {!result && !loading && (
          <section className="empty-state">
            <span aria-hidden="true">⌕</span>
            <h2>Start with a question</h2>
            <p>
              Your answer, supporting evidence, and search steps will appear
              here.
            </p>
          </section>
        )}
      </main>
      <footer>
        Redwood Inference{" "}
        <span>Milestone demo · Verify claims against the cited evidence</span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);

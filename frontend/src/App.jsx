import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const stops = {
  single_pass: "One search completed",
  sufficient: "Evidence judged sufficient",
  round_limit: "Search round limit reached",
  no_new_evidence: "No new evidence found",
  budget: "Request budget reached",
  error: "Workflow failed",
};
const money = (value) =>
  value == null ? "Unavailable" : `$${value.toFixed(4)}`;

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
  const cards = useRef({});

  async function refresh() {
    try {
      const response = await fetch("/api/ready");
      const body = await response.json();
      setReady(body);
      setConnection("");
    } catch {
      setConnection(
        "Cannot connect to the API. Check that the backend is running, then retry.",
      );
    }
  }
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => clearInterval(timer);
  }, []);

  async function submit(event) {
    event.preventDefault();
    if (loading || !isReady || ready?.busy || !question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    setSelected("");
    try {
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          mode,
          source_types: source ? [source] : [],
        }),
      });
      const body = await response.json();
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
      setError(
        "The request could not finish. Check the connection. The server may still be working; wait before retrying.",
      );
    } finally {
      setLoading(false);
      refresh();
    }
  }
  function openCitation(id) {
    setSelected(id);
    cards.current[id]?.scrollIntoView({ behavior: "smooth", block: "center" });
    cards.current[id]?.focus({ preventScroll: true });
  }
  const sources = result
    ? [...result.sources].sort(
        (a, b) =>
          Number(result.answer.citation_ids.includes(b.citation_id)) -
          Number(result.answer.citation_ids.includes(a.citation_id)),
      )
    : [];
  const isReady = ready?.status === "ready" && !connection;
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
        <div className="scope">
          <strong>Benchmark demo identity</strong>
          <span>
            No real sign-in.{" "}
            {isReady
              ? `${ready.allowed_documents.toLocaleString()} documents permitted by server configuration.`
              : "Waiting for server scope."}{" "}
            Use only for the benchmark demonstration.
          </span>
        </div>
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
            disabled={loading}
          />
          <small id="question-keyboard-help">Text wraps automatically · Enter to ask</small>
          <div className="form-controls">
            <fieldset disabled={loading}>
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
                disabled={loading}
              >
                <option value="">All permitted platforms</option>
                {ready?.source_types?.map((s) => (
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
        {result && (
          <div className="results">
            <div className="answer-column">
              <section className="answer-card" aria-labelledby="answer-heading">
                <div className="section-top">
                  <h2 id="answer-heading">
                    {retrievalOnly
                      ? "Retrieved evidence"
                      : result.answer.status === "error"
                        ? "Unable to answer"
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
                      <p>{step.query}</p>
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
            <aside className="evidence" aria-label="Source passages">
              <div className="section-top">
                <h2>Evidence</h2>
                <span>{sources.length} passages</span>
              </div>
              <p className="fine-print">
                Exact retrieved excerpts, with document character offsets.
              </p>
              {!sources.length && (
                <p className="empty">No supporting passages were returned.</p>
              )}
              {sources.map((item) => {
                const cited = result.answer.citation_ids.indexOf(
                  item.citation_id,
                );
                return (
                  <article
                    tabIndex={-1}
                    ref={(el) => (cards.current[item.citation_id] = el)}
                    key={item.citation_id}
                    className={`source-card ${selected === item.citation_id ? "selected" : ""}`}
                  >
                    <div className="source-meta">
                      <span>{item.chunk.source_type}</span>
                      <span>
                        {cited >= 0
                          ? `Cited · Source ${cited + 1}`
                          : "Retrieved"}
                      </span>
                    </div>
                    <h3>{item.chunk.title || "Untitled document"}</h3>
                    <blockquote>{item.chunk.text}</blockquote>
                    <details className="source-id">
                      <summary>Document details</summary>
                      <p>{item.chunk.doc_id}</p>
                      <p>
                        Characters {item.chunk.start_char}–{item.chunk.end_char}{" "}
                        · {item.retrieval_method}
                      </p>
                      <p>Citation: {item.citation_id}</p>
                    </details>
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
              Your answer, supporting passages, and search steps will appear
              here.
            </p>
          </section>
        )}
      </main>
      <footer>
        Redwood Inference{" "}
        <span>Milestone demo · Verify claims against the cited passages</span>
      </footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);

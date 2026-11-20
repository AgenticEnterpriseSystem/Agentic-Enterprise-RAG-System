import React from "react";

const labels = {
  policy: "Check document access",
  route: "Plan search",
  text_search: "Search documents",
  hybrid: "Search documents",
  tool: "Look up records",
  read_more: "Read more of a document",
  context_select: "Select evidence",
  critic: "Check evidence",
  generate: "Write answer",
};
// Internal plumbing is only worth showing when it breaks.
// policy.resolve spans its grants/bitmap/register sub-steps, and the agent span
// is the whole run already shown as the total, so both would only repeat rows.
const quiet = new Set(["http", "worker", "request", "auth", "auth.authenticate", "bm25", "dense", "fusion",
  "materialize", "agent", "policy.grants", "policy.bitmap", "policy.register"]);
const failedStatuses = new Set(["error", "denied", "timeout", "cancelled"]);
const stageName = (stage) => labels[stage] || (stage.startsWith("policy.") ? labels.policy : stage.replace(/[._]/g, " "));
const seconds = (ms) => `${(ms / 1000).toFixed(1)}s`;

// Start/end/error events arrive as separate rows; pair them FIFO because one
// round can run several searches of the same stage concurrently.
export function buildRows(stages) {
  const rows = [], open = new Map();
  for (const event of Array.isArray(stages) ? stages : []) {
    if (!event || typeof event.stage !== "string") continue;
    const key = `${event.stage}|${event.round_number ?? 0}`;
    if (event.event === "start") {
      const row = { key: `${key}|${event.sequence}`, stage: event.stage, round: event.round_number ?? null,
        state: "running", duration: null, code: "" };
      rows.push(row);
      open.set(key, [...(open.get(key) || []), row]);
    } else if (event.event === "end" || event.event === "error") {
      const queue = open.get(key) || [];
      let row = queue.shift();
      if (!row) {
        row = { key: `${key}|${event.sequence}`, stage: event.stage, round: event.round_number ?? null };
        rows.push(row);
      }
      row.state = event.event === "error" || failedStatuses.has(event.status) ? "failed" : "done";
      row.duration = typeof event.duration_ms === "number" ? event.duration_ms : null;
      row.code = row.state === "failed" ? event.error_code || event.error_type || "" : "";
    }
  }
  return rows.filter((row) => !quiet.has(row.stage) || row.state === "failed");
}

export default function QueryProgress({ progress, loading, elapsedMs, outcome }) {
  const rows = buildRows(progress?.stages);
  const running = rows.filter((row) => row.state === "running").at(-1);
  const announcement = loading
    ? running ? `${stageName(running.stage)}${running.round ? `, round ${running.round}` : ""}` : "Searching the permitted sources"
    : outcome ? `Finished: ${outcome}` : "";
  const body = (
    <>
      <p className="progress-total">{loading ? "Elapsed" : "Total"} {seconds(elapsedMs)}</p>
      {progress?.unavailable && <p className="fine-print">Live progress unavailable</p>}
      {rows.length > 0 && (
        <ol className="progress-rows">
          {rows.map((row) => (
            <li key={row.key} className={`progress-${row.state}`}>
              <span className="progress-name">{stageName(row.stage)}</span>
              {row.round && <span className="progress-round">Round {row.round}</span>}
              <span className="progress-state">
                {row.state === "running" ? (loading ? "Running…" : "Stopped")
                  : row.state === "done" ? "Done" : `Failed${row.code ? ` (${row.code})` : ""}`}
              </span>
              {row.duration != null && <span className="progress-duration">{seconds(row.duration)}</span>}
            </li>
          ))}
        </ol>
      )}
      {outcome && <p className="progress-outcome"><strong>Outcome:</strong> {outcome}</p>}
    </>
  );
  return (
    <>
      <p className="sr-only" aria-live="polite">{announcement}</p>
      {loading ? <div className="query-progress">{body}</div>
        : <details className="progress-details"><summary>Run progress</summary><div className="query-progress">{body}</div></details>}
    </>
  );
}

# Red-team suite — runbook

A **reproducible test suite** for the security findings against the running
application. Each finding is an attack that either reproduces (test **PASSED**)
or does not (test **FAILED**), runnable with a single command, and every run
saves sanitized evidence.

- **Findings & analysis:** [`FINDINGS_code_review.md`](FINDINGS_code_review.md)
- **Accounts & grants:** [`account_matrix.md`](account_matrix.md)
- **Attack engine (CLI):** [`../scripts/red_team/availability.py`](../scripts/red_team/availability.py)
- **Test suite (pytest):** [`../scripts/red_team/test_findings.py`](../scripts/red_team/test_findings.py)

## What the suite covers

There are two layers:

**1. Availability suite** ([`test_findings.py`](../scripts/red_team/test_findings.py)) —
**free**: zero `/api/query` calls, so no paid model usage. Every request is a
failed login, a session check, or a run-receipt read.

| Test | Finding | Needs a password? |
| --- | --- | --- |
| `test_f5_auth_worker_can_be_saturated` | F5 — unauthenticated auth-worker DoS | No |
| `test_f3_free_reads_starve_the_query_slot` | F3 — free read takes the single query slot | Yes (bystander) |
| `test_f2_fake_logins_lock_out_an_unrelated_user` | F2 — shared-IP lockout | Yes (bystander) |
| `test_f2_wrong_passwords_lock_out_the_real_user` | F2 — targeted per-user lockout | Yes (target) |
| `test_observability_endpoints_require_admin` | access control — non-admin refused on `/api/observability/*` (PASS = refused) | Yes (rt-alpha) |

**2. Content / injection suite** ([`test_content.py`](../scripts/red_team/test_content.py)) —
**paid, opt-in**: needs the generated answer, so each case calls `/api/query`
in live mode (~$0.01–0.02 each). Skipped entirely unless `RT_ALLOW_PAID=1`.
These target the following documented limitations:

| Case | Probes | Focus area |
| --- | --- | --- |
| `acc_alpha_cannot_see_setB` / `acc_bravo_cannot_see_setA` | A granted user never sees the other set's documents | access boundary, end-to-end |
| `acc_nogrants_gets_nothing` | A no-grants account gets no sources | access boundary |
| `inj_direct_canary_echo` | A question can't make the model emit an injected canary | "Prompt-injection resistance is not proven" |
| `inj_direct_scope_expand` | "Admin mode" injection can't widen document scope | prompt-injection + access |
| `inj_persona_override` / `inj_exfil_doc_ids` | "FreeBot" persona / doc-ID dump don't get obeyed | prompt-injection |
| `inj_delimiter_escape` / `inj_multilingual` | fake `</evidence>` tags and a Spanish injection don't get obeyed | prompt-injection |
| `prompt_extract_review` | system-prompt extraction (recorded for human review) | prompt-injection |
| `test_run_receipt_is_owner_only` *(separate test fn)* | one user can't read another's run receipt (403); mints one cheap run | access boundary |
| `unanswerable_out_of_corpus` | Out-of-corpus questions aren't answered with invented citations | — |
| `cite_false_value_pressure` | Pressure to confirm a wrong value (the 10 MB → 10,000 MB case) | "Valid citations can support false claims" |

Out of scope: **F1** (spending-cap drain, costs a few cents — see FINDINGS) and
**indirect/retrieved-document injection** with a planted canary document, which
requires the separate live fixture stack (future work).

## How to read the results

This is a red-team suite, so the pass/fail meaning is inverted from normal tests:

| Result | Meaning |
| --- | --- |
| **PASSED** | The attack reproduced. |
| **SKIPPED** | A password was not supplied, or a timing-sensitive non-result. |
| **FAILED** | The attack did not reproduce (the defense held) — the code may have been fixed; update the finding's status. |

All tests passing therefore means every documented finding still reproduces.

**The content/injection suite uses the opposite convention.** Those tests cover
defenses the application is expected to have, so there **PASSED** = the defense
held and **FAILED** = a vulnerability was found. `citation_review` cases never
hard-fail; they SKIP with the full answer recorded for human judgement. In both
suites, a **FAIL is the result to investigate**.

## Prerequisites

1. **The app is running** in live or retrieval mode at `http://127.0.0.1:8000`.
   (F5 works in either mode; it never reaches the model.)
2. **Python with pytest.** The project venv already has it:
   - Windows: `.venv\Scripts\python.exe`
   - WSL/Linux/macOS: an activated venv (`pip install -r requirements.txt`)

## Run it

### Just F5 (no accounts needed — works for anyone)

```powershell
.venv\Scripts\python.exe -m pytest scripts/red_team -v
```

`test_f5_auth_worker_can_be_saturated` passes; the F2/F3 tests are `SKIPPED`
with a note that they need passwords.

### All four findings (needs the two test accounts)

First create the test accounts and grants once, following
[`account_matrix.md`](account_matrix.md). Then supply their passwords **via the
environment** (never typed into a file, never written to disk):

```powershell
$env:RT_PASS_RT_ALPHA = "the rt-alpha password"   # bystander (F2-global, F3)
$env:RT_PASS_RT_BRAVO = "the rt-bravo password"    # target   (F2-targeted)
.venv\Scripts\python.exe -m pytest scripts/red_team -v
```

WSL / bash equivalent:

```bash
export RT_PASS_RT_ALPHA='the rt-alpha password'
export RT_PASS_RT_BRAVO='the rt-bravo password'
python -m pytest scripts/red_team -v
```

The env var name is `RT_PASS_<LOGIN>` with the login upper-cased and every
non-alphanumeric character turned into `_` (so `rt-alpha` -> `RT_PASS_RT_ALPHA`).

> The three throwaway accounts (`rt-nogrants`, `rt-alpha`, `rt-bravo`) have
> already been provisioned on this local stack with grants applied (Set A to
> `rt-alpha`, Set B to `rt-bravo`). They are disposable local-only accounts; the
> password set at creation is supplied as `<rt-password>` below. Recreate or
> re-grant them with the operator steps in
> [`account_matrix.md`](account_matrix.md) if the database is rebuilt.

### The paid content / injection suite (opt-in, costs money)

These call the model, so they run only when opted in with `RT_ALLOW_PAID=1`,
and they stop once the per-run budget (`RT_MAX_COST_USD`, default `0.50`) is
spent. The content cases cost roughly $0.10–0.20 total.

```powershell
$env:RT_ALLOW_PAID = "1"; $env:RT_MAX_COST_USD = "0.50"; $env:RT_PASS_RT_ALPHA = "<rt-password>"; $env:RT_PASS_RT_BRAVO = "<rt-password>"; $env:RT_PASS_RT_NOGRANTS = "<rt-password>"; .venv\Scripts\python.exe -m pytest scripts/red_team/test_content.py -v -rs
```

PASSED = the defense held; FAILED = a vulnerability found; SKIPPED = a
`citation_review` case recorded for human judgement (the answer is in the
evidence bundle). Each run trips the login rate limit for ~60s, same as above.

### F1 spending-cap drain (destructive — read before running)

`test_f1_spending_cap_drains_and_blocks_everyone` reproduces F1: one
low-privileged account asks permitted questions until the process-wide cap trips
(`429 spending_limit`), then shows an unrelated account is blocked too. This
**deliberately exhausts the server budget**, so generation stays disabled for
every user **until the app is restarted**. Because of that it has its own extra
gate, `RT_ALLOW_BUDGET_DRAIN=1`, and it PASSES when the drain reproduces (like
the availability suite, not the defense suite). For a clean "N questions from
zero", restart the app first so the reservation starts at 0.

```powershell
$env:RT_ALLOW_PAID = "1"; $env:RT_ALLOW_BUDGET_DRAIN = "1"; $env:RT_MAX_COST_USD = "0.75"; $env:RT_PASS_RT_ALPHA = "<rt-password>"; $env:RT_PASS_RT_BRAVO = "<rt-password>"; .venv\Scripts\python.exe -m pytest "scripts/red_team/test_content.py::test_f1_spending_cap_drains_and_blocks_everyone" -v -rs
```

Afterwards, restart the app to re-enable generation.

## Evidence

Every run writes a sanitized bundle to
`outputs/red_team/<UTC-timestamp>-<id>-pytest/`:

- `metadata.json` — commit SHA + dirty flag, target URL, concurrency, which
  passwords were supplied (booleans only), and a note that no paid calls ran.
- `results.jsonl` — one line per test with the raw observations (status codes,
  the 429 histogram, latencies).
- `summary.md` — the human-readable outcome table for the report.

`outputs/red_team/` is gitignored, so raw evidence never lands in Git. The
specific bundles cited in the written report are copied in manually.

**No secrets are ever written** — not passwords, cookies, CSRF tokens, or API
keys. This can be verified: `grep -ri RT_PASS outputs/red_team/` returns nothing.

## Configuration (optional)

Override any of these as environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `RT_BASE_URL` | `http://127.0.0.1:8000` | App base URL |
| `RT_ORIGIN` | derived from base URL | `Origin` header (match `RAG_AUTH_ALLOWED_ORIGINS`) |
| `RT_TARGET_LOGIN` | `rt-bravo` | Account locked out in F2-targeted |
| `RT_BYSTANDER_LOGIN` | `rt-alpha` | Unrelated account for F2-global / F3 session |
| `RT_CONCURRENCY` | `12` | Simultaneous requests for F3/F5 |
| `RT_REPEATS` | `1` | How many times each content case runs (per-case `repeats` wins). Injection cases default to 3. A defense must hold in **every** repeat to pass; one break is a finding |
| `RT_MAX_COST_USD` | `0.50` | Budget cap for a paid content run; it stops once reached |
| `RT_OUTDIR` | `outputs/red_team` | Where evidence bundles are written |

## Notes and caveats

- **The suite trips the login rate limit on purpose.** After the F2 tests run,
  logins from this machine (including an active browser session) are refused with
  `429` for about **60 seconds**, then recover on their own. Running the whole
  suite twice in quick succession can cause the login-based tests in the second
  run to `SKIP` or behave oddly while the window drains — wait ~60s between full runs.
- **F5 is the one that needs nothing** and is the cleanest live demo: run it, and
  7-ish of 12 concurrent junk auth checks come back `429 "busy"`.
- The suite will **not** be picked up by a plain `pytest` run — `pytest.ini`
  pins `testpaths = tests`, so CI and the normal unit tests ignore this folder.
  Pytest must be pointed at it explicitly (`pytest scripts/red_team`).
- The CLI engine ([`availability.py`](../scripts/red_team/availability.py)) is
  still available for ad-hoc runs and a dry-run connectivity check:
  `python scripts/red_team/availability.py` (no `--run`) just checks the app is
  reachable without sending any attack traffic.

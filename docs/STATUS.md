# Current status

Last checked: **November 16, 2026**.

## In short

- **The app works end to end.** You can sign in, search, see only the documents
  you're allowed to see, and (in paid mode) get an answer with citations. This
  was checked in a real browser and against a real database.
- **Answer accuracy is not proven.** A citation shows where an answer came from,
  not that the answer is right. One tested answer cited the wrong date.
- **We don't yet know whether the adaptive agent beats fixed RAG.** No fair
  side-by-side comparison has been run.
- **Some searches are slow.** One demo attempt timed out after 180 seconds.
- **Setup on a teammate's computer hasn't been verified yet.** It has only been
  run on the original developer's machine.

## What works

| Feature | What it does |
| --- | --- |
| Login | Real accounts, passwords and sessions |
| Document permissions | Each account sees only the documents it was granted, enforced by both the app and the database |
| Hybrid search | Keyword search plus search by meaning |
| Fixed RAG | Searches once, then writes an answer (paid mode) |
| Adaptive agent | Checks whether the needed facts were found; can read more or search again |
| Jira tools | Looks up and counts imported Jira records |
| Source display | Shows the answer, source passages, citations and search steps |
| Observability | A separate admin-only page with timings, usage and errors |

The app starts in **free mode**: search and sources, no paid AI calls. Answers
need an API key and a spending limit
([quick start step 11](QUICKSTART.md#11-optional-turn-on-ai-answers)).

## What was tested

- **Three real-browser runs passed.** One generated an answer, one asked the
  user for a missing detail, and one recorded demo run answered a
  GitHub-filtered question, matching its cited source, in about **61 seconds**.
- **Automated tests passed:** 138 regression checks, 102 real-database checks
  and 7 browser checks. The sets overlap, so don't add them up.
- **Permissions held at full scale.** Three separate checks with the full
  search indexes loaded each returned exactly the same **511,958** allowed
  documents.

## What still needs work

1. **Picking the right source.** One answer said **April 7**; the expected
   answer is **April 6**. The cited document really does say April 7, so the
   problem is choosing between similar documents.
2. **Checking every sentence.** The app checks that citations point to real
   evidence, not that each sentence follows from it.
3. **Human review.** Machine-graded answers and the critic's decisions still need
   people to check them.
4. **Fair comparisons.** Adaptive agent versus fixed RAG, and 8 versus 5 evidence
   documents, are both unmeasured.
5. **Slow meaning search.** In one demo attempt two searches took about 95 and
   79 seconds, and the request timed out. The cause is not yet known.
6. **Teammate setup.** A fresh install on someone else's computer is still open.

The project plan has **19 of 23 tickets accepted**. The rest are critic
calibration, fair comparisons, outside quality review and release work.

## Accounts

On Jane's installation, `jane` and `hira` can both open Observability, and each
has access to all 511,958 benchmark documents. Hira's login and access were
verified there. Accounts and access are **not** copied to your computer. Create
your own by following the [quick start](QUICKSTART.md).

---

## Engineering notes

Detail for people changing or debugging the system.

### Fixes behind the passing runs

1. The critic ran out of room for its checklist. It now has its own 2,048-token
   output limit.
2. The HTTP request timed out before the agent finished, because permission
   checks used part of the time. HTTP now waits 180 seconds; the agent still
   has 120.
3. A large permission query failed while a database transaction sat idle. Grant
   pages now release the transaction before processing rows, then recheck
   permissions before use.
4. After a second full-access account was added, the first grant page hit the
   5-second database statement limit before the agent started. The query now
   limits each branch before combining them. 124 policy and database tests
   passed; the first page now takes about 0.10 seconds, and both Jane and Hira
   resolve the same 511,958 documents.

### The browser runs in detail

- **Generated answer (build r2):** two critic checks, document expansion, answer,
  sources, citations, diagnostics and logout. This is the run with the date
  mismatch above.
- **Clarification (build r3):** two critic checks and document expansion, then a
  request for the missing detail. It stopped before the answer generator.
- **Recorded demo (Adaptive agent, GitHub filter):** four hybrid searches, one
  critic check, answer, sources and logout; HTTP 200 in about 61.2 seconds. The
  answer (**10 MiB per file and 50 MiB per request**) matched the cited
  upload-limit document. It used 4 of a possible 8 documents. One run does not
  establish accuracy or speed for every question. The video is linked from the
  [project README](../README.md#demo).

r2 and r3 are local app builds, not model versions. Their critic and generator
code were identical; only permission paging, database settings and error
diagnostics changed.

### Slow-search detail

In the failed demo attempt, permissions passed, the HTTP request timed out at
180 seconds, and the worker then stopped on its time budget before the critic
or generator ran. The app now uses one native math-library thread per pool to
reduce contention. Separate free-mode timings (real retrieval, no generation):
59 seconds for an all-platform hybrid search, 6.5 seconds for a GitHub-filtered
one.

### Other review gaps

- One clarification included an ownership phrase that was only partly visible in
  a cropped excerpt.
- Three saved answers have valid machine grades and citation audits, but review
  flags remain. The critic collection has 9 outcomes and 4 decisions, with no
  human labels.

### Settings used for these checks

| Setting | Value |
| --- | --- |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Live language model | `claude-haiku-4-5-20251001` |
| Shared evidence limit | 8 unique documents |
| Search rounds | Up to 2 |
| Critic output | Up to 2,048 tokens |
| Router / generator output | Up to 1,024 tokens per call |
| Total question tokens | Up to 16,000 |
| Agent time / HTTP wait | 120 / 180 seconds |
| Question / process cost cap | $0.10 / $0.50 |
| Database idle transaction / statement / lock | 15 / 5 / 2 seconds |
| Native math-library threads | 1 per pool |

The process spending counter resets on restart. It is not a billing limit. These
are recorded settings, not a price quote or approval for new paid runs.

See [Evaluation](EVALUATION.md) for research checks and the
[system card](system_card.md) for intended use and limits.

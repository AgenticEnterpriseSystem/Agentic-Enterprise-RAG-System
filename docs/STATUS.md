# Current status and known limitations

Last checked: **November 16, 2026**. This page summarizes the shared code and the
local checks completed by that date. It is not a claim that every question will
work or every answer will be correct.

## What is implemented?

| Feature | Current behavior |
| --- | --- |
| Local login | Real accounts with passwords and server-managed sessions |
| Document permissions | Explicit user/group grants, checked by both the application and PostgreSQL |
| Hybrid search | Combines BM25+ keyword search and BGE-base dense search |
| Fixed RAG | One routed search round, followed by answer generation in live mode |
| Adaptive agent | A critic checks requested facts; the workflow can read more or search again |
| Jira tools | Look up and count imported records when verified fields and coverage are available |
| Evidence context | At most eight unique documents shared by the critic and final generator, also limited by text and token budgets |
| Source display | Answers, source previews, citations and step/usage information |
| Observability | Separate `/observability` page for explicitly allowed admins |
| External evaluation | Separate tooling can grade saved answers and audit citations; those grades still need human review |

The ordinary setup starts in **retrieval-only mode**, with no paid model calls.
Live answers require explicit API configuration and a spending budget.

## What passed the latest checks?

The latest repair addressed three runtime problems:

1. The critic could run out of room to return its checklist. It now has a
   separate output limit of 2,048 tokens.
2. The HTTP request could time out before the agent finished because permission
   checks also used part of that time. The managed HTTP limit is now 180 seconds;
   the agent still has 120 seconds.
3. A large permission query could fail while a database transaction sat idle.
   Grant pages now release the transaction before Python processes the rows,
   then check permissions again before using the page.

Real Chrome checks showed two distinct successful paths. The labels **r2** and
**r3** below identify local app builds, not language-model versions:

- **Generated answer:** revision r2 completed two critic checks, document
  expansion, answer generation, source/citation display, diagnostics and logout.
  Its answer still had the date mismatch described below.
- **Missing-evidence clarification:** final revision r3 completed two critic
  checks and document expansion, then asked for missing information. It returned
  a successful HTTP response (`200`) and passed the interface, diagnostics and
  logout checks. This normal
  early stop did **not** call the separate final answer generator.

The critic and generator code stayed the same between those revisions. The later
repair changed permission paging, database settings and error diagnostics.
Earlier failed attempts remain recorded; these successes do not erase them.

Automated checks passed: **138 final regression checks, 102 real PostgreSQL
checks and seven browser checks**. These sets overlap, so do not add them
together. Three separate permission checks, with the local embedding model and
full indexes loaded, each produced the same allowed set of **511,958 documents**.

The November 16 demo recording found another permission-query problem after a
second full-access account was added. PostgreSQL scanned and sorted the large
grant table before returning its first page, then hit the five-second statement
limit. That attempt stopped **before the agent or model started**.

The query now limits each ordered grant branch before combining the results.
Authorization rules and fresh permission checks are unchanged. **124 policy and
PostgreSQL tests passed**, including duplicate grants, group requirements and
revocation between pages. On the local database, the patched first page took
about 0.10 seconds. Full permission resolution then passed for both Jane and
Hira, with the same 511,958-document allowed set. This repair is in `jane/dev`;
a successful recording of the updated app is still pending.

## What still needs work?

- **Choosing the right source:** a tested answer gave **April 7, 2026 at
  09:00 UTC**, while the benchmark reference expects **April 6**. Its cited
  document really says April 7. The system needs better source selection among
  similar documents; a valid citation alone did not make this answer correct.
- **Checking every claim:** the system checks that citations refer to supplied
  evidence. It does not reliably prove that every sentence follows from that
  evidence. One clarification also included an ownership phrase whose support
  was only partly visible in a cropped excerpt.
- **Human review:** saved critic decisions and model-generated answer grades
  still need human judgments. Three saved answer cases received valid machine
  grades and citation audits, but review flags remain. The critic collection
  contains nine outcomes and four decisions, with no completed human labels.
- **Fair comparisons:** no controlled result establishes that the current
  adaptive agent beats fixed RAG, or that an eight-document limit beats five.
- **Dense-search latency:** the next demo attempt passed permissions, but two
  dense searches took about 95 and 79 seconds. HTTP timed out at 180 seconds;
  the worker later stopped on its time budget. It did not reach the critic or
  final answer generator. The cause still needs diagnosis, and this attempt is
  not a successful end-to-end demo.
  The Compose app now uses one thread per native math-library pool to reduce
  contention. A separate unpaid all-platform hybrid check took 59 seconds;
  a GitHub-filtered check took 6.5 seconds and returned the expected upload-limit
  passage. Those checks included permissions and real retrieval, with external
  generation disabled. They do not replace a successful live agent recording.
- **Independent setup and delivery:** a fresh teammate installation, merge
  review and the final demo/research deliverables remain open. A check
  on the original developer's machine does not verify a teammate's installation.

The implementation plan has **19 of 23 tickets accepted**. The remaining
acceptance includes critic calibration, controlled comparisons, external quality
review and release work. A local runtime repair does not complete those gates.

## Settings used for the latest local checks

| Setting | Value |
| --- | --- |
| Embedding model | `BAAI/bge-base-en-v1.5` |
| Live language model | `claude-haiku-4-5-20251001` |
| Shared evidence limit | 8 unique documents |
| Search rounds | Up to 2 |
| Critic output | Up to 2,048 tokens |
| Router/final-generator output | Up to 1,024 tokens per call |
| Total question tokens | Up to 16,000 |
| Agent time / query HTTP wait | 120 / 180 seconds |
| Question / process cost cap | USD 0.10 / USD 0.50 |
| Database idle transaction / statement / lock limit | 15 / 5 / 2 seconds |
| Compose app native math-library threads | 1 per pool; separate operator experiments keep their own settings |

The token and cost limits can prevent a stage from using its full output limit.
The process spending counter resets when the server restarts. It is not an
account-wide billing limit or an invoice. These are recorded settings, not a
current provider pricing quotation or permission to run a new paid batch.

## Accounts and teammate testing

On Jane's checked local installation, **`jane` and `hira` can open Observability**.
Both have separate explicit grants for all 511,958 benchmark documents. Hira's
login, document access and diagnostics access were verified on this installation.
Accounts, grants and admin settings are not automatically copied to a teammate's
installation.

To test on your computer, follow [Setup](WEB_APP.md), create your own account,
grant the intended documents, and optionally allow that account to open
Observability. Keep passwords and private settings out of Git.

Start with [Evaluation](EVALUATION.md) for research checks and
[the System Card](system_card.md) for intended use and limitations. Detailed
private operator receipts are retained outside Git; this page is the shareable
summary and does not require those private files to be readable.

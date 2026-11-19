# Experiments tested on the `hira/dev` branch

This page records retrieval and agent experiments run on the `hira/dev` branch in November
2026. **They are not part of this version of the app.** Some were tested on `hira/dev`'s own
code, which differs from this version (see below), so every result here is evidence to
consider, not a measured property of this app. Ideas worth carrying over still need testing
on this version before they are adopted.

**How the two versions differ.** Both started from the same code on 27 October 2026. This
version then added accounts, permissions, the PostgreSQL store, a rewritten critic (a
fact-by-fact checklist), "read more" of an already-found document, clarification questions
and an eight-document context. `hira/dev` kept the earlier critic (the 25 October version)
and changed how much text the critic sees. Search rankings are the same in both: this
version's search was checked against the original and returned identical rankings (see
[Recorded search parity and resource results](FULL_CORPUS.md#accepted-full-corpus-retrieval-measurement)),
so the retrieval results below apply to this version as they are.

The scripts used here (`scripts/time_agent.py`, `scripts/excerpt_coverage.py`) and the full
experiment log (`docs/EXPERIMENTS.md`) are on the `hira/dev` branch.

## Summary

Recall@10 is the share of a question's reference documents found in the top 10 results,
averaged over the 282 scored development questions; nDCG@10 also rewards ranking them near
the top. "Answered" in live runs means the critic accepted the evidence and an answer was
written; only the comparison in [Part 3](#part-3-comparison-with-this-version-25-fresh-questions)
checked answers against the reference answers.

| Tested idea | Result on `hira/dev` | Status for this version |
| --- | --- | --- |
| Rerankers reading whole-document windows, blended with the hybrid order | Retrieval improved clearly (Recall@10 0.722 → up to 0.787) | Considering for GPU use; no gain in live agent runs on a laptop CPU, where it doubled the time |
| Path headers (file path as a line of context) in the BM25+ index or reranker windows | +0.001 and −0.006 Recall@10 | Not recommended |
| Ten results per search instead of five | No more answered questions, about 80% more cost | Not recommended as tested; this version's 8-document limit is a different design |
| Longer excerpts (6,000 characters instead of 3,000), with a larger context and token budget | More questions answered on tuning and fresh questions, at about 60–70% more cost | Considering; this version reads more of a document on request instead, and the two could be compared or combined |
| Keeping round-one evidence whole in round two | Part of a gain from 25 to 31 answered (of 70) on `hira/dev` | This version keeps supported passages across rounds by a different mechanism |
| 2,048 output tokens per call instead of 1,024 | Fixed an answer cut off at exactly 1,024 tokens | This version gives the critic 2,048 but the answer step 1,024; worth checking for cut-off answers |
| Releasing unused spending reservations after each reply | Fixed a session cap being reached at $0.86 of real spend on a $2 cap | This version also reserves worst-case cost per call; whether the unused part is released should be checked |
| A 200-second time limit instead of 120 | A ceiling only; without the reranker the slowest question took 182 s | Not needed here: this version's runs finished within 40 s |

## Part 1: Retrieval experiments

All retrieval experiments search all **511,958 documents** for the **300 development
questions**; 18 have no reference documents (`info_not_found`, `high_level`), leaving **282
scored questions**. Ranges are 95% paired bootstrap confidence intervals over questions.

### Results by question type

The current hybrid (BM25+ weight 0.7, BGE-base 0.3) broken down by the benchmark's question
types, next to each method alone. They show where each method is strong and where combining
them helps. Types with 12–24 questions are small; treat their differences as rough.

**Recall@10** (share of reference documents in the top 10):

| Question type | Questions | BM25+ | BGE-base | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Basic | 105 | 0.771 | 0.543 | **0.819** |
| Semantic (reworded) | 75 | **0.413** | 0.093 | 0.400 |
| Intra-document reasoning | 24 | **1.000** | 0.792 | **1.000** |
| Project related | 24 | 0.676 | 0.603 | **0.716** |
| Constrained | 18 | 0.917 | 0.861 | **0.972** |
| Completeness | 12 | 0.497 | 0.399 | **0.567** |
| Conflicting info | 12 | 0.792 | 0.792 | **0.917** |
| Miscellaneous | 12 | **0.917** | 0.833 | **0.917** |
| **All scored** | 282 | 0.692 | 0.487 | **0.722** |

**nDCG@10** (also rewards ranking reference documents near the top):

| Question type | BM25+ | BGE-base | Hybrid |
| --- | ---: | ---: | ---: |
| Basic | **0.674** | 0.442 | 0.660 |
| Semantic (reworded) | **0.281** | 0.058 | 0.253 |
| Intra-document reasoning | 0.911 | 0.710 | **0.958** |
| Project related | 0.634 | 0.567 | **0.676** |
| Constrained | 0.820 | 0.726 | **0.875** |
| Completeness | 0.516 | 0.414 | **0.572** |
| Conflicting info | 0.810 | 0.730 | **0.858** |
| Miscellaneous | **0.886** | 0.751 | 0.844 |
| **All scored** | **0.604** | 0.416 | **0.604** |

**Recall@5 and Recall@20** (BM25+ / BGE-base / hybrid):

| Question type | Recall@5 | Recall@20 |
| --- | --- | --- |
| Basic | 0.743 / 0.514 / 0.733 | 0.800 / 0.619 / 0.829 |
| Semantic (reworded) | 0.307 / 0.067 / 0.320 | 0.480 / 0.173 / 0.480 |
| Intra-document reasoning | 0.958 / 0.750 / 0.958 | 1.000 / 0.875 / 1.000 |
| Project related | 0.523 / 0.462 / 0.553 | 0.816 / 0.715 / 0.822 |
| Constrained | 0.806 / 0.861 / 0.944 | 0.972 / 0.944 / 0.972 |
| Completeness | 0.393 / 0.320 / 0.412 | 0.622 / 0.436 / 0.636 |
| Conflicting info | 0.792 / 0.750 / 0.875 | 0.917 / 0.833 / 0.958 |
| Miscellaneous | 0.917 / 0.833 / 0.917 | 0.917 / 0.833 / 0.917 |
| **All scored** | 0.625 / 0.448 / 0.641 | 0.747 / 0.562 / 0.760 |

- **BM25+ matches or beats BGE-base on Recall@10 in every type,** most of all on semantic
  (reworded) questions, where BGE-base finds almost nothing (0.093). This corpus is full of
  project codenames and IDs, which exact word matching handles well.
- **The hybrid's Recall@10 gains come from five types:** basic, project related,
  constrained, completeness and conflicting info. It ties on intra-document reasoning and
  miscellaneous, and is slightly below BM25+ on semantic questions.
- **Ranking (nDCG@10) is unchanged overall (0.604):** the hybrid ranks better on five types
  but slightly worse on basic, semantic and miscellaneous.
- **Semantic questions are the weakest type for every method,** and they remain the largest
  group of agent questions whose documents are never found (Part 2).

### Rerankers on whole-document windows

**Result: tested on development only; a clear retrieval gain, but no benefit in live agent
runs on a laptop CPU (Part 2). Considering for GPU use; not part of this version.**

The October rerankers saw one 3,000-character slice per document and made results worse
(see [Earlier experiments: fusion and reranking](FULL_CORPUS.md#hybrid-search-and-reranking)).
Three of the fixes suggested there were tested:

- **Whole-document windows (MaxP):** each candidate is scored on windows of 2,000 characters
  across its whole document, each starting with the title, and keeps its best window's score.
- **Blending:** the reranker's rank is combined with the hybrid's rank by weighted RRF instead
  of replacing it; the weight is the reranker's share.
- **Cross-validated weight:** the blend weight (1.0 to 0.6) is chosen by 5-fold
  cross-validation on development, so the reported score is not tuned on the same questions.
  Calibration questions were not used.

Three settings trade accuracy for speed. They differ in how many fused candidates are
reranked and how many windows of each document are read:

| Setting | Candidates reranked | Windows per document | Windows read per question | In plain terms |
| --- | --- | ---: | ---: | --- |
| Fast | 20: the hybrid's top 20 | up to 4 | about 74 | Re-sorts only the top 20, reading at most 4 chunks of each |
| Current | 50: the fused top 50, in practice BM25+'s top 50 | up to 8 | about 278 | Re-sorts the top 50, reading up to 8 chunks each |
| Deep | 100: BM25+'s 50 plus the dense model's 50 | up to 8 | about 500 | Also considers the dense model's own finds, so documents BM25+ never ranked can reach the top |

**`bge-reranker-v2-m3`** (568M parameters; cross-validated blend), with time per search
including the hybrid search itself:

| Setting | Recall@5 | No reference doc in top 5 (of 282) | Recall@10 | nDCG@10 | Recall@10 change vs hybrid | Per search, Colab GPU | Per search, laptop CPU |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| Hybrid, no reranker | 0.641 | 83 | 0.722 | 0.604 | — | ~1.0 s (measured) | 14.3 s (measured) |
| Fast | 0.702 | 67 | 0.745 | 0.679 | +0.023 (+0.006 to +0.043) | ~3.0 s (est.) | 82.9 s (measured) |
| Current | 0.721 | 62 | 0.773 | 0.687 | +0.051 (+0.027 to +0.079) | ~8.3 s (est.) | ~280 s (est.) |
| Deep | 0.741 | 56 | 0.787 | 0.703 | +0.066 (+0.034 to +0.100) | ~14.2 s (est.) | ~490 s (est.) |

**`bge-reranker-base`** (278M parameters; cross-validated blend) on the same questions:

| Setting | Recall@5 | No reference doc in top 5 (of 282) | Recall@10 | nDCG@10 | Recall@10 change vs hybrid | Reranking per search, Colab GPU |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Hybrid, no reranker | 0.641 | 83 | 0.722 | 0.604 | — | — |
| Fast | 0.691 | 70 | 0.736 | 0.659 | +0.015 (−0.007 to +0.037) | ~0.26 s (est.) |
| Current | 0.696 | 69 | 0.763 | 0.665 | +0.041 (+0.014 to +0.070) | ~0.9 s (est.) |
| Deep | 0.707 | 66 | 0.760 | 0.668 | +0.039 (+0.008 to +0.071) | ~1.6 s (est.) |

GPU times are estimated from the windows each setting reads and the measured seconds per
window on Colab GPUs; laptop times are on a 16 GB laptop CPU. With a fixed blend weight of 0.7
instead of a cross-validated one, base at the fast setting scored Recall@10 0.744 and nDCG@10
0.661; on a laptop CPU it added about 20–30 seconds per search.

- **Both rerankers improve ranking clearly** (nDCG@10 +0.05 to +0.10, all intervals above
  zero). Recall@10 gains are clear except for base at the fast setting.
- **v2-m3 is more accurate but much slower:** on a laptop CPU even the fast setting took about
  83 seconds per search.
- **The deep setting helps v2-m3 most,** but reads roughly twice as many windows as the
  current setting.

**Blending vs the reranker's order alone** (Recall@10 / nDCG@10):

| Reranker and setting | Hybrid | Reranker alone (weight 1.0) | Blend 0.7 | Cross-validated |
| --- | --- | --- | --- | --- |
| base, fast | 0.722 / 0.604 | 0.734 / 0.644 | 0.744 / 0.661 | 0.736 / 0.659 |
| base, current | 0.722 / 0.604 | 0.720 / 0.633 | 0.758 / 0.665 | 0.763 / 0.665 |
| v2-m3, fast | 0.722 / 0.604 | 0.745 / 0.674 | 0.745 / 0.671 | 0.745 / 0.679 |
| v2-m3, current | 0.722 / 0.604 | 0.756 / 0.678 | 0.773 / 0.687 | 0.773 / 0.687 |

- **Blending matters most for the weaker reranker:** base's order alone at 50 candidates
  matches the hybrid (0.720), while blended it reaches 0.758–0.763.
- **v2-m3 needs less help:** at 20 candidates its order alone equals every blend (0.745).
- **The blend never did clearly worse than the reranker alone.** Cross-validation chose
  weights of 0.6–0.8 for base and 0.7–0.9 for v2-m3.

### Path headers

**Result: no useful gain. Not recommended.**

A document's file path carries context its title often lacks; a Slack thread's title is only
its channel name. The test turned each path into a one-line header, for example
`slack/devex/1759999992-opaque-routing-shim-sdk-discuss.json` became
`Slack · #devex · 2025-10-09 · opaque routing shim sdk discuss` (platform, folders, date and the
file name's words), and put it in two places without re-embedding the dense index: in the
BM25+ index (header + title + content), which changes which documents become candidates, and
at the start of every reranker window, which changes how candidates are scored. All four
combinations used `bge-reranker-base` at the current setting (50 candidates, 8 windows,
cross-validated blend).

| Setup | BM25+ index | Reranker windows | Recall@10 | nDCG@10 | Recall@10 change vs hybrid |
| --- | --- | --- | ---: | ---: | --- |
| Hybrid, no reranker | plain | — | 0.722 | 0.604 | — |
| Hybrid, no reranker | **header** | — | 0.722 | 0.607 | +0.001 (+0.000 to +0.002; 2 better, 0 worse) |
| A | plain | title only | 0.763 | 0.665 | +0.041 (+0.014 to +0.070) |
| B | plain | **header** + title | 0.757 | 0.670 | +0.035 (+0.010 to +0.063) |
| C | **header** | title only | **0.770** | 0.667 | +0.049 (+0.020 to +0.079) |
| D | **header** | **header** + title | 0.761 | **0.672** | +0.040 (+0.013 to +0.069) |

- **BM25+ header:** +0.001 Recall@10 in the hybrid; +0.008 with the reranker (C vs A: three
  questions where the header brought the right document into the candidate set).
- **Reranker header:** −0.006 Recall@10 (B vs A) and +0.005 nDCG@10, within noise. Most
  candidates for a question share a channel or project, so their headers look alike.
- **Slack questions,** which should benefit most, did not improve.
- The header's dates may still be useful to the critic for choosing between conflicting
  announcements (see [considering](#considering-and-further-testing)).

## Part 2: Live agent experiments on `hira/dev`

These runs sent development questions through `hira/dev`'s web runtime in live mode
(`claude-haiku-4-5`, temperature 0), on a 16 GB laptop CPU, recording each question's time,
searches, how it stopped, its cost and where its reference documents were lost. The agent was
allowed 300 seconds so every question ran to its natural end. **The agent here is
`hira/dev`'s: the 25 October critic, not this version's checklist critic.** All runs are from
16 November 2026.

### What each kind of test measures

| | Recall tests (Part 1) | Live agent runs |
| --- | --- | --- |
| Question asked | Are the reference documents in the top 5/10 results? | Did the critic accept the evidence, so an answer was produced? |
| Questions | 300 development (282 scored) | 20, 70 and 25 development questions |
| Searches | The original question only, once | The original question, then up to 4 follow-up searches |
| Judge | A fixed answer key, the same every run | A language model whose verdicts vary between runs |

Identical runs differed by about 3–4 answered questions per 70, even at temperature 0, so
only larger changes, or changes confirmed on other questions, count.

### Reranker and ten results in the agent

**Result: neither answered more questions; the reranker doubled the time.** Four runs on the
same 20 questions (the 10 of the October 21 agent record plus 10 sampled with seed 5980).
"Top 10" also raised the context limit to 36,000 characters and the token budget to 32,000.

| Setup | Answered | Errors | Median time | Slowest | Over 200 s | Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **No reranker, top 5** | **8** | 1 | **67 s** | 93 s | 0 | **$0.33** |
| No reranker, top 10 | 6 | 2 | 73 s | 147 s | 0 | $0.59 |
| `bge-reranker-base` (fast, blend 0.7), top 5 | 5 | 0 | 126 s | 206 s | 1 | $0.36 |
| `bge-reranker-base`, top 10 | 6 | 0 | 138 s | 243 s | 4 | $0.61 |

- **Searching took 85–90% of the time:** about 12–16 seconds per search without the reranker
  and 30–48 seconds with it.
- **The answered counts are within run-to-run variation;** time and cost differences are not.
- **Why the reranker's retrieval gain did not show:** it raises Recall@5 by about 0.05, so on
  20 questions it changes about one; most failures happened after the right document was
  already found (next section); and the reranker chooses documents by whole-document windows,
  while the critic still reads one excerpt of each.

### Where unanswered questions failed

A diagnosis run on 70 development questions (the 20 above plus 50 sampled with seed 5980; no
reranker, top 5) checked, for each question, whether its reference documents appeared in any
search, in what the critic read, and in the answer's citations.

| Outcome | Questions | Share |
| --- | ---: | ---: |
| Answered | 25 | 36% |
| Correctly unanswered (`info_not_found`: no answer exists) | 4 | 6% |
| **Shown a reference document, judged insufficient** | **28** | **40%** |
| Reference never found by any search | 9 | 13% |
| Found but not shown to the critic | 2 | 3% |
| Error | 2 | 3% |

Of the 39 failures, 28 (72%) came after the critic had seen a reference document. Sorted by
hand from the critic's own reasons:

| | Reason | Count | Questions | What the critic said (examples) |
| --- | --- | ---: | --- | --- |
| A | The excerpt cut off the answer | 8 | 0431, 0391, 0226, 0357, 0430, 0215, 0133, 0152 | "excerpt cuts off mid-checklist"; "the code snippet is truncated" |
| B | A specific value missing; unclear whether it is elsewhere in the document | 8 | 0384, 0126, 0304, 0420, 0339, 0105, 0330, 0066 | "specific reduction amounts"; "explicit drift percentage threshold"; "the trace job ID" |
| C | Demands explicit confirmation or authority the question may not need | 7 | 0044, 0309, 0462, 0332, 0373, 0371, 0417 | "explicit written approval from Acme"; "only a hypothesis, not a confirmed root cause"; "a consolidated, explicit list" |
| D | Conflicting sources; refused to choose | 4 | 0176, 0149, 0342, 0354 | 0176 and 0149 are near-duplicates: three different dates for the same H200 announcement |
| E | Round 2 dropped the reference it had in round 1 | 1 | 0301 | Saw the thresholds in round 1; round 2 no longer had them |

**The excerpt test.** An offline check, over every development question, of whether the
reference answer's specific facts (numbers, dates, IDs, names) appear in the excerpt the
critic would see, compared with the whole reference documents. It matches exact tokens, so it
undercounts paraphrased answers. Of the 269 questions whose documents contain such facts:

| Excerpt size | Answer facts visible | Questions with every fact visible |
| ---: | ---: | ---: |
| 1,500 | 70% | 114 (42%) |
| **3,000** (this version's window size) | 85% | **169 (63%)** |
| 4,500 | 93% | 215 (80%) |
| 6,000 | 96% | 235 (87%) |
| 9,000 | 99% | 258 (96%) |
| 12,000 | 99% | 265 (99%) |
| Whole document | 100% | 269 (100%) |

Project-related questions were worst (4% complete at 3,000 characters, 79% at 9,000): they
need facts spread across several long documents. Reference documents have a median length of
6,072 characters; 21% fit in one 3,000-character excerpt. This version's "read more" feature
addresses the same gap by reading more of a matching document on request.

### Fixes and longer excerpts tested on `hira/dev`

**How rounds work.** The agent runs at most two rounds; a round is not a search:

| Round | What happens | Searches |
| --- | --- | --- |
| 1 | Search the question as asked; the critic judges the 5 results | 1 |
| 2 | If insufficient: run the critic's suggested follow-up queries, then the critic judges again | 2–4, depending on how many queries the critic suggests |

A `round_limit` stop means round 2 ended and the critic still judged the evidence
insufficient. More rounds would rarely help on `hira/dev`: round 2 rescued only 3 of 38
questions in its latest run.

Changes tested, in order:

- **Round two kept round-one evidence whole.** `hira/dev` had cut every round-two excerpt to
  1,500 characters, losing facts the critic had read in round one.
- **2,048 output tokens per call instead of 1,024.** One answer had stopped at exactly 1,024
  tokens and was rejected.
- **Unused spending reservations released after each reply.** Each call reserved its
  worst-case cost against the session cap; a run hit a $2 cap with $0.86 actually spent.
- **6,000-character excerpts** cut from the whole document around the matched passage (instead
  of a 3,000-character slice or a ~500-token BGE passage), with a 48,000-character context and
  a 32,000-token budget so they fit.

**Results across these changes, same 70 questions:**

| | Before (3,000-character excerpts) | + Round-two and output fixes | + 6,000-character excerpts |
| --- | ---: | ---: | ---: |
| **Answered** | 25 (36%) | 31 (44%) | **34 (49%)** |
| Correctly unanswered (no answer exists) | 4 | 4 | 4 |
| Answered in round 1 (one search) | 20 | 24 | 31 |
| Reference shown, judged insufficient | 28 | 24 | 20 |
| Reference never found | 9 | 9 | 9 |
| Errors | 2 | 1 | 1 |
| Median time per question | 69 s | — | 56 s |
| Cost for 70 questions | $1.16 | ~$1.1 | $1.81 |

The middle run was split across two sessions, so it has no single median time and its cost is
approximate. The 70 questions were used to choose these changes, so the longer excerpts were
also tested on 25 development questions not used for tuning:

| 25 fresh questions | 3,000-character excerpts | 6,000-character excerpts |
| --- | ---: | ---: |
| **Answered** | 6 (24%) | **9 (36%)** |
| Reference shown, judged insufficient | 11 | 8 |
| Never found | 8 | 8 |
| Median time | 80 s | 74 s |
| Cost per question | $0.017 | $0.029 |

- **Before → after, on the 70:** 13 gained and 4 lost (about 5% likely by chance), so the
  combined improvement is probably real for `hira/dev`'s agent.
- **Longer excerpts alone:** +3 on each set; consistent, but each step is within noise on its
  own. Cost rose by about 60–70% per question.
- **Side effect:** longer excerpts show more of each document, so the critic spotted more
  disagreements; refusals over conflicting sources rose from about 4 to 7 of 70.

## Part 3: Comparison with this version (25 fresh questions)

The same 25 fresh questions were run on this version of the agent (commit `27365b4`, its
default settings: up to 8 documents, 3,000-character windows with "read more", a 16,000-token
budget), with the same timing script and data. Each answered question was then compared with
the benchmark's reference answer by reading the first 300–400 characters of both. This is an
informal check, not the project's grader, and 25 questions is a small sample.

| 25 fresh questions | `hira/dev` (6,000-character excerpts) | This version |
| --- | ---: | ---: |
| Answered | 9 | 17 |
| **Correct (including mostly correct)** | **7** | **7** |
| Partly correct | 0 | 2 |
| **Wrong** | **2** | **8** |
| Not answered (insufficient, clarification, budget) | 16 | 8 (7 asked to clarify) |
| **Correct share of answers given** | **78%** | **41%** |
| Median time per question | 74 s | 29 s |
| Searches per question | median 4 | 1 for every question |
| Cost | $0.73 | $0.33 |

- **The same number of correct answers, different numbers of wrong ones.** This version's
  critic answered more readily: it answered 2 questions correctly (0406, 0328) that
  `hira/dev`'s critic refused, plus 2 partly, but also answered 7 questions without being shown
  a reference document (5 never found, 2 found but not shown), all wrongly.
- **Wrong answers came from retrieval misses in both versions.** Every answer that cited a
  reference document was correct or partly correct; every wrong answer cited a similar but
  different document (another incident, customer or design, with different numbers). Two
  questions (0143, 0087) were answered wrongly by both.
- **Longer excerpts helped once where 3,000-character windows did not:** in qst_0188 both
  versions had the reference document; only the 6,000-character excerpt contained the dates.
- **This version was about 2.5× faster and half the cost,** because it never ran a second
  search.

## Considering and further testing

| Idea | Why | What to test |
| --- | --- | --- |
| Grade answers with the project's grader | "Answered" is not "correct"; Part 3 shows the difference | Correct and wrong answers per version on the same questions |
| Longer excerpts in this version | 6,000 characters showed more answers on `hira/dev` and fixed qst_0188 | This version with longer windows, alone and alongside "read more" |
| A citation requirement | Every wrong answer in Part 3 cited a non-reference document | Whether requiring the answer to cite a document the critic judged a matching source reduces wrong answers without losing correct ones |
| Document dates in the evidence | Conflicting announcements (e.g. three H200 dates) can't be resolved without dates; the path headers can supply them | Whether the critic picks the latest source correctly |
| Better follow-up queries | Reworded (semantic) questions are most often never found | Queries built from names, IDs and terms in the first results |
| Reranker on a GPU | Clear retrieval gain, too slow on a laptop CPU | Live agent runs on a GPU |
| Check output and spending limits | Both caused failures on `hira/dev` | Whether 1,024-token answers are cut off here, and whether reservations are released |

## Caveats

- **"Answered" is not "correct"** except in Part 3, which was an informal check of 26 answers.
- **The 70 questions were used to choose `hira/dev`'s changes,** so their numbers are
  optimistic; the 25 fresh questions are the fairer check.
- **Noise:** identical runs differed by about 3–4 answered questions per 70.
- **Hardware and model:** all live timings are from a 16 GB laptop CPU with
  `claude-haiku-4-5`.
- **Retrieval metrics and agent results can disagree,** as the reranker showed.

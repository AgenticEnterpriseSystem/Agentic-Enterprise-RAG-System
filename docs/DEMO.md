# Record the project demo

Use this guide to make a **90-second screen recording** of one real app request,
then prepare a longer final presentation. The short recording should show the
product working, with readable text and a clear explanation.

The video and public watch link are **not published yet**. Add the real link to
the [project page](../README.md) after uploading and checking it. Do not use an
example URL or call a prepared script a completed demo.

## Before recording

1. Use the reviewed running version and note its commit and recording date in
   the private recording notes. Check [Current status](STATUS.md) for known limits.
2. Use a real account with reviewed document access. Observability also needs its
   own explicit admin permission. A username or role label does not grant either.
3. Prepare exactly one approved live question and its spending limit. The selected
   question below is derived from development question `qst_0001`; it is not an
   exact benchmark rerun. Use its approved wording. Its outcome has not been
   guaranteed. This guide authorizes no model calls, extra attempts or automatic
   retries.
4. Use a clean browser window, turn off notifications and hide other tabs. Enter
   the password off camera; capture the **Sign in** action afterward. Never show
   an environment file, password manager, API key, cookie or developer network panel.
5. Keep a private uncut recording and the request's real outcome. A request can
   take longer than 90 seconds. Shorten only the wait in the public edit and label
   it **“Waiting shortened — actual request: [measured duration]”**. Do not replace
   the result, join different requests into one, or present a replay as a new run.

The demo should use the actual **Adaptive agent** interface. Do not promise two
critic rounds, a reformulation or `read_more` before seeing the trace: those are
possible steps, not steps every request must take.

Selected question:

> What are the default max_file_size and max_total_request_size limits for multipart uploads on the OpenAI-compatible API endpoints?

The intended source states **10 MiB per file and 50 MiB per request**. Check the
new run's actual answer and cited passage before using that claim in narration.

## 90-second shot plan

The times below are positions in the edited video, not a latency claim.

| Time | Show | Action / point |
| --- | --- | --- |
| 0–12 s | Sign-in screen, then signed-in Search | Password entry stays off camera. Explain explicit account/document permissions. |
| 12–25 s | **Adaptive agent**, exact question, **Ask question** | Submit once. Explain keyword plus semantic search. |
| 25–38 s | Real loading state, with a labeled wait cut if needed | Explain the critic and its possible follow-up actions. |
| 38–55 s | Actual answer, citation and matching source card | Read the central result; open a real citation and show the supporting passage. |
| 55–65 s | Recorded search steps and **Evidence context** | Show what actually ran, the selected evidence and any exclusions. |
| 65–81 s | **Observability**, same request | Match its Request ID to Search; show stage timings and reported usage. |
| 81–90 s | Sign out; final title card | Clarify that evaluation is separate and answer accuracy is not guaranteed. |

If the real outcome is a clarification, error or missing citation, show it
honestly and adjust the narration. Do not label a clarification as a generated
final answer. A failed take is evidence to diagnose; it does not authorize another
paid attempt. A successful end-to-end answer clip requires an actual generated
answer and working citation/source display.

### Narration: 170 words

Read naturally and pause while pointing at the screen. The wait-shortening
sentence applies only when the wait really was edited. If another question is
approved, replace “upload limits” with its actual topic.

> This is our search app for a fictional company’s documents. I sign in with a real account. An operator grants documents explicitly; a role name does not grant access.
>
> I choose Adaptive agent and ask about upload limits. Hybrid search combines keyword and semantic matches within my permitted documents.
>
> The critic checks for missing facts. It can read more or reformulate the search, for up to two rounds. The waiting time here is shortened.
>
> Here is this request’s actual outcome. I open a citation to check its source, then inspect the recorded steps. Missing evidence can produce a clarification instead of a definitive answer.
>
> The Evidence context panel shows selected evidence and exclusions.
>
> Observability needs separate admin permission. It links this request to stage timings, usage and failures, without question or document contents.
>
> Alex’s evaluator grades saved answers separately. It does not run automatically when I ask a question, and its grades still need human review.
>
> I sign out. This shows one working path; it does not guarantee correct answers.

For a clarification without a clickable answer citation, replace the relevant
sentence with: **“Here is the actual clarification and the evidence returned with
it. I inspect the recorded steps to see what the system could establish.”** For
an error, explain the displayed error and diagnose it separately; do not retain
narration claiming an answer or a working path.

## Longer final demo: about four minutes

Reuse the approved recorded request when appropriate. Label it as recorded; a
presentation does not require another paid query.

| Time | Topic | What to demonstrate |
| --- | --- | --- |
| 0:00–0:30 | Problem and users | Search across a fictional company's documents; show the main interface. |
| 0:30–1:00 | Login and document access | Explain account → explicit grants → current request permissions → database checks. The app does not infer access from role names or folder hierarchy. |
| 1:00–2:15 | One real request | Show the exact question, hybrid search, actual critic decisions and any actual follow-up, then the answer or clarification. Label shortened waits. |
| 2:15–2:50 | Inspect the result | Open citations/source passages. Show the recorded steps and evidence selection. Explain that the critic and generator share the same selected evidence, bounded to at most eight documents and other budgets. |
| 2:50–3:20 | Diagnostics | Open the separate Observability page for the same Request ID. Explain stage timings, usage and explicit admin access. Unknown usage stays unavailable. |
| 3:20–3:45 | Evaluate saved answers | Explain Alex's separate grading and cited-evidence audit. If showing existing outputs, label them as a saved pilot: one question across three methods, with human review still pending. No live grader runs inside Search. |
| 3:45–4:00 | Limits and close | A cited source can still be the wrong source. Human validation and controlled comparisons remain open. Sign out and point to the project/video links. |

For a three-minute slot, shorten the explanation and use the 90-second clip.
For five minutes, spend the extra time inspecting one source and a critic decision;
do not add unapproved requests. See [How the system works](SYSTEM.md),
[Answer review](ANSWER_REVIEW.md) and [Current status](STATUS.md) for accurate details.

## Publish and check the link

Once the destination is approved and the real recording is ready:

- Review every frame and the audio for credentials, unrelated personal content
  and unintended document disclosure. Keep only the intended fictional demo material.
- Upload to YouTube as **unlisted**, or use Vimeo settings that allow anyone with
  the link to watch without a login or password. Check the resulting accessibility;
  a setting's name alone is not proof.
- Give the video a clear title and date. Disclose shortened waiting time in the
  video and description. Keep subtitles and UI text readable.
- Open the final watch URL in a signed-out/private browser. Confirm it plays from
  start to finish and has the intended approximately 90-second duration.
- Add a clickable **Watch the 90-second demo** link to the project page. An embed
  is optional; if used, test it on the published page and keep the watch link too.
- Test the link from the actual public project page, then record the real URL,
  duration and publication check. Until these pass, keep delivery marked pending.

Uploading the clip does not make the local app publicly hosted. Viewers should
not need access to the app, private database or repository credentials to watch.

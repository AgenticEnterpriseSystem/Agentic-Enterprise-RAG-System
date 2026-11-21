"""Content / answer-quality red-team suite (PAID — live mode only).

These reproduce or probe the following documented limitations:
prompt-injection resistance, valid-citation-backed false claims, and the
end-to-end access boundary. Each one calls POST /api/query, which costs money
in live mode, so the whole module is OPT-IN:

    RT_ALLOW_PAID=1 RT_MAX_COST_USD=0.50 \
    RT_PASS_RT_ALPHA=... RT_PASS_RT_BRAVO=... RT_PASS_RT_NOGRANTS=... \
    .venv/Scripts/python -m pytest scripts/red_team/test_content.py -v -rs

Without RT_ALLOW_PAID=1 every test here SKIPS, so a plain `pytest scripts/red_team`
run stays free (only the availability suite runs).

SEMANTICS (note — opposite of the availability suite):
  These test DEFENSES the system is supposed to have, so:
    PASSED   the defense held
    FAILED   a vulnerability was found
    SKIPPED  missing password, budget exhausted, or a `review` case recorded
             for human judgement
  (The availability suite is the reverse: PASSED means a known attack
  reproduced. In both suites, a FAIL is the result to investigate.)

Every run records the full answer text to the evidence bundle for the report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import content as content_engine
from conftest import password_for

pytestmark = pytest.mark.redteam

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE.parents[1] / "red_team" / "cases" / "content_cases.jsonl"
ACCOUNT_LOGIN = {"rt-alpha": "rt-alpha", "rt-bravo": "rt-bravo", "rt-nogrants": "rt-nogrants"}

# Categories where a human must look at the answer never hard-fail: any
# category whose name ends in "_review" (citation_review, prompt_extract_review).
def _is_review(category: str) -> bool:
    return category.endswith("_review")


def _load_cases():
    if not CASES_FILE.exists():
        return []
    cases = []
    for line in CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


CASES = _load_cases()


@pytest.fixture(scope="session")
def paid_allowed():
    if os.environ.get("RT_ALLOW_PAID") != "1":
        pytest.skip("Paid content tests are opt-in. Set RT_ALLOW_PAID=1 (live mode; costs money).")
    return True


@pytest.fixture(scope="session")
def content_budget():
    return {"spent_usd": 0.0, "cap_usd": content_engine.budget_cap_usd(), "unknown_cost_calls": 0}


@pytest.fixture(scope="session")
def content_sessions():
    return {}


def _session_for(account, client, content_sessions):
    login = ACCOUNT_LOGIN.get(account, account)
    if login in content_sessions:
        return content_sessions[login]
    password = password_for(login)
    if not password:
        pytest.skip(f"No password for '{login}'. Set RT_PASS_{login.upper().replace('-', '_')}=...")
    try:
        session = content_engine.Session(client, login, password)
    except RuntimeError as exc:
        pytest.skip(f"Could not establish a session for '{login}': {exc}")
    content_sessions[login] = session
    return session


def _repeats(case) -> int:
    """How many times to run a case. Model output varies, so a probe repeats
    rather than relying on a single sample. Per-case `repeats` wins, else
    RT_REPEATS, else 1."""
    try:
        default = int(os.environ.get("RT_REPEATS", "1"))
    except ValueError:
        default = 1
    return max(1, int(case.get("repeats", default)))


@pytest.mark.skipif(not CASES, reason="No content cases found.")
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_content_case(case, client, recorder, server_ready, paid_allowed,
                      content_budget, content_sessions):
    if content_budget["spent_usd"] >= content_budget["cap_usd"]:
        pytest.skip(f"Budget cap ${content_budget['cap_usd']:.2f} reached "
                    f"(spent ~${content_budget['spent_usd']:.3f}).")

    session = _session_for(case["account"], client, content_sessions)
    reps = _repeats(case)
    outcomes = []
    for i in range(1, reps + 1):
        if content_budget["spent_usd"] >= content_budget["cap_usd"]:
            break
        result = session.query(case["question"], mode=case.get("mode", "agentic"),
                               source_types=case.get("source_types"))
        cost = result.get("cost_usd")
        if isinstance(cost, (int, float)):
            content_budget["spent_usd"] += cost
        else:
            content_budget["unknown_cost_calls"] += 1
        scored = content_engine.score_case(case, result)
        scored.update(test_id=case["id"], category=case["category"], repeat=i, of=reps)
        recorder.add(scored)
        outcomes.append(scored)

    if not outcomes:
        pytest.skip(f"{case['id']}: budget cap reached before any run.")

    n = len(outcomes)
    succeeded = [o for o in outcomes if o["outcome"] == "attack_succeeded"]
    held = [o for o in outcomes if o["outcome"] == "defense_held"]
    setup = [o for o in outcomes if o["outcome"] == "setup_failure"]

    # citation_review and friends: never hard-fail unless a concrete marker rule
    # fired; record all runs for a human.
    if _is_review(case["category"]):
        if succeeded:
            pytest.fail(f"{case['id']}: {succeeded[0]['reasoning']}")
        pytest.skip(f"{case['id']} recorded for human review ({n} run(s)): {outcomes[0]['reasoning']}")

    if succeeded:  # worst-of-n: one break is a finding
        pytest.fail(f"VULNERABILITY — {case['id']}: broke in {len(succeeded)}/{n} run(s). "
                    f"{succeeded[0]['reasoning']}")
    if held and len(held) == n:
        print(f"\n[DEFENSE HELD {n}/{n}] {case['id']}: {held[0]['reasoning']}")
        return
    if setup and not held:
        pytest.skip(f"{case['id']} setup failed: {setup[0]['reasoning']}")
    pytest.skip(f"{case['id']} inconclusive: {len(held)}/{n} runs held; "
                f"{outcomes[-1]['reasoning']}")


# Permitted Set A questions for the F1 drain (rt-alpha is granted these docs).
F1_QUESTIONS = [
    "What does our documentation say about the max_total_request_size default and limit?",
    "Summarize the edge-sw-7 device configuration, including any cooldown_ms setting.",
    "What is the micro=tool:rollout plan described in our Linear records?",
    "What figure around 1,000,000 appears in the legal SOP, and in what context?",
    "What does the Confluence document say about an 18 months timeline or retention?",
    "Describe the GitHub repository settings that mention cooldown_ms.",
]


def test_f1_spending_cap_drains_and_blocks_everyone(client, rt_config, recorder,
                                                    server_ready, paid_allowed,
                                                    content_budget, content_sessions):
    """F1 (known vuln — PASS means it reproduced, like the availability suite).

    Ask one low-privileged account permitted questions until the process-wide
    cap trips (429 'spending_limit'), then show an UNRELATED account is also
    blocked. This DELIBERATELY exhausts the server budget: generation stays
    disabled for everyone until the app is restarted, so it needs its own extra
    opt-in and the app should be restarted afterwards.
    """
    if os.environ.get("RT_ALLOW_BUDGET_DRAIN") != "1":
        pytest.skip("F1 drains the server budget (disables generation until the app restarts). "
                    "Set RT_ALLOW_BUDGET_DRAIN=1 to run it, and restart the app afterwards.")

    attacker = _session_for("rt-alpha", client, content_sessions)
    try:
        max_q = int(os.environ.get("RT_F1_MAX_QUESTIONS", "80"))
    except ValueError:
        max_q = 80

    result = content_engine.drain_until_cap(
        attacker, F1_QUESTIONS, max_questions=max_q,
        rt_budget_cap_usd=content_budget["cap_usd"])
    content_budget["spent_usd"] += result["real_cost_usd"]

    # Process-wide effect: is an unrelated account now blocked too?
    bystander_blocked = None
    if result["cap_tripped"]:
        bystander = _session_for("rt-bravo", client, content_sessions)
        rb = bystander.query(F1_QUESTIONS[0], mode="agentic")
        bystander_blocked = (rb.get("http_status") == 429 and rb.get("code") != "busy")
        result["bystander_blocked"] = bystander_blocked
        result["bystander_http"] = rb.get("http_status")
        result["bystander_code"] = rb.get("code")

    outcome = "attack_succeeded" if result["cap_tripped"] else "inconclusive"
    reasoning = (
        f"Generation denied after {result['questions_until_cap']} questions "
        f"(trip={result.get('trip_code')}) with only ${result['real_cost_usd']:.4f} of real "
        f"spend vs the server's cap; unrelated account blocked={bystander_blocked}."
        if result["cap_tripped"] else
        f"Cap not reached within {result['questions_attempted']} questions / our "
        f"${content_budget['cap_usd']:.2f} spend cap.")
    recorder.add({"test_id": "F1", "category": "spending_cap_drain",
                  "outcome": outcome, "reasoning": reasoning, "observations": result})

    if not result["cap_tripped"]:
        pytest.skip("F1 did not reproduce within the question/spend limits "
                    "(raise RT_F1_MAX_QUESTIONS / RT_MAX_COST_USD, or restart the app for a clean count).")
    print(f"\n[REPRODUCED F1] {reasoning}")
    assert bystander_blocked is True, (
        "Cap tripped but the unrelated account was NOT blocked — F1's process-wide "
        f"impact not shown (bystander http={result.get('bystander_http')}, code={result.get('bystander_code')}).")


def test_run_receipt_is_owner_only(client, rt_config, recorder, server_ready, paid_allowed,
                                   content_budget, content_sessions):
    """Access control (PASS = defense HELD). A run receipt (GET /api/runs/{id})
    must be readable only by its owner. rt-alpha mints one run (one cheap live
    query), then rt-bravo, rt-nogrants and a random id must all be refused with
    403, while the owner can still read it. The GET itself calls no model.
    """
    if content_budget["spent_usd"] >= content_budget["cap_usd"]:
        pytest.skip(f"Budget cap ${content_budget['cap_usd']:.2f} reached.")

    alpha = _session_for("rt-alpha", client, content_sessions)
    minted = alpha.query("What does our documentation say about the max_total_request_size default?",
                         mode="agentic")
    cost = minted.get("cost_usd")
    if isinstance(cost, (int, float)):
        content_budget["spent_usd"] += cost
    run_id = minted.get("run_id")
    if not run_id:
        pytest.skip(f"Could not mint a run as rt-alpha (http={minted.get('http_status')}, "
                    f"code={minted.get('code')}).")

    import uuid as _uuid

    def read_receipt(login, rid):
        sess = _session_for(login, client, content_sessions)
        st, body, _, _, err = client.request(
            "GET", f"/api/runs/{rid}", cookies={content_engine.SESSION_COOKIE: sess.cookie})
        return {"login": login, "run": rid, "status": st,
                "code": content_engine.avail.error_code(body), "error": err}

    probes = {
        "foreign_rt-bravo": read_receipt("rt-bravo", run_id),
        "foreign_rt-nogrants": read_receipt("rt-nogrants", run_id),
        "random_uuid_rt-alpha": read_receipt("rt-alpha", str(_uuid.uuid4())),
        "owner_rt-alpha": read_receipt("rt-alpha", run_id),  # positive control
    }
    # A leak = anyone other than the owner (or a random id) getting 200.
    leaked = [k for k in ("foreign_rt-bravo", "foreign_rt-nogrants", "random_uuid_rt-alpha")
              if probes[k]["status"] == 200]
    owner_ok = probes["owner_rt-alpha"]["status"] == 200

    outcome = "attack_succeeded" if leaked else "defense_held"
    reasoning = (f"Run receipt leaked to {leaked}" if leaked else
                 f"Receipt refused to other accounts and a random id "
                 f"(bravo={probes['foreign_rt-bravo']['status']}, "
                 f"nogrants={probes['foreign_rt-nogrants']['status']}, "
                 f"random={probes['random_uuid_rt-alpha']['status']}); owner read="
                 f"{probes['owner_rt-alpha']['status']}.")
    recorder.add({"test_id": "ACCESS-receipt", "category": "access-control",
                  "outcome": outcome, "reasoning": reasoning,
                  "observations": {"run_id": run_id, "owner_ok": owner_ok, "probes": probes}})

    assert not leaked, reasoning
    if not owner_ok:
        print(f"\n[DEFENSE HELD, note] others refused, but owner got "
              f"{probes['owner_rt-alpha']['status']} not 200 (session/scope changed): {reasoning}")
    else:
        print(f"\n[DEFENSE HELD] run receipts are owner-only: {reasoning}")

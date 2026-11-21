"""Reproducible red-team suite for the application's availability findings.

Run it against a RUNNING app (not the unit tests):

    .venv/Scripts/python -m pytest scripts/red_team -v        # Windows
    python -m pytest scripts/red_team -v                      # venv activated

Each test reproduces one documented finding by attacking the live HTTP API the
same way the browser does. It makes NO /api/query calls, so there is NO paid
model usage. Read the semantics carefully:

    PASSED   the attack reproduced
    SKIPPED  missing account password, or a timing-sensitive non-result
    FAILED   the attack did NOT reproduce (the defense held) -> the code may
             have been fixed; update the finding's status in FINDINGS.

So for this suite, all tests passing means every documented finding still
reproduces. Every run also drops a sanitized evidence bundle under
outputs/red_team/ (no passwords/cookies/tokens on disk).

F5 needs no credentials and runs for anyone. F2 and F3 need test-account
passwords supplied via the environment (they SKIP without them):

    RT_PASS_RT_ALPHA=...   # bystander account (F2-global, F3 session)
    RT_PASS_RT_BRAVO=...   # target account   (F2-targeted lockout)

See red_team/README.md for the full runbook.
"""

from __future__ import annotations

import pytest

import availability as avail  # importable via conftest.py sys.path shim
from conftest import password_for

pytestmark = pytest.mark.redteam


def _assert_reproduced(result: dict) -> None:
    """Map a finding's outcome onto pytest pass/skip/fail with a clear reason."""
    outcome = result["outcome"]
    reasoning = result["reasoning"]
    if outcome == "attack_succeeded":
        print(f"\n[REPRODUCED] {result['test_id']}: {reasoning}")
        return
    if outcome in ("inconclusive", "not_run", "setup_failure"):
        pytest.skip(f"{result['test_id']} {outcome}: {reasoning}")
    if outcome == "defense_held":
        pytest.fail(
            f"{result['test_id']} did NOT reproduce -- the defense held. "
            f"If the code was fixed, update the finding's status. Detail: {reasoning}"
        )
    pytest.fail(f"{result['test_id']} unexpected outcome {outcome!r}: {reasoning}")


def test_server_is_reachable(server_ready):
    """Sanity gate: the app answers, auth is enabled, our Origin is accepted."""
    assert server_ready["outcome"] == "defense_held", server_ready["reasoning"]


# --------------------------------------------------------------------------- #
# Findings. Ordered least-disruptive first: F5 touches no login buckets; F3   #
# logs in once; the F2 lockout tests saturate the rate limiter, so they run   #
# last. (A fresh `pytest` run within ~60s of a previous one may SKIP the      #
# login-based tests while the rate-limit window drains -- see README.)        #
# --------------------------------------------------------------------------- #

def test_f5_auth_worker_can_be_saturated(client, rt_config, recorder, server_ready):
    """F5: an unauthenticated client floods the single auth worker.

    Concurrent junk /api/auth/me checks get 429 'Authentication is busy',
    which is the same worker every real login and every /api/query uses.
    """
    result = recorder.add(avail.test_f5_auth_saturation(client, rt_config["concurrency"]))
    _assert_reproduced(result)


def test_f3_free_reads_starve_the_query_slot(client, rt_config, recorder, server_ready):
    """F3: the single global 'busy' slot is taken by a free run-receipt read.

    Needs a real session (so RT_PASS_<bystander> must be set), then fires
    concurrent free GET /api/runs/{random-uuid} and counts the 429 'busy'.
    """
    login = rt_config["bystander_login"]
    password = password_for(login)
    if not password:
        pytest.skip(
            f"F3 needs a session. Set RT_PASS for '{login}' "
            f"(e.g. RT_PASS_{login.upper().replace('-', '_')}=...)."
        )
    result = recorder.add(
        avail.test_f3_query_slot(client, login, password, rt_config["concurrency"])
    )
    _assert_reproduced(result)


def test_f2_fake_logins_lock_out_an_unrelated_user(client, rt_config, recorder, server_ready):
    """F2 (global): fake logins with many different usernames trip the shared-IP
    bucket, then an UNRELATED real user is refused with 429 rate_limited."""
    login = rt_config["bystander_login"]
    password = password_for(login)
    if not password:
        pytest.skip(
            f"F2-global needs a real bystander. Set RT_PASS for '{login}' "
            f"(e.g. RT_PASS_{login.upper().replace('-', '_')}=...)."
        )
    result = recorder.add(
        avail.test_f2_global_lockout(client, login, password, global_attempts=20)
    )
    _assert_reproduced(result)


def test_f2_wrong_passwords_lock_out_the_real_user(client, rt_config, recorder, server_ready):
    """F2 (targeted): wrong passwords for a known username fill the per-login
    bucket, then the CORRECT password for that user is refused."""
    target = rt_config["target_login"]
    bystander = rt_config["bystander_login"]
    target_password = password_for(target)
    if not target_password:
        pytest.skip(
            f"F2-targeted needs the real password for '{target}'. "
            f"Set RT_PASS_{target.upper().replace('-', '_')}=..."
        )
    result = recorder.add(
        avail.test_f2_targeted_lockout(
            client, target, target_password, bystander, password_for(bystander),
            window_attempts=5,
        )
    )
    _assert_reproduced(result)


def test_observability_endpoints_require_admin(client, rt_config, recorder, server_ready):
    """Access control (note: PASS = defense HELD, the inverse of the findings
    above). A normal authenticated, non-admin user must NOT be able to read the
    admin-only observability endpoints — they should return 403. Free: no
    /api/query, no model call.
    """
    login = rt_config["bystander_login"]  # rt-alpha: a normal, non-admin account
    password = password_for(login)
    if not password:
        pytest.skip(f"Needs a session. Set RT_PASS for '{login}'.")
    status, cookie, _csrf, code = avail.do_login(client, login, password)
    if not cookie:
        pytest.skip(f"Could not log in as '{login}' (status {status}, {code}).")

    endpoints = ("/api/observability/access", "/api/observability/requests")
    observations, leaked = {}, []
    for path in endpoints:
        st, body, _, _, err = client.request("GET", path, cookies={avail.SESSION_COOKIE: cookie})
        observations[path] = {"status": st, "code": avail.error_code(body), "error": err}
        if st == 200:
            leaked.append(path)

    outcome = "attack_succeeded" if leaked else "defense_held"
    reasoning = (f"Non-admin '{login}' READ admin observability: {leaked}" if leaked else
                 f"Non-admin '{login}' refused on all admin observability endpoints ("
                 + ", ".join(f"{p.rsplit('/', 1)[-1]}={observations[p]['status']}" for p in endpoints) + ").")
    recorder.add(avail.result("ACCESS-observability", "access-control", outcome, reasoning, observations))
    assert not leaked, reasoning
    print(f"\n[DEFENSE HELD] observability admin-only: {reasoning}")

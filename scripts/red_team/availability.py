#!/usr/bin/env python3
"""Free, re-runnable availability red-team tests (findings F2, F3, F5).

These tests drive the running app over HTTP exactly as the UI does. They make
NO /api/query calls and therefore incur NO paid model usage: every request is a
failed login (no session, no model), an authentication status check, or a run
receipt read. Secrets (passwords, cookies, CSRF tokens, API keys) are never
written to the evidence files.

Findings exercised:
  F2  Login throttle locks out legitimate users (attempts counted before the
      password is checked; the loopback IP is shared behind Docker).
  F3  One global "busy" query slot with no per-user fairness; the free
      /api/runs/{id} endpoint also takes it.
  F5  Single auth worker (<=5 queued) shared by login and every query check.

Usage:
  # Default: connectivity + origin check only, no attack traffic.
  python3 scripts/red_team/availability.py --base-url http://127.0.0.1:8000

  # Run the free tests and save evidence:
  python3 scripts/red_team/availability.py --run \
      --target-login rt-bravo --bystander-login rt-alpha
  # Passwords (optional, strengthen F2/F3) come from the environment:
  #   RT_PASS_RT_BRAVO=...  RT_PASS_RT_ALPHA=...   (RT_PASS_<LOGIN upper, - and . -> _>)
  # or pass --prompt-passwords to type them (never echoed, never saved).

Outcomes per test: defense_held, attack_succeeded, inconclusive, setup_failure,
not_run. Absent evidence is never scored as success.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import ssl
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
CSRF_HEADER = "X-CSRF-Token"
SESSION_COOKIE = "rag_session"
BOOTSTRAP_COOKIE = "rag_login_csrf"
# Self-imposed safety caps so the harness itself can never run away.
MAX_LOCKOUT_ATTEMPTS = 40
MAX_CONCURRENCY = 16


class Client:
    """Minimal stdlib HTTP client. One connection per request (thread-safe use)."""

    def __init__(self, base_url: str, origin: str | None = None, timeout: float = 10.0):
        self.parsed = urlparse(base_url)
        if self.parsed.scheme not in ("http", "https") or not self.parsed.hostname:
            raise ValueError("base-url must be http(s)://host[:port]")
        self.origin = origin or f"{self.parsed.scheme}://{self.parsed.netloc}"
        self.timeout = timeout

    def _conn(self):
        if self.parsed.scheme == "https":
            port = self.parsed.port or 443
            return HTTPSConnection(self.parsed.hostname, port, timeout=self.timeout,
                                   context=ssl.create_default_context())
        return HTTPConnection(self.parsed.hostname, self.parsed.port or 80, timeout=self.timeout)

    def request(self, method, path, *, json_body=None, cookies=None, csrf=None,
                send_origin=True):
        """Return (status, body_obj_or_none, set_cookies_dict, latency_ms, error_str)."""
        headers = {"Accept": "application/json"}
        if send_origin:
            headers["Origin"] = self.origin
            headers["Sec-Fetch-Site"] = "same-origin"
        if cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())
        if csrf is not None:
            headers[CSRF_HEADER] = csrf
        payload = None
        if json_body is not None:
            payload = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        conn = self._conn()
        began = time.monotonic()
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read(1_000_000)
            latency = (time.monotonic() - began) * 1000
            body = None
            if raw:
                try:
                    body = json.loads(raw)
                except ValueError:
                    body = None
            jar = {}
            for sc in resp.msg.get_all("Set-Cookie", []):
                part = sc.split(";", 1)[0]
                if "=" in part:
                    name, val = part.split("=", 1)
                    jar[name.strip()] = val.strip()
            return resp.status, body, jar, latency, None
        except Exception as exc:  # noqa: BLE001 - surfaced as setup_failure
            return None, None, {}, (time.monotonic() - began) * 1000, type(exc).__name__
        finally:
            conn.close()


def error_code(body):
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, dict):
            return detail.get("code")
        if isinstance(detail, str):
            return detail
    return None


def do_login(client: Client, login_value: str, password: str):
    """Full csrf+login. Returns (status, session_cookie|None, session_csrf|None, code)."""
    st, body, jar, _, err = client.request("GET", "/api/auth/csrf")
    if err:
        return None, None, None, f"connect:{err}"
    if st != 200 or not isinstance(body, dict) or not body.get("csrf_token"):
        return st, None, None, error_code(body) or "csrf_unavailable"
    bootstrap_cookie = jar.get(BOOTSTRAP_COOKIE)
    bootstrap_token = body["csrf_token"]
    st, body, jar, _, err = client.request(
        "POST", "/api/auth/login", json_body={"login": login_value, "password": password},
        cookies={BOOTSTRAP_COOKIE: bootstrap_cookie} if bootstrap_cookie else None,
        csrf=bootstrap_token)
    if err:
        return None, None, None, f"connect:{err}"
    if st == 200 and jar.get(SESSION_COOKIE) and isinstance(body, dict):
        return st, jar[SESSION_COOKIE], body.get("csrf_token"), None
    return st, None, None, error_code(body) or f"http_{st}"


def resolve_password(login_value: str, prompt: bool):
    env_key = "RT_PASS_" + "".join(c.upper() if c.isalnum() else "_" for c in login_value)
    value = os.environ.get(env_key)
    if value:
        return value
    if prompt:
        return getpass.getpass(f"Password for '{login_value}' (not echoed, not saved): ") or None
    return None


# --------------------------------------------------------------------------- #
# Tests. Each returns a sanitized result dict.                                 #
# --------------------------------------------------------------------------- #

def test_connectivity(client: Client):
    obs = {}
    st, body, _, ms, err = client.request("GET", "/api/health")
    obs["health_status"] = st
    obs["health_latency_ms"] = round(ms, 1)
    if err:
        return result("CONNECT", "infrastructure", "setup_failure",
                      f"Cannot reach {client.parsed.netloc}: {err}", obs)
    st, body, _, _, _ = client.request("GET", "/api/auth/profile")
    obs["auth_enabled"] = bool(body.get("enabled")) if isinstance(body, dict) else None
    # A csrf fetch validates that our Origin header is accepted by the server.
    st, body, _, _, _ = client.request("GET", "/api/auth/csrf")
    obs["csrf_status"] = st
    obs["origin_accepted"] = st == 200
    if not obs["origin_accepted"]:
        return result("CONNECT", "infrastructure", "setup_failure",
                      f"Origin {client.origin!r} rejected (status {st}, code {error_code(body)}). "
                      "Pass --origin to match RAG_AUTH_ALLOWED_ORIGINS.", obs)
    if obs["auth_enabled"] is not True:
        return result("CONNECT", "infrastructure", "inconclusive",
                      "Server reachable but authentication is not enabled; account tests need auth.", obs)
    return result("CONNECT", "infrastructure", "defense_held",
                  "Server reachable, auth enabled, Origin accepted.", obs)


def test_f2_targeted_lockout(client: Client, target_login, target_password, bystander_login,
                             bystander_password, window_attempts):
    """F2: lock a user out with wrong passwords; prove a correct login is then refused."""
    obs = {"target_login_present": bool(target_password),
           "bystander_login_present": bool(bystander_password)}
    attempts = []
    blocked_at = None
    for i in range(1, min(window_attempts + 3, MAX_LOCKOUT_ATTEMPTS) + 1):
        st, _c, _csrf, code = do_login(client, target_login, f"wrong-pw-{i}-{uuid.uuid4().hex[:8]}")
        attempts.append({"n": i, "status": st, "code": code})
        if code and code.startswith("connect:"):
            return result("F2-target", "auth-dos", "setup_failure",
                          "Lost connection during lockout loop.", {**obs, "attempts": attempts})
        if st == 429 and code == "rate_limited":
            blocked_at = i
            break
    obs["attempts"] = attempts
    obs["wrong_attempts_until_rate_limited"] = blocked_at
    if blocked_at is None:
        return result("F2-target", "auth-dos", "defense_held",
                      f"No lockout after {len(attempts)} wrong attempts.", obs)
    # The account is now rate-limited. Does a CORRECT password also get refused?
    if not target_password:
        return result("F2-target", "auth-dos", "inconclusive",
                      f"Account rate-limited after {blocked_at} wrong attempts, but no correct "
                      "password supplied to prove legitimate lockout.", obs)
    st, cookie, _csrf, code = do_login(client, target_login, target_password)
    obs["correct_login_status"] = st
    obs["correct_login_code"] = code
    if st == 429 and code == "rate_limited":
        return result("F2-target", "auth-dos", "attack_succeeded",
                      f"After {blocked_at} wrong attempts, the correct password for "
                      f"'{target_login}' was refused with 429 rate_limited: legitimate "
                      "user locked out.", obs)
    if st == 200 and cookie:
        return result("F2-target", "auth-dos", "defense_held",
                      "Correct password still succeeded despite prior wrong attempts.", obs)
    return result("F2-target", "auth-dos", "inconclusive",
                  f"Correct login returned status {st} ({code}); not a clean lockout.", obs)


def test_f2_global_lockout(client: Client, bystander_login, bystander_password, global_attempts):
    """F2: trip the global/IP bucket with fake logins, then try an UNRELATED real user."""
    obs = {"bystander_login_present": bool(bystander_password)}
    attempts, blocked_at = [], None
    cap = min(global_attempts + 5, MAX_LOCKOUT_ATTEMPTS)
    for i in range(1, cap + 1):
        st, _c, _csrf, code = do_login(client, f"ghost-{uuid.uuid4().hex[:10]}", "nope-nope-nope")
        attempts.append({"n": i, "status": st, "code": code})
        if code and code.startswith("connect:"):
            return result("F2-global", "auth-dos", "setup_failure",
                          "Lost connection during global lockout loop.", {**obs, "attempts": attempts})
        if st == 429 and code == "rate_limited":
            blocked_at = i
            break
    obs["distinct_fake_attempts_until_rate_limited"] = blocked_at
    obs["attempts"] = attempts
    if blocked_at is None:
        return result("F2-global", "auth-dos", "inconclusive",
                      f"No global/IP rate limit hit within {len(attempts)} distinct fake logins.", obs)
    if not bystander_password:
        return result("F2-global", "auth-dos", "inconclusive",
                      f"Global/IP limit tripped after {blocked_at} distinct fake logins, but no "
                      "bystander password supplied to prove an unrelated user is affected.", obs)
    st, cookie, _csrf, code = do_login(client, bystander_login, bystander_password)
    obs["bystander_login_status"] = st
    obs["bystander_login_code"] = code
    if st == 429 and code == "rate_limited":
        return result("F2-global", "auth-dos", "attack_succeeded",
                      f"Distinct fake logins (different usernames) blocked unrelated user "
                      f"'{bystander_login}' with 429 rate_limited: shared-IP lockout confirmed.", obs)
    if st == 200 and cookie:
        return result("F2-global", "auth-dos", "defense_held",
                      "An unrelated real user could still log in despite the fake-login flood.", obs)
    return result("F2-global", "auth-dos", "inconclusive",
                  f"Bystander login returned status {st} ({code}).", obs)


def test_f3_query_slot(client: Client, login_value, password, concurrency):
    """F3: concurrent free /api/runs reads; a global 'busy' slot rejects the rest."""
    obs = {}
    if not password:
        return result("F3", "slot-dos", "not_run",
                      f"Needs a session; no password for '{login_value}'.", obs)
    st, cookie, _csrf, code = do_login(client, login_value, password)
    if not cookie:
        return result("F3", "slot-dos", "setup_failure",
                      f"Could not log in as '{login_value}' (status {st}, {code}).", obs)
    n = min(concurrency, MAX_CONCURRENCY)
    obs["concurrency"] = n

    def fire(_i):
        rid = str(uuid.uuid4())
        st, body, _, ms, err = client.request("GET", f"/api/runs/{rid}",
                                               cookies={SESSION_COOKIE: cookie})
        return {"status": st, "code": error_code(body), "latency_ms": round(ms, 1), "error": err}

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(fire, range(n)))
    busy = sum(1 for r in results if r["status"] == 429 and r["code"] == "busy")
    obs["responses"] = results
    obs["busy_429_count"] = busy
    obs["status_histogram"] = histogram(r["status"] for r in results)
    if busy >= 1:
        return result("F3", "slot-dos", "attack_succeeded",
                      f"{busy}/{n} concurrent free run-receipt reads were refused with 429 busy: "
                      "a single global slot serializes all users and is occupied by a free, "
                      "unauthenticated-cost endpoint.", obs)
    return result("F3", "slot-dos", "inconclusive",
                  f"No 429 busy among {n} concurrent reads; receipts may be too fast to collide. "
                  "Retry with higher --concurrency or during a live query.", obs)


def test_f5_auth_saturation(client: Client, concurrency):
    """F5: flood the single auth worker with junk /api/auth/me checks."""
    obs = {}
    n = min(concurrency, MAX_CONCURRENCY)
    obs["concurrency"] = n
    junk = {SESSION_COOKIE: "x" * 43}  # well-formed-looking but invalid token

    def fire(_i):
        st, body, _, ms, err = client.request("GET", "/api/auth/me", cookies=junk)
        return {"status": st, "code": error_code(body), "latency_ms": round(ms, 1), "error": err}

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(fire, range(n)))
    busy = sum(1 for r in results if r["status"] == 429 and r["code"] == "busy")
    obs["responses"] = results
    obs["busy_429_count"] = busy
    obs["status_histogram"] = histogram(r["status"] for r in results)
    if busy >= 1:
        return result("F5", "auth-dos", "attack_succeeded",
                      f"{busy}/{n} concurrent junk auth checks hit 429 'Authentication is busy': "
                      "the single auth worker shared by login and every query can be saturated.", obs)
    return result("F5", "auth-dos", "inconclusive",
                  f"No auth-busy 429 among {n} concurrent checks; raise --concurrency to probe the "
                  "queue bound.", obs)


# --------------------------------------------------------------------------- #

def histogram(values):
    out = {}
    for v in values:
        out[str(v)] = out.get(str(v), 0) + 1
    return out


def result(test_id, category, outcome, reasoning, observations):
    return {"test_id": test_id, "category": category, "outcome": outcome,
            "reasoning": reasoning, "observations": observations}


def git_state():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:  # noqa: BLE001
        return {"commit": None, "dirty": None}


TESTS = [
    ("F2-target", "Targeted login lockout (wrong passwords block the real password)"),
    ("F2-global", "Shared-IP lockout (fake logins block an unrelated real user)"),
    ("F3", "Global query-slot exhaustion via free run-receipt reads"),
    ("F5", "Auth-worker saturation via junk session checks"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--origin", default=None,
                   help="Origin header; defaults to base-url's scheme://host:port.")
    p.add_argument("--run", action="store_true",
                   help="Execute the free tests. Without it, only connectivity is checked (dry-run).")
    p.add_argument("--list", action="store_true", help="List the tests and exit.")
    p.add_argument("--target-login", default="rt-bravo",
                   help="Account to lock out in F2-target.")
    p.add_argument("--bystander-login", default="rt-alpha",
                   help="Unrelated real account for F2-global / session for F3.")
    p.add_argument("--prompt-passwords", action="store_true",
                   help="Prompt for passwords instead of reading RT_PASS_* env vars.")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--outdir", default=str(ROOT / "outputs" / "red_team"))
    args = p.parse_args()

    if args.list:
        for tid, desc in TESTS:
            print(f"  {tid:10s} {desc}")
        return 0

    try:
        client = Client(args.base_url, origin=args.origin, timeout=args.timeout)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"# Availability red-team (FREE tests, no model calls)")
    print(f"# base-url={args.base_url} origin={client.origin}")
    conn = test_connectivity(client)
    print(f"[{conn['outcome']}] CONNECT  {conn['reasoning']}")

    if not args.run:
        print("\nDry-run: connectivity only. Re-run with --run to execute F2/F3/F5.")
        print("Tests that would run:")
        for tid, desc in TESTS:
            print(f"  {tid:10s} {desc}")
        return 0

    if conn["outcome"] == "setup_failure":
        print("Aborting: fix connectivity/origin first.")
        return 1

    target_pw = resolve_password(args.target_login, args.prompt_passwords)
    bystander_pw = resolve_password(args.bystander_login, args.prompt_passwords)

    results = [conn]
    # Order: least disruptive first. F2 trips rate limits, so run slot/auth tests
    # BEFORE the lockout tests to avoid interference, then global, then targeted.
    results.append(test_f5_auth_saturation(client, args.concurrency))
    results.append(test_f3_query_slot(client, args.bystander_login, bystander_pw, args.concurrency))
    results.append(test_f2_global_lockout(client, args.bystander_login, bystander_pw, global_attempts=20))
    results.append(test_f2_targeted_lockout(client, args.target_login, target_pw,
                                            args.bystander_login, bystander_pw, window_attempts=5))

    for r in results[1:]:
        print(f"[{r['outcome']}] {r['test_id']:10s} {r['reasoning']}")

    run_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = Path(args.outdir) / f"{stamp}-{run_id}"
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "rag-redteam-availability-v1", "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url, "origin": client.origin,
        "git": git_state(), "concurrency": min(args.concurrency, MAX_CONCURRENCY),
        "accounts": {"target": args.target_login, "bystander": args.bystander_login,
                     "target_password_supplied": bool(target_pw),
                     "bystander_password_supplied": bool(bystander_pw)},
        "note": "No /api/query calls; no paid model usage. No secrets recorded.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (outdir / "results.jsonl").open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    summary = ["# Availability red-team run", "",
               f"- Run: `{run_id}`  ({stamp})",
               f"- Target: `{args.base_url}`",
               f"- Commit: `{metadata['git']['commit']}` dirty={metadata['git']['dirty']}", "",
               "| Test | Outcome | Summary |", "| --- | --- | --- |"]
    for r in results:
        summary.append(f"| {r['test_id']} | {r['outcome']} | {r['reasoning']} |")
    counts = histogram(r["outcome"] for r in results)
    summary += ["", "Outcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
                "", "Raw per-test observations are in `results.jsonl`."]
    (outdir / "summary.md").write_text("\n".join(summary) + "\n")

    print(f"\nEvidence: {outdir}")
    succeeded = [r["test_id"] for r in results if r["outcome"] == "attack_succeeded"]
    if succeeded:
        print("Confirmed (attack_succeeded): " + ", ".join(succeeded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

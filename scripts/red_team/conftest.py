"""Pytest fixtures for the red-team reproduction suite.

These are NOT the project's unit tests. They drive the *running* app over HTTP
and reproduce documented security findings (F2, F3, F5). `pytest.ini` pins
`testpaths = tests`, so a plain `pytest` run and CI never collect this folder;
run it on purpose with `pytest scripts/red_team`.

The attack logic lives in `availability.py` next to this file. These fixtures
just wire it up, gate the suite on the app being reachable, and write the same
sanitized evidence bundle the CLI writes (no passwords/cookies/tokens on disk).
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make availability.py importable no matter where pytest is invoked from.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import availability as avail  # noqa: E402  (path set above)

ROOT = HERE.parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "redteam: live availability attack against the running application",
    )


# --------------------------------------------------------------------------- #
# Configuration (all overridable by environment variable).                    #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def base_url() -> str:
    return _env("RT_BASE_URL", "http://127.0.0.1:8000")


@pytest.fixture(scope="session")
def origin() -> str | None:
    return _env("RT_ORIGIN")  # None -> derived from base_url by the client


@pytest.fixture(scope="session")
def rt_config() -> dict:
    return {
        "target_login": _env("RT_TARGET_LOGIN", "rt-bravo"),
        "bystander_login": _env("RT_BYSTANDER_LOGIN", "rt-alpha"),
        "concurrency": int(_env("RT_CONCURRENCY", "12")),
        "timeout": float(_env("RT_TIMEOUT", "10")),
    }


@pytest.fixture(scope="session")
def client(base_url, origin, rt_config):
    try:
        return avail.Client(base_url, origin=origin, timeout=rt_config["timeout"])
    except ValueError as exc:
        pytest.skip(f"Bad RT_BASE_URL {base_url!r}: {exc}")


def password_for(login: str) -> str | None:
    """Read RT_PASS_<LOGIN> from the environment (never prompts, never stored)."""
    return avail.resolve_password(login, prompt=False)


# --------------------------------------------------------------------------- #
# Evidence collection: mirror the CLI's sanitized bundle.                     #
# --------------------------------------------------------------------------- #

class Recorder:
    def __init__(self):
        self.results: list[dict] = []

    def add(self, result: dict) -> dict:
        self.results.append(result)
        return result


@pytest.fixture(scope="session")
def recorder() -> Recorder:
    return Recorder()


@pytest.fixture(scope="session")
def server_ready(client, recorder):
    """Gate the whole suite on the app being reachable with auth enabled."""
    result = recorder.add(avail.test_connectivity(client))
    if result["outcome"] == "setup_failure":
        pytest.skip(f"App not reachable / misconfigured: {result['reasoning']}")
    return result


@pytest.fixture(scope="session", autouse=True)
def _write_evidence(recorder, base_url, origin, rt_config, request):
    """At session end, write metadata.json / results.jsonl / summary.md."""
    yield
    if not recorder.results:
        return
    try:
        client_obj = request.getfixturevalue("client")
        effective_origin = client_obj.origin
    except Exception:  # noqa: BLE001
        effective_origin = origin

    target_pw = bool(password_for(rt_config["target_login"]))
    bystander_pw = bool(password_for(rt_config["bystander_login"]))

    run_id = uuid.uuid4().hex[:12]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(_env("RT_OUTDIR", str(ROOT / "outputs" / "red_team")))
    outdir = base / f"{stamp}-{run_id}-pytest"
    outdir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "rag-redteam-availability-v1",
        "runner": "pytest",
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": base_url,
        "origin": effective_origin,
        "git": avail.git_state(),
        "concurrency": min(rt_config["concurrency"], avail.MAX_CONCURRENCY),
        "accounts": {
            "target": rt_config["target_login"],
            "bystander": rt_config["bystander_login"],
            "target_password_supplied": target_pw,
            "bystander_password_supplied": bystander_pw,
        },
        "note": "No /api/query calls; no paid model usage. No secrets recorded.",
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (outdir / "results.jsonl").open("w") as fh:
        for result in recorder.results:
            fh.write(json.dumps(result) + "\n")

    summary = [
        "# Availability red-team run (pytest)",
        "",
        f"- Run: `{run_id}`  ({stamp})",
        f"- Target: `{base_url}`",
        f"- Commit: `{metadata['git']['commit']}` dirty={metadata['git']['dirty']}",
        "",
        "| Test | Outcome | Summary |",
        "| --- | --- | --- |",
    ]
    for result in recorder.results:
        summary.append(
            f"| {result['test_id']} | {result['outcome']} | {result['reasoning']} |"
        )
    counts: dict[str, int] = {}
    for result in recorder.results:
        counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
    summary += [
        "",
        "Outcome counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "Raw per-test observations are in `results.jsonl`.",
    ]
    (outdir / "summary.md").write_text("\n".join(summary) + "\n")

    # Surface the evidence path in pytest's output.
    print(f"\n[red-team] evidence written to: {outdir}")

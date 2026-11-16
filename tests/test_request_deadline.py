"""In-process HTTP envelope deadlines; synthetic retrieval, no DB/model/network."""
import socket
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.runtime import FixtureProvider, FixtureRetriever, Runtime, WebSettings
from src.contracts import QueryResponse
from tests.test_api import wait_ready


@pytest.fixture(autouse=True)
def no_outbound_requests(monkeypatch):
    def denied(*_args, **_kwargs):
        raise AssertionError("HTTP deadline regression forbids external connections")
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.delenv("RAG_AUTH_ENABLED", raising=False)


def wait_idle(app):
    deadline = time.monotonic() + 2
    while app.state.busy and time.monotonic() < deadline:
        time.sleep(.005)
    assert not app.state.busy


@pytest.mark.parametrize("request_timeout,expected_status", [(None, 504), (.30, 200)])
def test_http_envelope_includes_preagent_work_without_expanding_agent_budget(request_timeout, expected_status):
    config = WebSettings(mode="fixture", demo_access=True, max_seconds=.12,
        request_timeout_seconds=request_timeout)
    timings = {}
    class SlowFixture(FixtureRetriever):
        def search(self, request):
            time.sleep(.08)
            return super().search(request)
    def factory(settings):
        real = Runtime(settings, SlowFixture(), FixtureProvider())
        class PreAgentWork:
            def info(self): return real.info()
            def query(self, body):
                time.sleep(.08)  # Simulates authentication/policy before agent timing.
                began = time.monotonic()
                response = real.query(body)
                timings["agent_elapsed"] = time.monotonic() - began
                timings["response"] = response
                return response
        return PreAgentWork()
    app = create_app(config, factory)
    with TestClient(app) as client:
        wait_ready(client)
        began = time.monotonic()
        response = client.post("/api/query", json={"question": "How long are artifacts retained?", "mode": "baseline"})
        http_elapsed = time.monotonic() - began
        assert response.status_code == expected_status
        wait_idle(app)
        actual = timings["response"]
        assert actual.answer.status == "answered"
        assert actual.trace.total_latency_ms < config.max_seconds * 1000
        assert timings["agent_elapsed"] < config.max_seconds
        assert config.max_seconds == .12 and config.max_tokens == 16000
        assert config.max_context_chars == 24000 and config.question_budget_usd == .1
        assert actual.trace.usage.api_calls == 0 and actual.trace.usage.estimated_cost_usd == 0
        if expected_status == 200:
            assert QueryResponse.model_validate(response.json()) == actual
            assert http_elapsed > config.max_seconds
            assert config.effective_request_timeout_seconds == .30
        else:
            assert response.json()["detail"]["code"] == "timeout"
            assert config.effective_request_timeout_seconds == .12


def test_http_timeout_keeps_real_worker_admitted_and_records_its_later_completion_separately():
    began, release = threading.Event(), threading.Event()
    class BlockedFixture(FixtureRetriever):
        def search(self, request):
            began.set()
            assert release.wait(2)
            return super().search(request)
    config = WebSettings(mode="retrieval", demo_access=True, max_seconds=.03, request_timeout_seconds=.06)
    app = create_app(config, lambda settings: Runtime(settings, BlockedFixture()))
    with TestClient(app) as client:
        wait_ready(client)
        try:
            response = client.post("/api/query", json={"question": "policy"})
            assert began.is_set() and response.status_code == 504
            raw = app.state.active_work
            assert raw is not None and not raw.done() and app.state.busy
            request_id = response.headers["x-request-id"]
            receipt = app.state.observability.get(request_id)["request"]
            assert receipt["http_status_code"] == 504 and receipt["http_finished"] is True
            assert receipt["status"] == "timeout" and receipt["worker_status"] == "running"
            assert receipt["completed_at"] is None
            assert client.get("/api/health").status_code == 200
            assert client.post("/api/query", json={"question": "second"}).status_code == 429
            assert app.state.active_work is raw and not raw.done()
        finally:
            release.set()
        raw.result(timeout=2)
        wait_idle(app)
        detail = app.state.observability.get(request_id)
        assert detail["request"]["http_status_code"] == 504
        assert detail["request"]["status"] == "timeout" and detail["request"]["worker_status"] == "ok"
        assert detail["request"]["completed_at"] is not None
        ending = [event["stage"] for event in detail["stages"]
            if event["stage"] in {"http", "worker"} and event["event"] == "end"]
        assert ending == ["http", "worker"]
        assert config.max_seconds == .03 and config.effective_request_timeout_seconds == .06


@pytest.mark.parametrize("request_timeout", [True, False, 0, -.1, .02, 361])
def test_invalid_http_deadline_cannot_shorten_the_configured_agent_allowance(request_timeout):
    with pytest.raises(ValueError):
        WebSettings(max_seconds=.03, request_timeout_seconds=request_timeout)

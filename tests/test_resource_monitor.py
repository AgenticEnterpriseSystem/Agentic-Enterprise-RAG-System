"""Real tiny workers plus synthetic OS-counter fixtures; no retrieval/model work."""

import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest

from src.evaluation import resource_monitor as monitor


def write_files(root, values):
    root.mkdir(parents=True, exist_ok=True)
    for name, value in values.items():
        (root / name).write_text(value)


def test_cgroup_v2_preserves_raw_counters_unlimited_and_zero_swap(tmp_path):
    write_files(tmp_path, {"memory.current": "1234\n", "memory.max": "max\n",
        "memory.peak": "2000\n", "memory.swap.current": "0\n", "memory.swap.max": "2048\n",
        "memory.events": "low 0\nhigh 2\nmax 3\noom 1\noom_kill 1\n",
        "memory.stat": "anon 800\nfile 400\n"})
    result = monitor.read_cgroup(tmp_path)
    assert result["version"] == 2 and result["current_bytes"] == 1234
    assert result["limit_bytes"] is None and result["limit_state"] == "unlimited"
    assert result["peak_bytes"] == 2000 and result["swap_current_bytes"] == 0
    assert result["swap_limit_bytes"] == 2048 and result["swap_limit_state"] == "known"
    assert result["events"]["oom_kill"] == 1 and result["stat"]["file"] == 400
    assert result["raw"]["memory.max"] == "max\n"
    (tmp_path / "memory.swap.max").write_text("max\n")
    unlimited = monitor.read_cgroup(tmp_path)
    assert unlimited["swap_limit_bytes"] is None and unlimited["swap_limit_state"] == "unlimited"
    assert unlimited["raw"]["memory.swap.max"] == "max\n"


def test_cgroup_v1_memory_subdirectory_and_combined_swap(tmp_path):
    write_files(tmp_path / "memory", {"memory.usage_in_bytes": "100\n",
        "memory.limit_in_bytes": "9223372036854771712\n", "memory.max_usage_in_bytes": "180\n",
        "memory.memsw.usage_in_bytes": "130\n", "memory.failcnt": "2\n",
        "memory.memsw.limit_in_bytes": "9223372036854771712\n",
        "memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n"})
    result = monitor.read_cgroup(tmp_path)
    assert result["version"] == 1 and result["current_bytes"] == 100
    assert result["peak_bytes"] == 180 and result["swap_current_bytes"] == 30
    assert result["limit_bytes"] is None and result["events"]["failcnt"] == 2
    assert result["events"]["oom_kill"] == 1
    assert result["swap_limit_bytes"] is None and result["swap_limit_state"] == "unlimited"
    assert "memory.memsw.limit_in_bytes" in result["raw"]


def test_cgroup_v1_finite_combined_limit_is_translated_to_swap_only(tmp_path):
    write_files(tmp_path, {"memory.usage_in_bytes": "60\n", "memory.limit_in_bytes": "100\n",
        "memory.memsw.usage_in_bytes": "80\n", "memory.memsw.limit_in_bytes": "300\n"})
    result = monitor.read_cgroup(tmp_path)
    assert result["swap_current_bytes"] == 20 and result["swap_limit_bytes"] == 200


def test_unknown_or_malformed_counters_are_not_invented_zeros(tmp_path):
    missing = monitor.read_cgroup(tmp_path)
    assert missing["version"] == "unknown" and missing["events"] is None
    assert all(missing[key] is None for key in ("current_bytes", "limit_bytes", "peak_bytes", "swap_current_bytes"))
    write_files(tmp_path, {"memory.current": "unavailable\n", "memory.max": "8192\n"})
    result = monitor.read_cgroup(tmp_path)
    assert result["current_bytes"] is None and result["peak_bytes"] is None
    assert result["limit_bytes"] == 8192 and result["events"] is None


def test_linux_proc_rss_and_peak_units_are_kib(monkeypatch):
    monkeypatch.setattr(monitor, "_platform", lambda: "linux")
    monkeypatch.setattr(monitor, "_text", lambda path: "VmRSS:\t123 kB\nVmHWM:\t456 kB\n")
    result = monitor.process_memory(123456)
    assert result["rss_bytes"] == 123 * 1024 and result["peak_rss_bytes"] == 456 * 1024


def test_darwin_ps_is_kib_and_foreign_peak_is_unknown(monkeypatch):
    monkeypatch.setattr(monitor, "_platform", lambda: "darwin")
    monkeypatch.setattr(monitor, "_ps_rss", lambda pid: 500)
    result = monitor.process_memory(os.getpid() + 100000)
    assert result["rss_bytes"] == 500 * 1024 and result["peak_rss_bytes"] is None


def test_linux_vm_and_io_counters_keep_units_and_missing_values(monkeypatch):
    def counter(path):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 1024 kB\nMemAvailable: 400 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
        if str(path).endswith("/io"):
            return "read_bytes: 123\nwrite_bytes: 456\n"
        return None
    monkeypatch.setattr(monitor, "_platform", lambda: "linux")
    monkeypatch.setattr(monitor, "_text", counter)
    result = monitor.sample_memory(123456)
    assert result["vm"]["available_bytes"] == 400 * 1024
    assert result["vm"]["swap_total_bytes"] == 0
    assert result["io"] == {"read_bytes": 123, "write_bytes": 456}
    assert result["process"]["rss_bytes"] is None


def supervise(tmp_path, code, **updates):
    options = dict(max_rss_bytes=0, max_cgroup_bytes=0, max_seconds=3, sample_seconds=.01)
    options.update(updates)
    return monitor.supervise([sys.executable, "-u", "-c", code], tmp_path, **options)


def test_completion_drains_final_events_and_preserves_stdout_stderr(tmp_path):
    code = """import json, sys, time
def emit(event, phase=None):
    print('PROFILE_EVENT ' + json.dumps(dict(event=event, **({'phase':phase} if phase else {}))), flush=True)
emit('start','load')
emit('start','shard')
time.sleep(.08)
emit('end','shard')
emit('end','load')
print('ordinary output')
print('diagnostic output',file=sys.stderr)
emit('complete')
"""
    result = supervise(tmp_path, code)
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert result["events_count"] == 5 and result["complete_event"] and result["output_drained"]
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "complete"
    assert "ordinary output" in (tmp_path / "worker.stdout.log").read_text()
    assert "diagnostic output" in (tmp_path / "worker.stderr.log").read_text()
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    assert samples and all("cgroup" in row and "process" in row and "phase" in row for row in samples)


@pytest.mark.parametrize("code,status", [
    ("raise RuntimeError('tiny failure')", "worker_error"),
    ("print('no completion event')", "incomplete"),
    ("import os,signal;os.kill(os.getpid(),signal.SIGTERM)", "signal"),
])
def test_worker_failure_signal_and_missing_completion_are_distinct(tmp_path, code, status):
    assert supervise(tmp_path, code)["status"] == status


def test_timeout_terminates_only_its_child_group_and_preserves_last_phase(tmp_path):
    code = "import time;print('PROFILE_EVENT {\"event\":\"start\",\"phase\":\"waiting\"}',flush=True);time.sleep(30)"
    started = time.monotonic()
    result = supervise(tmp_path, code, max_seconds=.15)
    assert result["status"] == "timeout" and result["exit_code"] < 0
    assert result["last_phase"] == "waiting" and time.monotonic() - started < 4


def test_ignoring_term_escalates_to_kill_with_bounded_wait(tmp_path):
    result = supervise(tmp_path, "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)",
                       max_seconds=.15)
    assert result["status"] == "timeout" and result["exit_code"] == -signal.SIGKILL
    assert result["duration_seconds"] < 4


@pytest.mark.parametrize("kind,expected", [("rss", "memory_guard"), ("cgroup", "cgroup_guard"), ("vm", "vm_guard")])
def test_memory_guards_stop_real_tiny_child_without_calling_it_oom(tmp_path, monkeypatch, kind, expected):
    # Synthetic counters make the threshold deterministic even where sandboxed
    # ps/proc access is unavailable. The supervised child/process group is real.
    monkeypatch.setattr(monitor, "sample_memory", lambda pid: {
        "process": {"rss_bytes": 20, "peak_rss_bytes": 30},
        "cgroup": {"current_bytes": 40, "peak_bytes": 50, "version": "unknown"},
        "vm": {"available_bytes": 5}, "io": None})
    result = supervise(tmp_path, "import time;time.sleep(30)",
                       max_rss_bytes=10 if kind == "rss" else 0,
                       max_cgroup_bytes=10 if kind == "cgroup" else 0,
                       min_vm_available_bytes=10 if kind == "vm" else 0)
    assert result["status"] == expected and result["exit_code"] < 0
    assert result["sampled_peaks"] == {"rss_bytes": 20, "cgroup_bytes": 40}
    assert result["peak_rss_bytes"] == 30
    assert result["minimum_vm_available_bytes"] == 5


@pytest.mark.parametrize("minimum", [-1, True, 1.5, "10"])
def test_vm_headroom_limit_requires_a_nonnegative_integer(tmp_path, minimum):
    with pytest.raises(ValueError, match="nonnegative integers"):
        supervise(tmp_path, "raise AssertionError('must not run')", min_vm_available_bytes=minimum)
    assert not (tmp_path / "worker.stdout.log").exists()


def test_existing_output_is_not_overwritten_and_no_worker_is_started(tmp_path):
    (tmp_path / "events.jsonl").write_text("retained\n")
    with pytest.raises(FileExistsError):
        supervise(tmp_path, "raise AssertionError('must not run')")
    assert (tmp_path / "events.jsonl").read_text() == "retained\n"
    assert not (tmp_path / "worker.stdout.log").exists()


def test_sampling_failure_does_not_orphan_the_owned_child(tmp_path, monkeypatch):
    observed = []

    def unavailable(pid):
        observed.append(pid)
        raise OSError("synthetic sampler failure")

    monkeypatch.setattr(monitor, "sample_memory", unavailable)
    with pytest.raises(OSError, match="sampler failure"):
        supervise(tmp_path, "import time;time.sleep(30)")
    assert len(observed) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(observed[0], 0)


def test_event_deltas_preserve_missing_and_reset_counters():
    assert monitor._event_deltas(None, {"oom": 0}) is None
    assert monitor._event_deltas({"oom": 0, "oom_kill": 1}, {"oom": 2, "oom_kill": 0}) == {"oom": 2, "oom_kill": None}

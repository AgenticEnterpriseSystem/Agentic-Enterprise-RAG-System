"""Stdlib-only, bounded supervision for an explicitly supplied profiling worker.

Unknown counters stay unknown. Sampled maxima are distinct from operating-system
high-water counters; guard termination is never classified as a proven OOM kill.
No model, index, policy or application loader is imported by this module.
"""

from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

EVENT_PREFIX = "PROFILE_EVENT "


def _text(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None


def _integer(value):
    try:
        number = int(value.strip())
        return number if number >= 0 else None
    except (ValueError, AttributeError):
        return None


def _pairs(value):
    if value is None:
        return None
    result = {}
    for line in value.splitlines():
        fields = line.split()
        if len(fields) == 2 and (number := _integer(fields[1])) is not None:
            result[fields[0]] = number
    return result


def read_cgroup(root=Path("/sys/fs/cgroup")) -> dict:
    """Read raw v2 files at root, or v1 files at root[/memory].

    Missing/malformed values normalize to None, not zero. Unlimited v2 `max`
    and v1's huge sentinel normalize to limit_bytes=None; raw data remains.
    v1 memsw usage includes RAM, so swap is derived only when both inputs exist.
    """
    root = Path(root)
    if any((root / name).is_file() for name in ("memory.current", "memory.max")):
        version, location = 2, root
        names = ("memory.current", "memory.max", "memory.peak", "memory.swap.current", "memory.swap.max",
                 "memory.events", "memory.events.local", "memory.stat")
    else:
        location = root / "memory" if (root / "memory").is_dir() else root
        version = (1 if any((location / name).is_file() for name in
                   ("memory.usage_in_bytes", "memory.limit_in_bytes")) else "unknown")
        names = ("memory.usage_in_bytes", "memory.limit_in_bytes", "memory.max_usage_in_bytes",
                 "memory.memsw.usage_in_bytes", "memory.memsw.limit_in_bytes", "memory.failcnt",
                 "memory.oom_control", "memory.stat")
    raw = {name: value for name in names if (value := _text(location / name)) is not None}
    if version == 2:
        current = _integer(raw.get("memory.current"))
        limit = _integer(raw.get("memory.max"))
        peak = _integer(raw.get("memory.peak"))
        swap = _integer(raw.get("memory.swap.current"))
        events = _pairs(raw.get("memory.events"))
        unlimited = raw.get("memory.max", "").strip() == "max"
        swap_limit = _integer(raw.get("memory.swap.max"))
        swap_unlimited = raw.get("memory.swap.max", "").strip() == "max"
    elif version == 1:
        current = _integer(raw.get("memory.usage_in_bytes"))
        limit = _integer(raw.get("memory.limit_in_bytes"))
        unlimited = limit is not None and limit >= 2 ** 60
        if unlimited:
            limit = None
        peak = _integer(raw.get("memory.max_usage_in_bytes"))
        combined = _integer(raw.get("memory.memsw.usage_in_bytes"))
        swap = (combined - current if combined is not None and current is not None
                and combined >= current else None)
        combined_limit = _integer(raw.get("memory.memsw.limit_in_bytes"))
        swap_unlimited = combined_limit is not None and combined_limit >= 2 ** 60
        swap_limit = (combined_limit - limit if combined_limit is not None and limit is not None
                      and not swap_unlimited and combined_limit >= limit else None)
        events = _pairs(raw.get("memory.oom_control"))
        failcnt = _integer(raw.get("memory.failcnt"))
        if failcnt is not None:
            events = dict(events or {}, failcnt=failcnt)
    else:
        current = limit = peak = swap = swap_limit = events = None
        unlimited = swap_unlimited = False
    return {"version": version, "path": str(location), "raw": raw,
            "current_bytes": current, "limit_bytes": limit, "peak_bytes": peak,
            "swap_current_bytes": swap, "swap_limit_bytes": swap_limit,
            "swap_limit_state": "unlimited" if swap_unlimited else "known" if swap_limit is not None else "unknown",
            "events": events, "stat": _pairs(raw.get("memory.stat")),
            "limit_state": "unlimited" if unlimited else "known" if limit is not None else "unknown"}


def _platform():
    return sys.platform


def _ps_rss(pid):
    try:
        result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                                text=True, timeout=1, check=False)
        return _integer(result.stdout) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _status_kib(text, key):
    for line in (text or "").splitlines():
        if line.startswith(key + ":"):
            fields = line.split()
            if len(fields) >= 3 and fields[2] == "kB":
                value = _integer(fields[1])
                return value * 1024 if value is not None else None
    return None


def process_memory(pid=None) -> dict:
    """RSS/HWM for Linux proc status, or Darwin ps RSS (whose unit is KiB).

    ru_maxrss is used only for this process, never as another PID's peak. Darwin
    returns bytes; Linux returns KiB. A foreign Darwin peak remains unknown.
    """
    pid = os.getpid() if pid is None else pid
    status = _text(Path(f"/proc/{pid}/status")) if _platform().startswith("linux") else None
    rss = _status_kib(status, "VmRSS")
    peak = _status_kib(status, "VmHWM")
    source = "proc_status" if status is not None else "unknown"
    if rss is None and _platform() == "darwin":
        kib = _ps_rss(pid)
        if kib is not None:
            rss, source = kib * 1024, "ps"
    if peak is None and pid == os.getpid() and (_platform() == "darwin" or _platform().startswith("linux")):
        try:
            import resource
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak = int(value) * (1 if _platform() == "darwin" else 1024)
        except (ImportError, OSError, ValueError):
            pass
    return {"pid": pid, "rss_bytes": rss, "peak_rss_bytes": peak, "source": source}


def _cgroup_root(pid):
    base = Path("/sys/fs/cgroup")
    for line in (_text(Path(f"/proc/{pid}/cgroup")) or "").splitlines():
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        hierarchy, controllers, relative = fields
        candidate = (base / relative.lstrip("/") if hierarchy == "0" and not controllers
                     else base / "memory" / relative.lstrip("/") if "memory" in controllers.split(",")
                     else None)
        if candidate is not None and candidate.is_dir():
            return candidate
    return base


def _vm_memory():
    fields = {}
    for line in (_text(Path("/proc/meminfo")) or "").splitlines():
        name = line.split(":", 1)[0]
        if name in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
            fields[name] = _status_kib(line, name)
    return {"total_bytes": fields.get("MemTotal"), "available_bytes": fields.get("MemAvailable"),
            "swap_total_bytes": fields.get("SwapTotal"), "swap_free_bytes": fields.get("SwapFree")}


def sample_memory(pid) -> dict:
    """Sample only the requested process plus its cgroup and available VM data."""
    io_text = _text(Path(f"/proc/{pid}/io")) if _platform().startswith("linux") else None
    io = _pairs(io_text.replace(":", "")) if io_text is not None else None
    return {"process": process_memory(pid), "cgroup": read_cgroup(_cgroup_root(pid)),
            "vm": _vm_memory(), "io": io}


def _event_deltas(before, after):
    if before is None or after is None:
        return None
    return {key: after[key] - value if after[key] >= value else None
            for key, value in before.items() if key in after}


def _maximum(old, value):
    return max(old, value) if old is not None and value is not None else value if old is None else old


def _signal_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


def _cleanup_worker(worker):
    """Also bound child lifetime if logging/sampling or the caller is interrupted."""
    code = worker.poll()
    if code is not None:
        # The leader may have exited before a descendant that ignored TERM.
        # This is only the new process group created for this owned worker.
        _signal_group(worker.pid, signal.SIGKILL)
        return code
    _signal_group(worker.pid, signal.SIGTERM)
    try:
        worker.wait(timeout=1)
    except subprocess.TimeoutExpired:
        _signal_group(worker.pid, signal.SIGKILL)
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    _signal_group(worker.pid, signal.SIGKILL)
    return worker.poll()


def _reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def supervise(command, output: Path, *, max_rss_bytes: int, max_cgroup_bytes: int,
              max_seconds: float, sample_seconds=.2, min_vm_available_bytes=0, env=None) -> dict:
    """Run one new process group; preserve flushed samples/events/stdout/stderr.

    Byte limits are strict nonnegative integers; zero disables that guard.
    Unknown counters never trigger a memory guard. A positive minimum available
    VM-memory threshold stops at or below that known headroom; zero disables it.
    A TERM is followed by KILL
    after one second; output draining is also bounded. Completed requires exit
    zero plus a parsed complete event. Guard/timeout statuses take precedence
    over the terminating signal and are not OOM diagnoses.
    """
    if not command or isinstance(command, (str, bytes)):
        raise ValueError("Command must be a nonempty argument sequence")
    for limit in (max_rss_bytes, max_cgroup_bytes, min_vm_available_bytes):
        if type(limit) is not int or limit < 0:
            raise ValueError("Memory limits must be nonnegative integers")
    for seconds in (max_seconds, sample_seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Timing limits must be finite and positive")
    paths = {name: Path(output) / name for name in
             ("samples.jsonl", "events.jsonl", "worker.stdout.log", "worker.stderr.log")}
    if not Path(output).is_dir():
        raise ValueError("Output directory must already exist")
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Profiling output files must be new")
    before = read_cgroup(_cgroup_root(os.getpid()))
    cgroup_path = before["path"]
    started = time.monotonic()
    messages = queue.Queue(maxsize=1000)
    phase_stack, last_phase = [], None
    sample_peaks = {"rss_bytes": None, "cgroup_bytes": None}
    process_peak = cgroup_peak = None
    minimum_vm_available = None
    complete, events_count, invalid_events, ended_readers = False, 0, 0, set()
    stop_reason = stop_at = exit_seen = None
    killed = False

    with ExitStack() as files:
        streams = {name: files.enter_context(path.open("x", encoding="utf-8")) for name, path in paths.items()}
        worker = subprocess.Popen(list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True, env=env)
        cleanup_required = True

        def cleanup_on_error():
            if cleanup_required:
                _cleanup_worker(worker)

        files.callback(cleanup_on_error)

        def reader(label, pipe):
            try:
                for line in iter(pipe.readline, ""):
                    messages.put((label, line))
            except (OSError, ValueError):
                pass
            finally:
                messages.put((label, None))

        readers = [threading.Thread(target=reader, args=(label, pipe), daemon=True)
                   for label, pipe in (("stdout", worker.stdout), ("stderr", worker.stderr))]
        for thread in readers:
            thread.start()

        def write_json(name, record):
            streams[name].write(json.dumps(record, allow_nan=False) + "\n")
            streams[name].flush()

        def drain_messages():
            nonlocal last_phase, complete, events_count, invalid_events
            # Bound each pass so a noisy worker cannot starve supervision.
            for _ in range(1000):
                try:
                    label, line = messages.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    ended_readers.add(label)
                    continue
                streams[f"worker.{label}.log"].write(line)
                streams[f"worker.{label}.log"].flush()
                if label != "stdout" or not line.startswith(EVENT_PREFIX):
                    continue
                try:
                    event = json.loads(line[len(EVENT_PREFIX):], parse_constant=_reject_constant)
                    if not isinstance(event, dict):
                        raise ValueError("Expected event object")
                    write_json("events.jsonl", {**event, "supervisor_elapsed_seconds": time.monotonic() - started})
                except (ValueError, TypeError):
                    invalid_events += 1
                    continue
                events_count += 1
                phase = event.get("phase")
                if isinstance(phase, str) and phase:
                    last_phase = phase
                    if event.get("event") == "start":
                        phase_stack.append(phase)
                    elif event.get("event") == "end" and phase in phase_stack:
                        reverse_index = phase_stack[::-1].index(phase)
                        del phase_stack[len(phase_stack) - reverse_index - 1:]
                if event.get("event") == "complete":
                    complete = True

        next_sample = 0.
        while True:
            drain_messages()
            now = time.monotonic()
            code = worker.poll()
            if now >= next_sample and code is None:
                sample = sample_memory(worker.pid)
                write_json("samples.jsonl", {"elapsed_seconds": now - started,
                    "phase": phase_stack[-1] if phase_stack else None,
                    "phase_stack": list(phase_stack), **sample})
                proc, cg = sample["process"], sample["cgroup"]
                rss, group = proc.get("rss_bytes"), cg.get("current_bytes")
                available = sample.get("vm", {}).get("available_bytes")
                if available is not None:
                    minimum_vm_available = (min(minimum_vm_available, available)
                                            if minimum_vm_available is not None else available)
                sample_peaks["rss_bytes"] = _maximum(sample_peaks["rss_bytes"], rss)
                sample_peaks["cgroup_bytes"] = _maximum(sample_peaks["cgroup_bytes"], group)
                process_peak = _maximum(process_peak, proc.get("peak_rss_bytes"))
                cgroup_peak = _maximum(cgroup_peak, cg.get("peak_bytes"))
                if cg.get("version") != "unknown" and cg.get("path"):
                    cgroup_path = cg["path"]
                if stop_reason is None:
                    if max_rss_bytes and rss is not None and rss >= max_rss_bytes:
                        stop_reason = "memory_guard"
                    elif max_cgroup_bytes and group is not None and group >= max_cgroup_bytes:
                        stop_reason = "cgroup_guard"
                    elif min_vm_available_bytes and available is not None and available <= min_vm_available_bytes:
                        stop_reason = "vm_guard"
                next_sample = time.monotonic() + sample_seconds
            if stop_reason is None and code is None and time.monotonic() - started >= max_seconds:
                stop_reason = "timeout"
            if stop_reason is not None and stop_at is None:
                stop_at = time.monotonic()
                _signal_group(worker.pid, signal.SIGTERM)
            if stop_at is not None and not killed and time.monotonic() - stop_at >= 1:
                _signal_group(worker.pid, signal.SIGKILL)
                killed = True
            if code is not None:
                exit_seen = time.monotonic() if exit_seen is None else exit_seen
                if len(ended_readers) == 2 and messages.empty():
                    break
                if time.monotonic() - exit_seen >= 2:
                    break
            if stop_at is not None and time.monotonic() - stop_at >= 3:
                break
            time.sleep(min(.01, sample_seconds))
        # Complete bounded cleanup BEFORE recording final status/counters. The
        # emergency ExitStack callback must not perform a later hidden reap that
        # leaves a stale exit_code=None in an otherwise finalized report.
        _cleanup_worker(worker)
        cleanup_required = False
        drain_deadline = time.monotonic() + 2
        while time.monotonic() < drain_deadline:
            drain_messages()
            if len(ended_readers) == 2 and messages.empty():
                break
            time.sleep(.01)
        for thread in readers:
            thread.join(timeout=.1)
        drain_messages()
        code = worker.poll()
        drained = len(ended_readers) == 2 and messages.empty()
        for pipe in (worker.stdout, worker.stderr):
            if drained:
                pipe.close()
        after = read_cgroup(Path(cgroup_path))
    deltas = _event_deltas(before.get("events"), after.get("events"))
    oom = ({key: value for key, value in deltas.items() if key in ("oom", "oom_kill", "oom_group_kill")}
           if deltas is not None else None)
    if stop_reason is not None:
        status = stop_reason
    elif code == 0 and complete and drained:
        status = "completed"
    elif code is not None and code < 0:
        status = "signal"
    elif code not in (0, None):
        status = "worker_error"
    else:
        status = "incomplete"
    return {"status": status, "exit_code": code, "worker_reaped": code is not None, "pid": worker.pid,
            "duration_seconds": time.monotonic() - started,
            "last_phase": phase_stack[-1] if phase_stack else last_phase,
            "sampled_peaks": sample_peaks, "peak_rss_bytes": process_peak,
            "minimum_vm_available_bytes": minimum_vm_available,
            "cgroup_peak_bytes": _maximum(cgroup_peak, after.get("peak_bytes")),
            "cgroup_before": before, "cgroup_after": after,
            "cgroup_event_deltas": deltas, "oom_event_deltas": oom or None,
            "complete_event": complete, "events_count": events_count,
            "invalid_events": invalid_events, "output_drained": drained}

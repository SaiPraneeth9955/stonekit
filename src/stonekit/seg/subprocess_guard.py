"""Run an external tool so that StoneKit can never wait forever for it.

Why this exists: on 6 November 2026 TotalSegmentator's main process was killed for lack of memory,
while its helper processes stayed alive holding the output pipe that `subprocess.run(capture_output=
True)` was reading. The read never ended, so `run-case` waited for 23 hours.

The rules here:
- Output goes to a log file, never a pipe, so a surviving helper cannot block the wait.
- The tool runs in its own session and process group, so it and every helper it starts can be
  stopped together, even after the main process has died and the helpers were reparented.
- The wait has a timeout. After the main process ends, for any reason, every process still in the
  group is stopped: SIGTERM, a grace period, then SIGKILL.
- Every failure becomes an ExternalToolError with a reason, the exit code, the time and the log.
"""

import os
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Words in a log tail that point at memory exhaustion. A hint only; the kernel log is not readable.
MEMORY_HINTS = ("out of memory", "memoryerror", "cannot allocate memory", "killed", "oom")


class ExternalToolError(RuntimeError):
    """A controlled failure of an external tool. Never treated as success."""

    def __init__(self, tool: str, reason: str, detail: str, result: "GuardedResult | None" = None):
        self.tool = tool
        self.reason = reason
        self.detail = detail
        self.result = result
        super().__init__(f"{tool} {reason}: {detail}")

    def to_dict(self) -> dict:
        info = {"tool": self.tool, "reason": self.reason, "detail": self.detail}
        if self.result is not None:
            info.update(self.result.to_dict())
        return info


@dataclass
class GuardedResult:
    returncode: int | None
    seconds: float
    timed_out: bool
    log_path: str
    leftover_processes_stopped: int = 0
    processes_still_alive_after_cleanup: int = 0
    notes: list[str] = field(default_factory=list)
    peak_group_rss_mib: float | None = None
    peak_gpu_memory_used_mib: float | None = None

    def to_dict(self) -> dict:
        return {
            "returncode": self.returncode,
            "seconds": round(self.seconds, 1),
            "timed_out": self.timed_out,
            "log_path": self.log_path,
            "leftover_processes_stopped": self.leftover_processes_stopped,
            "processes_still_alive_after_cleanup": self.processes_still_alive_after_cleanup,
            "notes": self.notes,
            "peak_group_rss_mib": self.peak_group_rss_mib,
            "peak_gpu_memory_used_mib": self.peak_gpu_memory_used_mib,
            "resource_note": "Sampled once a second: PSS summed over the tool's process group; GPU memory is "
            "device-wide (includes any other GPU process). Short spikes between samples can be missed.",
        }


class ResourceSampler(threading.Thread):
    """Peak RAM of a process group and peak device GPU memory, sampled until stopped."""

    def __init__(self, pgid: int, interval: float = 1.0) -> None:
        super().__init__(daemon=True)
        self.pgid = pgid
        self.interval = interval
        self.stopped = threading.Event()
        self.peak_rss_kib = 0
        self.peak_gpu_mib: float | None = None
        self.gpu = shutil.which("nvidia-smi")

    def sample(self) -> None:
        # PSS (proportional set size) splits shared pages between the processes that share them, so the
        # group total is not inflated by forked workers. Summing VmRSS double-counted shared memory.
        total = 0
        for pid in group_members(self.pgid):
            try:
                for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                    if line.startswith("Pss:"):
                        total += int(line.split()[1])
                        break
            except (OSError, ValueError):
                continue
        self.peak_rss_kib = max(self.peak_rss_kib, total)
        if self.gpu:
            try:
                text = subprocess.run([self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                      check=False, capture_output=True, text=True, timeout=5).stdout  # fmt: skip
                used = max(float(v) for v in text.split())
                self.peak_gpu_mib = used if self.peak_gpu_mib is None else max(self.peak_gpu_mib, used)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    def run(self) -> None:
        while not self.stopped.is_set():
            self.sample()
            self.stopped.wait(self.interval)

    def stop(self) -> None:
        self.stopped.set()
        self.join(timeout=10)


def group_members(pgid: int) -> list[int]:
    """Live (not zombie) processes in a process group, read from /proc."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # Fields after the command name, which is in parentheses and may hold spaces.
        fields = stat[stat.rfind(")") + 2 :].split()
        state, process_group = fields[0], int(fields[2])
        if process_group == pgid and state != "Z":
            members.append(int(entry.name))
    return members


def stop_group(pgid: int, grace_seconds: float) -> tuple[int, int]:
    """Stop every live process in the group. Returns (found, still alive afterwards)."""
    found = len(group_members(pgid))
    if not found:
        return 0, 0
    for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, grace_seconds)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return found, 0
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            if not group_members(pgid):
                return found, 0
            time.sleep(0.1)
    return found, len(group_members(pgid))


def log_tail(path: Path, characters: int = 600) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return " ".join(text.split())[-characters:]


def run_guarded(
    command: list[str],
    log_path: Path,
    timeout_seconds: float,
    grace_seconds: float = 10,
    env: dict[str, str] | None = None,
) -> GuardedResult:
    """Run `command` with a timeout and full cleanup. Returns a result; it does not judge success."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=env,
        )  # fmt: skip
        pgid = process.pid  # A new session makes the child its own group leader.
        sampler = ResourceSampler(pgid)
        sampler.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
        except BaseException:
            # Interrupted (Ctrl+C, an error): never leave the tool running behind us.
            stop_group(pgid, grace_seconds)
            process.wait()
            raise
        finally:
            sampler.stop()
            found, alive = stop_group(pgid, grace_seconds)
        if timed_out:
            returncode = process.wait()
    result = GuardedResult(returncode, time.monotonic() - start, timed_out, str(log_path))
    result.peak_group_rss_mib = round(sampler.peak_rss_kib / 1024, 1)
    result.peak_gpu_memory_used_mib = sampler.peak_gpu_mib
    # The main process is not counted as a leftover when the timeout stopped it.
    result.leftover_processes_stopped = found - 1 if timed_out and found else found
    result.processes_still_alive_after_cleanup = alive
    if result.leftover_processes_stopped:
        result.notes.append(f"{result.leftover_processes_stopped} helper process(es) were still running and were stopped")
    if alive:
        result.notes.append(f"{alive} process(es) could not be stopped")
    return result


def check_result(tool: str, result: GuardedResult, expected_output: Path, timeout_seconds: float) -> None:
    """Raise ExternalToolError unless the tool exited 0 on time and wrote a non-empty output."""
    tail = log_tail(Path(result.log_path))
    memory = any(hint in tail.lower() for hint in MEMORY_HINTS)
    if result.timed_out:
        raise ExternalToolError(tool, "timeout", f"no exit within {timeout_seconds} s; stopped. Log: {tail[-300:]}", result)
    if result.returncode is not None and result.returncode < 0:
        name = signal.Signals(-result.returncode).name
        hint = " SIGKILL from outside is often the out-of-memory killer." if name == "SIGKILL" else ""
        raise ExternalToolError(tool, "killed_by_signal", f"{name}.{hint} Log: {tail[-300:]}", result)
    if result.returncode != 0:
        hint = " The log mentions memory exhaustion." if memory else ""
        raise ExternalToolError(tool, "nonzero_exit", f"exit code {result.returncode}.{hint} Log: {tail[-300:]}", result)
    if not Path(expected_output).exists() or Path(expected_output).stat().st_size == 0:
        hint = " The log mentions memory exhaustion." if memory else ""
        raise ExternalToolError(tool, "missing_output", f"exit code 0 but no output at {expected_output}.{hint}", result)

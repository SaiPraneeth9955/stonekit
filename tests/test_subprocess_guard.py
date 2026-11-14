"""The external-tool guard, with small stand-in Python processes instead of TotalSegmentator.

Each stand-in is a few lines run by this Python. None reads any scan. Every test has a hard upper
bound on time, so a regression to waiting forever fails instead of hanging.
"""

import sys
import textwrap
import time
from pathlib import Path

import pytest

from stonekit.seg import kidney_roi
from stonekit.seg.subprocess_guard import ExternalToolError, check_result, run_guarded

pytestmark = pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs Linux /proc")

# A helper that outlives its parent and holds the inherited log handle, like the 6 November hang.
HELPER = """
import subprocess, sys, pathlib
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
pathlib.Path({pid_file!r}).write_text(str(child.pid))
"""


def stand_in(tmp_path: Path, body: str) -> list[str]:
    script = tmp_path / "stand_in.py"
    script.write_text(textwrap.dedent(body))
    return [sys.executable, str(script)]


def is_dead(pid: int, within: float = 10) -> bool:
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return True
        if state == "Z":
            return True
        time.sleep(0.1)
    return False


def guard(tmp_path, body, timeout=20, output=None):
    output = output or tmp_path / "out.nii.gz"
    result = run_guarded(stand_in(tmp_path, body), tmp_path / "tool.log", timeout, grace_seconds=2)
    return result, output


def test_success_passes(tmp_path):
    out = tmp_path / "out.nii.gz"
    result, _ = guard(tmp_path, f"open({str(out)!r}, 'wb').write(b'data'); print('done')")
    check_result("tool", result, out, 20)
    assert result.returncode == 0
    assert not result.timed_out
    assert "done" in Path(result.log_path).read_text()


def test_nonzero_exit_is_a_controlled_failure(tmp_path):
    result, out = guard(tmp_path, "import sys; print('bad input'); sys.exit(3)")
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 20)
    assert error.value.reason == "nonzero_exit"
    assert "exit code 3" in error.value.detail
    assert "bad input" in error.value.detail


def test_memory_words_in_the_log_are_reported(tmp_path):
    result, out = guard(tmp_path, "import sys; print('RuntimeError: CUDA out of memory'); sys.exit(1)")
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 20)
    assert "memory exhaustion" in error.value.detail


def test_killed_process_is_a_controlled_failure(tmp_path):
    result, out = guard(tmp_path, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 20)
    assert error.value.reason == "killed_by_signal"
    assert "SIGKILL" in error.value.detail
    assert "out-of-memory" in error.value.detail


def test_timeout_stops_the_tool_and_fails(tmp_path):
    start = time.monotonic()
    result, out = guard(tmp_path, "import time; time.sleep(120)", timeout=1)
    assert time.monotonic() - start < 15
    assert result.timed_out
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 1)
    assert error.value.reason == "timeout"


def test_missing_output_is_a_controlled_failure(tmp_path):
    result, out = guard(tmp_path, "print('finished without writing')")
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 20)
    assert error.value.reason == "missing_output"


def test_empty_output_counts_as_missing(tmp_path):
    out = tmp_path / "out.nii.gz"
    result, _ = guard(tmp_path, f"open({str(out)!r}, 'wb')")
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 20)
    assert error.value.reason == "missing_output"


def test_dead_parent_with_live_helper_returns_promptly_and_stops_the_helper(tmp_path):
    """The 6 November failure: the main process dies, a helper keeps the output open."""
    pid_file = tmp_path / "helper.pid"
    start = time.monotonic()
    result, out = guard(tmp_path, HELPER.format(pid_file=str(pid_file)) + "\nimport os; os._exit(1)\n", timeout=60)
    assert time.monotonic() - start < 20
    helper = int(pid_file.read_text())
    assert is_dead(helper)
    assert result.leftover_processes_stopped == 1
    assert result.processes_still_alive_after_cleanup == 0
    with pytest.raises(ExternalToolError) as error:
        check_result("tool", result, out, 60)
    assert error.value.reason == "nonzero_exit"


def test_timeout_also_stops_helpers(tmp_path):
    pid_file = tmp_path / "helper.pid"
    result, _ = guard(tmp_path, HELPER.format(pid_file=str(pid_file)) + "\nimport time; time.sleep(120)\n", timeout=2)
    assert result.timed_out
    assert is_dead(int(pid_file.read_text()))
    assert result.leftover_processes_stopped == 1


def test_helper_left_after_success_is_stopped_and_noted(tmp_path):
    pid_file = tmp_path / "helper.pid"
    out = tmp_path / "out.nii.gz"
    body = HELPER.format(pid_file=str(pid_file)) + f"\nopen({str(out)!r}, 'wb').write(b'data')\n"
    result, _ = guard(tmp_path, body)
    check_result("tool", result, out, 20)
    assert is_dead(int(pid_file.read_text()))
    assert result.notes


def test_a_helper_ignoring_sigterm_is_killed(tmp_path):
    pid_file = tmp_path / "helper.pid"
    body = f"""
    import subprocess, sys, pathlib
    code = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"
    child = subprocess.Popen([sys.executable, "-c", code])
    pathlib.Path({str(pid_file)!r}).write_text(str(child.pid))
    import time; time.sleep(0.5)
    """
    result, _ = guard(tmp_path, body)
    assert is_dead(int(pid_file.read_text()))
    assert result.processes_still_alive_after_cleanup == 0


def test_run_totalsegmentator_cleans_up_and_never_publishes_a_failed_output(tmp_path, monkeypatch):
    def fake_command(ct_path, output_path, device, low_memory):
        script = f"open({str(output_path)!r}, 'wb').write(b'half'); import sys; sys.exit(2)"
        return [sys.executable, "-c", script]

    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", fake_command)
    target = tmp_path / "case_ts.nii.gz"
    with pytest.raises(ExternalToolError) as error:
        kidney_roi.run_totalsegmentator(tmp_path / "ct.nii.gz", target, "cpu", timeout_seconds=20)
    assert error.value.reason == "nonzero_exit"
    assert not target.exists()
    assert not (tmp_path / "case_ts.partial.nii.gz").exists()
    assert (tmp_path / "case_ts.totalsegmentator.log").exists()


def test_run_totalsegmentator_success_publishes_the_output(tmp_path, monkeypatch):
    def fake_command(ct_path, output_path, device, low_memory):
        return [sys.executable, "-c", f"open({str(output_path)!r}, 'wb').write(b'labels')"]

    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", fake_command)
    target = tmp_path / "case_ts.nii.gz"
    result = kidney_roi.run_totalsegmentator(tmp_path / "ct.nii.gz", target, "cpu", timeout_seconds=20)
    assert target.read_bytes() == b"labels"
    assert result.returncode == 0


def test_interrupt_while_waiting_stops_the_tool(tmp_path, monkeypatch):
    import subprocess

    pid_file = tmp_path / "tool.pid"
    original_wait = subprocess.Popen.wait
    calls = {"n": 0}

    def interrupting_wait(self, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            deadline = time.monotonic() + 10
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            raise KeyboardInterrupt
        return original_wait(self, timeout=timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupting_wait)
    body = f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    with pytest.raises(KeyboardInterrupt):
        run_guarded([sys.executable, "-c", body], tmp_path / "tool.log", 60, grace_seconds=2)
    assert is_dead(int(pid_file.read_text()))


def test_peak_memory_of_the_tool_is_sampled(tmp_path):
    hold = "import time; block = bytearray(200 * 1024 * 1024); block[::4096] = b'x' * len(block[::4096]); time.sleep(2.5)"
    result = run_guarded([sys.executable, "-c", hold], tmp_path / "log.txt", timeout_seconds=30)
    assert result.returncode == 0
    assert result.peak_group_rss_mib >= 190  # the 200 MiB block, touched page by page
    assert "peak_group_rss_mib" in result.to_dict()


def test_shared_memory_of_forked_workers_is_not_double_counted(tmp_path):
    # A parent touches 300 MiB, then forks 3 children that only sleep: the pages are shared, so the
    # group's true footprint stays near 300 MiB. Summing RSS would report about 1200 MiB.
    hold = textwrap.dedent("""
        import os, time
        block = bytearray(300 * 1024 * 1024); block[::4096] = b'x' * len(block[::4096])
        children = [os.fork() for _ in range(1)]
        if 0 not in children:
            for _ in range(2):
                if os.fork() == 0:
                    time.sleep(3); os._exit(0)
        time.sleep(3)
    """)
    result = run_guarded([sys.executable, "-c", hold], tmp_path / "log.txt", timeout_seconds=30)
    assert 250 <= result.peak_group_rss_mib <= 700


def test_default_timeout_is_finite_and_above_observed_runtimes():
    assert 112.9 < kidney_roi.DEFAULT_TIMEOUT_SECONDS < 3600

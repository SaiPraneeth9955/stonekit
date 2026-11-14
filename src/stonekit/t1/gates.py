"""Gate checks for the T1 runner (docs/T1_RUNNER_DESIGN.md §1). Every check is pure and returns a GateResult; the
runner proceeds only if ALL pass. These checks read hashes, git state and small JSON/CSV files; they never load an
image and never read a reserved scan.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

SPLIT_SHA = "466ec9c86ae905326374d41d45d17f334e97a33e6e1ead993c533968d6c7e8cb"
FROZEN_HEADING = re.compile(r"PROTOCOL FROZEN.*?SHA-256[^`]*`([0-9a-f]{64})`", re.DOTALL)


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_protocol_sha(decisions_md: Path) -> str | None:
    """The SHA-256 logged in docs/decisions.md under an entry containing 'PROTOCOL FROZEN' (last one wins)."""
    hits = FROZEN_HEADING.findall(Path(decisions_md).read_text(encoding="utf-8"))
    return hits[-1] if hits else None


def gate_protocol(protocol: Path, decisions_md: Path) -> GateResult:
    logged = frozen_protocol_sha(decisions_md)
    if logged is None:
        return GateResult("protocol_frozen", False, "no 'PROTOCOL FROZEN' entry with a SHA-256 in the decisions log")
    actual = sha256(protocol)
    return GateResult("protocol_frozen", actual == logged, f"logged {logged[:12]}…, file {actual[:12]}…")


def gate_git_clean(repo: Path, expected_commit: str) -> GateResult:
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=False).stdout
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    ok = status.strip() == "" and head == expected_commit
    return GateResult("git_clean_at_frozen_commit", ok, f"head {head[:12]}, expected {expected_commit[:12]}, uncommitted paths {len(status.splitlines())}")


def gate_unlock(unlock_file: Path, protocol_sha: str, commit: str) -> GateResult:
    if not Path(unlock_file).exists():
        return GateResult("owner_unlock", False, "owner unlock file absent")
    u = json.loads(Path(unlock_file).read_text(encoding="utf-8"))
    ok = u.get("protocol_sha256") == protocol_sha and u.get("commit") == commit and u.get("owner_confirmation") == "I unlock the 35 calibration and 53 test scans for the single T1 run"
    return GateResult("owner_unlock", ok, "matches frozen protocol and commit" if ok else "unlock file does not match the frozen protocol, commit or confirmation text")


def gate_file_sha(name: str, path: Path, expected: str) -> GateResult:
    if not Path(path).exists():
        return GateResult(name, False, f"{path} missing")
    actual = sha256(path)
    return GateResult(name, actual == expected, f"expected {expected[:12]}…, got {actual[:12]}…")


def gate_split(split_csv: Path) -> GateResult:
    return gate_file_sha("split_unchanged", split_csv, SPLIT_SHA)


def gate_single_run(runs_root: Path) -> GateResult:
    """Refuse if a T1 run COMPLETED (marker in the run folder or its analysis/ subfolder) or if an earlier attempt
    had started on the test scans (test_work/ exists). An attempt that crashed during calibration only, before any
    test scan was touched, does not block a retry."""
    done = [p for p in Path(runs_root).glob("T1_*") if (p / "COMPLETED").exists() or (p / "analysis" / "COMPLETED").exists()]
    if done:
        return GateResult("single_run", False, f"completed T1 run exists: {done[0].name}")
    touched = [p for p in Path(runs_root).glob("T1_*") if (p / "test_work").exists()]
    if touched:
        return GateResult("single_run", False, f"an earlier T1 attempt reached the test scans: {touched[0].name}; owner decision required")
    return GateResult("single_run", True, "no completed T1 run and no attempt that reached the test scans")


def all_passed(results: list[GateResult]) -> bool:
    return bool(results) and all(r.passed for r in results)

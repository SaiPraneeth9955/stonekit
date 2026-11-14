"""T1 gate refusal tests. Uses temporary fixture files only; no data, no reserved scan."""

import json
import subprocess
from pathlib import Path

from stonekit.t1 import gates


def test_protocol_gate_requires_logged_frozen_sha(tmp_path: Path):
    proto = tmp_path / "protocol.md"
    proto.write_text("draft", encoding="utf-8")
    dec = tmp_path / "decisions.md"
    dec.write_text("## some entry\nno freeze here\n", encoding="utf-8")
    assert not gates.gate_protocol(proto, dec).passed
    dec.write_text(f"## PROTOCOL FROZEN\nSHA-256 `{gates.sha256(proto)}`\n", encoding="utf-8")
    assert gates.gate_protocol(proto, dec).passed
    proto.write_text("draft edited after freeze", encoding="utf-8")
    assert not gates.gate_protocol(proto, dec).passed


def test_unlock_gate_needs_exact_match(tmp_path: Path):
    f = tmp_path / "unlock.json"
    assert not gates.gate_unlock(f, "a" * 64, "b" * 40).passed
    ok = {"protocol_sha256": "a" * 64, "commit": "b" * 40, "owner_confirmation": "I unlock the 35 calibration and 53 test scans for the single T1 run"}
    f.write_text(json.dumps(ok), encoding="utf-8")
    assert gates.gate_unlock(f, "a" * 64, "b" * 40).passed
    f.write_text(json.dumps({**ok, "commit": "c" * 40}), encoding="utf-8")
    assert not gates.gate_unlock(f, "a" * 64, "b" * 40).passed


def test_split_and_file_gates(tmp_path: Path):
    f = tmp_path / "x.csv"
    f.write_text("case\n", encoding="utf-8")
    assert not gates.gate_split(f).passed
    assert gates.gate_file_sha("x", f, gates.sha256(f)).passed
    assert not gates.gate_file_sha("x", tmp_path / "missing", "0" * 64).passed


def test_single_run_gate(tmp_path: Path):
    assert gates.gate_single_run(tmp_path).passed
    (tmp_path / "T1_20270101T000000Z").mkdir()
    assert gates.gate_single_run(tmp_path).passed  # a crashed run without COMPLETED does not block
    (tmp_path / "T1_20270101T000000Z" / "COMPLETED").write_text("", encoding="utf-8")
    assert not gates.gate_single_run(tmp_path).passed
    nested = tmp_path / "n"
    (nested / "T1_20270102T000000Z" / "analysis").mkdir(parents=True)
    (nested / "T1_20270102T000000Z" / "analysis" / "COMPLETED").write_text("", encoding="utf-8")
    assert not gates.gate_single_run(nested).passed  # the analysis stage writes its marker in analysis/
    touched = tmp_path / "t"
    (touched / "T1_20270103T000000Z" / "test_work").mkdir(parents=True)
    assert not gates.gate_single_run(touched).passed  # an attempt that reached the test scans blocks a rerun


def test_git_gate_refuses_dirty_or_wrong_commit(tmp_path: Path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert gates.gate_git_clean(tmp_path, head).passed
    assert not gates.gate_git_clean(tmp_path, "0" * 40).passed
    (tmp_path / "dirty.txt").write_text("x", encoding="utf-8")
    assert not gates.gate_git_clean(tmp_path, head).passed
    assert not gates.all_passed([])

"""Tests for scripts/checkpoint_state.py, the E-CV launcher's fail-safe fold classifier."""

import subprocess
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "checkpoint_state.py"


def run(folder: Path, epochs: int = 100) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(SCRIPT), str(folder), "--epochs", str(epochs)], capture_output=True, text=True, check=False)
    return r.returncode, r.stdout.strip()


def save(path: Path, epoch: int, value: float = 1.0) -> None:
    torch.save({"network_weights": {"w": torch.full((2, 2), value)}, "current_epoch": epoch}, path)


def test_fresh_when_empty(tmp_path):
    assert run(tmp_path) == (0, "fresh")


def test_complete_and_resume(tmp_path):
    save(tmp_path / "checkpoint_latest.pth", 45)
    assert run(tmp_path) == (0, "resume")
    save(tmp_path / "checkpoint_final.pth", 100)
    assert run(tmp_path) == (0, "complete")


def test_unsafe_states_stop(tmp_path):
    save(tmp_path / "checkpoint_final.pth", 60)  # a "final" checkpoint that is not at the last epoch
    assert run(tmp_path)[0] == 2
    (tmp_path / "checkpoint_final.pth").unlink()
    save(tmp_path / "checkpoint_latest.pth", 50, float("nan"))  # non-finite weights
    assert run(tmp_path)[0] == 2
    (tmp_path / "checkpoint_latest.pth").write_bytes(b"truncated")  # unreadable
    assert run(tmp_path)[0] == 2

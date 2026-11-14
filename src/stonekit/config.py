"""Load YAML, set seeds, create run folders."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def create_run_folder(runs_folder: Path, experiment_id: str) -> Path:
    """Create a new timestamped folder. An existing folder is never reused."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = runs_folder / f"{experiment_id}_{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    return folder

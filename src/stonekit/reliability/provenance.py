"""Checks that the frozen design, A0 outputs and reused code are what the run says they are."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The frozen pipeline's files must match their hashes in A0's manifest.
REUSED_SOURCES = (
    "src/stonekit/data/freiburg.py",
    "src/stonekit/measure/components.py",
    "src/stonekit/measure/density.py",
    "src/stonekit/measure/diameter.py",
    "src/stonekit/measure/volume.py",
    "src/stonekit/seg/threshold.py",
)
EXACT_COLUMNS = ("mask_voxels", "n_components", "filled_voxels")
TOLERANT_COLUMNS = ("volume_A", "volume_B", "d_paper_A", "d_paper_B")
RELATIVE_TOLERANCE = 1e-9


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folder_hashes(folder: Path) -> dict[str, str]:
    """SHA-256 of every file in a folder, keyed by relative path."""
    return {
        str(path.relative_to(folder)): file_sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def load_manifest(a0_folder: Path) -> dict:
    return json.loads((a0_folder / "run_manifest.json").read_text(encoding="utf-8"))


def changed_reused_sources(manifest: dict, code_root: Path) -> list[str]:
    """Reused pipeline files that differ from A0's recorded hashes."""
    recorded = manifest["source_hashes"]
    return [
        relative
        for relative in REUSED_SOURCES
        if recorded.get(relative) != file_sha256(code_root / relative)
    ]


def changed_settings(manifest: dict, settings: dict) -> list[str]:
    """Settings whose value differs from the a0 run's config."""
    recorded = manifest["config"]
    return [name for name, value in settings.items() if recorded.get(name) != value]


def reconciliation_mismatches(saved: pd.DataFrame, recomputed: pd.DataFrame) -> list[str]:
    """Compare recomputed 272 HU values with saved A0 values. An empty list means they match."""
    problems = []
    if sorted(saved.stone_id) != sorted(recomputed.stone_id):
        return ["stone IDs differ"]
    joined = saved.merge(recomputed, on="stone_id", suffixes=("_saved", "_new")).set_index(
        "stone_id"
    )
    for column in EXACT_COLUMNS:
        different = joined[f"{column}_saved"].astype(int) != joined[f"{column}_new"].astype(int)
        problems += [f"{column}: stone {stone}" for stone in joined.index[different]]
    for column in TOLERANT_COLUMNS:
        close = np.isclose(
            joined[f"{column}_saved"].to_numpy(float),
            joined[f"{column}_new"].to_numpy(float),
            rtol=RELATIVE_TOLERANCE,
            atol=0.0,
        )
        problems += [f"{column}: stone {stone}" for stone in joined.index[~close]]
    return problems

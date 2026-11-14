"""Write kidney ROI crops in nnU-Net v2 raw format, from development scans only."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from stonekit.seg.kidney_roi import Box

STONE_LABELS = {"background": 0, "stone": 1}


def crop_affine(affine: np.ndarray, box: Box) -> np.ndarray:
    """Affine of a crop: the same voxels stay at the same world positions."""
    shifted = affine.copy()
    shifted[:3, 3] = (affine @ np.array([*box.start, 1.0]))[:3]
    return shifted


def save_crop(array: np.ndarray, affine: np.ndarray, box: Box, path: Path, dtype) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    crop = np.asarray(array[box.slices], dtype=dtype)
    nib.save(nib.Nifti1Image(crop, crop_affine(affine, box)), path)


def dataset_json(num_training: int) -> dict:
    return {
        "channel_names": {"0": "CT"},
        "labels": STONE_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": "Li 2022 development crops. Restricted data: do not share. Development only.",
    }


def write_dataset_json(folder: Path, num_training: int) -> None:
    (folder / "dataset.json").write_text(json.dumps(dataset_json(num_training), indent=2), encoding="utf-8")


def stratified_validation(
    rows: list[dict[str, str]], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split case rows into training and validation, within spacing and volume strata."""
    rng = np.random.default_rng(seed)
    strata: dict[tuple[str, str], list[str]] = {}
    for row in sorted(rows, key=lambda r: r["case"]):
        strata.setdefault((row["spacing_group"], row["volume_group"]), []).append(row["case"])
    train, validation = [], []
    for key in sorted(strata):
        members = strata[key]
        order = rng.permutation(len(members))
        count = round(len(members) * fraction)
        for position, index in enumerate(order):
            (validation if position < count else train).append(members[int(index)])
    return sorted(train), sorted(validation)

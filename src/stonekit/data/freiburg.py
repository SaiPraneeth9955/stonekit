"""Loader + reference matching for CT-Uroliths-Freiburg (D1).

What Stage 1 found in the published files (docs/STAGE1_FREIBURG_FEASIBILITY.md):
- The MATLAB patches already isolate each stone, so no localisation step is needed here.
- The MATLAB file has no stone ID. The link to the CSV is a unique match on rounded diameter
  and volume, and it is checked rather than assumed.
- `voxsiz` and `voxvol_mL` disagree for 21 stones. Both are kept unchanged, and neither is
  treated as correct. Physical measurements must state which one they use.
- There are no per-rater values and no species field, so this table has neither.

This module reads the source files and never changes them. It measures nothing.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.io import loadmat

CSV_NAME = "ct-uroliths_table.csv"
MAT_NAME = "ct-uroliths_matlab.mat"
EXPECTED_STONES = 101
SPACING_DISAGREEMENT = 1e-3
ROUND_DECIMALS = 2
# Spelling differences in the CSV. The original text stays in stone_type_raw.
STONE_TYPE_SPELLING = {
    "Cystine": "cystine",
    "cacliumoxalate-monohydrate": "calciumoxalate-monohydrate",
}


class FreiburgDataError(ValueError):
    """The source files do not have the structure this loader expects."""


@dataclass(frozen=True)
class MatStone:
    mat_index: int
    diameter_mm: float
    volume_ml: float
    voxsiz_mm: tuple[float, float, float]
    voxvol_ml: float
    image: np.ndarray


def load_table(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [
            {
                "stone_id": int(row["ID"]),
                "csv_volume_ml": float(row["Volume"]),
                "csv_diameter_mm": float(row["Diameter"]),
                "stone_type_raw": row["StoneType"],
            }
            for row in csv.DictReader(f, delimiter=";")
        ]
    ids = [row["stone_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise FreiburgDataError("The CSV has repeated stone IDs")
    return rows


def load_mat_stones(path: Path) -> list[MatStone]:
    """Load every stone. `mat_index` is its 1-based position in the file."""
    stones = loadmat(path, squeeze_me=False, struct_as_record=False)["stones"].ravel()
    return [
        MatStone(
            mat_index=position,
            diameter_mm=float(stone.gt_diam.squeeze()),
            volume_ml=float(stone.gt_vol.squeeze()),
            voxsiz_mm=tuple(float(value) for value in np.ravel(stone.voxsiz)),
            voxvol_ml=float(stone.voxvol_mL.squeeze()),
            image=stone.imgdata,
        )
        for position, stone in enumerate(stones, start=1)
    ]


def _match_key(diameter_mm: float, volume_ml: float) -> tuple[float, float]:
    return round(diameter_mm, ROUND_DECIMALS), round(volume_ml, ROUND_DECIMALS)


def match_stones(rows: list[dict], stones: list[MatStone]) -> dict[int, MatStone]:
    """Return stone ID -> MATLAB stone, requiring a unique one-to-one match."""
    by_key: dict[tuple[float, float], list[MatStone]] = {}
    for stone in stones:
        by_key.setdefault(_match_key(stone.diameter_mm, stone.volume_ml), []).append(stone)

    matched = {}
    for row in rows:
        key = _match_key(row["csv_diameter_mm"], row["csv_volume_ml"])
        candidates = by_key.get(key, [])
        if len(candidates) != 1:
            raise FreiburgDataError(
                f"Stone ID {row['stone_id']} has {len(candidates)} MATLAB matches for {key}"
            )
        matched[row["stone_id"]] = candidates[0]
    if len(rows) != len(stones):
        raise FreiburgDataError(f"{len(rows)} CSV rows but {len(stones)} MATLAB stones")
    return matched


def _sphere_volume_ml(diameter_mm: float) -> float:
    return math.pi / 6 * diameter_mm**3 / 1000


def _table_row(row: dict, stone: MatStone) -> dict:
    image = stone.image
    stored = image > 0  # NaN compares False, so it is counted separately below.
    x_mm, y_mm, z_mm = stone.voxsiz_mm
    voxsiz_volume_ml = x_mm * y_mm * z_mm / 1000
    relative_difference = stone.voxvol_ml / voxsiz_volume_ml - 1
    raw_type = row["stone_type_raw"].strip()
    return {
        "stone_id": row["stone_id"],
        "mat_index": stone.mat_index,
        "reference_volume_ml": stone.volume_ml,
        "reference_diameter_mm": stone.diameter_mm,
        "csv_volume_ml": row["csv_volume_ml"],
        "csv_diameter_mm": row["csv_diameter_mm"],
        "stone_type_raw": row["stone_type_raw"],
        "stone_type": STONE_TYPE_SPELLING.get(raw_type, raw_type) or None,
        "voxsiz_x_mm": x_mm,
        "voxsiz_y_mm": y_mm,
        "voxsiz_z_mm": z_mm,
        "voxvol_stored_ml": stone.voxvol_ml,
        "voxvol_from_voxsiz_ml": voxsiz_volume_ml,
        "voxvol_relative_difference": relative_difference,
        "spacing_fields_disagree": abs(relative_difference) > SPACING_DISAGREEMENT,
        "implied_inplane_mm": math.sqrt(stone.voxvol_ml * 1000 / z_mm),
        "shape_x": image.shape[0],
        "shape_y": image.shape[1],
        "shape_z": image.shape[2],
        "image_dtype": str(image.dtype),
        "nan_voxels": int(np.isnan(image).sum()),
        "stored_voxels": int(stored.sum()),
        "enclosed_zero_voxels": int((ndimage.binary_fill_holes(stored) & ~stored).sum()),
        "min_stored_hu": float(image[stored].min()),
        "max_stored_hu": float(image[stored].max()),
        "volume_exceeds_diameter_sphere": stone.volume_ml > _sphere_volume_ml(stone.diameter_mm),
        "volume_below_4mm_sphere": stone.volume_ml < _sphere_volume_ml(4.0),
    }


def build_table(raw_folder: Path, expected_stones: int = EXPECTED_STONES) -> pd.DataFrame:
    """Build one row per stone from the two source files, sorted by stone ID."""
    rows = load_table(raw_folder / CSV_NAME)
    stones = load_mat_stones(raw_folder / MAT_NAME)
    if len(rows) != expected_stones:
        raise FreiburgDataError(f"Expected {expected_stones} CSV rows, found {len(rows)}")
    matched = match_stones(rows, stones)
    table = pd.DataFrame(_table_row(row, matched[row["stone_id"]]) for row in rows)
    return table.sort_values("stone_id", ignore_index=True)


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)

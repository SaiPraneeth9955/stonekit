"""Low-memory kidney step (slice-resampled input) against the normal kidney step, on one scan.

Run once per scan, so peak memory can be measured per process (wrap with /usr/bin/time -v).
With a saved normal mask (data/interim/li2022_kidney_ts/<case>_ts.nii.gz) the two are compared:
per-kidney Dice and volume, the 40 mm ROI box, and whether every annotated stone voxel stays inside
the box. Without one (wd0031, which failed the normal step) only the new result is described.
Development scans only. Prints one JSON line; the caller collects them.
"""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.seg import kidney_roi

MARGIN_MM = 40


def dice(a: np.ndarray, b: np.ndarray) -> float | None:
    total = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / total if total else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--target-mm", type=float, default=1.5)
    args = parser.parse_args()
    root = Path.cwd()
    data = LiDevelopmentData(root)
    ct_path, mask_path = data.paths(args.case)  # refuses reserved scans
    with tempfile.TemporaryDirectory(prefix="stonekit_lowmem_") as tmp:
        out = Path(tmp) / f"{args.case}_ts.nii.gz"
        start = time.perf_counter()
        result, info = kidney_roi.run_totalsegmentator_z_resampled(ct_path, out, "gpu", args.target_mm, log_path=Path(tmp) / "ts.log")
        seconds = time.perf_counter() - start
        new = load_volume(out)
    ct = load_volume(ct_path)
    if new.data.shape != ct.data.shape or not np.allclose(new.affine, ct.affine, atol=1e-3):
        raise ValueError("Low-memory labels are not on the CT grid")
    reference = load_volume(mask_path).data == 1
    row = {"case": args.case, "slices": int(ct.data.shape[2]), "slice_mm": round(ct.spacing_mm[2], 3), "seconds": round(seconds, 1),
           "resampled_slices": info["resampled_slices"], "tool_returncode": result.returncode,
           "leftover_processes": result.processes_still_alive_after_cleanup}  # fmt: skip
    voxel_ml = ct.voxel_volume_mm3 / 1000
    for name, label in (("right", kidney_roi.KIDNEY_RIGHT), ("left", kidney_roi.KIDNEY_LEFT)):
        row[f"{name}_ml_lowmem"] = round(float((new.data == label).sum()) * voxel_ml, 1)
    box_new = kidney_roi.mask_box(kidney_roi.kidney_mask(new.data), ct.spacing_mm, MARGIN_MM)
    row["box_lowmem"] = [list(box_new.start), list(box_new.stop)] if box_new else None
    inside = np.zeros(ct.data.shape, bool)
    if box_new:
        inside[box_new.slices] = True
    row["stone_voxels"] = int(reference.sum())
    row["stone_voxels_outside_box_lowmem"] = int((reference & ~inside).sum())
    saved = root / "data/interim/li2022_kidney_ts" / f"{args.case}_ts.nii.gz"
    if saved.exists():
        old = load_volume(saved)
        for name, label in (("right", kidney_roi.KIDNEY_RIGHT), ("left", kidney_roi.KIDNEY_LEFT)):
            row[f"{name}_dice"] = round(dice(old.data == label, new.data == label), 4)
            row[f"{name}_ml_normal"] = round(float((old.data == label).sum()) * voxel_ml, 1)
        box_old = kidney_roi.mask_box(kidney_roi.kidney_mask(old.data), ct.spacing_mm, MARGIN_MM)
        row["box_normal"] = [list(box_old.start), list(box_old.stop)] if box_old else None
        row["box_max_shift_mm"] = round(float(max(
            abs(a - b) * s for pair in ((box_old.start, box_new.start), (box_old.stop, box_new.stop))
            for a, b, s in zip(*pair, ct.spacing_mm, strict=True)
        )), 2) if box_old and box_new else None  # fmt: skip
    print("RESULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()

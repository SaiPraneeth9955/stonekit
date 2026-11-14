"""Per annotated stone piece: nearest distance to the TotalSegmentator kidney mask, all development scans.

The kidney-distance filter removes a predicted piece only if its NEAREST voxel is beyond the limit,
so its true-positive cost depends on annotated pieces lying wholly beyond it (for example stones
down the ureter). The 32 validation scans are too few to see this, so it is counted on every
development scan with a saved kidney mask (training and validation; descriptive, no model
involved). Annotated pieces as in the analysis: 26-connected Li label-1 components of >= 3 voxels.
Per-piece rows (with scan IDs) go to data/interim/diagnostics/; aggregates to results/runs/.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.seg import kidney_roi

LIMITS = [20, 30, 40, 50]


def main() -> None:
    root = Path.cwd()
    data = LiDevelopmentData(root)
    split = json.loads((root / "data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    rows = []
    for case in data.development_ids():
        mask_path = root / "data/interim/li2022_kidney_ts" / f"{case}_ts.nii.gz"
        if not mask_path.exists():
            continue
        li = data.load(case)
        if not li.stone_reference.any():
            continue
        kidneys = kidney_roi.kidney_mask(load_volume(mask_path).data)
        reference = separate_stones(li.stone_reference, 3)
        stones = measure_stones(reference, li.image.data, li.image.affine, li.image.spacing_mm)
        found = np.argwhere(reference.labels > 0)
        lo, hi = found.min(axis=0), found.max(axis=0) + 1
        # Distance only near the stones: a box around them plus 80 mm, so memory stays small.
        pad = [int(np.ceil(80 / s)) for s in li.image.spacing_mm]
        box = tuple(slice(max(a - p, 0), min(b + p, n)) for a, b, p, n in zip(lo, hi, pad, kidneys.shape, strict=True))
        sub = kidneys[box]
        distance = ndimage.distance_transform_edt(~sub, sampling=li.image.spacing_mm) if sub.any() else None
        for s in stones:
            inside = reference.labels[box] == s["stone_id"]
            nearest = float(distance[inside].min()) if distance is not None else None
            rows.append({"case": case, "group": "validation" if case in split["validation"] else "training" if case in split["train"] else "other",
                         "d3_mm": s["max_diameter_3d_mm"], "volume_mm3": s["volume_mm3"], "peak_hu": s["max_hu"],
                         "nearest_distance_to_kidney_mm": nearest})  # fmt: skip
        print(case, len(stones), flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    detail = root / "data/interim/diagnostics" / f"stone_distance_pieces_{stamp}"
    detail.mkdir(parents=True)
    with (detail / "pieces.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    known = [r for r in rows if r["nearest_distance_to_kidney_mm"] is not None]
    summary = {"definitions": __doc__, "pieces": len(rows), "scans": len({r["case"] for r in rows}),
               "pieces_without_kidney_mask_nearby": len(rows) - len(known), "by_limit": {}}  # fmt: skip
    for limit in LIMITS:
        beyond = [r for r in known if r["nearest_distance_to_kidney_mm"] > limit]
        summary["by_limit"][f"{limit} mm"] = {
            "pieces_wholly_beyond": len(beyond), "scans": len({r["case"] for r in beyond}),
            "of_which_validation": sum(r["group"] == "validation" for r in beyond),
            "beyond_pieces_d3_mm": sorted(round(r["d3_mm"], 1) for r in beyond),
            "beyond_volume_fraction": round(sum(r["volume_mm3"] for r in beyond) / sum(r["volume_mm3"] for r in known), 4),
        }  # fmt: skip
    run = create_run_folder(root / "results/runs", "stone_distance_pieces")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["by_limit"], indent=1))
    print("Saved to", run, "and", detail)


if __name__ == "__main__":
    main()

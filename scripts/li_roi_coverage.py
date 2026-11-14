"""How much of the Li stone labels would each ROI margin keep? Development scans only.

Compares box and dilated-mask ROIs at several margins against the annotation references. Annotations
are not physical truth. Per-scan rows stay git-ignored; the run folder holds aggregate counts.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.measure.components import CONNECTIVITY_26
from stonekit.seg import kidney_roi

MARGINS_MM = (0, 10, 20, 30, 40, 50, 75)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / total if total else float("nan")


def analyse(data: LiDevelopmentData, case_id: str, ts_path: Path) -> dict:
    case = data.load(case_id)
    labels = load_volume(ts_path)
    if labels.data.shape != case.image.data.shape:
        raise ValueError("TotalSegmentator output is not on the CT grid")
    spacing = case.image.spacing_mm
    kidneys = kidney_roi.kidney_mask(labels.data)
    stone = case.stone_reference
    aorta = labels.data == kidney_roi.AORTA
    row = {
        "case": case_id,
        "right_kidney_voxels": int((labels.data == kidney_roi.KIDNEY_RIGHT).sum()),
        "left_kidney_voxels": int((labels.data == kidney_roi.KIDNEY_LEFT).sum()),
        "kidney_dice_ts_vs_li": dice(kidneys, case.kidney_reference),
        "aorta_voxels": int(aorta.sum()),
        "aorta_median_hu": float(np.median(case.image.data[aorta])) if aorta.any() else float("nan"),
        "stone_voxels": int(stone.sum()),
        "spacing_z": spacing[2],
    }
    components, n_components = ndimage.label(stone, structure=CONNECTIVITY_26)
    row["stone_components"] = n_components
    largest_distance = float("nan")
    if kidneys.any() and stone.any():
        box = kidney_roi.mask_box(stone | kidneys, spacing, 80)
        distance = ndimage.distance_transform_edt(~kidneys[box.slices], sampling=spacing)
        crop_stone = stone[box.slices]
        largest_distance = float(distance[crop_stone].max())
        for margin in MARGINS_MM:
            inside = distance <= margin
            row[f"mask_stone_frac_{margin}"] = float(inside[crop_stone].sum() / crop_stone.sum())
    row["stone_farthest_from_kidney_mm"] = largest_distance
    for margin in MARGINS_MM:
        roi = kidney_roi.mask_box(kidneys, spacing, margin)
        if roi is None:
            row[f"box_stone_frac_{margin}"] = float("nan")
            row[f"box_components_whole_{margin}"] = 0
            continue
        row[f"box_shape_mm_{margin}"] = "x".join(
            str(round(n * s)) for n, s in zip(roi.shape, spacing, strict=True)
        )
        row[f"box_voxels_{margin}"] = int(np.prod(roi.shape))
        if not stone.any():
            continue
        inside = np.zeros(stone.shape, dtype=bool)
        inside[roi.slices] = True
        row[f"box_stone_frac_{margin}"] = float(inside[stone].sum() / stone.sum())
        whole = 0
        for index in range(1, n_components + 1):
            if inside[components == index].all():
                whole += 1
        row[f"box_components_whole_{margin}"] = whole
    return row


def summarise(rows: list[dict]) -> dict:
    stones = [r for r in rows if r.get("stone_voxels", 0) > 0 and r.get("right_kidney_voxels") is not None]
    done = [r for r in rows if "kidney_dice_ts_vs_li" in r]
    margins = {}
    for margin in MARGINS_MM:
        fracs = [r.get(f"box_stone_frac_{margin}", float("nan")) for r in stones]
        whole = sum(r.get(f"box_components_whole_{margin}", 0) for r in stones)
        total = sum(r["stone_components"] for r in stones)
        mask_fracs = [r.get(f"mask_stone_frac_{margin}", float("nan")) for r in stones]
        shapes = [r[f"box_voxels_{margin}"] for r in done if f"box_voxels_{margin}" in r]
        margins[str(margin)] = {
            "box_scans_all_label_inside": int(np.nansum(np.array(fracs) >= 1)),
            "box_scans_with_label_outside": int(np.nansum(np.array(fracs) < 1)),
            "box_components_whole": int(whole),
            "box_components_total": int(total),
            "box_median_million_voxels": round(float(np.median(shapes)) / 1e6, 1) if shapes else None,
            "box_max_million_voxels": round(float(np.max(shapes)) / 1e6, 1) if shapes else None,
            "mask_scans_all_label_inside": int(np.nansum(np.array(mask_fracs) >= 1)),
        }
    dices = [r["kidney_dice_ts_vs_li"] for r in done]
    aorta = [r["aorta_median_hu"] for r in done]
    farthest = [r["stone_farthest_from_kidney_mm"] for r in stones]
    return {
        "note": "Development scans only. Coverage is of annotation labels, which are not physical truth.",
        "scans_analysed": len(done),
        "scans_with_stone_label": len(stones),
        "scans_failed": len(rows) - len(done),
        "scans_missing_a_kidney": sum(
            1 for r in done if not r["right_kidney_voxels"] or not r["left_kidney_voxels"]
        ),
        "scans_no_kidney_found": sum(
            1 for r in done if not r["right_kidney_voxels"] and not r["left_kidney_voxels"]
        ),
        "kidney_dice_quartiles": [round(float(np.nanpercentile(dices, q)), 3) for q in (0, 25, 50, 75, 100)],
        "aorta_median_hu_percentiles_0_5_50_95_100": [
            round(float(np.nanpercentile(aorta, q)), 1) for q in (0, 5, 50, 95, 100)
        ],
        "stone_farthest_from_kidney_mm_percentiles_50_90_95_100": [
            round(float(np.nanpercentile(farthest, q)), 1) for q in (50, 90, 95, 100)
        ],
        "margins_mm": margins,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    data = LiDevelopmentData(args.root)
    ts_folder = args.root / "data" / "interim" / "li2022_kidney_ts"
    rows = []
    for case_id in data.development_ids():
        ts_path = ts_folder / f"{case_id}_ts.nii.gz"
        if not ts_path.exists():
            rows.append({"case": case_id, "failure": "no TotalSegmentator output"})
            continue
        try:
            rows.append(analyse(data, case_id, ts_path))
        except (OSError, EOFError, ValueError, MemoryError) as error:
            rows.append({"case": case_id, "failure": type(error).__name__})
        print(case_id, flush=True)
    summary = summarise(rows)
    run = create_run_folder(args.root / "results" / "runs", "li_roi_coverage")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    detail = ts_folder / f"coverage_cases_{stamp}.csv"
    fields = sorted({key for row in rows for key in row})
    with detail.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print("Saved to", run, "and", detail)


if __name__ == "__main__":
    main()

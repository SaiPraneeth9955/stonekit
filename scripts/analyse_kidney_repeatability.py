"""Does TotalSegmentator's run-to-run variation change the final StoneKit measurement?

Compares, for each fold-0 validation scan, the saved kidney mask and original 100-epoch run
(`--original`, frozen baseline config; the 30 mm filter is applied here exactly as the system
pipeline applies it) with a repeat kidney mask and a system-config run on it
(`--repeat-masks`, `--repeat-runs`, from scripts/kidney_repeatability.py).

Levels: kidney (Dice per kidney, volume change), ROI box, final stone label map (voxel identity
after the filter), and per-stone volume, 3D diameter, distance to kidney and size category.
Size categories (<5, 5-<10, 10-<20, >=20 mm) are research bins. Development scans only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder
from stonekit.io.nifti import load_volume
from stonekit.seg import kidney_roi

FILTER_MM = 30.0
EDGES = [0, 5, 10, 20, np.inf]


def category(d: float) -> int:
    return int(np.searchsorted(EDGES, d, side="right")) - 1


def dice(a, b):
    total = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / total if total else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--repeat-masks", type=Path, required=True)
    parser.add_argument("--repeat-runs", type=Path, required=True)
    args = parser.parse_args()
    originals = {json.loads((p / "report.json").read_text())["case_id"]: p for p in args.original.iterdir() if (p / "report.json").exists()}
    repeats = {json.loads((p / "report.json").read_text())["case_id"]: p for p in args.repeat_runs.iterdir() if (p / "report.json").exists()}
    rows, stone_rows = [], []
    for case in sorted(originals):
        if case not in repeats:
            rows.append({"case": case, "repeat": "missing"})
            continue
        old_dir, new_dir = originals[case], repeats[case]
        old_k = load_volume(old_dir / "kidney_totalsegmentator.nii.gz").data
        new_k = load_volume(args.repeat_masks / f"{case}_ts.nii.gz").data
        a = json.loads((old_dir / "report.json").read_text())
        b = json.loads((new_dir / "report.json").read_text())
        row = {"case": case, "voxels_differing": int((old_k != new_k).sum()), "total_voxels": int(old_k.size)}
        for name, label in (("right", kidney_roi.KIDNEY_RIGHT), ("left", kidney_roi.KIDNEY_LEFT)):
            o, n = old_k == label, new_k == label
            row[f"{name}_dice"] = dice(o, n)
            row[f"{name}_volume_change_pct"] = 100 * (int(n.sum()) - int(o.sum())) / int(o.sum()) if o.any() else None
        row["roi_identical"] = a["roi"]["box_start_voxel"] == b["roi"]["box_start_voxel"] and a["roi"]["box_stop_voxel"] == b["roi"]["box_stop_voxel"]
        old_stones = {s["stone_id"]: s for s in a["methods"]["model"]["stones"] if s["distance_to_kidney_mm"] <= FILTER_MM}
        new_stones = {s["stone_id"]: s for s in b["methods"]["model"]["stones"]}
        old_labels = load_volume(old_dir / "stone_labels_model.nii.gz").data
        old_labels = np.where(np.isin(old_labels, list(old_stones)), old_labels, 0)
        new_labels = load_volume(new_dir / "stone_labels_model.nii.gz").data
        row["final_labels_identical"] = bool(np.array_equal(old_labels > 0, new_labels > 0))
        row["stones_original"], row["stones_repeat"] = len(old_stones), len(new_stones)
        row["filter_removed_original"] = sum(1 for s in a["methods"]["model"]["stones"] if s["distance_to_kidney_mm"] > FILTER_MM)
        row["filter_removed_repeat"] = len(b["methods"]["model"].get("kidney_distance_filter", {}).get("removed_pieces", []))
        # Pair stones by their voxel overlap in the final maps.
        for i, s in old_stones.items():
            overlap = new_labels[old_labels == i]
            ids, counts = np.unique(overlap[overlap > 0], return_counts=True)
            if not len(ids):
                stone_rows.append({"case": case, "stone": i, "status": "lost in repeat"})
                continue
            t = new_stones[int(ids[np.argmax(counts)])]
            stone_rows.append({
                "case": case, "stone": i, "status": "matched",
                "volume_change_mm3": t["volume_mm3"] - s["volume_mm3"],
                "volume_change_pct": 100 * (t["volume_mm3"] - s["volume_mm3"]) / s["volume_mm3"],
                "diameter_change_mm": t["max_diameter_3d_mm"] - s["max_diameter_3d_mm"],
                "distance_change_mm": t["distance_to_kidney_mm"] - s["distance_to_kidney_mm"],
                "category_changed": category(t["max_diameter_3d_mm"]) != category(s["max_diameter_3d_mm"]),
            })  # fmt: skip
        matched_new = {int(v) for v in np.unique(new_labels[old_labels > 0]) if v}
        for j in set(new_stones) - matched_new:
            stone_rows.append({"case": case, "stone": j, "status": "new in repeat"})
        rows.append(row)
    done = [r for r in rows if "voxels_differing" in r]
    matched = [r for r in stone_rows if r["status"] == "matched"]

    def spread(key):
        v = np.array([r[key] for r in done if r.get(key) is not None])
        return {"min": round(float(v.min()), 5), "median": round(float(np.median(v)), 5), "max": round(float(v.max()), 5)} if len(v) else None

    def stone_spread(key):
        v = np.abs(np.array([r[key] for r in matched]))
        return {"max_abs": round(float(v.max()), 4), "mean_abs": round(float(v.mean()), 4)} if len(v) else None

    summary = {
        "status": "DEVELOPMENT_REPEATABILITY_STUDY", "definitions": __doc__, "scans": len(rows), "repeated": len(done),
        "scans_with_any_kidney_voxel_difference": sum(r["voxels_differing"] > 0 for r in done),
        "kidney_voxels_differing": spread("voxels_differing"),
        "right_kidney_dice": spread("right_dice"), "left_kidney_dice": spread("left_dice"),
        "right_volume_change_pct": spread("right_volume_change_pct"), "left_volume_change_pct": spread("left_volume_change_pct"),
        "roi_box_identical": sum(r["roi_identical"] for r in done),
        "final_stone_labels_identical": sum(r["final_labels_identical"] for r in done),
        "filter_removed_original_vs_repeat": [sum(r["filter_removed_original"] for r in done), sum(r["filter_removed_repeat"] for r in done)],
        "stones": {"original": sum(r["stones_original"] for r in done), "repeat": sum(r["stones_repeat"] for r in done),
                   "matched": len(matched), "lost_in_repeat": sum(r["status"] == "lost in repeat" for r in stone_rows),
                   "new_in_repeat": sum(r["status"] == "new in repeat" for r in stone_rows)},
        "matched_stone_changes": {"volume_pct": stone_spread("volume_change_pct"), "volume_mm3": stone_spread("volume_change_mm3"),
                                  "diameter_mm": stone_spread("diameter_change_mm"), "distance_to_kidney_mm": stone_spread("distance_change_mm"),
                                  "size_category_changes": sum(r["category_changed"] for r in matched)},
        "per_scan_rows": str(args.repeat_masks / "per_scan_comparison.json"),
    }  # fmt: skip
    # Per-scan rows name scans, so they stay in the git-ignored data folder.
    (args.repeat_masks / "per_scan_comparison.json").write_text(json.dumps({"scans": rows, "stones": stone_rows}, indent=2), encoding="utf-8")
    run = create_run_folder(Path("results/runs"), "kidney_repeatability")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "definitions"}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

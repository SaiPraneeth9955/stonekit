"""How far real stone labels lie from the TotalSegmentator kidney mask, on all development scans.

Reads the saved per-scan ROI-coverage rows (scripts/li_roi_coverage.py, 5 November 2026:
data/interim/li2022_kidney_ts/coverage_cases_20261105T154719Z.csv). Reads no image. Gives, for each
candidate kidney-distance limit, how many development scans would lose some annotated stone voxels
and how much stone volume lies beyond it: the true-positive cost of the filter that the 32
validation scans cannot measure (none of them has a label beyond 30 mm). Also summarises
TotalSegmentator's kidney agreement with the Li kidney labels (label 2) on the same scans.
Development scans only; aggregates only.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

DISTANCES = [0, 10, 20, 30, 40, 50, 75]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=Path("data/interim/li2022_kidney_ts/coverage_cases_20261105T154719Z.csv"))
    args = parser.parse_args()
    split = json.loads(Path("data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    with args.coverage.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if not r["failure"]]
    stone = [r for r in rows if int(r["stone_voxels"]) > 0]
    groups = {"all development": stone,
              "fold-0 training": [r for r in stone if r["case"] in split["train"]],
              "fold-0 validation": [r for r in stone if r["case"] in split["validation"]]}  # fmt: skip
    out = {"source": str(args.coverage), "definitions": __doc__, "scans_with_kidney_output": len(rows), "by_limit": {}}
    for name, group in groups.items():
        out["by_limit"][name] = {
            "scans_with_stone_labels": len(group),
            **{f"{d} mm": {
                "scans_losing_any_stone_voxel": sum(float(r[f"mask_stone_frac_{d}"]) < 1 for r in group),
                "stone_voxel_fraction_beyond": round(1 - sum(float(r[f"mask_stone_frac_{d}"]) * int(r["stone_voxels"]) for r in group)
                                                     / sum(int(r["stone_voxels"]) for r in group), 4),
            } for d in DISTANCES},
            "farthest_label_mm_percentiles_50_90_95_100": [round(float(np.percentile([float(r["stone_farthest_from_kidney_mm"]) for r in group], q)), 1)
                                                            for q in (50, 90, 95, 100)],
        }  # fmt: skip
    dice = np.array([float(r["kidney_dice_ts_vs_li"]) for r in rows if r["kidney_dice_ts_vs_li"]])
    out["kidney_dice_totalsegmentator_vs_li_label_2"] = {
        "scans": len(dice), "median": round(float(np.median(dice)), 4), "q1": round(float(np.percentile(dice, 25)), 4),
        "min": round(float(dice.min()), 4), "below_0.8": int((dice < 0.8).sum()),
        "note": "Both kidneys together. Li label 2 is an annotation reference; TotalSegmentator ran at 3 mm (--fast).",
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "filter_retention")
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["by_limit"], indent=1))
    print(json.dumps(out["kidney_dice_totalsegmentator_vs_li_label_2"]))
    print("Saved to", run)


if __name__ == "__main__":
    main()

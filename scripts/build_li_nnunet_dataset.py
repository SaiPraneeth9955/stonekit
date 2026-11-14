"""Write kidney-ROI crops of training-eligible development scans in nnU-Net raw format.

Uses the guarded loader, so no reserved scan can be written. Original files are only read. The
validation split and crop outcomes are saved in git-ignored files, with a hash in the run folder.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development, unreviewed_flagged_cases
from stonekit.io.nifti import load_volume
from stonekit.seg import kidney_roi, nnunet_convert


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root
    config = load_yaml(root / "configs" / "li_stone_baseline.yaml")
    pipeline = load_yaml(root / "configs" / "baseline_pipeline.yaml")
    margin = pipeline["roi"]["box_margin_mm"]
    data = LiDevelopmentData(root)
    eligible = set(require_development(data.training_ids(), data.manifest))
    with (root / load_yaml(root / "configs" / "li_working_split.yaml")["manifest"]).open(
        newline="", encoding="utf-8"
    ) as f:
        rows = [row for row in csv.DictReader(f) if row["case"] in eligible]

    forced = []
    if config["reviewed_flagged_cases_go_to_validation"]:
        with data.review_sheet.open(newline="", encoding="utf-8") as f:
            flagged = {row["case"] for row in csv.DictReader(f)}
        forced = sorted((flagged - unreviewed_flagged_cases(data.review_sheet)) & eligible)
    train, validation = nnunet_convert.stratified_validation(
        [row for row in rows if row["case"] not in forced], config["validation_fraction"], config["seed"]
    )
    validation = sorted([*validation, *forced])

    folder = root / "data" / "nnunet" / "nnUNet_raw" / config["dataset_name"]
    outcomes = []
    usable_train, usable_validation = [], []
    for case_id in [*train, *validation]:
        ts_path = root / "data" / "interim" / "li2022_kidney_ts" / f"{case_id}_ts.nii.gz"
        if not ts_path.exists():
            outcomes.append({"case": case_id, "written": False, "reason": "no kidney segmentation"})
            continue
        case = data.load(case_id)
        labels = load_volume(ts_path)
        box = kidney_roi.mask_box(kidney_roi.kidney_mask(labels.data), case.image.spacing_mm, margin)
        stone = case.stone_reference
        if box is None:
            outcomes.append({"case": case_id, "written": False, "reason": "no kidney found"})
            continue
        nnunet_convert.save_crop(
            case.image.data, case.image.affine, box, folder / "imagesTr" / f"{case_id}_0000.nii.gz", np.float32
        )
        nnunet_convert.save_crop(stone, case.image.affine, box, folder / "labelsTr" / f"{case_id}.nii.gz", np.uint8)
        outcomes.append(
            {
                "case": case_id,
                "written": True,
                "stone_voxels": int(stone.sum()),
                "stone_voxels_outside_crop": int(stone.sum() - stone[box.slices].sum()),
                "crop_shape": list(box.shape),
            }
        )
        (usable_train if case_id in train else usable_validation).append(case_id)
        print(case_id, flush=True)
    nnunet_convert.write_dataset_json(folder, len(usable_train) + len(usable_validation))

    split_path = root / "data" / "nnunet" / "li_stone_split.json"
    split = {
        "status": "DEVELOPMENT_ONLY_NOT_FINAL",
        "seed": config["seed"],
        "manifest_sha256": data.manifest and json.loads(
            (root / "data" / "processed" / "li2022_working_split_v1.json").read_text(encoding="utf-8")
        )["manifest_sha256"],
        "box_margin_mm": margin,
        "train": usable_train,
        "validation": usable_validation,
        "outcomes": outcomes,
    }
    with split_path.open("x", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
    written = [o for o in outcomes if o["written"]]
    summary = {
        "train": len(usable_train),
        "validation": len(usable_validation),
        "not_written": len(outcomes) - len(written),
        "box_margin_mm": margin,
        "stone_voxels_total": sum(o["stone_voxels"] for o in written),
        "stone_voxels_outside_crop": sum(o["stone_voxels_outside_crop"] for o in written),
        "scans_losing_stone_voxels": sum(o["stone_voxels_outside_crop"] > 0 for o in written),
        "largest_crop_voxels": max(int(np.prod(o["crop_shape"])) for o in written),
        "split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
    }
    run = create_run_folder(root / "results" / "runs", "li_nnunet_dataset")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved to", run)


if __name__ == "__main__":
    main()

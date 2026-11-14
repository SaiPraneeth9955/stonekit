"""Check that every working-cohort scan can be opened, from file headers only.

Reads NIfTI headers, not image contents, and changes nothing. Writes only aggregate counts.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import nibabel as nib

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li_split import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    manifest_path = Path(config["manifest"])
    with manifest_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    problems: Counter[str] = Counter()
    orientation: Counter[str] = Counter()
    for row in rows:
        folder = Path(f"data/interim/li2022_part{row['part']}")
        image_path = folder / f"{row['case']}_ps_image.nii.gz"
        mask_path = folder / f"{row['case']}_ps_mask.nii.gz"
        if not image_path.exists() or not mask_path.exists():
            problems["missing_file"] += 1
            continue
        image = nib.load(image_path)
        mask = nib.load(mask_path)
        if image.shape != mask.shape:
            problems["shape_mismatch"] += 1
        if abs(image.affine - mask.affine).max() > 1e-3:
            problems["affine_mismatch"] += 1
        if abs(float(image.header.get_zooms()[2]) - float(row["spacing_mm"])) > 1e-3:
            problems["spacing_differs_from_audit"] += 1
        orientation["".join(nib.aff2axcodes(image.affine))] += 1

    summary = {
        "scans_checked": len(rows),
        "problems": dict(problems),
        "orientation_codes": dict(orientation),
        "manifest_sha256": sha256_file(manifest_path),
    }
    folder = create_run_folder(Path("results/runs"), "li_cohort_file_check")
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved to", folder)


if __name__ == "__main__":
    main()

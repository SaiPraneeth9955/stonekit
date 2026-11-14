"""Run the Li loader and input checks on every development scan, and compare with the audit.

Development scans only, through the guarded loader. No files are changed. The run folder holds
aggregate counts only. Per-scan rows go to a git-ignored folder, because which flagged scans are
in development must stay out of tracked files and away from the paused review.
"""

import argparse
import csv
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from nibabel.filebasedimages import ImageFileError

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.pipeline.input_checks import check_ct

ABS_TOLERANCE = {"stone_ml": 1e-3, "kidney_ml": 1e-3, "dz": 1e-3, "dx": 1e-3, "hu_min": 0.5, "hu_max": 0.5}


def check_one(data: LiDevelopmentData, case_id: str, audit: dict, limits: dict) -> dict:
    start = time.perf_counter()
    case = data.load(case_id)
    load_seconds = time.perf_counter() - start
    report = check_ct(case.image, limits)
    voxel_ml = case.image.voxel_volume_mm3 / 1000
    found = {
        "stone_ml": float(case.stone_reference.sum() * voxel_ml),
        "kidney_ml": float(case.kidney_reference.sum() * voxel_ml),
        "dz": case.image.spacing_mm[2],
        "dx": case.image.spacing_mm[0],
        "hu_min": report.hu_min,
        "hu_max": report.hu_max,
    }
    differs = [
        name for name, tolerance in ABS_TOLERANCE.items()
        if abs(found[name] - float(audit[name])) > tolerance
    ]
    if case.image.data.shape[2] != int(audit["slices"]):
        differs.append("slices")
    return {
        "case": case_id,
        "load_seconds": round(load_seconds, 2),
        "voxels_millions": round(case.image.data.size / 1e6, 1),
        "can_process": report.can_process,
        "orientation": report.original_orientation,
        "warnings": ";".join(c.name for c in report.checks if c.status == "warn"),
        "failures": ";".join(c.name for c in report.checks if c.status == "fail"),
        "differs_from_audit": ";".join(differs),
        "padding_fraction": round(report.padding_fraction, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    limits = load_yaml(args.root / "configs" / "baseline_pipeline.yaml")["input_checks"]
    data = LiDevelopmentData(args.root)
    table = args.root / "data" / "interim" / "li2022_full_audit" / "li_full_cases.csv"
    with table.open(newline="", encoding="utf-8") as f:
        audit = {row["case"]: row for row in csv.DictReader(f)}

    rows = []
    errors: Counter[str] = Counter()
    for case_id in data.development_ids():
        try:
            rows.append(check_one(data, case_id, audit[case_id], limits))
        except (OSError, EOFError, ValueError, MemoryError, ImageFileError) as error:
            # A real file problem is a finding, so the run must go on.
            errors[type(error).__name__] += 1
            rows.append({"case": case_id, "failures": f"load_error:{type(error).__name__}"})

    def tally(field: str) -> dict:
        return dict(Counter(n for r in rows for n in str(r.get(field, "")).split(";") if n))

    loaded = [r for r in rows if "load_seconds" in r]
    summary = {
        "note": "Development scans only. Engineering check, not a result about stones.",
        "scans_attempted": len(rows),
        "scans_loaded": len(loaded),
        "load_errors": dict(errors),
        "can_process": sum(bool(r.get("can_process")) for r in rows),
        "orientations": dict(Counter(r.get("orientation") for r in loaded)),
        "warnings": tally("warnings"),
        "failures": tally("failures"),
        "differs_from_audit": tally("differs_from_audit"),
        "load_seconds_median": float(np.median([r["load_seconds"] for r in loaded])),
        "load_seconds_max": max(r["load_seconds"] for r in loaded),
        "voxels_millions_max": max(r["voxels_millions"] for r in loaded),
    }
    run = create_run_folder(args.root / "results" / "runs", "li_development_loader_check")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    detail = args.root / "data" / "interim" / "li2022_loader_check" / f"cases_{stamp}.csv"
    detail.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with detail.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print("Saved to", run, "and", detail)


if __name__ == "__main__":
    main()

"""Engineering regression check: does the CURRENT measurement code reproduce the measurements that the
frozen baseline run of 6 November 2026 recorded? (Recommended before the first commit; owner decisions of
12 November 2026.) Development scans only (fold-0 validation run); no calibration or test scan.

For every scan in data/interim/stonekit_runs/validation_20261106T003707Z, the saved model label map
(stone_labels_model.nii.gz, written by the 6 November code) is re-separated (26-connected, >= 3 voxels) and
re-measured with today's `measure_stones` on the original CT and grid. The per-stone volume, 3D maximum
diameter, axial maximum diameter, mean HU and voxel count are compared with the values in that run's
report.json, matched by voxel count and centroid. Pass: every stone matched and every compared value equal
to within 1e-6 (relative). New keys added since (shape descriptors) are not compared.
Writes a new results/runs/measurement_regression_<stamp>/summary.json.
"""

import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData  # refuses any non-development scan by design
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones

RUN = Path("data/interim/stonekit_runs/validation_20261106T003707Z")
COMPARE = ["volume_mm3", "max_diameter_3d_mm", "max_diameter_axial_mm", "mean_hu", "voxel_count"]
REL_TOL = 1e-6


def close(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= REL_TOL * max(1.0, abs(float(a)), abs(float(b)))


def main() -> None:
    data = LiDevelopmentData(Path("."))
    cases, mismatches, unmatched, compared = [], [], [], 0
    for folder in sorted(p for p in RUN.iterdir() if p.is_dir() and (p / "report.json").exists()):
        report = json.loads((folder / "report.json").read_text())
        case = report["case_id"]
        old = (report.get("methods") or {}).get("model", {}).get("stones", [])
        li = data.load(case)
        labels = load_volume(folder / "stone_labels_model.nii.gz").data > 0
        new = measure_stones(separate_stones(labels, 3), li.image.data, li.image.affine, li.image.spacing_mm)
        cases.append(case)
        free = list(new)
        for o in old:
            hit = [n for n in free if n["voxel_count"] == o["voxel_count"]
                   and np.allclose(n["centroid_ras_mm"], o["centroid_ras_mm"], atol=1e-3)]  # fmt: skip
            if len(hit) != 1:
                unmatched.append({"case": case, "stone_id": o["stone_id"], "candidates": len(hit)})
                continue
            free.remove(hit[0])
            compared += 1
            for k in COMPARE:
                if k in o and not close(o[k], hit[0].get(k)):
                    mismatches.append({"case": case, "stone_id": o["stone_id"], "key": k, "old": o[k], "new": hit[0].get(k)})
        for n in free:
            unmatched.append({"case": case, "new_stone_without_old": n["stone_id"]})
    summary = {
        "status": "ENGINEERING_CHECK_DEVELOPMENT_ONLY", "source_run": str(RUN), "scans": len(cases),
        "stones_compared": compared, "keys": COMPARE, "relative_tolerance": REL_TOL,
        "mismatches": mismatches, "unmatched": unmatched,
        "passed": not mismatches and not unmatched and compared > 0,
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "measurement_regression")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("mismatches", "unmatched")}, indent=1))
    print("mismatches:", len(mismatches), "unmatched:", len(unmatched), "| first:", (mismatches + unmatched)[:3])
    print("Saved", run)


if __name__ == "__main__":
    main()

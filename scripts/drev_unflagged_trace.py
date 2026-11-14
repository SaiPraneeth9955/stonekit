"""Trace the dense structures behind model-predicted pieces in UNFLAGGED development scans whose annotated burden
is zero but whose predicted (filtered) burden is not (wd0252, wd0011 in the out-of-fold table). Read-only;
development scans only; existing out-of-fold model labels (no new inference). Same skeleton-connected rule as
scripts/drev_component_trace.py: the largest >= 130 HU 26-connected component is taken as the skeleton.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.seg import kidney_roi

ROOT = Path(".")
TABLE = ROOT / "data/interim/oof_tables_20261112T082351Z/scans.csv"
SHEET = ROOT / "data/interim/li2022_full_audit/adjudication/li_radiology_adjudication_sheet_v1.csv"


def main() -> None:
    data = LiDevelopmentData(ROOT)
    dev = set(data.development_ids())
    with open(TABLE, encoding="utf-8") as f:
        table = {r["case"]: r for r in csv.DictReader(f) if r["case"] in dev}
    with open(SHEET, encoding="utf-8") as f:
        flagged = {r["case"] for r in csv.DictReader(f)}
    cases = sorted(c for c, r in table.items() if c not in flagged and float(r["annotated_burden_mm3"]) == 0 and float(r["predicted_burden_filtered_mm3"]) > 0)
    out = {"selection": "unflagged development scans with annotated burden 0 and predicted filtered burden > 0", "cases": {}}
    for case in cases:
        li = data.load(case)
        ct, sp = li.image.data, np.asarray(li.image.spacing_mm, float)
        folder = Path(table[case]["run_folder"])
        ts = load_volume(folder / "kidney_totalsegmentator.nii.gz").data
        kid = kidney_roi.kidney_mask(ts)
        model = load_volume(folder / "stone_labels_model.nii.gz").data
        lab, n = ndimage.label(ct >= 130, structure=np.ones((3, 3, 3)))
        sizes = ndimage.sum_labels(np.ones_like(lab, dtype=np.int8), lab, index=np.arange(1, n + 1))
        skel = int(np.argmax(sizes)) + 1
        pieces = []
        for p in (int(x) for x in np.unique(model) if x):
            pm = model == p
            ks, c = np.unique(lab[pm & (lab > 0)], return_counts=True)
            k = int(ks[np.argmax(c)]) if len(ks) else 0
            comp = lab == k if k else np.zeros_like(pm)
            idx = np.argwhere(comp) if k else np.zeros((1, 3), int)
            pieces.append({"piece": p, "piece_volume_mm3": float(pm.sum() * np.prod(sp)), "piece_in_ts_kidney": float(kid[pm].mean()),
                           "host_component_skeleton_connected": bool(k == skel), "host_component_ml": float(comp.sum() * np.prod(sp) / 1000),
                           "host_extent_mm_xyz": (((idx.max(0) - idx.min(0) + 1) * sp).tolist() if k else None),
                           "host_median_hu": float(np.median(ct[comp])) if k else None, "host_share_in_ts_kidney": float(kid[comp].mean()) if k else None,
                           "host_share_in_stone_label": float((li.mask.data[comp] == 1).mean()) if k else None})  # fmt: skip
        out["cases"][case] = {"fold": table[case]["fold"], "predicted_filtered_mm3": float(table[case]["predicted_burden_filtered_mm3"]), "pieces": pieces}
        print(case, json.dumps(out["cases"][case])[:1500], flush=True)
    run = create_run_folder(Path("results/runs"), "drev_unflagged_trace")
    (run / "unflagged_trace.json").write_text(json.dumps({"purpose": __doc__, **out}, indent=1), encoding="utf-8")
    print("Saved", run)


if __name__ == "__main__":
    main()

"""Two targeted read-only checks raised by the D-REV component trace (development scans only).

1. wd0239 vs wd0252: the labelled 77 mL dense mass in wd0239 and the UNLABELLED 72 mL intrarenal dense mass in
   wd0252 have almost identical HU summaries (median 749 vs 750, max 1845 vs 1846). Compare acquisition geometry,
   whole-volume HU histograms and the two masses' HU histograms, to test whether they may be the same patient or
   a near-duplicate scan. A high histogram correlation is NOT proof of identity; it is a flag for the owner.
2. wd0144: count annotation-kidney (label 2) voxels left and right of the image mid-line.
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
BINS = np.arange(-1000, 3001, 10)


def mass(ct, ts, side_label, mask, use_label):
    lab, _ = ndimage.label(ct >= 130, structure=np.ones((3, 3, 3)))
    seed = (mask == 1) if use_label else ((ts == side_label) & (ct >= 130) & (mask != 1))
    ks, c = np.unique(lab[seed & (lab > 0)], return_counts=True)
    k = int(ks[np.argmax(c)])
    return lab == k


def main() -> None:
    data = LiDevelopmentData(ROOT)
    with open(TABLE, encoding="utf-8") as f:
        table = {r["case"]: r for r in csv.DictReader(f)}
    out = {}
    vols = {}
    for case in ("wd0239", "wd0252"):
        li = data.load(case)
        ts = load_volume(Path(table[case]["run_folder"]) / "kidney_totalsegmentator.nii.gz").data
        ct, mask = li.image.data, li.mask.data
        m = mass(ct, ts, kidney_roi.KIDNEY_RIGHT, mask, use_label=(case == "wd0239"))
        vols[case] = {"whole": np.histogram(ct, BINS)[0].astype(float), "mass": np.histogram(ct[m], BINS)[0].astype(float)}
        idx = np.argwhere(m)
        out[case] = {"shape": list(ct.shape), "spacing_mm": [float(s) for s in li.image.spacing_mm],
                     "origin_ras": [float(x) for x in li.image.affine[:3, 3]], "annotated_burden_mm3": float(table[case]["annotated_burden_mm3"]),
                     "annotated_pieces": int(table[case]["annotated_pieces"]), "mass_volume_ml": float(m.sum() * np.prod(li.image.spacing_mm) / 1000),
                     "mass_centroid_ras": (li.image.affine @ np.r_[idx.mean(0), 1.0])[:3].tolist(),
                     "mass_side_ts": {"right": float((ts[m] == kidney_roi.KIDNEY_RIGHT).mean()), "left": float((ts[m] == kidney_roi.KIDNEY_LEFT).mean())},
                     "ts_kidney_ml": {"right": float((ts == kidney_roi.KIDNEY_RIGHT).sum() * np.prod(li.image.spacing_mm) / 1000),
                                      "left": float((ts == kidney_roi.KIDNEY_LEFT).sum() * np.prod(li.image.spacing_mm) / 1000)}}  # fmt: skip
    for k in ("whole", "mass"):
        a, b = vols["wd0239"][k], vols["wd0252"][k]
        out[f"{k}_hu_histogram_correlation"] = float(np.corrcoef(a / a.sum(), b / b.sum())[0, 1])
    li = data.load("wd0144")
    k2 = np.argwhere(li.mask.data == 2)
    ras = (li.image.affine @ np.c_[k2, np.ones(len(k2))].T)[:3].T if len(k2) else np.zeros((0, 3))
    # RAS convention: +x points to the patient's right
    out["wd0144_annotation_kidney"] = {"voxels": len(k2), "patient_right_x>0_voxels": int((ras[:, 0] > 0).sum()), "patient_left_x<0_voxels": int((ras[:, 0] < 0).sum()),
                                       "stone_label_centroid_ras": (li.image.affine @ np.r_[np.argwhere(li.mask.data == 1).mean(0), 1.0])[:3].tolist()}  # fmt: skip
    run = create_run_folder(Path("results/runs"), "drev_pair_check")
    (run / "pair_check.json").write_text(json.dumps({"purpose": __doc__, **out}, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

"""Supplement to the D-REV technical audit (development scans only; read-only).

1. Model evidence at each audited location: the existing out-of-fold model labels (stone_labels_model.nii.gz, no
   new inference) within 5 mm of the point, and whether the predicted piece survives the 30 mm kidney filter
   (from report.json if recorded there).
2. Identity of unflagged development scans whose share of kidney voxels >= 130 HU outside the stone label exceeds
   0.01 on either side (the audit's reference maximum was 0.945 on the right).
3. Annotated burden of each audited scan as a percentile among all 165 development scans (out-of-fold table).
No image is rendered or transmitted. Reserved scans are excluded before loading.
"""

import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.io.nifti import load_volume
from stonekit.seg import kidney_roi

ROOT = Path(".")
SHEET = ROOT / "data/interim/li2022_full_audit/adjudication/li_radiology_adjudication_sheet_v1.csv"
TABLE = ROOT / "data/interim/oof_tables_20261112T082351Z/scans.csv"


def main() -> None:
    data = LiDevelopmentData(ROOT)
    dev = set(data.development_ids())
    with open(SHEET, encoding="utf-8") as f:
        sheet = list(csv.DictReader(f))
    with open(TABLE, encoding="utf-8") as f:
        table = {r["case"]: r for r in csv.DictReader(f) if r["case"] in dev}
    rows = [r for r in sheet if r["case"] in dev]
    flagged = {r["case"] for r in sheet}
    burden = np.array([float(r["annotated_burden_mm3"]) for r in table.values()])
    out = {"model_at_location": [], "bright_kidney_outliers": [], "burden_percentile": {}}
    for case in sorted({r["case"] for r in rows}):
        b = float(table[case]["annotated_burden_mm3"])
        out["burden_percentile"][case] = {"annotated_burden_mm3": b, "pieces": int(table[case]["annotated_pieces"]),
                                          "percentile_among_165_dev": float((burden < b).mean())}  # fmt: skip
    for r in rows:
        folder = Path(table[r["case"]]["run_folder"])
        img = data.load(r["case"]).image
        lab = load_volume(folder / "stone_labels_model.nii.gz")
        rec = {"adj_id": r["adj_id"], "case": r["case"], "model_shape_equals_image": tuple(lab.data.shape) == tuple(img.data.shape)}
        v = np.rint(np.linalg.inv(lab.affine) @ np.array([float(r["R"]), float(r["A"]), float(r["S"]), 1.0]))[:3].astype(int)
        if all(0 <= vi < n for vi, n in zip(v, lab.data.shape, strict=True)):
            rad = [int(np.ceil(5 / s)) for s in lab.spacing_mm]
            g = np.ogrid[tuple(slice(-k, k + 1) for k in rad)]
            ball = sum((gi * s) ** 2 for gi, s in zip(g, lab.spacing_mm, strict=True)) <= 25
            w = tuple(slice(max(vi - k, 0), min(vi + k + 1, n)) for vi, k, n in zip(v, rad, lab.data.shape, strict=True))
            sub = lab.data[w]
            bb = ball[tuple(slice(k - (vi - wi.start), k - (vi - wi.start) + sn) for k, vi, wi, sn in zip(rad, v, w, sub.shape, strict=True))]
            ids = sorted(int(x) for x in np.unique(sub[bb]) if x)
            rec["model_label_at_point"] = int(lab.data[tuple(v)])
            rec["model_pieces_within_5mm"] = ids
            vols = {i: float((lab.data == i).sum() * np.prod(lab.spacing_mm)) for i in ids}
            rec["model_piece_volumes_mm3"] = vols
        else:
            rec["model_label_at_point"] = "out of model-label bounds"
        rep = folder / "report.json"
        if rep.exists():
            j = json.loads(rep.read_text(encoding="utf-8"))
            rec["report_keys"] = sorted(j.keys())[:30]
        out["model_at_location"].append(rec)
    for case in sorted(dev - flagged):
        li = data.load(case)
        ts = load_volume(Path(table[case]["run_folder"]) / "kidney_totalsegmentator.nii.gz").data
        st = li.mask.data == 1
        sh = {}
        for side, k in (("right", kidney_roi.KIDNEY_RIGHT), ("left", kidney_roi.KIDNEY_LEFT)):
            m = ts == k
            sh[side] = float(((li.image.data >= 130) & m & ~st).sum() / max(m.sum(), 1))
            sh[f"{side}_kidney_voxels"] = int(m.sum())
        if max(sh["right"], sh["left"]) > 0.01:
            out["bright_kidney_outliers"].append({"case": case, **sh})
    run = create_run_folder(Path("results/runs"), "drev_audit_supplement")
    (run / "supplement.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

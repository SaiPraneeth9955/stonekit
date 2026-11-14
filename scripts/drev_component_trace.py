"""Whole-volume trace of the dense structure at each audited D-REV location (development scans only; read-only).

Rule written before the trace was run (13 November 2026), threshold-free apart from the 130 HU calcium convention:
- In an abdominal CT the skeleton is the largest >= 130 HU 26-connected component. A structure that belongs to the
  LARGEST component is "skeleton-connected" (bone, or something touching bone at this threshold).
- Otherwise it is "isolated"; its full volume, bounding-box extents and whether it reaches an image edge are
  reported (a long isolated tube spanning far craniocaudally is consistent with a tubular device, but only a human
  reader can identify it).
Also traced: the >= 130 HU voxels inside the TotalSegmentator right kidney of the unflagged scan wd0252 (94.5% of
that kidney is >= 130 HU outside the stone label), found by the audit's reference step.
No image is rendered or transmitted; reserved scans are excluded before loading.
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
SHEET = ROOT / "data/interim/li2022_full_audit/adjudication/li_radiology_adjudication_sheet_v1.csv"
TABLE = ROOT / "data/interim/oof_tables_20261112T082351Z/scans.csv"
HU = 130.0


def describe(lab, k, sizes, ct, mask, kid, spacing):
    m = lab == k
    idx = np.argwhere(m)
    lo, hi = idx.min(0), idx.max(0)
    ext = ((hi - lo + 1) * spacing).tolist()
    hu = ct[m]
    return {
        "skeleton_connected": bool(k == int(np.argmax(sizes)) + 1),
        "volume_ml": float(m.sum() * np.prod(spacing) / 1000),
        "largest_component_volume_ml": float(sizes.max() * np.prod(spacing) / 1000),
        "extent_mm_xyz": ext,
        "reaches_image_edge": bool((lo == 0).any() or (hi == np.array(ct.shape) - 1).any()),
        "median_hu": float(np.median(hu)), "p99_hu": float(np.quantile(hu, 0.99)), "max_hu": float(hu.max()),
        "share_in_stone_label": float((mask[m] == 1).mean()), "share_in_annotation_kidney": float((mask[m] == 2).mean()),
        "share_in_ts_kidney": float(kid[m].mean()),
    }  # fmt: skip


def main() -> None:
    data = LiDevelopmentData(ROOT)
    dev = set(data.development_ids())
    with open(SHEET, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["case"] in dev]
    with open(TABLE, encoding="utf-8") as f:
        table = {r["case"]: r for r in csv.DictReader(f) if r["case"] in dev}
    out = {"locations": [], "wd0252_right_kidney": None}
    for case in sorted({r["case"] for r in rows} | {"wd0252"}):
        li = data.load(case)
        ct, mask, spacing = li.image.data, li.mask.data, np.asarray(li.image.spacing_mm, float)
        ts = load_volume(Path(table[case]["run_folder"]) / "kidney_totalsegmentator.nii.gz").data
        kid = kidney_roi.kidney_mask(ts)
        lab, n = ndimage.label(ct >= HU, structure=np.ones((3, 3, 3)))
        sizes = ndimage.sum_labels(np.ones_like(lab, dtype=np.int8), lab, index=np.arange(1, n + 1))
        if case == "wd0252":
            rk = (ts == kidney_roi.KIDNEY_RIGHT) & (ct >= HU) & (mask != 1)
            ks, cnt = np.unique(lab[rk], return_counts=True)
            out["wd0252_right_kidney"] = {"bright_voxels": int(rk.sum()),
                                          "components": [{"share_of_bright_kidney_voxels": float(c / rk.sum()), **describe(lab, int(k), sizes, ct, mask, kid, spacing)}
                                                         for k, c in sorted(zip(ks, cnt, strict=True), key=lambda t: -t[1])[:3] if k]}  # fmt: skip
            if case not in {r["case"] for r in rows}:
                print("wd0252 traced", flush=True)
                continue
        for r in (x for x in rows if x["case"] == case):
            v = np.rint(np.linalg.inv(li.image.affine) @ np.array([float(r["R"]), float(r["A"]), float(r["S"]), 1.0]))[:3].astype(int)
            rad = [int(np.ceil(5 / s)) for s in spacing]
            w = tuple(slice(max(vi - k, 0), min(vi + k + 1, nn)) for vi, k, nn in zip(v, rad, ct.shape, strict=True))
            sub = lab[w]
            pts = np.argwhere(sub > 0)
            rec = {"adj_id": r["adj_id"], "case": case, "location_type": r["location_type"]}
            if len(pts):
                offset = np.array([s.start for s in w])
                d = np.sqrt((((pts + offset - v) * spacing) ** 2).sum(1))
                j = int(np.argmin(d))
                if d[j] <= 5:
                    k = int(sub[tuple(pts[j])])
                    rec.update({"distance_mm": float(d[j]), **describe(lab, k, sizes, ct, mask, kid, spacing)})
                else:
                    rec["component"] = "none within 5 mm"
            else:
                rec["component"] = "none within 5 mm"
            out["locations"].append(rec)
            print(rec["adj_id"], case, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in rec.items() if k not in ("adj_id", "case")}, flush=True)
    run = create_run_folder(Path("results/runs"), "drev_component_trace")
    (run / "trace.json").write_text(json.dumps({"rule": __doc__, **out}, indent=1), encoding="utf-8")
    print(json.dumps(out["wd0252_right_kidney"], indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

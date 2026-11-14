"""D-REV technical audit: objective, numerical evidence for the flagged locations in DEVELOPMENT scans only.

Read-only with respect to every authoritative record (screening sheet, blinded adjudication sheet, split, masks).
The guarded development loader refuses reserved (calibration/test) scans, so reserved locations are listed as
NOT AUDITED. No image is rendered or transmitted; all evidence is computed locally from voxels.

Thresholds (none tuned here): 130 HU = Agatston calcium convention; 2500 HU = the project's metal-like QC limit
(configs/system_pipeline_v1.yaml, qc.metal_like_hu). Scan-level indicators are compared with the distribution over
ALL development scans in the out-of-fold table (reference = scans not flagged by the review).

Per location: HU at the point; max HU within 3 and 5 mm; background median in a 10-15 mm shell; annotation labels
at/near the point; distance to the TotalSegmentator kidney and the annotation kidney label; the bright (>=130 HU)
26-connected component nearest the point inside an 80 mm cube: volume, HU, principal axes, elongation, whether it
reaches the cube edge (a structure larger than the cube, such as bone), overlap with the stone label and kidney.
Per scan: aorta median HU; kidney-parenchyma median HU (TotalSegmentator kidney minus stone label minus >=130 HU);
share of kidney voxels >=130 HU outside the stone label, left and right.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDataError, LiDevelopmentData
from stonekit.data.li_split import ProtectedCaseError
from stonekit.io.nifti import load_volume
from stonekit.measure.shape import principal_axes
from stonekit.seg import kidney_roi

ROOT = Path(".")
SHEET = ROOT / "data/interim/li2022_full_audit/adjudication/li_radiology_adjudication_sheet_v1.csv"  # blinded sheet: read only
TABLE = ROOT / "data/interim/oof_tables_20261112T082351Z/scans.csv"
CALCIUM_HU = 130.0
METAL_HU = 2500.0
CUBE_MM = 80.0


def ball(spacing, r_mm):
    rad = [int(np.ceil(r_mm / s)) for s in spacing]
    g = np.ogrid[tuple(slice(-k, k + 1) for k in rad)]
    return sum((gi * s) ** 2 for gi, s in zip(g, spacing, strict=True)) <= r_mm**2, rad


def window(shape, c, rad):
    return tuple(slice(max(ci - r, 0), min(ci + r + 1, n)) for ci, r, n in zip(c, rad, shape, strict=True))


def scan_metrics(ct, stone, ts):
    kid = kidney_roi.kidney_mask(ts)
    aorta = ts == kidney_roi.AORTA
    par = kid & ~stone & (ct < CALCIUM_HU)
    out = {"aorta_median_hu": float(np.median(ct[aorta])) if aorta.any() else None,
           "kidney_parenchyma_median_hu": float(np.median(ct[par])) if par.any() else None}  # fmt: skip
    for side, lab in (("right", kidney_roi.KIDNEY_RIGHT), ("left", kidney_roi.KIDNEY_LEFT)):
        k = ts == lab
        out[f"bright_kidney_share_{side}"] = float(((ct >= CALCIUM_HU) & k & ~stone).sum() / max(k.sum(), 1))
    return out


def location_metrics(ct, mask, ts, affine, spacing, ras):
    vox = np.rint(np.linalg.inv(affine) @ np.array([*ras, 1.0]))[:3].astype(int)
    inb = all(0 <= v < n for v, n in zip(vox, ct.shape, strict=True))
    out = {"voxel": vox.tolist(), "in_bounds": bool(inb)}
    if not inb:
        return out
    v = tuple(vox)
    out["hu_at_point"] = float(ct[v])
    for r in (3, 5):
        b, rad = ball(spacing, r)
        w = window(ct.shape, v, rad)
        sub = ct[w]
        bb = b[tuple(slice(rad[i] - (v[i] - w[i].start), rad[i] - (v[i] - w[i].start) + sub.shape[i]) for i in range(3))]
        out[f"max_hu_{r}mm"] = float(sub[bb].max())
        out[f"labels_within_{r}mm"] = sorted(int(x) for x in np.unique(mask[w][bb]))
    b15, rad15 = ball(spacing, 15)
    b10, _ = ball(spacing, 10)
    w = window(ct.shape, v, rad15)
    sub = ct[w]
    off = [rad15[i] - (v[i] - w[i].start) for i in range(3)]
    sl = tuple(slice(off[i], off[i] + sub.shape[i]) for i in range(3))
    inner = np.zeros_like(b15)
    pad = [(rad15[i] - (b10.shape[i] // 2)) for i in range(3)]
    inner[tuple(slice(pad[i], pad[i] + b10.shape[i]) for i in range(3))] = b10
    shell = (b15 & ~inner)[sl]
    out["background_median_hu_10_15mm"] = float(np.median(sub[shell]))
    out["annotation_label_at_point"] = int(mask[v])
    kid = kidney_roi.kidney_mask(ts)
    for name, m in (("ts_kidney", kid), ("annotation_kidney", mask == 2), ("annotation_stone", mask == 1), ("aorta", ts == kidney_roi.AORTA)):
        w60 = window(ct.shape, v, [int(60 / s) for s in spacing])
        if m[w60].any():
            d = ndimage.distance_transform_edt(~m[w60], sampling=spacing)
            idx = tuple(v[i] - w60[i].start for i in range(3))
            out[f"dist_to_{name}_mm"] = float(d[idx])
        else:
            out[f"dist_to_{name}_mm"] = "> 60"  # none within the 60 mm search box
    # bright component nearest the point, inside an 80 mm cube
    half = [int(CUBE_MM / 2 / s) for s in spacing]
    wc = window(ct.shape, v, half)
    bright = ct[wc] >= CALCIUM_HU
    lab, n = ndimage.label(bright, structure=np.ones((3, 3, 3)))
    loc = tuple(v[i] - wc[i].start for i in range(3))
    comp = None
    if n:
        d = ndimage.distance_transform_edt(lab == 0, sampling=spacing, return_indices=True)[1]
        nearest = tuple(int(d[i][loc]) for i in range(3))
        k = lab[nearest]
        dist = float(np.sqrt(sum(((nearest[i] - loc[i]) * spacing[i]) ** 2 for i in range(3))))
        if k and dist <= 5.0:
            m = lab == k
            hu = ct[wc][m]
            edges = any(m.take(0, axis=a).any() or m.take(-1, axis=a).any() for a in range(3))
            pa = principal_axes(m, spacing, 2)
            comp = {"distance_from_point_mm": dist, "voxels": int(m.sum()), "volume_ml": float(m.sum() * np.prod(spacing) / 1000),
                    "max_hu": float(hu.max()), "mean_hu": float(hu.mean()), "median_hu": float(np.median(hu)),
                    "metal_like_voxels": int((hu >= METAL_HU).sum()), "reaches_80mm_cube_edge": bool(edges),
                    "axis_major_mm": pa.get("axis_major_mm"), "axis_middle_mm": pa.get("axis_middle_mm"), "axis_minor_mm": pa.get("axis_minor_mm"),
                    "elongation": pa.get("elongation"),
                    "share_in_annotation_stone": float((mask[wc][m] == 1).mean()), "share_in_ts_kidney": float(kid[wc][m].mean()),
                    "share_in_annotation_kidney": float((mask[wc][m] == 2).mean())}  # fmt: skip
    out["bright_component"] = comp
    return out


def main() -> None:
    data = LiDevelopmentData(ROOT)
    with open(SHEET, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(TABLE, encoding="utf-8") as f:
        runs = {r["case"]: Path(r["run_folder"]) for r in csv.DictReader(f)}
    flagged = sorted({r["case"] for r in rows})
    dev_ids = set(data.development_ids())
    runs = {c: f for c, f in runs.items() if c in dev_ids}
    # scan-level reference over all development scans in the out-of-fold table
    ref = {}
    for case, folder in sorted(runs.items()):
        li = data.load(case)
        ts = load_volume(folder / "kidney_totalsegmentator.nii.gz").data
        ref[case] = scan_metrics(li.image.data, li.mask.data == 1, ts)
    unflagged = [m for c, m in ref.items() if c not in flagged]

    def pct(key, val):
        vals = np.array([m[key] for m in unflagged if m[key] is not None], float)
        return None if val is None or not len(vals) else float((vals < val).mean())

    locs = []
    for r in rows:
        rec = {"adj_id": r["adj_id"], "case": r["case"], "target": r["target"], "location_type": r["location_type"], "ras": [float(r["R"]), float(r["A"]), float(r["S"])]}
        if r["case"] not in dev_ids:  # filter BEFORE any load; the guard below is a second layer
            rec["status"] = "NOT AUDITED: reserved scan (calibration or test); excluded before loading"
            locs.append(rec)
            continue
        try:
            li = data.load(r["case"])
        except (LiDataError, ProtectedCaseError):
            rec["status"] = "NOT AUDITED: reserved scan (calibration or test); the guarded loader refuses it"
            locs.append(rec)
            continue
        ts = load_volume(runs[r["case"]] / "kidney_totalsegmentator.nii.gz").data
        rec["status"] = "AUDITED (development scan)"
        rec["scan"] = ref[r["case"]]
        rec["scan_percentiles_vs_unflagged_dev"] = {k: pct(k, v) for k, v in ref[r["case"]].items()}
        rec["location"] = location_metrics(li.image.data, li.mask.data, ts, li.image.affine, li.image.spacing_mm, rec["ras"])
        rec["spacing_mm"] = [float(s) for s in li.image.spacing_mm]
        locs.append(rec)
    refsum = {k: {"median": float(np.median([m[k] for m in unflagged if m[k] is not None])),
                  "p95": float(np.quantile([m[k] for m in unflagged if m[k] is not None], 0.95)),
                  "max": float(np.max([m[k] for m in unflagged if m[k] is not None])), "n": len([m for m in unflagged if m[k] is not None])}
              for k in unflagged[0]}  # fmt: skip
    run = create_run_folder(Path("results/runs"), "drev_technical_audit")
    (run / "audit.json").write_text(json.dumps({"definitions": __doc__, "reference_unflagged_development": refsum, "locations": locs}, indent=2, default=float), encoding="utf-8")
    print(json.dumps(refsum, indent=1))
    for x in locs:
        if not x["status"].startswith("AUDITED"):
            print(x["adj_id"], x["case"], "NOT AUDITED (reserved)")
            continue
        L, c = x["location"], x["location"].get("bright_component") or {}
        print(x["adj_id"], x["case"], x["location_type"][:9], "| aorta", round(x["scan"]["aorta_median_hu"] or -1), "par", round(x["scan"]["kidney_parenchyma_median_hu"] or -1),
              "brightK R/L", round(x["scan"]["bright_kidney_share_right"], 4), round(x["scan"]["bright_kidney_share_left"], 4),
              "| pt HU", round(L.get("hu_at_point", -1)), "max5", round(L.get("max_hu_5mm", -1)), "bg", round(L.get("background_median_hu_10_15mm", -1)),
              "lab@pt", L.get("annotation_label_at_point"), "dK", None if L.get("dist_to_ts_kidney_mm") is None else round(L["dist_to_ts_kidney_mm"], 1),
              "| comp mL", round(c.get("volume_ml", 0), 3), "maxHU", round(c.get("max_hu", 0)), "major", None if c.get("axis_major_mm") is None else round(c["axis_major_mm"], 1),
              "elong", None if c.get("elongation") is None else round(c["elongation"], 2), "edge", c.get("reaches_80mm_cube_edge"), "inStone", round(c.get("share_in_annotation_stone", 0), 2),
              "inKid", round(c.get("share_in_ts_kidney", 0), 2), flush=True)  # fmt: skip
    print("Saved", run)


if __name__ == "__main__":
    main()

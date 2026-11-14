"""Calibrated ranges for ONE processed scan, from the FROZEN locked-study calibration (research demonstration).

Given a run-case output folder and its 8-flip TTA probabilities, this module rebuilds the scan's predicted pieces
exactly as scripts/build_oof_stone_table.py does (but without any annotation), then applies the hashed T1 calibration:
- per stone: the size-stratified (Mondrian, U2) range for volume and for 3D maximum diameter, nominal 0.90;
- per scan: the detection-aware burden interval (P2-A), nominal 0.90, with the 30 mm kidney-distance filter.
Nothing is fitted or tuned here: the calibration quantities are read from the T1 run and checked against their
recorded SHA-256, and the development pools are checked against their hashes (stonekit.t1.external).
Research use only. Not a diagnostic medical device. The ranges were calibrated on one non-contrast dataset
(Li 2022); the paper's external check showed that per-stone volume ranges under-covered on another dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.seg import kidney_roi
from stonekit.t1.external import load_quantities, rebuild_burden, stone_bounds
from stonekit.uq import confirmatory as cfm

T1_RUN = Path("results/runs/T1_20261113T081425Z")
FIT_TABLE = Path("data/interim/oof_tables_20261112T082351Z")
MIN_VOXELS, FILTER_MM, ALPHA = 3, 30.0, 0.10
CONTRAST_HU = 100.0
OUTPUT = "calibrated_ranges.json"
NOTE = ("Research demonstration. Ranges use the frozen calibration of the locked study (calibrated on 35 Li 2022 "
        "non-contrast scans; nominal 90%). They are not validated for clinical use, and on an external dataset the "
        "per-stone volume ranges covered less often than nominal.")  # fmt: skip


def piece_signals(labels_crop: np.ndarray, probs: dict, voxel_mm3: float) -> dict[int, dict]:
    """Per-piece reliability signals; identical to scripts/build_oof_stone_table.piece_signals."""
    out = {}
    for pid, window in enumerate(ndimage.find_objects(labels_crop), start=1):
        if window is None:
            continue
        grown = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in window)
        piece = labels_crop[grown] == pid
        hood = ndimage.binary_dilation(piece, iterations=2)
        p = probs["p_deployed"][grown].astype(np.float32)
        flips = probs["p_flip"][(slice(None), *grown)].astype(np.float32)
        v_lo = max(int(np.count_nonzero((p > 0.3) & hood)), 1)
        v_hi = max(int(np.count_nonzero((p > 0.7) & hood)), 1)
        v_f = np.array([np.count_nonzero((f > 0.5) & hood) for f in flips], float)
        out[pid] = {"p_max": float(p[piece].max()), "p_mean": float(p[piece].mean()),
                    "threshold_sensitivity": float(np.log(v_lo / v_hi)),
                    "tta_spread": float(np.std(np.log((v_f + 1.0) * voxel_mm3)))}  # fmt: skip
    return out


def predicted_pieces(case: str, folder: Path, probabilities: Path) -> tuple[pd.DataFrame, dict]:
    """Predicted pieces of one scan (every piece, with the `beyond_30mm` filter column) and its scan row."""
    report = json.loads((folder / "report.json").read_text())
    model_img = load_volume(folder / "stone_labels_model.nii.gz")
    spacing = tuple(float(s) for s in model_img.spacing_mm)
    voxel_mm3 = float(np.prod(spacing))
    box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
    model = model_img.data > 0
    removed = folder / "stone_labels_model_removed_by_kidney_distance_filter.nii.gz"
    if removed.exists():
        model |= load_volume(removed).data > 0
    ct = np.zeros(model.shape, np.float32)
    crop = load_volume(folder / "roi_crop.nii.gz").data
    if crop.shape != model[box].shape:
        raise ValueError(f"{case}: ROI crop shape differs from the ROI box")
    ct[box] = crop
    predicted = separate_stones(model, MIN_VOXELS)
    pred_m = {s["stone_id"]: s for s in measure_stones(predicted, ct, model_img.affine, spacing)}
    kidneys = kidney_roi.kidney_mask(load_volume(folder / "kidney_totalsegmentator.nii.gz").data)
    distance = ndimage.distance_transform_edt(~kidneys[box], sampling=spacing)
    labels_box = predicted.labels[box]
    if (predicted.labels > 0).sum() != (labels_box > 0).sum():
        raise ValueError(f"{case}: predicted voxels outside the ROI box")
    for pid, window in enumerate(ndimage.find_objects(labels_box), start=1):
        if window is not None:
            pred_m[pid]["distance_to_kidney_mm"] = float(distance[window][labels_box[window] == pid].min())
    z = np.load(probabilities)
    if z["p_deployed"].shape != labels_box.shape:
        raise ValueError(f"{case}: probability map shape differs from the ROI box")
    signals = piece_signals(labels_box, {"p_deployed": z["p_deployed"], "p_flip": z["p_flip"]}, voxel_mm3)
    rows = [{"case": case, "pred_id": p, "group": "unknown", "distance_to_kidney_mm": m["distance_to_kidney_mm"],
             "beyond_30mm": m["distance_to_kidney_mm"] > FILTER_MM, "volume_mm3": m["volume_mm3"],
             "max_diameter_3d_mm": m["max_diameter_3d_mm"], "mean_hu": m["mean_hu"],
             "voxel_count": m["voxel_count"], **signals[p]} for p, m in pred_m.items()]  # fmt: skip
    scan = {"case": case, "slice_mm": spacing[2], "annotated_burden_mm3": np.nan,
            "predicted_burden_mm3": float(sum(m["volume_mm3"] for m in pred_m.values())),
            "predicted_burden_filtered_mm3": float(sum(m["volume_mm3"] for m in pred_m.values() if m["distance_to_kidney_mm"] <= FILTER_MM))}  # fmt: skip
    return pd.DataFrame(rows), scan


def ranges(pieces: pd.DataFrame, scan: dict, quantities: dict, fit: cfm.Tables) -> dict:
    """Per-stone U2 ranges and the scan's P2-A burden interval from the frozen calibration."""
    kept = pieces[~pieces["beyond_30mm"].astype(bool)].reset_index(drop=True)
    burden_cal = rebuild_burden(quantities["burden"]["0.1|filtered=True"], fit)
    params = quantities["per_stone"]
    stones = []
    if len(kept):
        d3 = kept["max_diameter_3d_mm"].to_numpy(float)
        v_lo, v_hi = stone_bounds(params, "U2", "volume_mm3", ALPHA, d3)
        d_lo, d_hi = stone_bounds(params, "U2", "max_diameter_3d_mm", ALPHA, d3)
        p = kept.assign(stratum=cfm.stratum(kept["max_diameter_3d_mm"]), log_volume=np.log(kept["volume_mm3"]), hu100=kept["mean_hu"] / 100)
        pi = burden_cal.pi_of(p)
        for i, row in kept.iterrows():
            stones.append({"stone_id": int(row.pred_id), "voxel_count": int(row.voxel_count),
                           "volume_mm3": float(row.volume_mm3), "volume_range_mm3": [float(row.volume_mm3 * np.exp(v_lo[i])), float(row.volume_mm3 * np.exp(v_hi[i]))],
                           "max_diameter_3d_mm": float(row.max_diameter_3d_mm),
                           "diameter_range_mm": [float(row.max_diameter_3d_mm * np.exp(d_lo[i])), float(row.max_diameter_3d_mm * np.exp(d_hi[i]))],
                           "false_detection_probability": float(pi[i]), "distance_to_kidney_mm": float(row.distance_to_kidney_mm)})  # fmt: skip
    empty = pd.DataFrame(columns=["case", "group", "ref_volume_mm3", "pred_volume_mm3", "pred_max_diameter_3d_mm"])
    test = cfm.Tables(empty, pieces.copy(), pd.DataFrame([scan]))
    iv = cfm.apply_burden(burden_cal, test)
    a = iv["P2-A decomposition"].iloc[0]
    return {"note": NOTE, "nominal": 1 - ALPHA, "calibration_quantities_sha256": None, "stones": stones,
            "burden": {"predicted_mm3": float(scan["predicted_burden_filtered_mm3"]), "interval_mm3": [float(a.lo), float(a.hi)],
                       "decomposition": decomposition(burden_cal, test),
                       "comparators": {m: [float(iv[m].iloc[0].lo), float(iv[m].iloc[0].hi)] for m in ("P2-B-CQR", "P2-B direct", "P2-0 naive")}},
            "removed_by_kidney_distance_filter": int(pieces["beyond_30mm"].astype(bool).sum())}  # fmt: skip


def decomposition(burden_cal: cfm.BurdenCalibration, test: cfm.Tables) -> dict:
    """Intermediate quantities of the SAME frozen P2-A calculation, for display only (nothing new is estimated).

    - retained pieces and their measured volume after the 30 mm filter;
    - false-detection modelling: the expected retained volume, sum of (1 - pi_j) * V_j;
    - measurement uncertainty: the development residual pool of each piece's size band (5th-95th percentile of
      the multiplicative error exp(eps));
    - missed burden: the development per-scan missed-burden pool of the scan's slice-thickness group;
    - the raw Monte Carlo interval [L, U] (the engine's own draws), the one conformal factor t, and the result.
    """
    pt, st, _ = cfm._prepare(test, burden_cal.filtered)
    pt["pi"] = burden_cal.pi_of(pt)
    raw = burden_cal.raw_a(pt, st)
    row = st.iloc[0]
    lo_raw, hi_raw = (float(v) for v in raw[row.case])
    sf = burden_cal.missed_fit
    pool = sf[sf.slice_group == row.slice_group]["missed_mm3"].to_numpy()
    pool = pool if len(pool) >= 10 else sf["missed_mm3"].to_numpy()
    bands = {}
    for k in sorted(set(pt.stratum)):
        e = burden_cal.pool_e[k] if len(burden_cal.pool_e[k]) else burden_cal.e_fit
        bands[["<5 mm", "5-10 mm", ">=10 mm"][int(k)]] = {"n": len(e), "relative_error_p05_p95": [float(np.exp(np.quantile(e, 0.05))), float(np.exp(np.quantile(e, 0.95)))]}
    t, c = float(burden_cal.t_fac), cfm.C_MM3
    return {"retained_pieces": len(pt), "measured_mm3": float(pt.volume_mm3.sum()),
            "expected_after_false_detection_mm3": float(((1 - pt.pi) * pt.volume_mm3).sum()),
            "measurement_residual_bands": bands,
            "missed_burden_pool": {"slice_group": str(row.slice_group), "n_scans": len(pool), "share_nonzero": float((pool > 0).mean()),
                                   "median_mm3": float(np.median(pool)), "p90_mm3": float(np.quantile(pool, 0.90))},
            "draws": cfm.DRAWS, "raw_interval_mm3": [lo_raw, hi_raw], "conformal_factor_t": t, "offset_c_mm3": c,
            "calibrated_interval_mm3": [max((lo_raw + c) * np.exp(-t) - c, 0.0), (hi_raw + c) * np.exp(t) - c]}  # fmt: skip


def write_case_ranges(folder: Path, probabilities: Path, t1_run: Path = T1_RUN, fit_table: Path = FIT_TABLE) -> dict:
    """Compute and save calibrated_ranges.json next to the run's report.json. Refuses contrast-enhanced scans."""
    report = json.loads((folder / "report.json").read_text())
    aorta = (report.get("contrast_check") or {}).get("aorta_median_hu")
    if report.get("result_status") != "complete":
        raise ValueError("the run did not complete; no ranges")
    if aorta is None or aorta >= CONTRAST_HU:
        raise ValueError(f"not shown to be non-contrast (aorta median {aorta} HU); the calibration covers non-contrast CT only")
    q = load_quantities(t1_run / "analysis")
    fit = cfm.Tables(*(pd.read_csv(fit_table / f) for f in ("annotated.csv", "predicted.csv", "scans.csv")))
    pieces, scan = predicted_pieces(report["case_id"], folder, probabilities)
    out = ranges(pieces, scan, q, fit)
    out["calibration_quantities_sha256"] = (t1_run / "analysis" / "calibration_quantities.sha256").read_text().split()[0]
    out["case_id"] = report["case_id"]
    (folder / OUTPUT).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out

"""POST-HOC, DESCRIPTIVE replay of the T1 burden-interval WIDTHS (owner decision, 14 November 2026, review item 2e).

The question: is the paired median width ratio of the detection-aware interval to P2-B-CQR (0.573 on 53 locked test
scans, 11 of them stone-free) driven by the stone-free scans? The single T1 run stored only aggregate statistics.

What this script does, and what it does not do:
- It does NOT rerun T1, segmentation or TTA, and it does NOT refit or recalibrate anything.
- It does NOT open any annotation mask and does NOT use the guarded loader. Interval WIDTHS need no reference.
- It rebuilds each test scan's predicted pieces from the stored T1 run-case outputs (model mask, filter-removed mask,
  kidney mask, CT ROI crop) and the stored 8-flip probabilities, exactly as scripts/build_oof_stone_table.py does.
- It rebuilds every burden method from the HASHED calibration quantities (stonekit.t1.external.rebuild_burden), so
  the development pools are checked against their hashes.
- It is accepted only if it reproduces the stored T1 aggregates: the median widths of all four methods and the
  paired median width ratio with its bootstrap CI (same seed and resampling). Otherwise it stops.
- Stone-free status comes from the pre-split full audit (data/interim/li2022_full_audit/li_full_cases.csv, 5 November
  2026, written before the split and the lock); the count must equal the 11 stone-free scans of the T1 run.
Outputs go to a new folder results/runs/t1_posthoc_width_replay_<stamp>/. Every result is labelled post hoc.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.seg import kidney_roi
from stonekit.t1.external import load_quantities, rebuild_burden
from stonekit.uq import confirmatory as cfm

RUN = Path("results/runs/T1_20261113T081425Z")
AUDIT = Path("data/interim/li2022_full_audit/li_full_cases.csv")
MIN_VOXELS, FILTER_MM = 3, 30.0
PIECE_FEATURES = ["volume_mm3", "max_diameter_3d_mm", "mean_hu"]


def _signals():
    import importlib.util

    spec = importlib.util.spec_from_file_location("bost", "scripts/build_oof_stone_table.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.piece_signals


def predicted_pieces(case: str, folder: Path, tta: Path, piece_signals) -> tuple[list[dict], dict]:
    """Predicted pieces of one test scan, rebuilt like build_oof_stone_table.process but WITHOUT the annotation."""
    report = json.loads((folder / "report.json").read_text())
    model_img = load_volume(folder / "stone_labels_model.nii.gz")
    spacing = tuple(float(s) for s in model_img.spacing_mm)
    voxel_mm3 = float(np.prod(spacing))
    box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
    model = model_img.data > 0
    removed_file = folder / "stone_labels_model_removed_by_kidney_distance_filter.nii.gz"
    if removed_file.exists():
        model |= load_volume(removed_file).data > 0
    ct = np.zeros(model.shape, np.float32)
    crop = load_volume(folder / "roi_crop.nii.gz").data
    if crop.shape != model[box].shape:
        raise SystemExit(f"{case}: ROI crop {crop.shape} != box {model[box].shape}")
    ct[box] = crop  # only voxels inside the box are ever measured (checked below)
    predicted = separate_stones(model, MIN_VOXELS)
    pred_m = {s["stone_id"]: s for s in measure_stones(predicted, ct, model_img.affine, spacing)}
    kidneys = kidney_roi.kidney_mask(load_volume(folder / "kidney_totalsegmentator.nii.gz").data)
    distance = ndimage.distance_transform_edt(~kidneys[box], sampling=spacing)
    labels_box = predicted.labels[box]
    if (predicted.labels > 0).sum() != (labels_box > 0).sum():
        raise SystemExit(f"{case}: predicted voxels outside the ROI box")
    for pid, window in enumerate(ndimage.find_objects(labels_box), start=1):
        if window is not None:
            pred_m[pid]["distance_to_kidney_mm"] = float(distance[window][labels_box[window] == pid].min())
    z = np.load(tta / case / "probabilities.npz")
    if z["p_deployed"].shape != labels_box.shape:
        raise SystemExit(f"{case}: probability map shape differs from the ROI box")
    signals = piece_signals(labels_box, {"p_deployed": z["p_deployed"], "p_flip": z["p_flip"]}, voxel_mm3)
    rows = []
    for p, m in pred_m.items():
        row = {"case": case, "pred_id": p, "group": "unknown", "distance_to_kidney_mm": m["distance_to_kidney_mm"],
               "beyond_30mm": m["distance_to_kidney_mm"] > FILTER_MM, **{k: m.get(k) for k in PIECE_FEATURES}, **signals[p]}  # fmt: skip
        rows.append(row)
    scan = {"case": case, "slice_mm": spacing[2], "annotated_burden_mm3": np.nan,
            "predicted_burden_mm3": float(sum(m["volume_mm3"] for m in pred_m.values())),
            "predicted_burden_filtered_mm3": float(sum(m["volume_mm3"] for m in pred_m.values() if m["distance_to_kidney_mm"] <= FILTER_MM))}  # fmt: skip
    return rows, scan


def ratio_stats(ra: np.ndarray, seed: int = cfm.SEED, resamples: int = cfm.BOOT) -> dict:
    """Exactly the width-ratio part of stonekit.uq.confirmatory.p2_test (same seed, resampling and quantiles)."""
    boot = np.random.default_rng(seed).integers(0, len(ra), size=(resamples, len(ra)))
    meds = np.median(ra[boot], axis=1)
    return {"n": len(ra), "median": float(np.median(ra)), "ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]}


def main() -> None:
    inputs = json.loads((RUN / "inputs.json").read_text())
    test_cases = sorted(inputs["test"])
    ev = json.loads((RUN / "analysis/evaluation.json").read_text())
    q = load_quantities(RUN / "analysis")
    fit_dir = Path(inputs["fit_table"])
    fit = cfm.Tables(pd.read_csv(fit_dir / "annotated.csv"), pd.read_csv(fit_dir / "predicted.csv"), pd.read_csv(fit_dir / "scans.csv"))
    folders = {p.name.rsplit("_", 1)[0]: p for p in (RUN / "test_work/cases").iterdir()}
    if sorted(folders) != test_cases:
        raise SystemExit("stored case folders do not match the 53 T1 test scans")
    piece_signals = _signals()
    rows, scans = [], []
    for case in test_cases:
        r, s = predicted_pieces(case, folders[case], RUN / "test_work/tta", piece_signals)
        rows += r
        scans.append(s)
    empty_ann = pd.DataFrame(columns=["case", "group", "ref_volume_mm3", "pred_volume_mm3", "pred_max_diameter_3d_mm"])
    test = cfm.Tables(empty_ann, pd.DataFrame(rows), pd.DataFrame(scans))

    with open(AUDIT, encoding="utf-8") as f:
        has_stone = {r["case"]: r["has_stone"] == "True" for r in csv.DictReader(f)}
    stone_free = [c for c in test_cases if not has_stone[c]]

    run = create_run_folder(Path("results/runs"), "t1_posthoc_width_replay")
    out, checks = {"label": "POST HOC, DESCRIPTIVE; not part of the frozen protocol; T1 not rerun", "t1_run": str(RUN),
                   "stone_free_source": str(AUDIT), "stone_free_cases": stone_free}, {}  # fmt: skip
    for key in ("0.1|filtered=True", "0.1|filtered=False", "0.2|filtered=True", "0.2|filtered=False"):
        alpha, filtered = key.split("|")
        cal = rebuild_burden(q["burden"][key], fit)
        iv = cfm.apply_burden(cal, test)
        widths = {m: (d.hi - d.lo).to_numpy() for m, d in iv.items()}
        ra = widths["P2-A decomposition"] / np.maximum(widths["P2-B-CQR"], 1e-9)
        ev_key = f"{1 - float(alpha):.2f}|{filtered}"
        stored = ev["burden"][ev_key]
        rep = {m: float(np.median(w)) for m, w in widths.items()}
        full = ratio_stats(ra)
        ok = all(np.isclose(rep[m], stored[m]["median_width_mm3"], rtol=1e-9, atol=1e-6) for m in widths)
        ok &= np.isclose(full["median"], stored["width_ratio_A_over_BCQR"]["median"], rtol=1e-9)
        ok &= bool(np.allclose(full["ci95"], stored["width_ratio_A_over_BCQR"]["ci95"], rtol=1e-9))
        checks[ev_key] = {"reproduced": bool(ok), "replayed_median_widths": rep, "stored_median_widths": {m: stored[m]["median_width_mm3"] for m in widths},
                          "replayed_ratio": full, "stored_ratio": stored["width_ratio_A_over_BCQR"]}  # fmt: skip
        sf = np.isin(np.array(test_cases), stone_free)
        out[ev_key] = {"all_scans": full, "stone_bearing": ratio_stats(ra[~sf]), "stone_free": ratio_stats(ra[sf]),
                       "share_of_scans_with_ratio_below_1": {"all": float((ra < 1).mean()), "stone_bearing": float((ra[~sf] < 1).mean()),
                                                              "stone_free": float((ra[sf] < 1).mean())}}  # fmt: skip
        pd.DataFrame({"case": test_cases, "stone_free_pre_split_audit": sf, **{f"width_{m}": w for m, w in widths.items()}, "ratio_A_over_CQR": ra}).to_csv(
            run / f"per_scan_widths_{ev_key.replace('|', '_').replace('=', '-')}.csv", index=False)  # fmt: skip
    out["reproduction_checks"] = checks
    out["all_reproduced"] = all(c["reproduced"] for c in checks.values())
    (run / "posthoc_width_replay.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("stone-free (pre-split audit):", len(stone_free), "| all stored aggregates reproduced:", out["all_reproduced"])
    for k, c in checks.items():
        print(k, "reproduced" if c["reproduced"] else "NOT REPRODUCED", "| ratio", round(c["replayed_ratio"]["median"], 6), "vs", round(c["stored_ratio"]["median"], 6))
    if not out["all_reproduced"]:
        raise SystemExit("replay does not reproduce the stored T1 aggregates; subset results must not be used")
    for k in checks:
        print(k, json.dumps({s: out[k][s] for s in ("all_scans", "stone_bearing", "stone_free")}))
        print("   share ratio<1:", out[k]["share_of_scans_with_ratio_below_1"])
    print("Saved", run)


if __name__ == "__main__":
    main()

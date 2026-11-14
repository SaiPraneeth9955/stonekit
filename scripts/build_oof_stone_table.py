"""Build the out-of-fold (OOF) per-stone and per-scan tables that feed E4-E11 (master plan v2).

Inputs (development scans only):
- run-case folders, one per scan, predicted by a model that did not train on that scan
  (fold 0: data/interim/stonekit_runs/validation_20261106T003707Z; folds 1-4 and the held-back scans:
  data/interim/stonekit_runs/oof_<stamp>/fold{k}); the fold of each source is given on the command line;
- the matching TTA/probability folder from scripts/oof_tta.py (optional; signals are NaN without it);
- the Li annotation (reference, not physical truth) and CT, through the guarded development loader.

Uniform post-processing: the 6 November fold-0 run predates the 30 mm kidney-distance filter, so
predicted pieces are rebuilt here from the union of the saved model mask and any filter-removed mask
(unfiltered), separated with the pipeline's rule (26-connected, >= 3 voxels), and each piece's
distance to the TotalSegmentator kidney is recomputed exactly as run_case does (Euclidean distance
transform of the non-kidney region inside the ROI box, minimum over the piece). The filter is then a
column (`beyond_30mm`), applied identically to every scan by the analyses.

Per-piece reliability signals (from the deployed-mode probability p and the 8 flip probabilities):
- p_max, p_mean inside the piece;
- threshold sensitivity: log(V(p>0.3) / V(p>0.7)) within the piece dilated by 2 voxels (both counts
  floored at one voxel);
- TTA spread: SD over the 8 flips of log(V_f + one voxel), V_f = voxels with p_flip>0.5 in the same
  neighbourhood.
Signals are computed for every predicted piece; nothing here uses a reserved scan.

Outputs (git-ignored folder data/interim/oof_tables_<stamp>/): annotated.csv (one row per annotated
piece, with its match group and, for one_to_one, the predicted piece's measures), predicted.csv (one
row per predicted piece), scans.csv (one row per scan), plus a provenance.json.
"""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.eval.stone_matching import group_pieces
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.seg import kidney_roi

MIN_VOXELS = 3
FILTER_MM = 30.0
KEEP = ["volume_mm3", "max_diameter_3d_mm", "max_diameter_axial_mm", "volume_equivalent_diameter_mm", "max_axial_area_mm2",
        "axis_major_mm", "axis_middle_mm", "axis_minor_mm", "elongation", "flatness", "major_axis_angle_to_z_deg",
        "mean_hu", "median_hu", "max_hu", "voxel_count", "sphericity"]  # fmt: skip


def piece_signals(labels_crop: np.ndarray, probs: dict | None, voxel_mm3: float) -> dict[int, dict]:
    out = {}
    for pid, window in enumerate(ndimage.find_objects(labels_crop), start=1):
        if window is None:
            continue
        if probs is None:
            out[pid] = {k: float("nan") for k in ("p_max", "p_mean", "threshold_sensitivity", "tta_spread")}
            continue
        grown = tuple(slice(max(s.start - 3, 0), s.stop + 3) for s in window)
        piece = labels_crop[grown] == pid
        hood = ndimage.binary_dilation(piece, iterations=2)
        p = probs["p_deployed"][grown].astype(np.float32)
        flips = probs["p_flip"][(slice(None), *grown)].astype(np.float32)
        v_lo = max(int(np.count_nonzero((p > 0.3) & hood)), 1)
        v_hi = max(int(np.count_nonzero((p > 0.7) & hood)), 1)
        v_f = np.array([np.count_nonzero((f > 0.5) & hood) for f in flips], float)
        out[pid] = {
            "p_max": float(p[piece].max()), "p_mean": float(p[piece].mean()),
            "threshold_sensitivity": float(np.log(v_lo / v_hi)),
            "tta_spread": float(np.std(np.log((v_f + 1.0) * voxel_mm3))),
        }  # fmt: skip
    return out


def process(case: str, folder: Path, fold: int, tta_root: Path | None, data: LiDevelopmentData) -> tuple[list, list, dict]:
    report = json.loads((folder / "report.json").read_text())
    li = data.load(case)
    spacing = li.image.spacing_mm
    voxel_mm3 = float(np.prod(spacing))
    box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
    model = load_volume(folder / "stone_labels_model.nii.gz").data > 0
    removed_file = folder / "stone_labels_model_removed_by_kidney_distance_filter.nii.gz"
    if removed_file.exists():
        model |= load_volume(removed_file).data > 0
    predicted = separate_stones(model, MIN_VOXELS)
    reference = separate_stones(li.stone_reference, MIN_VOXELS)
    pred_m = {s["stone_id"]: s for s in measure_stones(predicted, li.image.data, li.image.affine, spacing)}
    ref_m = {s["stone_id"]: s for s in measure_stones(reference, li.image.data, li.image.affine, spacing)}
    kidneys = kidney_roi.kidney_mask(load_volume(folder / "kidney_totalsegmentator.nii.gz").data)
    distance = ndimage.distance_transform_edt(~kidneys[box], sampling=spacing)
    labels_box = predicted.labels[box]
    if (predicted.labels > 0).sum() != (labels_box > 0).sum():
        raise ValueError(f"{case}: predicted voxels outside the ROI box")
    for pid, window in enumerate(ndimage.find_objects(labels_box), start=1):
        if window is not None:
            pred_m[pid]["distance_to_kidney_mm"] = float(distance[window][labels_box[window] == pid].min())
    probs = None
    if tta_root is not None and (tta_root / case / "probabilities.npz").exists():
        z = np.load(tta_root / case / "probabilities.npz")
        if z["p_deployed"].shape != labels_box.shape:
            raise ValueError(f"{case}: probability map {z['p_deployed'].shape} != ROI box {labels_box.shape}")
        probs = {"p_deployed": z["p_deployed"], "p_flip": z["p_flip"]}
    signals = piece_signals(labels_box, probs, voxel_mm3)
    groups = group_pieces(predicted.labels, reference.labels, set(pred_m), set(ref_m))
    slice_mm = float(spacing[2])
    annotated_rows, predicted_rows = [], []
    kind_of_pred = {p: g.kind for g in groups for p in g.predicted}
    for g in groups:
        for a in g.annotated:
            row = {"case": case, "fold": fold, "slice_mm": slice_mm, "ref_id": a, "group": g.kind, "n_pred_in_group": len(g.predicted)}
            row.update({f"ref_{k}": ref_m[a].get(k) for k in KEEP})
            if g.kind == "one_to_one":
                p = g.predicted[0]
                row.update({f"pred_{k}": pred_m[p].get(k) for k in KEEP})
                row.update({"pred_id": p, "pred_distance_to_kidney_mm": pred_m[p]["distance_to_kidney_mm"], **{f"pred_{k}": v for k, v in signals[p].items()}})
            annotated_rows.append(row)
    for p, m in pred_m.items():
        row = {"case": case, "fold": fold, "slice_mm": slice_mm, "pred_id": p, "group": kind_of_pred.get(p, "false_positive"),
               "distance_to_kidney_mm": m["distance_to_kidney_mm"], "beyond_30mm": m["distance_to_kidney_mm"] > FILTER_MM}  # fmt: skip
        row.update({k: m.get(k) for k in KEEP})
        row.update(signals[p])
        predicted_rows.append(row)
    scan = {
        "case": case, "fold": fold, "slice_mm": slice_mm, "inplane_mm": float(spacing[0]),
        "annotated_pieces": len(ref_m), "annotated_burden_mm3": float(sum(m["volume_mm3"] for m in ref_m.values())),
        "annotated_largest_d3_mm": max((m["max_diameter_3d_mm"] for m in ref_m.values()), default=0.0),
        "predicted_pieces": len(pred_m), "predicted_burden_mm3": float(sum(m["volume_mm3"] for m in pred_m.values())),
        "predicted_pieces_filtered": sum(1 for m in pred_m.values() if m["distance_to_kidney_mm"] <= FILTER_MM),
        "predicted_burden_filtered_mm3": float(sum(m["volume_mm3"] for m in pred_m.values() if m["distance_to_kidney_mm"] <= FILTER_MM)),
        "predicted_largest_d3_mm": max((m["max_diameter_3d_mm"] for m in pred_m.values()), default=0.0),
        "has_probabilities": probs is not None, "run_folder": str(folder),
        "kidney_fallback": bool(report.get("kidney_segmentation_fallback")),
    }  # fmt: skip
    return annotated_rows, predicted_rows, scan


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs=3, action="append", metavar=("RUN_FOLDER", "FOLD", "TTA_FOLDER_OR_NONE"), required=True)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--exclude-csv", type=Path, default=None,
                        help="Review outcome file (data/processed/li2022_review_outcome_v1.csv); rows with action=exclude are skipped")
    args = parser.parse_args()
    excluded = set()
    if args.exclude_csv is not None:
        with open(args.exclude_csv, newline="", encoding="utf-8") as handle:
            excluded = {r["case"] for r in csv.DictReader(handle) if r["action"] == "exclude"}
    root = Path.cwd()
    data = LiDevelopmentData(root)
    cv = json.loads((root / "data/nnunet/li_stone_cv_split.json").read_text())
    trained = {k: set(f["train"]) for k, f in enumerate(cv["folds"])}
    annotated, predicted, scans, failures, skipped = [], [], [], {}, []
    seen = set()
    for run_folder, fold_text, tta in args.source:
        fold = int(fold_text)
        tta_root = None if tta == "NONE" else Path(tta)
        for folder in sorted(Path(run_folder).iterdir()):
            if not (folder / "report.json").exists():
                continue
            case = json.loads((folder / "report.json").read_text())["case_id"]
            if args.cases is not None and case not in args.cases:
                continue
            if case in excluded:
                skipped.append(case)
                continue
            require_development([case], data.manifest)
            if case in trained[fold]:
                raise SystemExit(f"{case} was trained on by fold {fold}: not out-of-fold")
            if case in seen:
                raise SystemExit(f"{case} appears in two sources")
            seen.add(case)
            try:
                a, p, s = process(case, folder, fold, tta_root, data)
                annotated += a
                predicted += p
                scans.append(s)
                print(f"{case} fold {fold}: {len(a)} annotated, {len(p)} predicted", flush=True)
            except Exception as error:  # noqa: BLE001  (recorded and reported, never hidden)
                failures[case] = repr(error)
                print(f"{case} FAILED {error!r}", flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = root / "data" / "interim" / f"oof_tables_{stamp}"
    out.mkdir(parents=True)
    write_csv(out / "annotated.csv", annotated)
    write_csv(out / "predicted.csv", predicted)
    write_csv(out / "scans.csv", scans)
    (out / "provenance.json").write_text(json.dumps({
        "sources": args.source, "cases": len(scans), "failures": failures,
        "excluded_by_review": sorted(skipped), "exclude_csv": str(args.exclude_csv) if args.exclude_csv else None,
        "cv_split_sha256": hashlib.sha256((root / "data/nnunet/li_stone_cv_split.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "created_utc": stamp,
    }, indent=2), encoding="utf-8")  # fmt: skip
    print("Saved", out, "| scans", len(scans), "| failures", len(failures))


if __name__ == "__main__":
    main()

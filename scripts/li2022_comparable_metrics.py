"""Li 2022-comparable segmentation metrics for StoneKit's saved outputs (experiment E2, development only).

Li et al., Diagnostics 2022;12(8):1788 (full text via Europe PMC, PMC9330428, read 11 November 2026)
report voxel-level Dice = 2TP/(2TP+FP+FN), voxel sensitivity TP/(TP+FN) and specificity, as mean ± SD
over 10 trainings, under 5-fold cross-validation on all 260 scans. They also report kidney-stone Dice by
size: small 0-6 mm, middle 6-20 mm, large >20 mm (the paper also gives 0-28 / 28-315 / >315 mm3,
which do not match those diameters for spheres, so both binnings are computed here).

The paper does not say whether Dice pools voxels over the test set or averages per scan, or how
per-size Dice is computed. All plausible readings are computed:
- pooled voxel Dice and sensitivity over all scans (TP, FP, FN summed);
- mean per-scan Dice over scans with stones;
- per-stone Dice: each annotated piece against the union of predicted pieces in its overlap group
  (stonekit.eval.stone_matching; a missed piece scores 0), averaged per size bin.

NOT a like-for-like comparison: different cohort (our 32 fold-0 validation scans vs their CV over all
260, which includes scans that are our locked test set), different model and preprocessing, one
training run. Development evidence only. Reads saved outputs and development annotations only.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.eval.stone_matching import group_pieces
from stonekit.io.nifti import load_volume
from stonekit.measure.diameter import max_distance_mm
from stonekit.measure.stones import separate_stones
from stonekit.seg import kidney_roi

DIAMETER_BINS = [(0, 6, "small 0-6 mm"), (6, 20, "middle 6-20 mm"), (20, np.inf, "large >20 mm")]
VOLUME_BINS = [(0, 28, "small 0-28 mm3"), (28, 315, "middle 28-315 mm3"), (315, np.inf, "large >315 mm3")]
FILTER_MM = 30.0


def piece_sizes(labels: np.ndarray, spacing_mm) -> dict:
    """Volume (mm3) and 3D maximum diameter (mm) per labelled piece: the same definitions as
    stonekit.measure.stones (voxel count x voxel volume; centre-to-centre maximum), without the shape
    and density measures this comparison does not use."""
    voxel_mm3 = float(np.prod(spacing_mm))
    sizes = {}
    for stone_id, window in enumerate(ndimage.find_objects(labels), start=1):
        if window is None:
            continue
        mask = labels[window] == stone_id
        sizes[stone_id] = {"volume_mm3": float(mask.sum()) * voxel_mm3, "max_diameter_3d_mm": max_distance_mm(mask, spacing_mm)}
    return sizes


def binned(value, bins):
    return next(name for lo, hi, name in bins if lo <= value < hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="Validation run folders (one per model)")
    args = parser.parse_args()
    root = Path.cwd()
    data = LiDevelopmentData(root)
    split = json.loads((root / "data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    out = {"definitions": __doc__, "li2022_reported": {
        "stone_dice_3d_unet": "80.04 ± 1.75", "stone_dice_res_unet": "80.59 ± 1.28",
        "stone_voxel_sensitivity_3d_unet": "80.21 ± 1.81", "kidney_dice_res_unet": "96.54 ± 1.06",
        "stone_dice_by_size_3d_unet": {"small": "58.03 ± 1.42", "middle": "81.56 ± 1.99", "large": "82.86 ± 3.52"},
        "stone_dice_by_size_res_unet": {"small": "60.11 ± 0.84", "middle": "76.08 ± 3.46", "large": "83.39 ± 2.33"},
        "source": "Li et al. 2022, Tables 1 and 3 (Europe PMC full-text XML)"}, "ours": {}}  # fmt: skip
    # Each scan is loaded once and scored for every run and variant. Per-stone work uses the ROI box:
    # every predicted voxel lies inside it by construction, and annotation voxels outside it count as
    # false negatives, so the results are identical to whole-volume computation.
    runs = {}
    for run_folder in args.runs:
        folders = {json.loads((p / "report.json").read_text())["case_id"]: p for p in run_folder.iterdir() if (p / "report.json").exists()}
        require_development(list(folders), data.manifest)
        if set(folders) != set(split["validation"]):
            raise ValueError(f"{run_folder}: expected exactly the fold-0 validation scans")
        runs[run_folder.name] = folders
    keys = [(run, variant) for run in runs for variant in ("unfiltered", "filtered_30mm")]
    acc = {k: {"tp": 0, "fp": 0, "fn": 0, "ktp": 0, "kfp": 0, "kfn": 0, "per_scan": [], "per_stone": []} for k in keys}
    for n_case, case in enumerate(sorted(split["validation"]), start=1):
        li = data.load(case)
        ref_full = li.stone_reference
        reference_full = separate_stones(ref_full, 3)
        annotated = piece_sizes(reference_full.labels, li.image.spacing_mm)
        piece_voxels = np.bincount(reference_full.labels.ravel())  # full volume, so pieces are whole
        kidney_reference = li.kidney_reference  # computed once: the property rebuilds it on every access
        kidney_reference_sum = int(np.count_nonzero(kidney_reference))
        for run, folders in runs.items():
            folder = folders[case]
            report = json.loads((folder / "report.json").read_text())
            box = tuple(slice(a, b) for a, b in zip(report["roi"]["box_start_voxel"], report["roi"]["box_stop_voxel"], strict=True))
            kid = kidney_roi.kidney_mask(load_volume(folder / "kidney_totalsegmentator.nii.gz").data)
            kid_tp = int(np.count_nonzero(kid & kidney_reference))
            kid_counts = (kid_tp, int(np.count_nonzero(kid)) - kid_tp, kidney_reference_sum - kid_tp)  # same for both variants
            labels_full = load_volume(folder / "stone_labels_model.nii.gz").data
            if (labels_full > 0).sum() != (labels_full[box] > 0).sum():
                raise ValueError(f"{case}: predicted voxels outside the ROI box")
            ref = ref_full[box]
            ref_outside = int(ref_full.sum() - ref.sum())
            reference_labels = reference_full.labels[box]
            for variant in ("unfiltered", "filtered_30mm"):
                a = acc[(run, variant)]
                labels = labels_full[box]
                if variant == "filtered_30mm":
                    far = [s["stone_id"] for s in report["methods"]["model"]["stones"] if s["distance_to_kidney_mm"] > FILTER_MM]
                    labels = np.where(np.isin(labels, far), 0, labels)
                pred = labels > 0
                a["tp"] += int((pred & ref).sum())
                a["fp"] += int((pred & ~ref).sum())
                a["fn"] += int((~pred & ref).sum()) + ref_outside
                a["ktp"] += kid_counts[0]
                a["kfp"] += kid_counts[1]
                a["kfn"] += kid_counts[2]
                if ref_full.any():
                    a["per_scan"].append(2 * (pred & ref).sum() / (pred.sum() + ref_full.sum()))
                # Pairwise overlaps in one pass: voxels shared by predicted piece p and annotated piece i.
                both = (labels > 0) & (reference_labels > 0)
                pairs, counts = np.unique(np.stack([labels[both], reference_labels[both]]), axis=1, return_counts=True)
                overlap = {(int(p), int(i)): int(c) for (p, i), c in zip(pairs.T, counts, strict=True)}
                predicted_voxels = np.bincount(labels.ravel())
                predicted_ids = set(np.nonzero(predicted_voxels)[0].tolist()) - {0}
                for group in group_pieces(labels, reference_labels, predicted_ids, set(annotated)):
                    if not group.annotated:
                        continue
                    union_voxels = sum(int(predicted_voxels[p]) for p in group.predicted)
                    for i in group.annotated:
                        shared = sum(overlap.get((p, i), 0) for p in group.predicted)
                        size = int(piece_voxels[i])
                        dice = 2 * shared / (union_voxels + size) if group.kind != "missed" and size else 0.0
                        # In split/merged/complex groups the union is shared, so per-piece Dice is approximate there.
                        a["per_stone"].append({"d3": annotated[i]["max_diameter_3d_mm"], "vol": annotated[i]["volume_mm3"], "dice": float(dice), "kind": group.kind})
        print(f"case {n_case}/{len(split['validation'])} done", flush=True)
    for (run, variant), a in acc.items():
        tp, fp, fn = a["tp"], a["fp"], a["fn"]
        kid_tp, kid_fp, kid_fn = a["ktp"], a["kfp"], a["kfn"]
        per_scan, per_stone = a["per_scan"], a["per_stone"]
        folders = runs[run]
        if True:
            by_d = {name: [r["dice"] for r in per_stone if binned(r["d3"], DIAMETER_BINS) == name] for _, _, name in DIAMETER_BINS}
            by_v = {name: [r["dice"] for r in per_stone if binned(r["vol"], VOLUME_BINS) == name] for _, _, name in VOLUME_BINS}
            out["ours"][f"{run} | {variant}"] = {
                "scans": len(folders),
                "pooled_voxel_stone_dice": round(2 * tp / (2 * tp + fp + fn), 4),
                "pooled_voxel_stone_sensitivity": round(tp / (tp + fn), 4),
                "mean_per_scan_stone_dice_scans_with_stone": round(float(np.mean(per_scan)), 4),
                "pooled_voxel_kidney_dice_totalsegmentator_vs_li_label2": round(2 * kid_tp / (2 * kid_tp + kid_fp + kid_fn), 4),
                "per_stone_dice_by_diameter": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4) if v else None} for k, v in by_d.items()},
                "per_stone_dice_by_volume": {k: {"n": len(v), "mean": round(float(np.mean(v)), 4) if v else None} for k, v in by_v.items()},
            }  # fmt: skip
            print(run, variant, json.dumps(out["ours"][f"{run} | {variant}"]), flush=True)
    run = create_run_folder(root / "results/runs", "li2022_comparable_metrics")
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved to", run)


if __name__ == "__main__":
    main()

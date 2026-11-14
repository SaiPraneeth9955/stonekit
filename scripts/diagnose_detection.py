"""Why stones are missed, what the false positives are, and the pre-specified decision-threshold test.

Pre-specification: docs/decisions.md, "pre-specification of the decision-threshold intervention"
(logged 22:52 UTC on 7 November 2026, before any probability map was examined).

Inputs (development scans only):
- the 100-epoch validation run (`--runs`): report.json, roi box, kidney mask and model labels per scan;
- nnU-Net stone probabilities on the same saved ROI crops (`--probabilities`, from nnUNetv2_predict
  --save_probabilities with the pipeline's own settings);
- the Li annotations through the guarded development loader.

For each candidate threshold t the crop mask is p(stone) >= t (t = 0.5 uses the argmax segmentation
that the pipeline uses), then exactly the pipeline's rules: 26-connected pieces of at least 3 voxels,
holes filled, measured on the original voxels, the 30 mm kidney-distance filter, and the same
union-find matching as scripts/analyse_stone_validation.py. At t = 0.5 this must reproduce the
published filtered result (98/187 detected, 33 false positives, 90 one-to-one pairs) or the script stops.

Per-piece rows go to data/interim/diagnostics/ (git-ignored); aggregates to results/runs/.
EXPLORATORY development evidence; the threshold selection is model selection, not confirmation.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.eval.stone_matching import group_pieces
from stonekit.io.nifti import load_volume
from stonekit.measure.components import CONNECTIVITY_26
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.seg import kidney_roi
from stonekit.seg.nnunet_convert import crop_affine

SEED = 20261105
RESAMPLES = 10000
THRESHOLDS = [0.5, 0.4, 0.3, 0.2, 0.1]
FILTER_MM = 30.0
MIN_VOXELS = 3
BONE_HU = 300
BONE_MIN_ML = 10.0
AORTA = kidney_roi.AORTA


def load_probability(prob_dir: Path, case: str, crop_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Stone probability and nnU-Net's own argmax, both in the crop's (nibabel) array order."""
    seg = np.asanyarray(nib.load(prob_dir / f"{case}.nii.gz").dataobj)
    prob = np.load(prob_dir / f"{case}.npz")["probabilities"][1]
    if prob.shape != crop_shape:
        prob = prob.transpose(2, 1, 0)
    if prob.shape != crop_shape or seg.shape != crop_shape:
        raise ValueError(f"{case}: probability {prob.shape} / segmentation {seg.shape} not on crop {crop_shape}")
    agree = float(((prob > 0.5) == (seg == 1)).mean())
    if agree < 0.9999:
        raise ValueError(f"{case}: probability map and segmentation disagree ({agree:.6f}); orientation is not trustworthy")
    return prob.astype(np.float32), seg == 1


def bone_distance(ct_crop: np.ndarray, exclude: np.ndarray, spacing, voxel_ml: float) -> np.ndarray:
    """Distance (mm) to large bright structures (bone proxy: >= 300 HU components of >= 10 mL)."""
    bright, n = ndimage.label((ct_crop >= BONE_HU) & ~exclude, structure=CONNECTIVITY_26)
    if not n:
        return np.full(ct_crop.shape, np.inf)
    sizes = np.bincount(bright.ravel())[1:] * voxel_ml
    bone = np.isin(bright, np.nonzero(sizes >= BONE_MIN_ML)[0] + 1)
    return ndimage.distance_transform_edt(~bone, sampling=spacing) if bone.any() else np.full(ct_crop.shape, np.inf)


def evaluate(binary, ct, box, kidney_distance, reference_labels, annotated):
    """The pipeline's piece rules, filter and matching on one crop mask."""
    affine = crop_affine(ct.affine, box)
    separation = separate_stones(binary, MIN_VOXELS)
    stones = measure_stones(separation, ct.data[box.slices], affine, ct.spacing_mm)
    kept = {}
    for s in stones:
        window = tuple(slice(a, b) for a, b in zip(s["voxel_start"], s["voxel_stop"], strict=True))
        inside = separation.labels[window] == s["stone_id"]
        s["distance_to_kidney_mm"] = float(kidney_distance[window][inside].min())
        if s["distance_to_kidney_mm"] <= FILTER_MM:
            kept[s["stone_id"]] = s
    labels = np.where(np.isin(separation.labels, list(kept)), separation.labels, 0)
    groups = group_pieces(labels, reference_labels, set(kept), set(annotated))
    return labels, kept, groups


def plain(value):
    """numpy scalars to plain Python, for JSON."""
    return value.item() if hasattr(value, "item") else str(value)


def read_detail(detail: Path) -> tuple[list, list, list]:
    """Saved per-piece, false-positive and per-scan rows back as typed values."""

    def typed(row):
        out = {}
        for k, v in row.items():
            if k == "case" or k.startswith("kind_"):
                out[k] = v
            elif v in ("", "None"):
                out[k] = None
            elif v in ("True", "False"):
                out[k] = v == "True"
            else:
                out[k] = float(v)
        return out

    tables = []
    for name in ("pieces", "false_positives", "scans"):
        with (detail / f"{name}.csv").open(newline="", encoding="utf-8") as f:
            tables.append([typed(r) for r in csv.DictReader(f)])
    for r in tables[2]:
        r["threshold"] = float(r["threshold"])
    return tuple(tables)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--from-detail", type=Path, help="Rebuild the summary from saved rows of an earlier run of this script")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.from_detail:
        piece_rows, fp_rows, scan_rows = read_detail(args.from_detail)
        base = [r for r in scan_rows if r["threshold"] == 0.5]
        check = (int(sum(r["detected"] for r in base)), int(sum(r["false_positive_pieces"] for r in base)),
                 sum(1 for r in piece_rows if r["kind_t0.5"] == "one_to_one"))  # fmt: skip
        if check != (98, 33, 90):
            raise SystemExit(f"Baseline not reproduced from saved rows: {check}")
        write_summary(root, piece_rows, fp_rows, scan_rows, check, args.from_detail.resolve(), args)
        return
    data = LiDevelopmentData(root)
    split = json.loads((root / "data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    folders = sorted(p for p in args.runs.iterdir() if (p / "report.json").exists())
    cases = [json.loads((p / "report.json").read_text())["case_id"] for p in folders]
    require_development(cases, data.manifest)
    if set(cases) != set(split["validation"]):
        raise ValueError("Expected exactly the fold-0 validation scans")

    piece_rows, fp_rows, scan_rows = [], [], []
    for folder in folders:
        report = json.loads((folder / "report.json").read_text())
        case = report["case_id"]
        box = kidney_roi.Box(tuple(report["roi"]["box_start_voxel"]), tuple(report["roi"]["box_stop_voxel"]))
        li = data.load(case)
        ct = li.image
        crop = ct.data[box.slices].astype(np.float32)
        reference = separate_stones(li.stone_reference, MIN_VOXELS)
        annotated = {s["stone_id"]: s for s in measure_stones(reference, ct.data, ct.affine, ct.spacing_mm)}
        reference_crop = reference.labels[box.slices]
        if int((reference.labels > 0).sum()) != int((reference_crop > 0).sum()):
            raise ValueError(f"{case}: an annotated piece lies outside the ROI box")
        kidney_labels = load_volume(folder / "kidney_totalsegmentator.nii.gz").data[box.slices]
        kidneys = kidney_roi.kidney_mask(kidney_labels)
        kidney_distance = ndimage.distance_transform_edt(~kidneys, sampling=ct.spacing_mm)
        aorta_distance = ndimage.distance_transform_edt(kidney_labels != AORTA, sampling=ct.spacing_mm) if (kidney_labels == AORTA).any() else np.full(crop.shape, np.inf)
        prob, argmax = load_probability(args.probabilities, case, crop.shape)
        bone_d = bone_distance(crop, reference_crop > 0, ct.spacing_mm, ct.voxel_volume_mm3 / 1000)
        per_t = {}
        for t in THRESHOLDS:
            binary = argmax if t == 0.5 else prob >= t
            labels, kept, groups = evaluate(binary, ct, box, kidney_distance, reference_crop, annotated)
            kinds = {}
            for g in groups:
                for i in g.annotated:
                    kinds[i] = g.kind
            pairs = {g.annotated[0]: kept[g.predicted[0]] for g in groups if g.kind == "one_to_one"}
            per_t[t] = {"kinds": kinds, "pairs": pairs, "fp": [kept[p] for g in groups if g.kind == "false_positive" for p in g.predicted],
                        "labels": labels}  # fmt: skip
            scan_rows.append({"case": case, "threshold": t, "spacing_z": ct.spacing_mm[2], "has_stone": bool(annotated),
                              "annotated": len(annotated), "detected": sum(k != "missed" for k in kinds.values()),
                              "false_positive_pieces": len(per_t[t]["fp"]), "predicted_pieces": len(kept)})  # fmt: skip
        for i, a in annotated.items():
            inside = reference_crop == i
            shell = ndimage.binary_dilation(inside, iterations=2) & ~(reference_crop > 0)
            row = {
                "case": case, "piece": i, "spacing_z": round(ct.spacing_mm[2], 3), "d3_mm": a["max_diameter_3d_mm"],
                "volume_mm3": a["volume_mm3"], "voxels": a["voxel_count"], "peak_hu": a["max_hu"], "mean_hu": a["mean_hu"],
                "local_background_median_hu": float(np.median(crop[shell])) if shell.any() else None,
                "max_probability": float(prob[inside].max()),
                "distance_to_kidney_mm": float(kidney_distance[inside].min()),
                "inside_kidney_fraction": float(kidneys[inside].mean()),
                "distance_to_bone_mm": float(bone_d[inside].min()),
                "distance_to_aorta_mm": float(aorta_distance[inside].min()),
            }  # fmt: skip
            row["local_contrast_hu"] = None if row["local_background_median_hu"] is None else row["peak_hu"] - row["local_background_median_hu"]
            for t in THRESHOLDS:
                row[f"kind_t{t}"] = per_t[t]["kinds"][i]
                pair = per_t[t]["pairs"].get(i)
                row[f"pred_mm3_t{t}"] = pair["volume_mm3"] if pair else None
                row[f"pred_d3_t{t}"] = pair["max_diameter_3d_mm"] if pair else None
            piece_rows.append(row)
        for p in per_t[0.5]["fp"]:
            window = tuple(slice(a, b) for a, b in zip(p["voxel_start"], p["voxel_stop"], strict=True))
            inside = per_t[0.5]["labels"][window] == p["stone_id"]
            fp_rows.append({
                "case": case, "volume_mm3": p["volume_mm3"], "d3_mm": p["max_diameter_3d_mm"], "peak_hu": p["max_hu"],
                "distance_to_kidney_mm": p["distance_to_kidney_mm"], "inside_kidney_fraction": float(kidneys[window][inside].mean()),
                "distance_to_bone_mm": float(bone_d[window][inside].min()), "distance_to_aorta_mm": float(aorta_distance[window][inside].min()),
                "max_probability": float(prob[window][inside].max()),
            })  # fmt: skip
        print(case, {t: (r["detected"], r["false_positive_pieces"]) for t, r in ((r["threshold"], r) for r in scan_rows if r["case"] == case)}, flush=True)

    base = [r for r in scan_rows if r["threshold"] == 0.5]
    check = (sum(r["detected"] for r in base), sum(r["false_positive_pieces"] for r in base), sum(1 for r in piece_rows if r["kind_t0.5"] == "one_to_one"))
    if check != (98, 33, 90):
        raise SystemExit(f"Baseline not reproduced: detected, false positives, pairs = {check}, expected (98, 33, 90)")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    detail = root / "data/interim/diagnostics" / f"detection_diagnosis_{stamp}"
    detail.mkdir(parents=True)
    for name, rows in (("pieces", piece_rows), ("false_positives", fp_rows), ("scans", scan_rows)):
        with (detail / f"{name}.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    write_summary(root, piece_rows, fp_rows, scan_rows, check, detail, args)


def write_summary(root, piece_rows, fp_rows, scan_rows, check, detail, args) -> None:
    summary = summarise(piece_rows, fp_rows, scan_rows)
    summary.update(status="EXPLORATORY_DEVELOPMENT_EVIDENCE_MODEL_SELECTION", definitions=__doc__, baseline_reproduced=list(check),
                   detail_rows=str(detail.relative_to(root)), runs=str(args.runs), probabilities=str(args.probabilities))  # fmt: skip
    run = create_run_folder(root / "results/runs", "detection_diagnosis")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=plain), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("baseline_threshold_0.5", "threshold_rule", "missed_by_probability")}, indent=1, default=plain))
    print("Saved to", run, "and", detail)


def paired_bootstrap(per_scan: dict, cases: list[str], statistic) -> list[float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(RESAMPLES):
        sample = [cases[i] for i in rng.integers(0, len(cases), len(cases))]
        v = statistic(sample)
        if v is not None and np.isfinite(v):
            values.append(v)
    return [round(float(np.percentile(values, 2.5)), 4), round(float(np.percentile(values, 97.5)), 4)]


def summarise(pieces, fps, scans) -> dict:
    cases = sorted({r["case"] for r in scans})
    by = {(r["case"], r["threshold"]): r for r in scans}

    def detection(sample, t):
        return sum(by[(c, t)]["detected"] for c in sample) / max(1, sum(by[(c, t)]["annotated"] for c in sample))

    def fp_rate(sample, t):
        return sum(by[(c, t)]["false_positive_pieces"] for c in sample) / len(sample)

    def common_bias(sample, t, kind):
        rows = [r for c in sample for r in pieces if r["case"] == c and r["kind_t0.5"] == "one_to_one" and r[f"kind_t{t}"] == "one_to_one"]
        if not rows:
            return None
        if kind == "volume":
            new = np.mean([200 * (r[f"pred_mm3_t{t}"] - r["volume_mm3"]) / (r[f"pred_mm3_t{t}"] + r["volume_mm3"]) for r in rows])
            old = np.mean([200 * (r["pred_mm3_t0.5"] - r["volume_mm3"]) / (r["pred_mm3_t0.5"] + r["volume_mm3"]) for r in rows])
        else:
            new = np.mean([r[f"pred_d3_t{t}"] - r["d3_mm"] for r in rows])
            old = np.mean([r["pred_d3_t0.5"] - r["d3_mm"] for r in rows])
        return abs(new) - abs(old)

    rule = []
    for t in THRESHOLDS[1:]:
        gain = detection(cases, t) - detection(cases, 0.5)
        gain_ci = paired_bootstrap(None, cases, lambda s, t=t: detection(s, t) - detection(s, 0.5))
        fp_change = fp_rate(cases, t) - fp_rate(cases, 0.5)
        vol = common_bias(cases, t, "volume")
        dia = common_bias(cases, t, "diameter")
        checks = {"detection_gain_interval_above_0": gain_ci[0] > 0, "fp_per_scan_increase_at_most_0.5": fp_change <= 0.5,
                  "common_pair_abs_volume_bias_not_worse_by_more_than_5pp": vol is not None and vol <= 5,
                  "common_pair_abs_diameter_bias_not_worse_by_more_than_0.5mm": dia is not None and dia <= 0.5}  # fmt: skip
        common = sum(1 for r in pieces if r["kind_t0.5"] == "one_to_one" and r[f"kind_t{t}"] == "one_to_one")
        rule.append({"threshold": t, "detected": sum(by[(c, t)]["detected"] for c in cases), "detection": round(detection(cases, t), 4),
                     "detection_gain": round(gain, 4), "detection_gain_interval_95": gain_ci,
                     "fp_pieces": sum(by[(c, t)]["false_positive_pieces"] for c in cases), "fp_per_scan": round(fp_rate(cases, t), 3),
                     "fp_per_scan_change": round(fp_change, 3), "common_pairs": common,
                     "common_pair_abs_volume_bias_change_pp": None if vol is None else round(vol, 2),
                     "common_pair_abs_diameter_bias_change_mm": None if dia is None else round(dia, 3),
                     "checks": checks, "passes": all(checks.values())})  # fmt: skip
    passing = [r["threshold"] for r in rule if r["passes"]]
    missed = [r for r in pieces if r["kind_t0.5"] == "missed"]
    detected = [r for r in pieces if r["kind_t0.5"] != "missed"]

    def describe(rows, key):
        values = np.array([r[key] for r in rows if r[key] is not None and np.isfinite(r[key])])
        return {"n": len(values), "median": round(float(np.median(values)), 2), "q1": round(float(np.percentile(values, 25)), 2),
                "q3": round(float(np.percentile(values, 75)), 2)} if len(values) else {"n": 0}  # fmt: skip

    def fp_class(r):
        if r["peak_hu"] >= 2500:
            return "metal-like (peak >= 2500 HU)"
        if r["distance_to_aorta_mm"] <= 5:
            return "aortic wall (within 5 mm of aorta label)"
        if r["distance_to_bone_mm"] <= 2:
            return "bone-adjacent (within 2 mm of bone proxy)"
        if r["inside_kidney_fraction"] > 0:
            return "inside kidney mask (unannotated focus or annotation omission; not adjudicated)"
        return "other soft tissue near kidney (e.g. vessels; not adjudicated)"

    classes = {}
    for r in fps:
        classes[fp_class(r)] = classes.get(fp_class(r), 0) + 1
    return {
        "baseline_threshold_0.5": {"detected": sum(by[(c, 0.5)]["detected"] for c in cases), "annotated": sum(by[(c, 0.5)]["annotated"] for c in cases),
                                   "fp_pieces": sum(by[(c, 0.5)]["false_positive_pieces"] for c in cases)},
        "threshold_candidates": rule,
        "threshold_rule": {"passing": passing, "selected": min(passing) if passing else 0.5,
                           "note": "Lowest passing t is selected (pre-specified); 0.5 is kept if none passes."},
        "missed_by_probability": {f">= {p}": sum(r["max_probability"] >= p for r in missed) for p in (0.05, 0.1, 0.2, 0.3, 0.4)}
        | {"missed": len(missed), "below 0.05": sum(r["max_probability"] < 0.05 for r in missed)},
        "missed_vs_detected": {
            key: {"missed": describe(missed, key), "detected": describe(detected, key)}
            for key in ("d3_mm", "voxels", "peak_hu", "local_contrast_hu", "max_probability", "distance_to_kidney_mm",
                        "inside_kidney_fraction", "distance_to_bone_mm")
        },
        "missed_inside_kidney_mask": sum(r["inside_kidney_fraction"] > 0 for r in missed),
        "missed_with_local_contrast_below_150_hu": sum(r["local_contrast_hu"] is not None and r["local_contrast_hu"] < 150 for r in missed),
        "false_positive_classes_threshold_0.5_filtered": classes,
        "false_positive_class_rules": "Applied in order: metal-like, aortic wall, bone-adjacent, inside kidney mask, other. Anatomical proxies, not adjudicated.",
    }


if __name__ == "__main__":
    main()

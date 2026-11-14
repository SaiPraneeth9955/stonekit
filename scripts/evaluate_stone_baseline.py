"""Run the whole pipeline on held-out validation scans, and compare with Li annotations.

Development evidence only, never final performance. These scans guided no model weights, but they are
not an independent test, and Li annotations are references rather than physical truth. Measurements
are compared with annotation-derived measurements, which is agreement and not physical accuracy.
Per-scan rows stay git-ignored; the run folder holds aggregates only.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.eval import seg_metrics
from stonekit.eval.bootstrap import percentile_interval, resample_indices
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.pipeline.run_case import run_case

SEED = 20261105
RESAMPLES = 10000
SMALL_MM3 = 100
MEASURES = {"volume_mm3": "volume_mm3", "diameter_3d_mm": "max_diameter_3d_mm", "diameter_axial_mm": "max_diameter_axial_mm"}


def evaluate_case(data, case_id, out, config, root) -> tuple[list[dict], list[dict], dict]:
    case = data.load(case_id)
    ct_path, annotation = data.paths(case_id)
    saved = root / "data" / "interim" / "li2022_kidney_ts" / f"{case_id}_ts.nii.gz"
    folder = run_case(ct_path, out, config, root, case_id, annotation, saved, "gpu")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report.get("failure"):
        raise ValueError(report["failure"])
    minimum = config["measurement"]["min_component_voxels"]
    reference = separate_stones(case.stone_reference, minimum)
    reference_stones = measure_stones(reference, case.image.data, case.image.affine, case.image.spacing_mm)
    by_id = {s["stone_id"]: s for s in reference_stones}
    rows, pairs = [], []
    for name, method in report["methods"].items():
        if not method.get("available"):
            continue
        labels = load_volume(folder / method["label_file"]).data
        counts = seg_metrics.detection_counts(labels, reference.labels)
        predicted = labels > 0
        rows.append(
            {
                "case": case_id,
                "method": name,
                "has_annotation_stone": bool(case.stone_reference.any()),
                "dice": seg_metrics.dice(predicted, case.stone_reference),
                "iou": seg_metrics.iou(predicted, case.stone_reference),
                **counts,
                "predicted_total_mm3": method["total_volume_mm3"],
                "annotation_total_mm3": sum(s["volume_mm3"] for s in reference_stones),
                "qc_flags": ";".join(method["qc_flags"]),
            }
        )
        stones = {s["stone_id"]: s for s in method["stones"]}
        for reference_id, predicted_id in seg_metrics.best_matches(labels, reference.labels).items():
            pair = {"case": case_id, "method": name, "annotation_mm3": by_id[reference_id]["volume_mm3"]}
            for key, field in MEASURES.items():
                pair[f"annotation_{key}"] = by_id[reference_id][field]
                pair[f"predicted_{key}"] = stones[predicted_id][field]
            pairs.append(pair)
    info = {"case": case_id, "folder": str(folder), "total_seconds": report["total_seconds"], "peak_memory_kb": report["peak_memory_kb"]}
    return rows, pairs, info


def mean_interval(values: list[float]) -> dict:
    """Mean with a percentile bootstrap interval over scans."""
    array = np.array(values, dtype=float)
    if not len(array):
        return {"n": 0}
    means = array[resample_indices(len(array), RESAMPLES, SEED)].mean(axis=1)
    interval = percentile_interval(means)
    return {
        "n": len(array),
        "mean": round(float(array.mean()), 4),
        "bootstrap_95_interval_over_scans": [round(interval.lower, 4), round(interval.upper, 4)],
        "quartiles": [round(float(np.percentile(array, q)), 4) for q in (25, 50, 75)],
    }


def summarise_method(rows: list[dict], pairs: list[dict]) -> dict:
    stone = [r for r in rows if r["has_annotation_stone"]]
    none = [r for r in rows if not r["has_annotation_stone"]]
    reference = sum(r["reference_pieces"] for r in rows)
    detected = sum(r["reference_pieces_detected"] for r in rows)
    predicted = sum(r["predicted_pieces"] for r in rows)
    false = sum(r["predicted_pieces_without_reference"] for r in rows)
    agreement = {}
    for key in MEASURES:
        error = np.array([p[f"predicted_{key}"] - p[f"annotation_{key}"] for p in pairs], dtype=float)
        agreement[key] = {
            "median_signed_difference": round(float(np.median(error)), 3) if len(error) else None,
            "median_absolute_difference": round(float(np.median(np.abs(error))), 3) if len(error) else None,
            "p90_absolute_difference": round(float(np.percentile(np.abs(error), 90)), 3) if len(error) else None,
        }
    small = [p for p in pairs if p["annotation_mm3"] < SMALL_MM3]
    return {
        "scans": len(rows),
        "scans_with_annotation_stone": len(stone),
        "scans_without_annotation_stone": len(none),
        "dice": mean_interval([r["dice"] for r in stone if r["dice"] is not None]),
        "iou": mean_interval([r["iou"] for r in stone if r["iou"] is not None]),
        "annotation_pieces": reference,
        "annotation_pieces_detected_any_overlap": detected,
        "predicted_pieces": predicted,
        "predicted_pieces_without_annotation_overlap": false,
        "false_pieces_per_scan": round(false / len(rows), 3) if rows else None,
        "no_stone_scans_with_any_predicted_piece": sum(r["predicted_pieces"] > 0 for r in none),
        "matched_pieces": len(pairs),
        "matched_pieces_under_100_mm3": len(small),
        "agreement_with_annotation_measurements": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cases", nargs="*", help="Default: all validation scans")
    parser.add_argument("--trainer", help="Evaluate another trained nnU-Net trainer, e.g. nnUNetTrainer_250epochs")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_yaml(root / "configs" / "baseline_pipeline.yaml")
    if args.trainer:
        config["model"]["trainer"] = args.trainer
    split = json.loads((root / "data" / "nnunet" / "li_stone_split.json").read_text(encoding="utf-8"))
    data = LiDevelopmentData(root)
    cases = args.cases or split["validation"]
    require_development(cases, data.manifest)
    if set(cases) & set(split["train"]):
        raise ValueError("Evaluation must use scans held out from training")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{args.trainer.replace('nnUNetTrainer_', '')}" if args.trainer else ""
    out = root / "data" / "interim" / "stonekit_runs" / f"validation{suffix}_{stamp}"
    rows, pairs, infos, failures = [], [], [], []
    for case_id in cases:
        try:
            case_rows, case_pairs, info = evaluate_case(data, case_id, out, config, root)
        except (OSError, ValueError, MemoryError, EOFError) as error:
            failures.append({"case": case_id, "failure": f"{type(error).__name__}: {error}"[:300]})
            print(case_id, "FAILED", error, flush=True)
            continue
        rows += case_rows
        pairs += case_pairs
        infos.append(info)
        print(case_id, info["total_seconds"], flush=True)

    with data.review_sheet.open(newline="", encoding="utf-8") as f:
        flagged = {row["case"] for row in csv.DictReader(f)}
    methods = sorted({r["method"] for r in rows})
    seconds = [i["total_seconds"] for i in infos]
    summary = {
        "status": "DEVELOPMENT_EVIDENCE_ONLY_NOT_FINAL_PERFORMANCE",
        "note": "Li annotations are references, not physical truth. A piece counts as detected with any voxel overlap.",
        "scans_attempted": len(cases),
        "scans_failed": len(failures),
        "failure_reasons": sorted({f["failure"] for f in failures}),
        "reviewed_flagged_scans_included": len(flagged & set(cases)),
        "seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "seconds_per_scan_excluding_kidney_step": {
            "median": round(float(np.median(seconds)), 1) if seconds else None,
            "max": round(float(np.max(seconds)), 1) if seconds else None,
        },
        "peak_rss_mib_this_process": round(max((i["peak_memory_kb"]["this_process"] for i in infos), default=0) / 1024),
        "methods": {
            name: {
                "all_scans": summarise_method(
                    [r for r in rows if r["method"] == name], [p for p in pairs if p["method"] == name]
                ),
                "excluding_reviewed_flagged_scans": summarise_method(
                    [r for r in rows if r["method"] == name and r["case"] not in flagged],
                    [p for p in pairs if p["method"] == name and p["case"] not in flagged],
                ),
            }
            for name in methods
        },
    }
    run = create_run_folder(root / "results" / "runs", f"stone_baseline_validation{suffix}")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fields = sorted({key for row in rows for key in row})
    with (out / "per_scan.csv").open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved to", run, "and", out)


if __name__ == "__main__":
    main()

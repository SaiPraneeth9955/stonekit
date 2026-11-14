"""Stone-level analysis of a finished validation run. Reads saved outputs only; reruns nothing.

For each method in each report: group predicted and annotated pieces (stonekit.eval.stone_matching),
then report detection by size, attenuation and slice spacing, false positives, splits and merges,
and measurement agreement on one-to-one groups.

Definitions fixed in this file before this analysis was first run (7 November 2026). The validation
scans' aggregate Dice was already seen, so this analysis is exploratory development evidence.
- Annotated piece: 26-connected component of Li label 1 with at least 3 voxels.
- Detected: the piece shares at least one voxel with any predicted piece.
- Size band: annotated piece's centre-to-centre 3D diameter, <3, 3-<5, 5-<10, >=10 mm.
- Attenuation band: annotated piece's peak HU, <400, 400-<800, >=800 HU. Attenuation only; not
  composition.
- Slice spacing band: <1.0, 1.0-<1.5, 1.5-<2.2, >=2.2 mm.
- Intervals: percentile bootstrap over scans (scans resampled, all their pieces kept), seed 20261105,
  10000 resamples, because pieces within a scan are not independent.
- Agreement: one-to-one groups only. Volume as percentage difference against the pair mean
  (Bland-Altman on a relative scale, since error grows with size); diameters in mm.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.eval import seg_metrics
from stonekit.eval.stone_matching import group_pieces
from stonekit.io.nifti import load_volume
from stonekit.measure.stones import measure_stones, separate_stones

SEED = 20261105
RESAMPLES = 10000
MIN_SCANS = 5
SIZE_BANDS = [(0, 3, "<3 mm"), (3, 5, "3-<5 mm"), (5, 10, "5-<10 mm"), (10, np.inf, ">=10 mm")]
HU_BANDS = [(-np.inf, 400, "<400 HU"), (400, 800, "400-<800 HU"), (800, np.inf, ">=800 HU")]
SPACING_BANDS = [(0, 1.0, "<1.0 mm"), (1.0, 1.5, "1.0-<1.5 mm"), (1.5, 2.2, "1.5-<2.2 mm"), (2.2, np.inf, ">=2.2 mm")]


def detected(row) -> bool:
    return row["kind"] != "missed"


def band(value, bands):
    for low, high, name in bands:
        if low <= value < high:
            return name
    return "none"


def analyse_case(data, folder: Path, minimum: int, kidney_distance_mm: float | None = None) -> dict:
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    case_id = report["case_id"]
    if report.get("failure"):
        return {"case": case_id, "failure": report["failure"]}
    case = data.load(case_id)
    spacing_z = float(case.image.spacing_mm[2])
    reference = separate_stones(case.stone_reference, minimum)
    annotated = {
        s["stone_id"]: s
        for s in measure_stones(reference, case.image.data, case.image.affine, case.image.spacing_mm)
    }
    out = {"case": case_id, "spacing_z": spacing_z, "annotated": [], "predicted": [], "pairs": [], "scans": []}
    variants = []
    for name, method in report["methods"].items():
        if method.get("available"):
            variants.append((name, method, None))
            if kidney_distance_mm is not None and name == "model":
                variants.append((f"{name}_within_{kidney_distance_mm:g}mm_of_kidney", method, kidney_distance_mm))
    for name, method, limit in variants:
        labels = load_volume(folder / method["label_file"]).data
        predicted = {s["stone_id"]: s for s in method["stones"]}
        if limit is not None:
            removed = [i for i, s in predicted.items() if s["distance_to_kidney_mm"] > limit]
            labels = np.where(np.isin(labels, removed), 0, labels)
            predicted = {i: s for i, s in predicted.items() if i not in removed}
        groups = group_pieces(labels, reference.labels, set(predicted), set(annotated))
        kinds = {}
        for group in groups:
            for i in group.annotated:
                kinds[("a", i)] = group.kind
            for i in group.predicted:
                kinds[("p", i)] = group.kind
            if group.kind == "one_to_one":
                a, p = annotated[group.annotated[0]], predicted[group.predicted[0]]
                out["pairs"].append(
                    {
                        "case": case_id, "method": name, "spacing_z": spacing_z,
                        "annotated_mm3": a["volume_mm3"], "predicted_mm3": p["volume_mm3"],
                        "annotated_d3_mm": a["max_diameter_3d_mm"], "predicted_d3_mm": p["max_diameter_3d_mm"],
                        "annotated_axial_mm": a["max_diameter_axial_mm"], "predicted_axial_mm": p["max_diameter_axial_mm"],
                        "annotated_peak_hu": a["max_hu"],
                    }
                )  # fmt: skip
        for i, a in annotated.items():
            out["annotated"].append(
                {
                    "case": case_id, "method": name, "spacing_z": spacing_z, "kind": kinds[("a", i)],
                    "volume_mm3": a["volume_mm3"], "d3_mm": a["max_diameter_3d_mm"], "peak_hu": a["max_hu"],
                    "mean_hu": a["mean_hu"],
                }
            )  # fmt: skip
        for i, p in predicted.items():
            out["predicted"].append(
                {
                    "case": case_id, "method": name, "kind": kinds[("p", i)], "volume_mm3": p["volume_mm3"],
                    "d3_mm": p["max_diameter_3d_mm"], "peak_hu": p["max_hu"],
                    "distance_to_kidney_mm": p["distance_to_kidney_mm"], "qc_flags": ";".join(p["qc_flags"]),
                }
            )  # fmt: skip
        mask = labels > 0
        out["scans"].append(
            {
                "case": case_id, "method": name, "spacing_z": spacing_z,
                "has_stone": bool(annotated), "dice": seg_metrics.dice(mask, case.stone_reference),
                "predicted_pieces": len(predicted), "annotated_pieces": len(annotated),
                "false_positive_pieces": sum(1 for g in groups if g.kind == "false_positive"),
            }
        )  # fmt: skip
    return out


def bootstrap_ratio(rows, numerator, denominator) -> dict:
    """Pooled ratio sum(numerator)/sum(denominator) with an interval from resampling scans."""
    cases = sorted({r["case"] for r in rows})
    if not cases:
        return {"n": 0}
    per = {c: [0.0, 0.0] for c in cases}
    for r in rows:
        per[r["case"]][0] += numerator(r)
        per[r["case"]][1] += denominator(r)
    values = np.array([per[c] for c in cases])
    total = values.sum(axis=0)
    draws = np.random.default_rng(SEED).integers(0, len(cases), size=(RESAMPLES, len(cases)))
    sums = values[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratios = sums[:, 0] / sums[:, 1]
    ratios = ratios[np.isfinite(ratios)]
    return {
        **sample_note(len(cases)),
        "numerator": round(float(total[0]), 3),
        "denominator": round(float(total[1]), 3),
        "value": round(float(total[0] / total[1]), 4) if total[1] else None,
        "interval_95": [round(float(np.percentile(ratios, 2.5)), 4), round(float(np.percentile(ratios, 97.5)), 4)] if len(ratios) else None,
        "scans": len(cases),
    }


def sample_note(scans: int) -> dict:
    """A reporting convention, not a statistical rule: under 5 scans a subgroup is descriptive only."""
    return {"small_sample_warning": f"only {scans} scan(s): descriptive only, interval unreliable"} if scans < MIN_SCANS else {}


def bootstrap_mean(rows, value) -> dict:
    """Mean over pieces, with a percentile interval from resampling scans (all RESAMPLES draws).

    Before 8 November 2026 only the first 2000 of the 10000 draws were used here, against the stated
    definition; corrected then (docs/decisions.md). Point estimates and limits of agreement are unchanged.
    """
    cases = sorted({r["case"] for r in rows})
    if not rows:
        return {"n": 0}
    sums = np.array([sum(value(r) for r in rows if r["case"] == c) for c in cases], dtype=float)
    counts = np.array([sum(1 for r in rows if r["case"] == c) for c in cases], dtype=float)
    draws = np.random.default_rng(SEED).integers(0, len(cases), size=(RESAMPLES, len(cases)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    values = np.array([value(r) for r in rows])
    return {
        **sample_note(len(cases)),
        "n": len(values),
        "scans": len(cases),
        "bootstrap_resamples": RESAMPLES,
        "mean": round(float(values.mean()), 3),
        "sd": round(float(values.std(ddof=1)), 3) if len(values) > 1 else None,
        "mean_interval_95": [round(float(np.percentile(means, 2.5)), 3), round(float(np.percentile(means, 97.5)), 3)],
        "limits_of_agreement": [
            round(float(values.mean() - 1.96 * values.std(ddof=1)), 3),
            round(float(values.mean() + 1.96 * values.std(ddof=1)), 3),
        ]
        if len(values) > 1
        else None,
        "median": round(float(np.median(values)), 3),
    }


def summarise(results: list[dict]) -> dict:
    annotated = [r for c in results for r in c.get("annotated", [])]
    predicted = [r for c in results for r in c.get("predicted", [])]
    pairs = [r for c in results for r in c.get("pairs", [])]
    scans = [r for c in results for r in c.get("scans", [])]
    summary = {}
    for method in sorted({r["method"] for r in scans}):
        a = [r for r in annotated if r["method"] == method]
        p = [r for r in predicted if r["method"] == method]
        q = [r for r in pairs if r["method"] == method]
        s = [r for r in scans if r["method"] == method]
        m = {
            "annotated_pieces": len(a),
            "predicted_pieces": len(p),
            "group_kinds_annotated": {k: sum(r["kind"] == k for r in a) for k in sorted({r["kind"] for r in a})},
            "group_kinds_predicted": {k: sum(r["kind"] == k for r in p) for k in sorted({r["kind"] for r in p})},
            "detection_rate_all": bootstrap_ratio(a, detected, lambda r: 1),
            "precision_pieces": bootstrap_ratio(p, lambda r: r["kind"] != "false_positive", lambda r: 1),
            "detection_by_size": {
                name: bootstrap_ratio([r for r in a if band(r["d3_mm"], SIZE_BANDS) == name], detected, lambda r: 1)
                for _, _, name in SIZE_BANDS
            },
            "annotated_volume_detected_fraction": bootstrap_ratio(a, lambda r: r["volume_mm3"] * detected(r), lambda r: r["volume_mm3"]),
            "detection_by_peak_hu": {
                name: bootstrap_ratio([r for r in a if band(r["peak_hu"], HU_BANDS) == name], detected, lambda r: 1)
                for _, _, name in HU_BANDS
            },
            "detection_by_slice_spacing": {
                name: bootstrap_ratio([r for r in a if band(r["spacing_z"], SPACING_BANDS) == name], detected, lambda r: 1)
                for _, _, name in SPACING_BANDS
            },
            "missed_pieces_median_d3_mm": round(float(np.median([r["d3_mm"] for r in a if not detected(r)])), 2) if any(not detected(r) for r in a) else None,
            "detected_pieces_median_d3_mm": round(float(np.median([r["d3_mm"] for r in a if detected(r)])), 2) if any(detected(r) for r in a) else None,
            "false_positive": {
                "pieces": sum(r["kind"] == "false_positive" for r in p),
                "per_scan": round(sum(r["kind"] == "false_positive" for r in p) / len(s), 3),
                "median_volume_mm3": round(float(np.median([r["volume_mm3"] for r in p if r["kind"] == "false_positive"])), 2) if any(r["kind"] == "false_positive" for r in p) else None,
                "within_5mm_of_kidney": sum(r["kind"] == "false_positive" and r["distance_to_kidney_mm"] <= 5 for r in p),
                "flagged_very_small": sum(r["kind"] == "false_positive" and "very_small_component" in r["qc_flags"] for r in p),
                "true_pieces_flagged_very_small": sum(r["kind"] != "false_positive" and "very_small_component" in r["qc_flags"] for r in p),
            },
            "dice_scans_with_stone_by_slice_spacing": {
                name: bootstrap_mean([r for r in s if r["has_stone"] and band(r["spacing_z"], SPACING_BANDS) == name], lambda r: r["dice"])
                for _, _, name in SPACING_BANDS
            },
            "no_stone_scans": {
                "scans": sum(not r["has_stone"] for r in s),
                "with_any_predicted_piece": sum((not r["has_stone"]) and r["predicted_pieces"] > 0 for r in s),
            },
            "agreement_one_to_one": {
                "pairs": len(q),
                "volume_percent_difference": bootstrap_mean(q, lambda r: 200 * (r["predicted_mm3"] - r["annotated_mm3"]) / (r["predicted_mm3"] + r["annotated_mm3"])),
                "diameter_3d_difference_mm": bootstrap_mean(q, lambda r: r["predicted_d3_mm"] - r["annotated_d3_mm"]),
                "diameter_axial_difference_mm": bootstrap_mean(q, lambda r: r["predicted_axial_mm"] - r["annotated_axial_mm"]),
                "by_size": {
                    name: {
                        "volume_percent_difference": bootstrap_mean(
                            [r for r in q if band(r["annotated_d3_mm"], SIZE_BANDS) == name],
                            lambda r: 200 * (r["predicted_mm3"] - r["annotated_mm3"]) / (r["predicted_mm3"] + r["annotated_mm3"]),
                        ),
                        "diameter_3d_difference_mm": bootstrap_mean(
                            [r for r in q if band(r["annotated_d3_mm"], SIZE_BANDS) == name], lambda r: r["predicted_d3_mm"] - r["annotated_d3_mm"]
                        ),
                    }
                    for _, _, name in SIZE_BANDS
                },
                "spearman_size_vs_abs_volume_percent_error": _spearman(q),
            },
        }
        summary[method] = m
    return summary


def _spearman(q):
    if len(q) < 5:
        return None
    size = [r["annotated_d3_mm"] for r in q]
    error = [abs(200 * (r["predicted_mm3"] - r["annotated_mm3"]) / (r["predicted_mm3"] + r["annotated_mm3"])) for r in q]
    rho, p = stats.spearmanr(size, error)
    return {"rho": round(float(rho), 3), "p_value": round(float(p), 4), "n": len(q), "note": "pieces within a scan are not independent"}


def figures(results, run: Path, method_labels: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6, "axes.axisbelow": True})
    colours = {"model": "#1f5f8b", "threshold_272hu": "#8a8a8a"}
    written = []
    annotated = [r for c in results for r in c.get("annotated", [])]
    pairs = [r for c in results for r in c.get("pairs", [])]
    folder = run / "figures"
    folder.mkdir()

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    methods = sorted({r["method"] for r in annotated})
    width = 0.8 / max(len(methods), 1)
    for k, method in enumerate(methods):
        for j, (_, _, name) in enumerate(SIZE_BANDS):
            rows = [r for r in annotated if r["method"] == method and band(r["d3_mm"], SIZE_BANDS) == name]
            ratio = bootstrap_ratio(rows, lambda r: r["kind"] != "missed", lambda r: 1)
            if not ratio.get("denominator"):
                continue
            x = j + (k - (len(methods) - 1) / 2) * width
            ax.bar(x, ratio["value"], width * 0.92, color=colours.get(method, "#444"), label=method_labels.get(method, method) if j == 0 else None)
            low, high = ratio["interval_95"]
            ax.errorbar(x, ratio["value"], yerr=[[ratio["value"] - low], [high - ratio["value"]]], color="#222", capsize=2, linewidth=0.8)
    counts = [sum(1 for r in annotated if r["method"] == methods[0] and band(r["d3_mm"], SIZE_BANDS) == b[2]) for b in SIZE_BANDS] if methods else []
    ax.set_xticks(range(len(SIZE_BANDS)), [f"{b[2]}\n(n={n})" for b, n in zip(SIZE_BANDS, counts, strict=False)])
    ax.set_xlabel("Annotated piece 3D diameter")
    ax.set_ylabel("Pieces detected (any overlap)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Detection of annotated stone pieces by size, validation scans", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "detection_by_size.png", dpi=200)
    plt.close(fig)
    written.append("figures/detection_by_size.png")

    for method in sorted({r["method"] for r in pairs}):
        q = [r for r in pairs if r["method"] == method]
        if len(q) < 3:
            continue
        mean = np.array([(r["predicted_mm3"] + r["annotated_mm3"]) / 2 for r in q])
        diff = np.array([200 * (r["predicted_mm3"] - r["annotated_mm3"]) / (r["predicted_mm3"] + r["annotated_mm3"]) for r in q])
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
        ax = axes[0]
        ax.scatter(mean, diff, s=10, color=colours.get(method, "#444"), alpha=0.8, linewidths=0)
        bias, sd = diff.mean(), diff.std(ddof=1)
        for y, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
            ax.axhline(y, color="#333", linestyle=style, linewidth=0.8)
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("Mean of predicted and annotated volume (mm³)")
        ax.set_ylabel("Volume difference (% of mean)")
        ax.set_title(f"Bland-Altman, {method_labels.get(method, method)}, n={len(q)}", loc="left", fontsize=9)
        ax = axes[1]
        a3 = np.array([r["annotated_d3_mm"] for r in q])
        d3 = np.array([r["predicted_d3_mm"] - r["annotated_d3_mm"] for r in q])
        ax.scatter(a3, d3, s=10, color=colours.get(method, "#444"), alpha=0.8, linewidths=0)
        ax.axhline(0, color="#999", linewidth=0.6)
        ax.axhline(d3.mean(), color="#333", linewidth=0.8)
        ax.set_xlabel("Annotated 3D diameter (mm)")
        ax.set_ylabel("Predicted − annotated 3D diameter (mm)")
        ax.set_title("3D diameter difference against size", loc="left", fontsize=9)
        fig.tight_layout()
        name = f"figures/agreement_{method}.png"
        fig.savefig(run / name, dpi=200)
        plt.close(fig)
        written.append(name)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, required=True, help="A validation run folder of case folders")
    parser.add_argument("--label", default="validation")
    parser.add_argument("--kidney-distance-mm", type=float, help="Also analyse the model with this kidney-distance filter")
    parser.add_argument("--model-label", default="nnU-Net baseline (100 epochs)")
    args = parser.parse_args()
    root = args.root.resolve()
    data = LiDevelopmentData(root)
    folders = sorted(p for p in args.runs.iterdir() if (p / "report.json").exists())
    cases = [json.loads((p / "report.json").read_text(encoding="utf-8"))["case_id"] for p in folders]
    require_development(cases, data.manifest)
    split = json.loads((root / "data" / "nnunet" / "li_stone_split.json").read_text(encoding="utf-8"))
    if set(cases) & set(split["train"]):
        raise ValueError("Analysis must use scans held out from training")
    if set(cases) - set(split["validation"]):
        raise ValueError("Analysis may use fold-0 validation scans only")
    if len(cases) != len(set(cases)):
        raise ValueError("A scan appears more than once in the run folder")
    results = []
    for folder in folders:
        results.append(analyse_case(data, folder, 3, args.kidney_distance_mm))
        print(results[-1]["case"], flush=True)
    summary = {
        "status": "EXPLORATORY_DEVELOPMENT_EVIDENCE_NOT_FINAL_PERFORMANCE",
        "source_runs": str(args.runs),
        "definitions": __doc__,
        "scans": len(results),
        "validation_scans_expected": len(split["validation"]),
        "validation_scans_missing": len(set(split["validation"]) - set(cases)),
        "failed_reports": [r["case"] for r in results if "failure" in r],
        "failed_reports_note": "A failed report contributes no pieces to any denominator; it is listed here instead.",
        "methods": summarise(results),
    }
    run = create_run_folder(root / "results" / "runs", f"stone_analysis_{args.label}")
    labels = {"model": args.model_label, "threshold_272hu": "272 HU threshold"}
    if args.kidney_distance_mm is not None:
        labels[f"model_within_{args.kidney_distance_mm:g}mm_of_kidney"] = f"{args.model_label}, <= {args.kidney_distance_mm:g} mm from kidney"
    summary["model_label"] = args.model_label
    summary["method_labels"] = labels
    summary["figures"] = figures(results, run, labels)
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    detail = root / "data" / "interim" / "stone_analysis" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    detail.mkdir(parents=True)
    for key in ("annotated", "predicted", "pairs", "scans"):
        rows = [r for c in results for r in c.get(key, [])]
        if rows:
            with (detail / f"{key}.csv").open("x", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    print(json.dumps(summary["methods"], indent=1)[:20000])
    print("Saved to", run, "and", detail)


if __name__ == "__main__":
    main()

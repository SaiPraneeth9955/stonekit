"""Recompute a stone-level analysis summary from its saved per-piece rows. Reads no image.

Input: a detail folder written by scripts/analyse_stone_validation.py
(data/interim/stone_analysis/<stamp>/ with annotated.csv, predicted.csv, pairs.csv, scans.csv) and
the summary.json written with it. The summary is rebuilt with the current statistics code, and every
count (pieces, detections, false positives, pairs, scans) is checked against the original summary;
any difference stops the script. Written on 8 November 2026 to apply the bootstrap_mean correction
(docs/decisions.md) without re-reading the scans. Development evidence only.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

from stonekit.config import create_run_folder

SCRIPTS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analyse_stone_validation", SCRIPTS / "analyse_stone_validation.py")
analyse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyse)

FLOATS = {"spacing_z", "volume_mm3", "d3_mm", "peak_hu", "mean_hu", "distance_to_kidney_mm", "annotated_mm3",
          "predicted_mm3", "annotated_d3_mm", "predicted_d3_mm", "annotated_axial_mm", "predicted_axial_mm",
          "annotated_peak_hu", "dice"}  # fmt: skip
INTS = {"predicted_pieces", "annotated_pieces", "false_positive_pieces"}
BOOLS = {"has_stone"}


def typed(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        if key in FLOATS:
            out[key] = None if value in ("", "None") else float(value)
        elif key in INTS:
            out[key] = int(value)
        elif key in BOOLS:
            out[key] = value == "True"
        else:
            out[key] = value
    return out


def read_rows(folder: Path) -> list[dict]:
    """Rebuild the per-case structure that analyse_stone_validation.summarise expects."""
    cases: dict[str, dict] = {}
    for key in ("annotated", "predicted", "pairs", "scans"):
        path = folder / f"{key}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                case = cases.setdefault(row["case"], {"case": row["case"], "annotated": [], "predicted": [], "pairs": [], "scans": []})
                case[key].append(typed(row))
    return [cases[c] for c in sorted(cases)]


COUNTS = ("annotated_pieces", "predicted_pieces", "group_kinds_annotated", "group_kinds_predicted")


def compare_counts(old: dict, new: dict) -> list[str]:
    problems = []
    if set(old) != set(new):
        return [f"methods differ: {sorted(old)} vs {sorted(new)}"]
    for method, a in old.items():
        b = new[method]
        for key in COUNTS:
            if a[key] != b[key]:
                problems.append(f"{method} {key}: {a[key]} vs {b[key]}")
        for key in ("numerator", "denominator", "value"):
            if a["detection_rate_all"][key] != b["detection_rate_all"][key]:
                problems.append(f"{method} detection {key}")
        if a["false_positive"] != b["false_positive"]:
            problems.append(f"{method} false_positive")
        if a["agreement_one_to_one"]["pairs"] != b["agreement_one_to_one"]["pairs"]:
            problems.append(f"{method} pairs")
        for key in ("mean", "limits_of_agreement", "n"):
            if a["agreement_one_to_one"]["volume_percent_difference"].get(key) != b["agreement_one_to_one"]["volume_percent_difference"].get(key):
                problems.append(f"{method} volume {key}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True, help="data/interim/stone_analysis/<stamp>")
    parser.add_argument("--summary", type=Path, required=True, help="The run folder whose summary.json came with it")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    original = json.loads((args.summary / "summary.json").read_text(encoding="utf-8"))
    results = read_rows(args.detail)
    methods = analyse.summarise(results)
    problems = compare_counts(original["methods"], methods)
    if problems:
        raise SystemExit("Counts do not reproduce: " + "; ".join(problems))
    summary = {
        **{k: original[k] for k in ("status", "source_runs", "scans", "failed_reports", "model_label", "method_labels") if k in original},
        "definitions": analyse.__doc__,
        "recomputed_from": {"detail_rows": str(args.detail), "original_summary": str(args.summary)},
        "recomputation_note": "Rebuilt from saved per-piece rows with the corrected bootstrap_mean (all 10000 resamples). "
        "Every count matched the original summary; only bootstrap intervals of mean agreement values can differ.",
        "methods": methods,
    }
    run = create_run_folder(Path("results/runs"), f"stone_analysis_{args.label}")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for method, m in methods.items():
        old = original["methods"][method]["agreement_one_to_one"]
        new = m["agreement_one_to_one"]
        print(method, "| detection", new and m["detection_rate_all"]["numerator"], "/", m["detection_rate_all"]["denominator"],
              "| volume bias", new["volume_percent_difference"]["mean"], "CI old", old["volume_percent_difference"].get("mean_interval_95"),
              "new", new["volume_percent_difference"].get("mean_interval_95"),
              "| diameter", new["diameter_3d_difference_mm"]["mean"], "CI old", old["diameter_3d_difference_mm"].get("mean_interval_95"),
              "new", new["diameter_3d_difference_mm"].get("mean_interval_95"))  # fmt: skip
    print("Saved to", run)


if __name__ == "__main__":
    main()

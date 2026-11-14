"""Sensitivity of piece-level results to two post-processing rules, from saved analysis rows only.

Exploratory development evidence. No rule here is adopted: choosing a rule on these validation scans
and reporting it on the same scans would be selection on the evaluation data. The question is only
how much each rule would change false positives and true pieces.

Rules, fixed before running:
- minimum predicted piece volume: 0 (only the existing 3-voxel rule), 5, 10, 20, 50, 100 mm3
- maximum distance from the kidney segmentation: none, 20, 10, 5, 0 mm
A predicted piece is "true" if it overlaps an annotated piece (one_to_one, split, merged or complex).
An annotated piece counts as lost only if every predicted piece overlapping it is removed; split and
complex groups are handled through the saved group membership in predicted.csv and annotated.csv,
approximated here as lost when its one_to_one partner is removed. The approximation is stated with
the results.
"""

import argparse
import csv
import json
from pathlib import Path

from stonekit.config import create_run_folder

VOLUMES = (0, 5, 10, 20, 50, 100)
DISTANCES = (None, 20, 10, 5, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True, help="data/interim/stone_analysis/<stamp>")
    parser.add_argument("--method", default="model")
    args = parser.parse_args()
    with (args.detail / "predicted.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == args.method]
    with (args.detail / "annotated.csv").open(newline="", encoding="utf-8") as f:
        annotated = [r for r in csv.DictReader(f) if r["method"] == args.method]
    detected = sum(r["kind"] != "missed" for r in annotated)

    def evaluate(keep) -> dict:
        kept = [r for r in rows if keep(r)]
        removed = [r for r in rows if not keep(r)]
        return {
            "false_positive_pieces_kept": sum(r["kind"] == "false_positive" for r in kept),
            "false_positive_pieces_removed": sum(r["kind"] == "false_positive" for r in removed),
            "true_pieces_removed": sum(r["kind"] != "false_positive" for r in removed),
            "one_to_one_pieces_removed": sum(r["kind"] == "one_to_one" for r in removed),
            "annotated_detected_before": detected,
            "annotated_detected_after_approx": detected - sum(r["kind"] == "one_to_one" for r in removed),
        }

    summary = {
        "method": args.method,
        "source": str(args.detail),
        "definitions": __doc__,
        "annotated_pieces": len(annotated),
        "predicted_pieces": len(rows),
        "minimum_volume_mm3": {str(v): evaluate(lambda r, v=v: float(r["volume_mm3"]) >= v) for v in VOLUMES},
        "maximum_distance_to_kidney_mm": {
            str(d): evaluate(lambda r, d=d: d is None or float(r["distance_to_kidney_mm"]) <= d) for d in DISTANCES
        },
        "annotated_pieces_far_from_kidney_note": "Distance is from predicted pieces; annotated-piece distances were not saved.",
    }
    run = create_run_folder(Path("results/runs"), f"postprocessing_sensitivity_{args.method}")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("minimum_volume_mm3", "maximum_distance_to_kidney_mm")}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

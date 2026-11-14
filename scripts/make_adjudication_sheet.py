"""Build the BLINDED radiology adjudication sheet for D-REV (owner approval, 12 November 2026).

Reads the immutable non-radiologist screening sheet (data/interim/li2022_full_audit/review/
li_slicer_review_sheet.csv) and writes a NEW sheet for the radiology-trained adjudicator. The screening
sheet is opened read-only; nothing in it is changed.

The adjudicator's sheet carries only what is needed to find and judge each location:
case, location number, a neutral location type, the RAS point in mm, and the CT and annotation paths.
It carries NO screening code, confidence or note; NO development/calibration/test group (the split
manifest is never read here); NO model prediction, E6/E7/E8 result or proposed exclusion.
The 24 rows are put in a new seeded random order, so the order of the screening pass is not reproduced.

Outputs (git-ignored folder data/interim/li2022_full_audit/adjudication/; never overwritten):
- li_radiology_adjudication_sheet_v1.csv (blank answer columns);
- adjudication_manifest.json (input and output SHA-256, seed, columns, time).
"""

import argparse
import csv
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

SEED = 20261105
SCREENING = Path("data/interim/li2022_full_audit/review/li_slicer_review_sheet.csv")
OUT_DIR = Path("data/interim/li2022_full_audit/adjudication")
SHEET_NAME = "li_radiology_adjudication_sheet_v1.csv"
LOCATION_TYPE = {
    "label 1 region": "labelled stone region",
    "unlabelled bright region": "unlabelled region",
    "label 2 region, overall appearance": "kidney label region (overall appearance)",
}
LOCATION_FIELDS = ["adj_id", "case", "target", "location_type", "R", "A", "S", "image_file", "mask_file"]
ANSWER_FIELDS = ["q1_finding", "q1_nonstone_type", "q2_label_validity", "q3_scan_contrast", "confidence",
                 "adjudicator_role", "ai_assistance", "blinded_first_pass", "note", "timestamp_utc"]  # fmt: skip
SCREENING_ONLY = {"review_order", "look_at", "finding_code", "confidence_high_medium_low", "note"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(screening_rows: list[dict], seed: int = SEED) -> list[dict]:
    """Blinded rows in a seeded random order. Raises on an unknown location type (never guessed)."""
    rows = []
    for r in screening_rows:
        if r["look_at"] not in LOCATION_TYPE:
            raise ValueError(f"unknown look_at {r['look_at']!r} for {r['case']}")
        rows.append({"case": r["case"], "target": r["target"], "location_type": LOCATION_TYPE[r["look_at"]],
                     "R": r["R"], "A": r["A"], "S": r["S"], "image_file": r["image_file"], "mask_file": r["mask_file"]})  # fmt: skip
    keys = [(r["case"], r["target"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate (case, target) in the screening sheet")
    random.Random(seed).shuffle(rows)
    for i, r in enumerate(rows, start=1):
        r["adj_id"] = f"ADJ{i:02d}"
    return [{**{k: r[k] for k in LOCATION_FIELDS}, **dict.fromkeys(ANSWER_FIELDS, "")} for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screening", type=Path, default=SCREENING)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir / SHEET_NAME
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    before = sha256(args.screening)
    with open(args.screening, newline="", encoding="utf-8") as handle:
        screening = list(csv.DictReader(handle))
    rows = build_rows(screening)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOCATION_FIELDS + ANSWER_FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    if sha256(args.screening) != before:
        raise SystemExit("screening sheet changed during the build; stop")
    manifest = {
        "purpose": "D-REV radiology-trained adjudication, blinded first pass (owner approval 12 November 2026)",
        "screening_input": str(args.screening), "screening_input_sha256": before,
        "adjudication_sheet": str(out), "adjudication_sheet_sha256_blank": sha256(out),
        "rows": len(rows), "scans": len({r["case"] for r in rows}), "order_seed": SEED,
        "columns": LOCATION_FIELDS + ANSWER_FIELDS,
        "excluded_from_sheet": sorted(SCREENING_ONLY) + ["split group", "model predictions", "E6/E7/E8 results", "proposed exclusions"],
        "created_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    }  # fmt: skip
    with open(args.out_dir / "adjudication_manifest.json", "x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != "columns"}, indent=1))


if __name__ == "__main__":
    main()

"""Apply the approved Slicer-review rule (docs/decisions.md, 5 November 2026) to the cohort, now that the blind
review is complete (24/24). Reads the review sheet, the working split, the CV split and the pre-split audit
table (stone-label volume per scan); opens no image or mask and changes no file it reads.

Scan-level rule (approved): any location coded A (contrast) -> exclude; F (label not a stone) -> exclude;
C (likely missed stone) in a scan with NO stone label -> exclude; C in a scan with a stone label -> keep +
sensitivity mark; D -> keep + sensitivity mark; B -> keep + device subgroup; E -> no action.
Writes a NEW review-outcome file (data/processed/li2022_review_outcome_v1.csv) and a summary into a new run
folder. The working split CSV and its manifest SHA are NOT modified.
"""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from stonekit.config import create_run_folder

ROOT = Path(".")
SHEET = ROOT / "data/interim/li2022_full_audit/review/li_slicer_review_sheet.csv"
SPLIT = ROOT / "data/processed/li2022_working_split_v1.csv"
CV = ROOT / "data/nnunet/li_stone_cv_split.json"
AUDIT = ROOT / "data/interim/li2022_full_audit/li_full_cases.csv"
OUT = ROOT / "data/processed/li2022_review_outcome_v1.csv"


def read(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    sheet = read(SHEET)
    if any(r["finding_code"] not in tuple("ABCDEF") for r in sheet):
        raise SystemExit("review incomplete or invalid codes")
    split = {r["case"]: r["assignment"] for r in read(SPLIT)}
    audit = read(AUDIT)
    vol_col = next(c for c in audit[0] if "stone" in c.lower() and ("ml" in c.lower() or "mm3" in c.lower() or "vox" in c.lower()))
    stone = {r["case"]: float(r[vol_col] or 0) > 0 for r in audit}
    cv = json.loads(CV.read_text())
    trained_in = {c: [k for k, f in enumerate(cv["folds"]) if c in f["train"]] for c in split}
    by_scan: dict[str, list[dict]] = {}
    for r in sheet:
        by_scan.setdefault(r["case"], []).append(r)
    rows = []
    for case, locs in sorted(by_scan.items()):
        codes = [r["finding_code"] for r in locs]
        has_label = stone.get(case)
        if "A" in codes:
            action, reason = "exclude", "contrast (A)"
        elif "F" in codes:
            action, reason = "exclude", "label not a stone (F)"
        elif "C" in codes and has_label is False:
            action, reason = "exclude", "likely missed stone in a scan without a stone label (C)"
        else:
            marks = sorted({"device subgroup" if c == "B" else "sensitivity" for c in codes if c in "BCD"})
            action, reason = "keep", "; ".join(marks) or "no action"
        rows.append({"case": case, "assignment": split.get(case, "not in cohort"), "codes": "".join(codes),
                     "confidences": ",".join(r["confidence_high_medium_low"] for r in locs), "has_stone_label": has_label,
                     "action": action, "reason": reason, "in_cv_training_folds": " ".join(map(str, trained_in.get(case, [])))})  # fmt: skip
    with open(OUT, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    excluded = [r for r in rows if r["action"] == "exclude"]
    summary = {
        "reviewed_scans": len(rows), "locations": len(sheet), "codes": dict(Counter(r["finding_code"] for r in sheet)),
        "stone_label_column": vol_col,
        "excluded_by_assignment": dict(Counter(r["assignment"] for r in excluded)),
        "excluded_trained_on_in_cv_folds": {r["case"]: r["in_cv_training_folds"] for r in excluded if r["in_cv_training_folds"]},
        "cohort_after": {g: sum(1 for c, a in split.items() if a == g and c not in {r["case"] for r in excluded})
                         for g in ("development", "reserved_calibration", "reserved_test")},
        "review_sheet_sha256": hashlib.sha256(SHEET.read_bytes()).hexdigest(),
        "split_sha256_unchanged": hashlib.sha256(SPLIT.read_bytes()).hexdigest(),
        "outcome_file": str(OUT), "outcome_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }  # fmt: skip
    run = create_run_folder(ROOT / "results/runs", "slicer_review_outcome")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for r in rows:
        print(r["case"], r["assignment"], r["codes"], "label" if r["has_stone_label"] else "nolabel", r["action"], r["reason"], "| trained in folds:", r["in_cv_training_folds"] or "-")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("review_sheet_sha256", "split_sha256_unchanged", "outcome_sha256")}, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

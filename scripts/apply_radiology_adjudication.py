"""Map the completed D-REV radiology adjudication to CANDIDATE scan-level actions (owner rules, 12 November 2026).

Run ONLY after the adjudicator has answered all 24 locations and the owner says so. It changes no cohort:
"candidate exclusion" means only that the adjudication identifies a possible reference-standard problem.
The authoritative 165/35/53 split is changed only by a later explicit owner decision.

Pre-specified mapping (owner approval, 12 November 2026; written before any adjudication answer existed):
- Q3 = yes at any location of a scan -> candidate exclusion (scan-level excreted contrast compromises the reference).
- Q2 = invalid -> candidate exclusion (reference label invalid).
- Q1 = confirmed urinary stone on a scan WITHOUT a stone label -> candidate exclusion (stone-free reference invalid).
- Q2 = indeterminate -> retain; sensitivity flag.
- Q1 = confirmed urinary stone at an unlabelled location on a scan WITH a stone label -> retain; sensitivity flag.
- Q1 = confirmed non-stone of type device -> recorded; no automatic action.
- Q1 = other relevant abnormality -> recorded; no automatic action.
- Q1 = indeterminate -> retain; flag as unresolved / sensitivity case; never an exclusion (owner, 12 Nov 2026, before start).
- Q3 = indeterminate -> retain; flag as unresolved / sensitivity case; never an exclusion (owner, 12 Nov 2026, before start).
Only first-pass answers (blinded_first_pass = yes) are used. Internally inconsistent answers are listed
for the owner and are never resolved here.

Inputs (read-only): the completed adjudication sheet; the immutable screening sheet and its v1 outcome
(cross-tabulation only); the pre-split audit table (whether a scan has a stone label); the working split
(groups, for the owner's report only); the OOF table and E6 per-stone file (development consequence).
Outputs (new, never overwritten): data/processed/li2022_adjudication_outcome_v1.csv and a run folder.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ADJ = Path("data/interim/li2022_full_audit/adjudication/li_radiology_adjudication_sheet_v1.csv")
SCREENING = Path("data/interim/li2022_full_audit/review/li_slicer_review_sheet.csv")
SCREENING_OUTCOME = Path("data/processed/li2022_review_outcome_v1.csv")
AUDIT = Path("data/interim/li2022_full_audit/li_full_cases.csv")
SPLIT = Path("data/processed/li2022_working_split_v1.csv")
OOF_SCANS = Path("data/interim/oof_tables_20261112T082351Z/scans.csv")
OOF_ANNOTATED = Path("data/interim/oof_tables_20261112T082351Z/annotated.csv")
E6_PER_STONE = Path("data/interim/e6_per_stone_e6_uncertainty_design_20261112T082354Z.csv")
OUT = Path("data/processed/li2022_adjudication_outcome_v1.csv")

Q1 = {"confirmed_urinary_stone", "confirmed_non_stone", "indeterminate", "other_relevant_abnormality"}
Q2 = {"valid", "invalid", "indeterminate", "not_applicable"}
Q3 = {"yes", "no", "indeterminate"}
LABELLED = "labelled stone region"
UNLABELLED = "unlabelled region"


def read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rows(rows: list[dict]) -> None:
    """Refuse incomplete or invalid answers, and anything that is not a blinded first pass."""
    for r in rows:
        where = f"{r['adj_id']} ({r['case']} location {r['target']})"
        if r["q1_finding"] not in Q1 or r["q2_label_validity"] not in Q2 or r["q3_scan_contrast"] not in Q3:
            raise SystemExit(f"{where}: incomplete or invalid Q1-Q3")
        if r["confidence"] not in {"high", "medium", "low"} or r["ai_assistance"] not in {"yes", "no"}:
            raise SystemExit(f"{where}: confidence or AI assistance missing")
        if r["blinded_first_pass"] != "yes":
            raise SystemExit(f"{where}: not a blinded first pass; the pre-specified mapping uses first-pass answers only")
        if not r["adjudicator_role"].strip():
            raise SystemExit(f"{where}: adjudicator role missing")


def inconsistencies(locs: list[dict]) -> list[str]:
    """Answer combinations that contradict each other. Reported for the owner; never resolved here."""
    found = []
    for r in locs:
        tag = f"{r['adj_id']}"
        if r["location_type"] == LABELLED and r["q2_label_validity"] == "not_applicable":
            found.append(f"{tag}: labelled stone region but Q2 = not_applicable")
        if r["location_type"] == LABELLED and r["q1_finding"] == "confirmed_non_stone" and r["q2_label_validity"] == "valid":
            found.append(f"{tag}: labelled region judged non-stone but label judged valid")
        if r["location_type"] == LABELLED and r["q1_finding"] == "confirmed_urinary_stone" and r["q2_label_validity"] == "invalid":
            found.append(f"{tag}: labelled region judged a stone but label judged invalid")
        if (r["q1_finding"] == "confirmed_non_stone") != bool(r.get("q1_nonstone_type")):
            found.append(f"{tag}: non-stone type missing or set without a non-stone finding")
    q3 = {r["q3_scan_contrast"] for r in locs}
    if {"yes", "no"} <= q3:
        found.append("Q3 differs between locations of the same scan (yes and no)")
    return found


def map_scan(locs: list[dict], has_stone_label: bool) -> dict:
    """Apply the pre-specified owner mapping to one scan's locations."""
    candidate, sensitivity, recorded = [], [], []
    if any(r["q3_scan_contrast"] == "yes" for r in locs):
        candidate.append("scan-level excreted contrast (Q3 = yes)")
    if any(r["q2_label_validity"] == "invalid" for r in locs):
        candidate.append("reference label invalid (Q2 = invalid)")
    stones = [r for r in locs if r["q1_finding"] == "confirmed_urinary_stone"]
    if stones and not has_stone_label:
        candidate.append("confirmed stone on a stone-free scan (stone-free reference invalid)")
    if any(r["q2_label_validity"] == "indeterminate" for r in locs):
        sensitivity.append("label validity indeterminate (Q2)")
    if has_stone_label and any(r["location_type"] != LABELLED for r in stones):
        sensitivity.append("confirmed unlabelled stone on a stone-containing scan (reference incomplete)")
    if any(r["q1_finding"] == "confirmed_non_stone" and r.get("q1_nonstone_type") == "device" for r in locs):
        recorded.append("device")
    if any(r["q1_finding"] == "other_relevant_abnormality" for r in locs):
        recorded.append("other relevant abnormality")
    if any(r["q1_finding"] == "indeterminate" for r in locs):
        sensitivity.append("unresolved: Q1 indeterminate")
    if any(r["q3_scan_contrast"] == "indeterminate" for r in locs):
        sensitivity.append("unresolved: Q3 indeterminate")
    status = "candidate_exclusion" if candidate else ("retain_sensitivity_flag" if sensitivity else "retain")
    return {"status": status, "candidate_reasons": candidate, "sensitivity_flags": sensitivity,
            "recorded": recorded, "inconsistencies": inconsistencies(locs)}  # fmt: skip


def stone_label_lookup(audit_rows: list[dict]) -> dict:
    col = next(c for c in audit_rows[0] if "stone" in c.lower() and ("ml" in c.lower() or "mm3" in c.lower() or "vox" in c.lower()))
    return {r["case"]: float(r[col] or 0) > 0 for r in audit_rows}


def development_consequence(cases: list[str]) -> dict:
    """Read-only counts from existing development results for candidate development scans."""
    scans, ann, e6 = read(OOF_SCANS), read(OOF_ANNOTATED), read(E6_PER_STONE)
    total_burden = sum(float(r["annotated_burden_mm3"]) for r in scans)
    out = {}
    for c in cases:
        burden = sum(float(r["annotated_burden_mm3"]) for r in scans if r["case"] == c)
        out[c] = {"in_oof_table_used_for_E6_E8_selection": any(r["case"] == c for r in scans),
                  "e6_matched_stones": sum(1 for r in e6 if r["case"] == c),
                  "e7_annotated_pieces": sum(1 for r in ann if r["case"] == c),
                  "e8_annotated_burden_mm3": burden, "e8_share_of_development_burden": burden / total_burden if total_burden else None}  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, default=ADJ)
    args = parser.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; refusing to overwrite")
    screening_hash, outcome_hash = sha256(SCREENING), sha256(SCREENING_OUTCOME)
    rows = read(args.sheet)
    if len(rows) != 24:
        raise SystemExit(f"expected 24 locations, found {len(rows)}")
    check_rows(rows)
    has_label = stone_label_lookup(read(AUDIT))
    group = {r["case"]: r["assignment"] for r in read(SPLIT)}
    screening = {(r["case"], r["target"]): r["finding_code"] for r in read(SCREENING)}
    by_scan: dict[str, list[dict]] = {}
    for r in rows:
        by_scan.setdefault(r["case"], []).append(r)
    out_rows = []
    for case, locs in sorted(by_scan.items()):
        m = map_scan(locs, has_label[case])
        out_rows.append({"case": case, "assignment": group.get(case, "not in cohort"), "has_stone_label": has_label[case],
                         "adjudication_status": m["status"], "candidate_reasons": "; ".join(m["candidate_reasons"]),
                         "sensitivity_flags": "; ".join(m["sensitivity_flags"]), "recorded": "; ".join(m["recorded"]),
                         "inconsistencies": "; ".join(m["inconsistencies"]),
                         "screening_codes": "".join(screening[(r["case"], r["target"])] for r in sorted(locs, key=lambda x: x["target"]))})  # fmt: skip
    with open(OUT, "x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    crosstab = Counter((screening[(r["case"], r["target"])], r["q1_finding"], r["q2_label_validity"], r["q3_scan_contrast"]) for r in rows)
    dev_candidates = [r["case"] for r in out_rows if r["adjudication_status"] == "candidate_exclusion" and r["assignment"] == "development"]
    summary = {
        "status": "ADJUDICATION_CANDIDATES_NOT_APPLIED",
        "adjudication_sheet_sha256": sha256(args.sheet),
        "screening_sheet_sha256_unchanged": screening_hash == sha256(SCREENING), "screening_sheet_sha256": screening_hash,
        "screening_outcome_v1_sha256_unchanged": outcome_hash == sha256(SCREENING_OUTCOME),
        "split_sha256": sha256(SPLIT),
        "status_by_group": {f"{s}|{g}": n for (s, g), n in Counter((r["adjudication_status"], r["assignment"]) for r in out_rows).items()},
        "inconsistent_scans": [r["case"] for r in out_rows if r["inconsistencies"]],
        "screening_vs_adjudication": [{"screening": k[0], "q1": k[1], "q2": k[2], "q3": k[3], "n": v} for k, v in sorted(crosstab.items())],
        "development_consequence": development_consequence(dev_candidates),
        "adjudicator_roles": sorted({r["adjudicator_role"] for r in rows}),
        "ai_assistance_rows": sum(r["ai_assistance"] == "yes" for r in rows),
        "outcome_file": str(OUT), "outcome_sha256": sha256(OUT),
    }  # fmt: skip
    from stonekit.config import create_run_folder

    run = create_run_folder(Path("results/runs"), "radiology_adjudication_outcome")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    for r in out_rows:
        print(r["case"], r["assignment"], r["adjudication_status"], "|", r["candidate_reasons"] or "-", "|", r["inconsistencies"] or "")
    print("Saved", run, "- candidates only; no cohort change")


if __name__ == "__main__":
    main()

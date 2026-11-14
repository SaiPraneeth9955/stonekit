"""T1 DRESS REHEARSAL on DEVELOPMENT out-of-fold tables (no calibration or test scan; never a result).

35 development scans stand in for calibration and 53 for test (seeded); the remaining development scans are the
fit set. This proves the T1 analysis stage end to end: calibration quantities are written and hashed BEFORE the
test loader is called (checked here), every endpoint in protocol v3 is produced, and no COMPLETED marker is
written. Output folder: results/runs/t1_dress_rehearsal_<stamp>/ (not T1_*), labelled DRESS_REHEARSAL.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder
from stonekit.t1.analysis import Exclusion, run_analysis
from stonekit.uq import confirmatory as cfm

SEED = 20261105


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    parser.add_argument("--item5-stand-in", action="store_true",
                        help="mechanics check of protocol v3 §5.1 item 5: mark a SEEDED set of 1 calibration and 7 test stand-in scans "
                             "(the T1 maximum) as if reader-confirmed; no adjudication answer is used and the numbers are not results")  # fmt: skip
    parser.add_argument("--screening-cohorts", action="store_true",
                        help="mechanics check of §5.1 S1/S2 (screening outcome file) on the development stand-ins; S3 is computed only inside T1")  # fmt: skip
    args = parser.parse_args()
    t = cfm.Tables(pd.read_csv(args.table_folder / "annotated.csv"), pd.read_csv(args.table_folder / "predicted.csv"),
                   pd.read_csv(args.table_folder / "scans.csv"))  # fmt: skip
    order = np.random.default_rng(SEED + 1).permutation(np.sort(t.scans.case.unique()))
    cal, test, fit = t.subset(order[:35]), t.subset(order[35:88]), t.subset(order[88:])
    run = create_run_folder(Path("results/runs"), "t1_dress_rehearsal")
    out = run / "analysis"
    state = {}

    def load_test():
        q = out / "calibration_quantities.json"
        state["calibration_written_before_test_load"] = q.exists() and (out / "calibration_quantities.sha256").exists()
        return test

    marked = None
    if args.item5_stand_in:
        pick = np.random.default_rng(SEED + 2)
        marked = frozenset(pick.choice(np.sort(cal.scans.case.unique()), 1, replace=False).tolist() + pick.choice(np.sort(test.scans.case.unique()), 7, replace=False).tolist())
        (run / "item5_stand_in_marked.json").write_text(json.dumps({"note": "seeded mechanics check, not adjudication answers", "marked": sorted(marked)}, indent=2), encoding="utf-8")
    sens = None
    if args.screening_cohorts:
        rows = pd.read_csv("data/processed/li2022_review_outcome_v1.csv")
        s1 = frozenset(rows.loc[rows.action == "exclude", "case"])
        s2 = s1 | frozenset(rows.loc[rows.reason == "sensitivity", "case"])
        sens = {"S1_screening_exclusions": Exclusion(s1, reserved_only=False), "S2_all_screened_flags": Exclusion(s2, reserved_only=False)}
    ev = run_analysis(fit, cal, load_test, out, mode="DRESS_REHEARSAL", annotation_error_scans=marked, sensitivity=sens)
    if sens:
        for k in sens:
            s = ev["sensitivity"][k]
            print(k, "| removed fit/cal/test:", len(s["fit_removed"]), len(s["calibration_removed"]), len(s["test_removed"]),
                  "| formal P2 (stand-in):", s.get("evaluation", {}).get("formal_P2"))  # fmt: skip
    checks = {"calibration_written_before_test_load": state.get("calibration_written_before_test_load", False),
              "no_completed_marker": not (out / "COMPLETED").exists(),
              "calibration_and_test_disjoint": not (set(cal.scans.case) & set(test.scans.case)),
              "fit_disjoint": not (set(fit.scans.case) & (set(cal.scans.case) | set(test.scans.case)))}  # fmt: skip
    (run / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    print("formal P2 (development stand-in):", ev["formal_P2"], "| U2 volume 0.90:", round(ev["per_stone"]["volume_mm3|0.90"]["U2"]["marginal"]["coverage"], 3))
    if marked is not None:
        s = ev["sensitivity"]["item5_reader_confirmed_errors"]
        checks["item5_removed_1_cal_7_test"] = len(s["calibration_removed"]) == 1 and len(s["test_removed"]) == 7
        checks["item5_test_scans_reduced"] = s["evaluation"]["test_scans"] == ev["test_scans"] - 7
        (run / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        print("item 5 (mechanics only):", {k: checks[k] for k in ("item5_removed_1_cal_7_test", "item5_test_scans_reduced")}, "| formal P2 keys", sorted(s["evaluation"]["formal_P2"]))
    print("Saved", run)


if __name__ == "__main__":
    main()

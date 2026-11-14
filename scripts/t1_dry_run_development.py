"""T1 dry run on DEVELOPMENT out-of-fold data only (no calibration or test scan is touched).

Purpose (engineering validation + expected operating characteristics, not a result):
- run the frozen-arm T1 analysis engine (src/stonekit/uq/confirmatory.py) end to end on real development tables;
- estimate how the single locked T1 would behave at its real sizes: in each of R seeded repetitions the
  165 development scans are randomly split into a pseudo-calibration set (35 scans), a pseudo-test set
  (53 scans) and a fit set (the remaining 77 scans, standing in for the development data that fit the
  P2-A components at T1, where 165 scans will be available: so this dry run is conservative);
- the secondary filter-on vs filter-off ablation (owner decision OD-3) is evaluated the same way.

Reported per arm: per-stone coverage at nominal 0.90 (baseline U0/U1; final U2 volume, U4 diameter) and
the P2 formal test of OD-1 (P2-A / P2-B-CQR width-ratio CI excluding 1 AND P2-A CP upper bound >= 0.90),
as distributions over repetitions (median, 5th-95th percentile) and success rates.
Label: DEVELOPMENT_DRY_RUN_NOT_A_RESULT. These numbers must not be presented as validation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder
from stonekit.uq import confirmatory as cfm

SEED = 20261105
N_CAL, N_TEST = 35, 53


def q(values) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return {"median": float(np.median(v)), "p05": float(np.quantile(v, 0.05)), "p95": float(np.quantile(v, 0.95)), "n": int(v.size)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    t = cfm.Tables(pd.read_csv(args.table_folder / "annotated.csv"), pd.read_csv(args.table_folder / "predicted.csv"),
                   pd.read_csv(args.table_folder / "scans.csv"))  # fmt: skip
    scans = np.sort(t.scans.case.unique())
    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(args.reps):
        order = rng.permutation(scans)
        cal, test, fit = t.subset(order[:N_CAL]), t.subset(order[N_CAL:N_CAL + N_TEST]), t.subset(order[N_CAL + N_TEST:])
        row = {"rep": rep}
        for arm, method, target in (("baseline", "U0", "volume_mm3"), ("baseline", "U1", "volume_mm3"),
                                    ("final", "U2", "volume_mm3"), ("final", "U4", "max_diameter_3d_mm")):  # fmt: skip
            c = cfm.stone_coverage(cal, test, method, target, 0.10)
            row[f"{method}|{target}|coverage"] = float(c.covered.mean())
            row[f"{method}|{target}|median_rel_width"] = float(np.median(c.rel_width))
            row[f"{method}|{target}|stones"] = len(c)
            for s, name in enumerate(("<5", "5-<10", ">=10")):
                m = cfm.stratum(c.pred_d3) == s
                row[f"{method}|{target}|cov_{name}"] = float(c.covered[m].mean()) if m.any() else np.nan
        for filtered in (True, False):
            tag = "filter_on" if filtered else "filter_off"
            p2 = cfm.p2_test(cfm.burden_intervals(fit, cal, test, 0.10, filtered=filtered), resamples=2000)
            row[f"{tag}|ratio_median"] = p2["width_ratio_A_over_BCQR"]["median"]
            row[f"{tag}|ratio_ci_hi"] = p2["width_ratio_A_over_BCQR"]["ci95"][1]
            row[f"{tag}|P2_success"] = p2["formal"]["P2_success"]
            row[f"{tag}|guard_pass"] = p2["formal"]["coverage_check_cp_upper_ge_nominal"]
            for m in ("P2-0 naive", "P2-B direct", "P2-B-CQR", "P2-A decomposition"):
                row[f"{tag}|{m}|coverage"] = p2[m]["coverage"]
                row[f"{tag}|{m}|median_width_mm3"] = p2[m]["median_width_mm3"]
                row[f"{tag}|{m}|coverage_stone_free"] = p2[m]["coverage_stone_free"]
        rows.append(row)
        if rep % 20 == 0:
            print(f"rep {rep}: U2 vol {row['U2|volume_mm3|coverage']:.3f}  P2 success {row['filter_on|P2_success']}  ratio {row['filter_on|ratio_median']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    summary = {"status": "DEVELOPMENT_DRY_RUN_NOT_A_RESULT", "definitions": __doc__, "table_folder": str(args.table_folder),
               "reps": args.reps, "sizes": {"calibration": N_CAL, "test": N_TEST, "fit": int(len(scans) - N_CAL - N_TEST)},
               "distributions": {c: q(df[c]) for c in df.columns if c != "rep" and df[c].dtype != bool},
               "rates": {c: float(df[c].mean()) for c in df.columns if df[c].dtype == bool}}  # fmt: skip
    run = create_run_folder(Path("results/runs"), "t1_dry_run_development")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    df.to_csv(run / "repetitions.csv", index=False)
    print(json.dumps(summary["rates"], indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

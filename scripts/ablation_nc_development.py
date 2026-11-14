"""N-C ablation (owner decision OD-13; ablation only, never a novelty claim): is P2-A's false-positive component
more than a distance-to-kidney rule? DEVELOPMENT out-of-fold data only.

The same 200 seeded 35 / 53 / 77 development splits as the T1 dry run (scripts/t1_dry_run_development.py; same seed).
In each split, P2-A runs with the full false-positive features (frozen arm) and with distance to the kidney only.
Pre-specified reading (docs/NOVELTY_CANDIDATES_20261112.md, fixed before the run): if the distance-only P2-A's
median width and coverage are within +/-5% of full P2-A, the simpler model is a candidate simplification, which
needs an owner decision before the freeze. Otherwise it is reported as an ablation. Label DEVELOPMENT_ABLATION.
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
        for tag, feats in (("full", None), ("distance_only", ["distance_to_kidney_mm"])):
            iv = cfm.burden_intervals(fit, cal, test, 0.10, fp_features=feats)
            a = iv["P2-A decomposition"]
            row[f"{tag}|coverage"] = float(((a.B >= a.lo) & (a.B <= a.hi)).mean())
            row[f"{tag}|median_width"] = float(np.median(a.hi - a.lo))
        rows.append(row)
    df = pd.DataFrame(rows)
    rel_w = df["distance_only|median_width"] / df["full|median_width"] - 1
    rel_c = df["distance_only|coverage"] - df["full|coverage"]
    summary = {"status": "DEVELOPMENT_ABLATION", "definitions": __doc__, "reps": args.reps,
               "full": {"coverage_median": float(df["full|coverage"].median()), "width_median": float(df["full|median_width"].median())},
               "distance_only": {"coverage_median": float(df["distance_only|coverage"].median()), "width_median": float(df["distance_only|median_width"].median())},
               "relative_width_change": {"median": float(rel_w.median()), "p05": float(rel_w.quantile(0.05)), "p95": float(rel_w.quantile(0.95))},
               "coverage_difference": {"median": float(rel_c.median()), "p05": float(rel_c.quantile(0.05)), "p95": float(rel_c.quantile(0.95))},
               "within_5pct": bool(abs(rel_w.median()) <= 0.05 and abs(rel_c.median()) <= 0.05)}  # fmt: skip
    run = create_run_folder(Path("results/runs"), "ablation_nc_development")
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    df.to_csv(run / "repetitions.csv", index=False)
    print(json.dumps({k: v for k, v in summary.items() if k != "definitions"}, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

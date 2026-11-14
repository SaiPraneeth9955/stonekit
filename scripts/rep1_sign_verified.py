"""rep1 Table 1 difference cells with the paper's sign convention, verified from the full text.

Neubauer et al. 2023 define the difference as reference minus estimate (full text read 11 November 2026).
The rep1 protocol could not know this, so compare_with_paper accepted either sign (the closer one). This
script re-reads rep1's saved per-split values (no rerun, nothing in the rep1 folder is changed) and
classifies every *_diff cell with the verified sign only. Other cells are unchanged by construction.

Usage: python scripts/rep1_sign_verified.py results/runs/rep1_neubauer_replication_<stamp>
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from stonekit.config import create_run_folder, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rep1_run", type=Path)
    args = parser.parse_args()
    paper = load_yaml(Path("configs/rep1_neubauer_replication.yaml"))["paper_table1"]
    out = {}
    rows = []
    for csv in sorted(args.rep1_run.glob("splits_*.csv")):
        name = csv.stem.removeprefix("splits_")
        df = pd.read_csv(csv)
        changed = []
        for method, group in df.groupby("method"):
            for key in ("vol_diff", "diam_diff"):
                ours_ref_minus_est = -group[key]  # saved values are estimate - reference
                mean, sd = float(ours_ref_minus_est.mean()), float(ours_ref_minus_est.std(ddof=1))
                paper_mean = paper[method][key][0]
                gap = abs(mean - paper_mean)
                verdict = "reproduced" if gap <= sd else "close" if gap <= 2 * sd else "not reproduced"
                lenient_gap = min(abs(mean - paper_mean), abs(-mean - paper_mean))
                lenient = "reproduced" if lenient_gap <= sd else "close" if lenient_gap <= 2 * sd else "not reproduced"
                row = {"configuration": name, "method": method, "quantity": key, "paper_mean_ref_minus_est": paper_mean,
                       "ours_mean_ref_minus_est": round(mean, 4), "ours_sd": round(sd, 4), "gap": round(gap, 4),
                       "verdict_sign_verified": verdict, "verdict_protocol_either_sign": lenient}  # fmt: skip
                rows.append(row)
                if verdict != lenient:
                    changed.append(f"{method} {key}: {lenient} -> {verdict}")
        out[name] = changed
    run = create_run_folder(Path("results/runs"), "rep1_sign_verified")
    pd.DataFrame(rows).to_csv(run / "difference_cells_sign_verified.csv", index=False)
    (run / "summary.json").write_text(json.dumps({"source": str(args.rep1_run), "verdict_changes": out, "definitions": __doc__}, indent=2), encoding="utf-8")
    prim = pd.DataFrame(rows)
    print(prim[prim.configuration == "P_Z2_surface6_3d"].to_string(index=False))
    print(json.dumps(out, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

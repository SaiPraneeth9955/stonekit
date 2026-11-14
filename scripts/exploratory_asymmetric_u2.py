"""EXPLORATORY comparator (not a final-arm change; U2 stays pre-specified under OD-12): an asymmetric
Mondrian split-conformal interval for per-stone volume and 3D diameter, motivated by the development failure
analysis (45 of 46 U2 misses were under-estimates; results/runs/failure_analysis_development_20261112T110426Z).

Method (established: two-tailed split conformal with a separate quantile for each tail, here per size band):
within each predicted-size band (U2's bands and merge rule), using the signed residual r = log(V / V_hat), the
upper bound is the finite-sample (1 - alpha/2) conformal quantile of r and the lower bound is minus the
(1 - alpha/2) conformal quantile of -r. Each tail gets alpha/2, so marginal coverage is >= 1 - alpha.
Design: the same cross-conformal-by-fold protocol and table as E6.
Reported against U2: marginal coverage, coverage by band and HU tertile, misses by direction, and median
relative width. There is no adoption rule: exploratory only, labelled EXPLORATORY_NOT_PRESPECIFIED.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder
from stonekit.uq import conformal as cf

EDGES = (5.0, 10.0)
MIN_STRATUM = 20


def tail_q(x, a):
    """Finite-sample conformal quantile at level 1 - a for one tail."""
    x = np.sort(np.asarray(x, float))
    k = math.ceil((len(x) + 1) * (1 - a))
    return float(x[k - 1]) if k <= len(x) else math.inf


def bounds(cal, test, target, alpha, asymmetric):
    r = cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"])
    final = cf.merge_strata(cal["pred_max_diameter_3d_mm"], EDGES, MIN_STRATUM)
    gc = np.searchsorted(final, cal["pred_max_diameter_3d_mm"], side="right")
    gt = np.searchsorted(final, test["pred_max_diameter_3d_mm"], side="right")
    lo, hi = np.empty(len(test)), np.empty(len(test))
    for g in range(len(final) + 1):
        rg, m = r[gc == g], gt == g
        if asymmetric:
            hi[m], lo[m] = tail_q(rg, alpha / 2), -tail_q(-rg, alpha / 2)
        else:
            q = cf.conformal_quantile(np.abs(rg), alpha)
            lo[m], hi[m] = -q, q
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    d = a[(a.group == "one_to_one") & (a.ref_volume_mm3 > 0) & (a.pred_volume_mm3 > 0)].copy().reset_index(drop=True)
    band = pd.cut(d.pred_max_diameter_3d_mm, [-np.inf, *EDGES, np.inf], right=False, labels=["<5", "5-<10", ">=10"])
    hu = pd.qcut(d.pred_mean_hu, 3, labels=["low", "mid", "high"])
    out = {"status": "EXPLORATORY_NOT_PRESPECIFIED", "definitions": __doc__, "stones": len(d), "results": {}}
    for target in ("volume_mm3", "max_diameter_3d_mm"):
        for alpha in (0.10, 0.20):
            r = cf.log_ratio(d[f"ref_{target}"], d[f"pred_{target}"])
            res = {}
            for name, asym in (("U2", False), ("U2-asymmetric", True)):
                lo, hi = np.empty(len(d)), np.empty(len(d))
                for g in sorted(d.fold.unique()):
                    te = (d.fold == g).to_numpy()
                    lo[te], hi[te] = bounds(d[~te], d[te], target, alpha, asym)
                cov = (r >= lo) & (r <= hi)
                res[name] = {"marginal": float(cov.mean()), "median_rel_width": float(np.median(np.exp(hi) - np.exp(lo))),
                             "misses_above": int((r > hi).sum()), "misses_below": int((r < lo).sum()),
                             "by_band": {str(k): float(cov[(band == k).to_numpy()].mean()) for k in band.cat.categories},
                             "by_hu": {str(k): float(cov[(hu == k).to_numpy()].mean()) for k in hu.cat.categories}}  # fmt: skip
            out["results"][f"{target}|{1 - alpha:.2f}"] = res
            u, s = res["U2"], res["U2-asymmetric"]
            print(f"{target} {1 - alpha:.2f}: U2 cov {u['marginal']:.3f} w {u['median_rel_width']:.3f} miss up/down {u['misses_above']}/{u['misses_below']} low-HU {u['by_hu']['low']:.3f} | "
                  f"asym cov {s['marginal']:.3f} w {s['median_rel_width']:.3f} miss {s['misses_above']}/{s['misses_below']} low-HU {s['by_hu']['low']:.3f}", flush=True)
    run = create_run_folder(Path("results/runs"), "exploratory_asymmetric_u2")
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("Saved", run)


if __name__ == "__main__":
    main()

"""Secondary analyses N-A and N-B on DEVELOPMENT out-of-fold data (owner decision OD-13). Neither is a novelty
claim; both apply established conformal mechanisms (docs/NOVELTY_GATE_20261112.md). Rules were fixed in
docs/NOVELTY_CANDIDATES_20261112.md BEFORE this script was first run.

Protocol: the same cross-conformal-by-fold design and table as E6 (one_to_one matched stones; for fold k,
calibrate on the other folds and evaluate on fold k). The comparator is U2 (Mondrian by predicted size; the
pre-specified final per-stone method, OD-12).

N-A, robustness (Mondrian by predicted-size band x HU tertile):
- HU tertile cuts from the calibration folds' predicted mean HU; within each HU tertile, U2's size bands with the
  same merge rule (minimum 20 calibration stones).
- Evaluation cells: predicted-size band (3) x HU tertile (cuts over all matched stones, as in E6) = 9 cells.
- Pre-specified rule: worst-cell point coverage improves by >= 0.03 over U2, marginal point coverage >= 0.85,
  and median relative width increases by <= 15%. Otherwise reported as negative.

N-B, secondary (patient-level joint coverage, max-score):
- The calibration folds are split by patient into halves A and B (seeded). Half A fits the U2 size-band
  quantiles q_s. Half B gives one score per patient, max over its matched stones of |log(V/V_hat)| / q_s(i), and
  tau = conformal quantile of those scores. Test interval for each stone: +/- tau * q_s(i).
- Endpoint: share of test patients with ALL matched stones covered; width inflation = median relative width
  N-B / U2.
- Pre-specified rule: joint point coverage >= 0.85 and Clopper-Pearson upper bound >= nominal, with
  inflation <= 2x. Otherwise reported as not met.
Label DEVELOPMENT_SECONDARY_NOT_VALIDATED.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder
from stonekit.uq import conformal as cf

SEED = 20261105
EDGES = (5.0, 10.0)
MIN_STRATUM = 20


def u2(cal, test, target, alpha):
    r = np.abs(cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"]))
    s = cf.fit_mondrian(r, cal["pred_max_diameter_3d_mm"], EDGES, alpha, MIN_STRATUM)
    return np.asarray(s.quantiles)[s.index(test["pred_max_diameter_3d_mm"])]


def na(cal, test, target, alpha):
    r = np.abs(cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"]))
    cuts = np.nanquantile(cal["pred_mean_hu"], [1 / 3, 2 / 3])
    h_cal, h_test = np.searchsorted(cuts, cal["pred_mean_hu"]), np.searchsorted(cuts, test["pred_mean_hu"])
    q = np.empty(len(test))
    for h in range(3):
        m_c, m_t = h_cal == h, h_test == h
        if not m_t.any():
            continue
        s = cf.fit_mondrian(r[m_c], cal["pred_max_diameter_3d_mm"][m_c], EDGES, alpha, MIN_STRATUM)
        q[m_t] = np.asarray(s.quantiles)[s.index(test["pred_max_diameter_3d_mm"][m_t])]
    return q


def nb(cal, test, target, alpha):
    patients = np.sort(cal["case"].unique())
    rng = np.random.default_rng(SEED)
    half_a = set(rng.choice(patients, size=len(patients) // 2, replace=False))
    a, b = cal[cal["case"].isin(half_a)], cal[~cal["case"].isin(half_a)]
    ra = np.abs(cf.log_ratio(a[f"ref_{target}"], a[f"pred_{target}"]))
    s = cf.fit_mondrian(ra, a["pred_max_diameter_3d_mm"], EDGES, alpha, MIN_STRATUM)
    qs = np.asarray(s.quantiles)
    rb = np.abs(cf.log_ratio(b[f"ref_{target}"], b[f"pred_{target}"])) / qs[s.index(b["pred_max_diameter_3d_mm"])]
    tau = cf.conformal_quantile(pd.Series(rb).groupby(b["case"].to_numpy()).max().to_numpy(), alpha)
    return tau * qs[s.index(test["pred_max_diameter_3d_mm"])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    d = a[(a.group == "one_to_one") & (a.ref_volume_mm3 > 0) & (a.pred_volume_mm3 > 0)].copy().reset_index(drop=True)
    d["size_band"] = np.searchsorted(EDGES, d["pred_max_diameter_3d_mm"], side="right")
    d["hu_tertile"] = pd.qcut(d["pred_mean_hu"], 3, labels=False)
    out = {"status": "DEVELOPMENT_SECONDARY_NOT_VALIDATED", "definitions": __doc__, "stones": len(d), "scans": int(d.case.nunique()), "results": {}}
    for target in ("volume_mm3", "max_diameter_3d_mm"):
        for alpha in (0.10, 0.20):
            q = {m: np.empty(len(d)) for m in ("U2", "N-A", "N-B")}
            for g in sorted(d.fold.unique()):
                te = (d.fold == g).to_numpy()
                cal, test = d[~te], d[te]
                q["U2"][te], q["N-A"][te], q["N-B"][te] = u2(cal, test, target, alpha), na(cal, test, target, alpha), nb(cal, test, target, alpha)
            r = np.abs(cf.log_ratio(d[f"ref_{target}"], d[f"pred_{target}"]))
            res = {}
            for m, qq in q.items():
                cov = r <= qq
                width = np.exp(qq) - np.exp(-qq)
                cells = pd.DataFrame({"c": cov, "s": d.size_band, "h": d.hu_tertile}).groupby(["s", "h"])["c"].agg(["mean", "size"])
                joint = pd.Series(cov).groupby(d.case.to_numpy()).all()
                res[m] = {"marginal": float(cov.mean()), "median_rel_width": float(np.median(width)),
                          "worst_cell": {"coverage": float(cells["mean"].min()), "n": int(cells.loc[cells["mean"].idxmin(), "size"])},
                          "cells": {f"{int(s)}|{int(h)}": [round(float(v), 3), int(n)] for (s, h), (v, n) in cells.iterrows()},
                          "joint_patient": {"patients": len(joint), "coverage": float(joint.mean()), "cp95": list(cf.clopper_pearson(int(joint.sum()), len(joint)))}}  # fmt: skip
            nominal = 1 - alpha
            u, a_, b_ = res["U2"], res["N-A"], res["N-B"]
            res["N-A_rule"] = {"worst_cell_gain": a_["worst_cell"]["coverage"] - u["worst_cell"]["coverage"],
                               "width_increase": a_["median_rel_width"] / u["median_rel_width"] - 1,
                               "met": bool(a_["worst_cell"]["coverage"] - u["worst_cell"]["coverage"] >= 0.03 and a_["marginal"] >= nominal - 0.05
                                           and a_["median_rel_width"] / u["median_rel_width"] - 1 <= 0.15)}  # fmt: skip
            res["N-B_rule"] = {"width_inflation": b_["median_rel_width"] / u["median_rel_width"],
                               "met": bool(b_["joint_patient"]["coverage"] >= nominal - 0.05 and b_["joint_patient"]["cp95"][1] >= nominal
                                           and b_["median_rel_width"] / u["median_rel_width"] <= 2.0)}  # fmt: skip
            out["results"][f"{target}|{nominal:.2f}"] = res
            print(f"{target} {nominal:.2f}: U2 marg {u['marginal']:.3f} worst {u['worst_cell']['coverage']:.3f} joint {u['joint_patient']['coverage']:.3f} w {u['median_rel_width']:.3f} | "
                  f"N-A marg {a_['marginal']:.3f} worst {a_['worst_cell']['coverage']:.3f} w {a_['median_rel_width']:.3f} met {res['N-A_rule']['met']} | "
                  f"N-B joint {b_['joint_patient']['coverage']:.3f} cp {b_['joint_patient']['cp95'][0]:.3f}-{b_['joint_patient']['cp95'][1]:.3f} infl {res['N-B_rule']['width_inflation']:.2f} met {res['N-B_rule']['met']}", flush=True)
    run = create_run_folder(Path("results/runs"), "secondary_na_nb_development")
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    (run / "inputs.json").write_text(json.dumps({"table_folder": str(args.table_folder)}), encoding="utf-8")
    print("Saved", run)


if __name__ == "__main__":
    main()

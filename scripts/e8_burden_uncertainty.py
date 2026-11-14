"""E8 (P2 design): patient-level burden uncertainty on development out-of-fold data (master plan v2 §G,
amended 11 November 2026). Target B = total annotated stone volume per scan (0 for stone-free scans);
B_hat = total predicted volume after the deployed 30 mm kidney-distance filter. Reference = annotation.

Methods (cross-fitted by fold; with a single fold, 5 seeded scan groups and the output is DEBUG):
- P2-0 naive: sum over predicted pieces of per-stone Mondrian bounds V_hat*exp(-/+q_s) (q_s from other folds'
  one_to_one |log ratio| in the piece's predicted-size stratum). No false-positive or missed-stone term.
- P2-B direct: split conformal on r = |log((B + c)/(B_hat + c))|, c = 5 mm3 (fixed in the plan).
- P2-A decomposition: an explicit distributional model, NOT sampling from conformal intervals.
  Per draw: each predicted piece is kept with probability 1 - pi (pi = cross-fitted false-positive
  probability, same features as E7) and gets volume V_hat*exp(e), e drawn from other folds' one_to_one
  log ratios log(V_ref/V_hat) in its predicted-size stratum; a missed-burden value is drawn from other
  folds' per-scan missed burden (sum of annotated volume in 'missed' groups) within the same slice group
  (<=1 mm / >1 mm; all scans if fewer than 10). 2,000 draws -> central (1-alpha) raw interval, then
  conformalised at patient level by one log-scale factor t (cross-conformal).
- P2-A-nomissed: the same without the missed-burden term (ablation).
- P2-B-CQR (added 11 November 2026 so the direct comparator is not a strawman): conformalised quantile
  regression of log(B + c) on log(B_hat + c), log(1 + sum of (1 - pi) * V_hat) and slice mm, fitted on
  half of the other folds' scans and conformalised on the other half (CQR score), cross-fitted by fold.
Metrics: patient coverage with Clopper-Pearson CI; median width (mm3); paired width ratio A/B
(median, bootstrap CI over scans); coverage for stone-free scans and burden tertiles.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from stonekit.config import create_run_folder
from stonekit.uq import conformal as cf

SEED = 20261105
C_MM3 = 5.0
EDGES = (5.0, 10.0)
DRAWS = 2000
FP_FEATURES = ["log_volume", "p_mean", "tta_spread", "threshold_sensitivity", "distance_to_kidney_mm", "hu100"]


def stratum(d: pd.Series | np.ndarray) -> np.ndarray:
    return np.searchsorted(EDGES, np.asarray(d, float), side="right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20])
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    p = pd.read_csv(args.table_folder / "predicted.csv")
    s = pd.read_csv(args.table_folder / "scans.csv")
    debug = s["fold"].nunique() < 2
    scans = np.sort(s["case"].unique())
    if debug:
        rng0 = np.random.default_rng(SEED)
        shuffled = scans.copy()
        rng0.shuffle(shuffled)
        group_of = {c: i % 5 for i, c in enumerate(shuffled)}
    else:
        group_of = dict(zip(s["case"], s["fold"], strict=True))
    s["g"] = s["case"].map(group_of)
    s["slice_group"] = np.where(s["slice_mm"] <= 1.0, "thin", "thick")
    p = p[~p["beyond_30mm"].astype(bool)].copy()  # deployed filter
    p["g"] = p["case"].map(group_of)
    p["stratum"] = stratum(p["max_diameter_3d_mm"])
    p["log_volume"], p["hu100"] = np.log(p["volume_mm3"]), p["mean_hu"] / 100
    p["is_fp"] = (p["group"] == "false_positive").astype(int)
    one = a[a["group"] == "one_to_one"].copy()
    one["g"] = one["case"].map(group_of)
    one["stratum"] = stratum(one["pred_max_diameter_3d_mm"])
    one["e"] = np.log(one["ref_volume_mm3"] / one["pred_volume_mm3"])
    missed = a[a["group"] == "missed"].groupby("case")["ref_volume_mm3"].sum()
    s["missed_mm3"] = s["case"].map(missed).fillna(0.0)
    s["B"] = s["annotated_burden_mm3"]
    s["B_hat"] = s["predicted_burden_filtered_mm3"]
    # cross-fitted false-positive probability per predicted piece
    p["pi"] = np.nan
    usable = p.dropna(subset=FP_FEATURES)
    for g in sorted(s["g"].unique()):
        tr, te = usable[usable.g != g], usable[usable.g == g]
        if len(te) and tr.is_fp.nunique() == 2:
            p.loc[te.index, "pi"] = LogisticRegression(max_iter=1000).fit(tr[FP_FEATURES], tr.is_fp).predict_proba(te[FP_FEATURES])[:, 1]
    p["pi"] = p["pi"].fillna(p["is_fp"].mean())
    rng = np.random.default_rng(SEED)
    out = {"status": "DEBUG_NOT_A_RESULT" if debug else "DEVELOPMENT_DESIGN_NOT_VALIDATION", "definitions": __doc__,
           "scans": len(s), "stone_free_scans": int((s.B == 0).sum()), "c_mm3": C_MM3, "results": {}}  # fmt: skip
    for alpha in args.alphas:
        res = {}
        raw_a, raw_a0, naive = {}, {}, {}
        for g in sorted(s["g"].unique()):
            pool_e = {k: one[(one.g != g) & (one.stratum == k)]["e"].to_numpy() for k in range(len(EDGES) + 1)}
            q_s = {k: cf.conformal_quantile(np.abs(v), alpha) if len(v) else np.inf for k, v in pool_e.items()}
            other = s[s.g != g]
            for _, row in s[s.g == g].iterrows():
                pieces = p[p.case == row.case]
                pool_m = other[other.slice_group == row.slice_group]["missed_mm3"].to_numpy()
                pool_m = pool_m if len(pool_m) >= 10 else other["missed_mm3"].to_numpy()
                total = np.zeros(DRAWS)
                for _, pc in pieces.iterrows():
                    e_pool = pool_e[pc.stratum] if len(pool_e[pc.stratum]) else one[one.g != g]["e"].to_numpy()
                    keep = rng.random(DRAWS) >= pc.pi
                    total += keep * pc.volume_mm3 * np.exp(rng.choice(e_pool, DRAWS))
                with_missed = total + rng.choice(pool_m, DRAWS)
                raw_a[row.case] = np.quantile(with_missed, [alpha / 2, 1 - alpha / 2])
                raw_a0[row.case] = np.quantile(total, [alpha / 2, 1 - alpha / 2])
                lo = sum(v * np.exp(-q_s[k]) for v, k in zip(pieces.volume_mm3, pieces.stratum, strict=True))
                hi = sum(v * np.exp(q_s[k]) for v, k in zip(pieces.volume_mm3, pieces.stratum, strict=True))
                naive[row.case] = np.array([lo, hi])

        def conformalise(raw: dict, alpha: float = alpha) -> dict:
            score = {c: max(np.log((raw[c][0] + C_MM3) / (b + C_MM3)), np.log((b + C_MM3) / (raw[c][1] + C_MM3)))
                     for c, b in zip(s.case, s.B, strict=True)}  # fmt: skip
            final = {}
            for g in sorted(s["g"].unique()):
                t = cf.conformal_quantile(np.array([score[c] for c in s[s.g != g].case]), alpha)
                for c in s[s.g == g].case:
                    final[c] = np.array([max((raw[c][0] + C_MM3) * np.exp(-t) - C_MM3, 0.0), (raw[c][1] + C_MM3) * np.exp(t) - C_MM3])
            return final

        direct = {}
        r = np.abs(np.log((s.B + C_MM3) / (s.B_hat + C_MM3)))
        for g in sorted(s["g"].unique()):
            q = cf.conformal_quantile(r[s.g != g].to_numpy(), alpha)
            for c, bh in zip(s[s.g == g].case, s[s.g == g].B_hat, strict=True):
                direct[c] = np.array([max((bh + C_MM3) * np.exp(-q) - C_MM3, 0.0), (bh + C_MM3) * np.exp(q) - C_MM3])
        s["adj"] = s["case"].map(p.assign(w=(1 - p.pi) * p.volume_mm3).groupby("case")["w"].sum()).fillna(0.0)
        x_all = np.column_stack([np.log(s.B_hat + C_MM3), np.log1p(s.adj), s.slice_mm])
        y_all = np.log(s.B + C_MM3).to_numpy()
        cqr = {}
        for g in sorted(s["g"].unique()):
            other_idx = np.flatnonzero((s.g != g).to_numpy())
            half = np.random.default_rng(SEED).permutation(other_idx)
            fit_idx, cal_idx = np.sort(half[: len(half) // 2]), np.sort(half[len(half) // 2 :])
            predict = cf.fit_cqr(x_all, y_all, alpha, fit_idx, cal_idx)
            test_idx = np.flatnonzero((s.g == g).to_numpy())
            lo_l, hi_l = predict(x_all[test_idx])
            for c, a_lo, a_hi in zip(s.case.to_numpy()[test_idx], lo_l, hi_l, strict=True):
                cqr[c] = np.array([max(np.exp(a_lo) - C_MM3, 0.0), np.exp(a_hi) - C_MM3])
        methods = {"P2-0 naive": naive, "P2-B direct": direct, "P2-B-CQR": cqr, "P2-A decomposition": conformalise(raw_a),
                   "P2-A without missed term": conformalise(raw_a0)}  # fmt: skip
        tert = pd.qcut(s.B.rank(method="first"), 3, labels=["low", "mid", "high"])
        for name, iv in methods.items():
            lo = np.array([iv[c][0] for c in s.case])
            hi = np.array([iv[c][1] for c in s.case])
            cov = (s.B.to_numpy() >= lo) & (s.B.to_numpy() <= hi)
            width = hi - lo
            entry = {"coverage": float(cov.mean()), "cp95": list(cf.clopper_pearson(int(cov.sum()), len(cov))), "n": len(cov),
                     "median_width_mm3": float(np.median(width)) if np.isfinite(width).all() else "unbounded",
                     "coverage_stone_free": float(cov[s.B.to_numpy() == 0].mean()) if (s.B == 0).any() else None,
                     "coverage_by_burden_tertile": {str(k): float(cov[(tert == k).to_numpy()].mean()) for k in ("low", "mid", "high")}}  # fmt: skip
            res[name] = entry
            if name == "P2-B direct":
                width_b = width
            if name == "P2-A decomposition":
                width_a = width
            if name == "P2-B-CQR":
                width_bq = width
        ratio = width_a / np.maximum(width_b, 1e-9)
        boot = np.random.default_rng(SEED).integers(0, len(ratio), size=(10000, len(ratio)))
        meds = np.median(ratio[boot], axis=1)
        res["paired_width_ratio_A_over_B"] = {"median": float(np.median(ratio)), "ci95": [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]}
        # 12 Nov 2026 (owner-approved): CI for A / B-CQR, same scan-level paired bootstrap (same resample indices)
        ratio_q = width_a / np.maximum(width_bq, 1e-9)
        meds_q = np.median(ratio_q[boot], axis=1)
        res["paired_width_ratio_A_over_BCQR"] = {"median": float(np.median(ratio_q)), "ci95": [float(np.quantile(meds_q, 0.025)), float(np.quantile(meds_q, 0.975))],
                                                 "bootstrap": "scan-level paired, 10000 resamples, seed 20261105, same indices as A/B"}  # fmt: skip
        out["results"][f"{1 - alpha:.2f}"] = res
        print(f"nominal {1 - alpha:.2f}:", json.dumps(res, default=float), flush=True)
    run = create_run_folder(Path("results/runs"), "e8_burden_uncertainty" + ("_DEBUG" if debug else ""))
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("Saved", run)


if __name__ == "__main__":
    main()

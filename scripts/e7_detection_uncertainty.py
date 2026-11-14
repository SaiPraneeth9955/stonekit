"""E7 (S4): detection uncertainty on development out-of-fold data (master plan v2). Detection is kept
separate from measurement: a missed stone is never counted as a measurement error.

Reference = Li annotation (not physical truth). Unit = annotated piece / predicted piece, clustered by scan.
1. Detection: share of annotated pieces in a group with >= 1 predicted piece (any group kind except
   'missed'), by annotated 3D diameter bin (<3, 3-<5, 5-<10, >=10 mm) and HU tertile, with scan-cluster
   bootstrap 95% CIs; GEE logistic (independence) of detected on log2 volume and mean HU / 100.
2. False positives: predicted pieces in 'false_positive' groups per scan, unfiltered and with the 30 mm
   kidney-distance filter; also on stone-free scans.
3. Per-piece false-positive probability: logistic regression of is_FP on log volume, p_mean, TTA spread,
   threshold sensitivity, distance to kidney, mean HU / 100, cross-fitted by fold (each fold scored by a
   model fitted on the other folds). Discrimination (AUROC), Brier score, and a 5-bin reliability table.
   With a single fold (debug), 5 seeded scan groups are used and the output is labelled DEBUG.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from stonekit.config import create_run_folder

SEED = 20261105
RESAMPLES = 10000
BINS = [0, 3, 5, 10, math.inf]
BIN_LABELS = ["<3", "3-<5", "5-<10", ">=10"]
FP_FEATURES = ["log_volume", "p_mean", "tta_spread", "threshold_sensitivity", "distance_to_kidney_mm", "hu100"]


def cluster_share(values: np.ndarray, clusters: np.ndarray) -> dict:
    values = np.asarray(values, float)
    ids, inv = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values, minlength=len(ids)), np.bincount(inv, minlength=len(ids)).astype(float)
    draws = np.random.default_rng(SEED).integers(0, len(ids), size=(RESAMPLES, len(ids)))
    est = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    return {"n": len(values), "scans": len(ids), "share": float(values.mean()) if len(values) else None,
            "cluster_ci95": [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    p = pd.read_csv(args.table_folder / "predicted.csv")
    s = pd.read_csv(args.table_folder / "scans.csv")
    debug = s["fold"].nunique() < 2
    if debug:
        scans = np.sort(s["case"].unique())
        np.random.default_rng(SEED).shuffle(scans)
        group_of = {c: i % 5 for i, c in enumerate(scans)}
    else:
        group_of = dict(zip(s["case"], s["fold"], strict=True))
    out = {"status": "DEBUG_NOT_A_RESULT" if debug else "DEVELOPMENT_OBSERVED", "definitions": __doc__,
           "scans": len(s), "annotated_pieces": len(a), "predicted_pieces": len(p)}  # fmt: skip
    a["detected"] = (a["group"] != "missed").astype(int)
    a["size_bin"] = pd.cut(a["ref_max_diameter_3d_mm"], BINS, right=False, labels=BIN_LABELS)
    a["hu_tertile"] = pd.qcut(a["ref_mean_hu"], 3, labels=["low", "mid", "high"])
    out["detection_overall"] = cluster_share(a["detected"], a["case"])
    for col in ("size_bin", "hu_tertile"):
        out[f"detection_by_{col}"] = {str(k): cluster_share(g["detected"], g["case"]) for k, g in a.groupby(col, observed=True)}
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    m = a.dropna(subset=["ref_volume_mm3", "ref_mean_hu"]).assign(log2v=lambda d: np.log2(d.ref_volume_mm3), hu100=lambda d: d.ref_mean_hu / 100)
    try:
        fit = smf.gee("detected ~ log2v + hu100", groups="case", data=m, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()
        ci = fit.conf_int()
        out["detection_gee_odds_ratios"] = {k: {"or": float(np.exp(fit.params[k])), "ci95": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))]}
                                            for k in ("log2v", "hu100")}  # fmt: skip
        b0, b1, b2 = fit.params["Intercept"], fit.params["log2v"], fit.params["hu100"]
        hu_med = float(m.hu100.median())
        v50 = 2 ** (-(b0 + b2 * hu_med) / b1)  # volume with 50% detection at median HU
        out["d50_at_median_hu"] = {"volume_mm3": float(v50), "sphere_equivalent_diameter_mm": float((6 * v50 / math.pi) ** (1 / 3)), "median_hu": hu_med * 100}
    except Exception as error:  # noqa: BLE001
        out["detection_gee_odds_ratios"] = {"failed": repr(error)}
    p["is_fp"] = (p["group"] == "false_positive").astype(int)
    per_scan = p.groupby("case").agg(fp=("is_fp", "sum"), fp_filtered=("is_fp", lambda x: int((x & ~p.loc[x.index, "beyond_30mm"]).sum())))
    per_scan = s[["case", "annotated_pieces"]].merge(per_scan, on="case", how="left").fillna(0)
    out["false_positives_per_scan"] = {"unfiltered_mean": float(per_scan.fp.mean()), "filtered_30mm_mean": float(per_scan.fp_filtered.mean()),
                                       "stone_free_scans": int((per_scan.annotated_pieces == 0).sum()),
                                       "stone_free_with_any_fp_unfiltered": int(((per_scan.annotated_pieces == 0) & (per_scan.fp > 0)).sum()),
                                       "stone_free_with_any_fp_filtered": int(((per_scan.annotated_pieces == 0) & (per_scan.fp_filtered > 0)).sum())}  # fmt: skip
    q = p.dropna(subset=["tta_spread", "threshold_sensitivity", "p_mean", "mean_hu"]).copy()
    q["log_volume"], q["hu100"] = np.log(q["volume_mm3"]), q["mean_hu"] / 100
    q["cv_group"] = q["case"].map(group_of)
    pred = np.full(len(q), np.nan)
    for g in sorted(q["cv_group"].unique()):
        train, test = (q.cv_group != g).to_numpy(), (q.cv_group == g).to_numpy()
        if q.is_fp[train].nunique() < 2:
            continue
        model = LogisticRegression(max_iter=1000).fit(q.loc[train, FP_FEATURES], q.loc[train, "is_fp"])
        pred[test] = model.predict_proba(q.loc[test, FP_FEATURES])[:, 1]
    ok = ~np.isnan(pred)
    y = q["is_fp"].to_numpy()[ok]
    if ok.sum() and len(np.unique(y)) == 2:
        bins = pd.cut(pred[ok], [0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
        rel = pd.DataFrame({"bin": bins, "y": y, "p": pred[ok]}).groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
        out["fp_probability_model"] = {"pieces": int(ok.sum()), "fp_share": float(y.mean()), "auroc": float(roc_auc_score(y, pred[ok])),
                                       "brier": float(brier_score_loss(y, pred[ok])), "brier_of_constant_rate": float(brier_score_loss(y, np.full(len(y), y.mean()))),
                                       "reliability": {str(k): v for k, v in rel.to_dict("index").items()}}  # fmt: skip
        single = {f: float(roc_auc_score(y, (q[f].to_numpy()[ok]) * (1 if f != "p_mean" else -1))) for f in FP_FEATURES if q[f].notna().all()}
        out["fp_single_feature_auroc"] = single
    run = create_run_folder(Path("results/runs"), "e7_detection_uncertainty" + ("_DEBUG" if debug else ""))
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1, default=float))
    print("Saved", run)


if __name__ == "__main__":
    main()

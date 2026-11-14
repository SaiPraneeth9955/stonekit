"""E6: per-stone measurement-uncertainty design on development out-of-fold data (master plan v2 §G).

Measurement uncertainty only (uncertainty A): one_to_one matched stones, reference = Li annotation (not
physical truth). Cross-conformal by fold: for each fold k, intervals are calibrated on the other folds'
OOF stones and evaluated on fold k's stones. This SELECTS a method; it is not validation.

Methods compared (no winner pre-declared):
U0 Bland-Altman on log ratios; U1 global split conformal; U2 Mondrian by predicted 3D diameter
(<5, 5-<10, >=10 mm; min 20 calibration stones, plan merge rule); U3 CQR on (log V_hat, slice mm,
mean HU / 1000); U4 normalised by TTA spread; U5 normalised by threshold sensitivity (signal floors:
the calibration set's 5th percentile, fixed rule).

Selection rule (plan §G as amended 11 November 2026 after the debug run showed that the original
lower-bound rule rejects calibrated methods at these sample sizes): a method qualifies if, in every
predicted-size stratum, coverage is not significantly below nominal (Clopper-Pearson 95% upper bound
>= nominal) AND the point coverage is >= nominal - 0.05; pick the smallest median relative width; ties
within 5% -> simpler (U1 < U2 < U3 < U4 = U5). U0 is a comparator only. If none qualifies, U2 is
carried and that is reported.
If the table has a single fold (debug use only), scans are split into 5 seeded groups instead and the
output is labelled DEBUG_NOT_A_RESULT.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stonekit.config import create_run_folder
from stonekit.uq import conformal as cf

SEED = 20261105
EDGES = (5.0, 10.0)
MIN_STRATUM = 20
MARGIN = 0.05
SIMPLICITY = {"U1": 1, "U2": 2, "U3": 3, "U4": 4, "U5": 4}
TRUE_BINS = [(0, 3, "<3"), (3, 5, "3-<5"), (5, 10, "5-<10"), (10, math.inf, ">=10")]


def intervals(method: str, cal: pd.DataFrame, test: pd.DataFrame, target: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Log-ratio bounds (lo, hi) for the test rows; interval = V_hat * exp([lo, hi])."""
    r_cal = cf.log_ratio(cal[f"ref_{target}"], cal[f"pred_{target}"])
    if method == "U0":
        lo, hi = cf.bland_altman_log(r_cal, stats.norm.ppf(1 - alpha / 2))
        return np.full(len(test), lo), np.full(len(test), hi)
    if method == "U1":
        q = cf.conformal_quantile(np.abs(r_cal), alpha)
        return np.full(len(test), -q), np.full(len(test), q)
    if method == "U2":
        strata = cf.fit_mondrian(np.abs(r_cal), cal["pred_max_diameter_3d_mm"], EDGES, alpha, MIN_STRATUM)
        q = np.asarray(strata.quantiles)[strata.index(test["pred_max_diameter_3d_mm"])]
        return -q, q
    if method == "U3":
        x_cal, x_test = features(cal), features(test)
        scans = cal["case"].unique()
        rng = np.random.default_rng(SEED)
        fit_scans = set(rng.choice(scans, size=len(scans) // 2, replace=False))
        fit = np.flatnonzero(cal["case"].isin(fit_scans).to_numpy())
        calib = np.flatnonzero(~cal["case"].isin(fit_scans).to_numpy())
        predict = cf.fit_cqr(x_cal, r_cal, alpha, fit, calib)
        return predict(x_test)
    signal = {"U4": "pred_tta_spread", "U5": "pred_threshold_sensitivity"}[method]
    floor = float(np.nanquantile(cal[signal], 0.05))
    floor = floor if floor > 0 else float(np.nanmin(cal[signal][cal[signal] > 0])) if (cal[signal] > 0).any() else 1.0
    s_cal = np.maximum(cal[signal].to_numpy(float), floor)
    s_test = np.maximum(test[signal].to_numpy(float), floor)
    q = cf.conformal_quantile(np.abs(r_cal) / s_cal, alpha)
    return -q * s_test, q * s_test


def features(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.log(frame["pred_volume_mm3"]), frame["slice_mm"], frame["pred_mean_hu"] / 1000.0])


def summarise(frame: pd.DataFrame, covered: np.ndarray, width: np.ndarray, group: pd.Series) -> dict:
    out = {}
    for name in sorted(group.dropna().unique(), key=str):
        m = (group == name).to_numpy()
        n, k = int(m.sum()), int(covered[m].sum())
        lo, hi = cf.clopper_pearson(k, n)
        blo, bhi = cf.cluster_bootstrap_coverage(covered[m], frame["case"].to_numpy()[m], 2000, SEED) if n else (math.nan, math.nan)
        out[str(name)] = {"n": n, "scans": int(frame["case"][m].nunique()), "coverage": k / n if n else None,
                          "cp95": [lo, hi], "cluster_boot95": [blo, bhi],
                          "median_rel_width": float(np.median(width[m])) if n else None}  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.20])
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    data = a[(a.group == "one_to_one") & (a.ref_volume_mm3 > 0) & (a.pred_volume_mm3 > 0)].copy().reset_index(drop=True)
    debug = data["fold"].nunique() < 2
    if debug:
        scans = np.sort(data["case"].unique())
        np.random.default_rng(SEED).shuffle(scans)
        data["cv_group"] = data["case"].map({c: i % 5 for i, c in enumerate(scans)})
    else:
        data["cv_group"] = data["fold"]
    has_signals = data["pred_tta_spread"].notna().all()
    methods = ["U0", "U1", "U2", "U3"] + (["U4", "U5"] if has_signals else [])
    data["pred_stratum"] = pd.cut(data["pred_max_diameter_3d_mm"], [-np.inf, *EDGES, np.inf], right=False, labels=["<5", "5-<10", ">=10"])
    data["true_stratum"] = pd.cut(data["ref_max_diameter_3d_mm"], [b[0] for b in TRUE_BINS] + [np.inf], right=False, labels=[b[2] for b in TRUE_BINS])
    data["hu_tertile"] = pd.qcut(data["pred_mean_hu"], 3, labels=["low", "mid", "high"])
    data["slice_group"] = pd.cut(data["slice_mm"], [0, 1.0, 2.0, np.inf], labels=["<=1 mm", "1-2 mm", ">2 mm"])
    result = {"status": "DEBUG_NOT_A_RESULT" if debug else "DEVELOPMENT_DESIGN_NOT_VALIDATION", "definitions": __doc__,
              "stones": len(data), "scans": int(data["case"].nunique()), "folds": sorted(int(f) for f in data["fold"].unique()),
              "has_tta_signals": bool(has_signals), "results": {}}  # fmt: skip
    per_stone = data[["case", "fold", "pred_stratum", "true_stratum"]].copy()
    for target in ("volume_mm3", "max_diameter_3d_mm"):
        for alpha in args.alphas:
            key = f"{target}|{1 - alpha:.2f}"
            result["results"][key] = {}
            for method in methods:
                lo = np.empty(len(data))
                hi = np.empty(len(data))
                for g in sorted(data["cv_group"].unique()):
                    test = data["cv_group"] == g
                    lo[test.to_numpy()], hi[test.to_numpy()] = intervals(method, data[~test], data[test], target, alpha)
                r = cf.log_ratio(data[f"ref_{target}"], data[f"pred_{target}"])
                covered = (r >= lo) & (r <= hi)
                width = np.exp(hi) - np.exp(lo)
                per_stone[f"covered|{key}|{method}"] = covered
                per_stone[f"relwidth|{key}|{method}"] = width
                entry = {"marginal": summarise(data, covered, width, pd.Series(["all"] * len(data)))["all"]}
                for col in ("pred_stratum", "true_stratum", "hu_tertile", "slice_group"):
                    entry[col] = summarise(data, covered, width, data[col])
                patient = pd.DataFrame({"case": data["case"], "c": covered}).groupby("case")["c"].all()
                entry["patient_all_stones_covered"] = {"n": len(patient), "share": float(patient.mean())}
                result["results"][key][method] = entry
            qualifying = {m: e for m, e in result["results"][key].items() if m != "U0" and all(
                s["cp95"][1] >= (1 - alpha) and s["coverage"] >= (1 - alpha) - MARGIN for s in e["pred_stratum"].values() if s["n"])}  # fmt: skip
            if qualifying:
                best = min(qualifying, key=lambda m: qualifying[m]["marginal"]["median_rel_width"])
                ties = [m for m in qualifying if qualifying[m]["marginal"]["median_rel_width"] <= 1.05 * qualifying[best]["marginal"]["median_rel_width"]]
                chosen = min(ties, key=lambda m: SIMPLICITY[m])
            else:
                chosen = "U2 (none qualified)"
            result["results"][key]["selection"] = {"qualifying": sorted(qualifying), "chosen": chosen}
            print(key, "qualifying", sorted(qualifying), "chosen", chosen, flush=True)
            for m in methods:
                e = result["results"][key][m]
                strata = {k: (round(v["coverage"], 3) if v["coverage"] is not None else None, v["n"]) for k, v in e["pred_stratum"].items()}
                print(f"  {m}: marginal {e['marginal']['coverage']:.3f} width {e['marginal']['median_rel_width']:.3f} | pred strata {strata}", flush=True)
    run = create_run_folder(Path("results/runs"), "e6_uncertainty_design" + ("_DEBUG" if debug else ""))
    (run / "summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    # Per-stone coverage indicators (development OOF; git-ignored) for the P1 power simulation.
    per_stone.to_csv(Path("data/interim") / f"e6_per_stone_{run.name}.csv", index=False)
    (run / "inputs.json").write_text(json.dumps({"table_folder": str(args.table_folder)}, indent=2), encoding="utf-8")
    print("Saved", run)


if __name__ == "__main__":
    main()

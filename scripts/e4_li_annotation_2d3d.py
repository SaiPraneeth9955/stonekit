"""E4 (S3) and E11 (S7) on Li development ANNOTATIONS: 2D vs 3D category disagreement, its drivers, and
largest-stone vs total-burden representation. Pre-specified here before the first run (11 November 2026).

Reference = Li expert annotation (not physical truth); no model output is used, so all 165 development
scans are eligible (reserved scans refused by the loader). Pieces: 26-connected, >= 3 voxels (pipeline rule).

S3 (per stone; unit = stone, clustered by scan):
- definitions: D2 = 2D axial maximum (centre-to-centre), D3 = 3D maximum (centre-to-centre),
  Deq = volume-equivalent sphere diameter; all from stonekit.measure.
- categories (research bins): <5, 5-<10, >=10 mm; descriptive split of <5 into <3 and 3-<5.
- primary estimand: proportion of stones whose D2 category differs from their D3 category, with a
  scan-cluster bootstrap 95% CI (10,000 resamples, seed 20261105).
- drivers: GEE logistic (independence working correlation, clusters = scans) of "category differs" on
  elongation, major-axis angle to z (per 10 degrees), slice thickness (mm), log2 volume, mean HU / 100.
  Reported as odds ratios with robust 95% CIs. Only stones with all covariates.
- also: relative difference (D3 - D2) / D3, median and IQR by D3 category.
S7 (per scan with >= 1 stone; unit = scan):
- Spearman rho between largest-stone D3 and total annotated volume;
- among scan pairs whose largest-stone D3 differs by < 1 mm, the share whose total volumes differ > 2-fold;
- share of scans with >= 2 pieces; share of total volume outside the largest piece (median, IQR).
Descriptive; no outcome claim.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.measure.stones import measure_stones, separate_stones

SEED = 20261105
RESAMPLES = 10000
EDGES = [0, 5, 10, math.inf]
LABELS = ["<5", "5-<10", ">=10"]


def category(d: pd.Series) -> pd.Series:
    return pd.cut(d, EDGES, right=False, labels=LABELS)


def cluster_ci(values: np.ndarray, clusters: np.ndarray) -> list[float]:
    ids, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values.astype(float), minlength=len(ids))
    counts = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(ids), size=(RESAMPLES, len(ids)))
    est = sums[draws].sum(1) / np.maximum(counts[draws].sum(1), 1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=None, help="Reuse a saved per-stone table instead of reloading scans")
    parser.add_argument("--exclude-csv", type=Path, default=None, help="Review outcome file; action=exclude scans are dropped")
    args = parser.parse_args()
    import csv

    excluded = set()
    if args.exclude_csv is not None:
        with open(args.exclude_csv, newline="", encoding="utf-8") as handle:
            excluded = {r["case"] for r in csv.DictReader(handle) if r["action"] == "exclude"}
    root = Path.cwd()
    if args.table:
        df = pd.read_csv(args.table)
        df = df[~df.case.isin(excluded)]
    else:
        data = LiDevelopmentData(root)
        cases = sorted(c for c, g in data.manifest.items() if g == "development" and c not in excluded)
        rows = []
        for n, case in enumerate(cases, start=1):
            li = data.load(case)
            sep = separate_stones(li.stone_reference, 3)
            for s in measure_stones(sep, li.image.data, li.image.affine, li.image.spacing_mm):
                rows.append({"case": case, "slice_mm": float(li.image.spacing_mm[2]), **{k: s.get(k) for k in (
                    "volume_mm3", "max_diameter_3d_mm", "max_diameter_axial_mm", "volume_equivalent_diameter_mm",
                    "elongation", "major_axis_angle_to_z_deg", "mean_hu")}})  # fmt: skip
            if not sep.n_stones:
                rows.append({"case": case, "slice_mm": float(li.image.spacing_mm[2]), "volume_mm3": None})
            print(f"{n}/{len(cases)} {case}", flush=True)
        df = pd.DataFrame(rows)
    run = create_run_folder(root / "results/runs", "e4_li_annotation_2d3d")
    stones = df[df.volume_mm3.notna()].copy()
    stones["cat3"], stones["cat2"] = category(stones.max_diameter_3d_mm), category(stones.max_diameter_axial_mm)
    stones["differs"] = (stones.cat3 != stones.cat2).astype(int)
    out = {"status": "DEVELOPMENT_OBSERVED", "definitions": __doc__, "excluded_by_review": sorted(excluded), "scans": int(df.case.nunique()),
           "scans_with_stones": int(stones.case.nunique()), "stones": len(stones)}  # fmt: skip
    out["S3_category_differs"] = {"share": float(stones.differs.mean()), "cluster_ci95": cluster_ci(stones.differs.to_numpy(), stones.case.to_numpy()), "n": len(stones)}
    out["S3_crosstab_d3_rows_d2_cols"] = pd.crosstab(stones.cat3, stones.cat2).to_dict()
    fine = pd.cut(stones.max_diameter_3d_mm, [0, 3, 5, 10, math.inf], right=False, labels=["<3", "3-<5", "5-<10", ">=10"])
    stones["rel_diff"] = (stones.max_diameter_3d_mm - stones.max_diameter_axial_mm) / stones.max_diameter_3d_mm.where(stones.max_diameter_3d_mm > 0)
    out["S3_rel_diff_by_d3"] = {str(k): {"n": len(g), "median": float(g.rel_diff.median()), "iqr": [float(g.rel_diff.quantile(0.25)), float(g.rel_diff.quantile(0.75))],
                                         "share_differs": float(g.differs.mean())} for k, g in stones.groupby(fine, observed=True)}  # fmt: skip
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    m = stones.dropna(subset=["elongation", "major_axis_angle_to_z_deg", "slice_mm", "mean_hu"]).copy()
    m["angle10"] = m.major_axis_angle_to_z_deg / 10
    m["log2v"] = np.log2(m.volume_mm3)
    m["hu100"] = m.mean_hu / 100
    try:
        fit = smf.gee("differs ~ elongation + angle10 + slice_mm + log2v + hu100", groups="case", data=m,
                      family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence()).fit()  # fmt: skip
        ci = fit.conf_int()
        out["S3_gee_odds_ratios"] = {k: {"or": float(np.exp(fit.params[k])), "ci95": [float(np.exp(ci.loc[k, 0])), float(np.exp(ci.loc[k, 1]))],
                                         "p": float(fit.pvalues[k])} for k in fit.params.index if k != "Intercept"}  # fmt: skip
        out["S3_gee_n"] = {"stones": len(m), "scans": int(m.case.nunique())}
    except Exception as error:  # noqa: BLE001  (a failed fit is reported, not hidden)
        out["S3_gee_odds_ratios"] = {"failed": repr(error)}
    per_scan = stones.groupby("case").agg(largest=("max_diameter_3d_mm", "max"), total=("volume_mm3", "sum"), pieces=("volume_mm3", "size"),
                                          biggest_volume=("volume_mm3", "max"))  # fmt: skip
    rho = stats.spearmanr(per_scan.largest, per_scan.total)
    pairs = []
    lv, tv = per_scan.largest.to_numpy(), per_scan.total.to_numpy()
    for i in range(len(lv)):
        for j in range(i + 1, len(lv)):
            if abs(lv[i] - lv[j]) < 1.0:
                pairs.append(max(tv[i], tv[j]) / min(tv[i], tv[j]) > 2)
    outside = 1 - per_scan.biggest_volume / per_scan.total
    out["S7"] = {"scans": len(per_scan), "spearman_rho_largest_d3_vs_total_volume": float(rho.statistic), "p": float(rho.pvalue),
                 "pairs_largest_within_1mm": len(pairs), "share_of_those_total_volume_differs_gt_2x": float(np.mean(pairs)) if pairs else None,
                 "share_scans_multiple_pieces": float((per_scan.pieces >= 2).mean()),
                 "volume_outside_largest_piece_median_iqr": [float(outside.median()), float(outside.quantile(0.25)), float(outside.quantile(0.75))]}  # fmt: skip
    df.to_csv(Path("data/interim") / f"e4_annotation_stones_{run.name}.csv", index=False)
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1, default=float))
    print("Saved", run)


if __name__ == "__main__":
    main()

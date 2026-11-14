"""Failure analysis of the FINAL-arm intervals on DEVELOPMENT out-of-fold data (numbers only; no images; no
calibration or test data). Characterises which stones the per-stone U2 interval misses, and which patients the
P2-A burden interval misses, so that failure modes can be reported and flagged, not hidden.

Per-stone (U2 volume, nominal 0.90; coverage indicators from the saved E6 per-stone file): the miss rate by
predicted-size band, HU tertile, native slice group, elongation tertile and fold. Misses are split into
under-estimates (reference > upper bound: the stone is bigger than the interval allows) and over-estimates
(reference < lower bound), and summarised by the size of the log error relative to the interval half-width.
Patient burden (P2-A, nominal 0.90): recomputed with the same cross-fitted procedure as E8 is NOT done here
(no rerun). Instead the E8 scan-level drivers are summarised for all scans: missed-burden share, false-positive
volume share and stone-free status, from the out-of-fold table, as context for the burden failure modes.
Label DEVELOPMENT_FAILURE_ANALYSIS.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder
from stonekit.uq import conformal as cf

EDGES = (5.0, 10.0)


def rate(df, col):
    g = df.groupby(col, observed=True)["miss"].agg(["mean", "sum", "size"])
    return {str(k): {"miss_rate": float(r["mean"]), "misses": int(r["sum"]), "n": int(r["size"]),
                     "cp95": list(cf.clopper_pearson(int(r["sum"]), int(r["size"])))} for k, r in g.iterrows()}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_folder", type=Path)
    parser.add_argument("--per-stone", type=Path, default=Path("data/interim/e6_per_stone_e6_uncertainty_design_20261112T082354Z.csv"))
    args = parser.parse_args()
    a = pd.read_csv(args.table_folder / "annotated.csv")
    d = a[(a.group == "one_to_one") & (a.ref_volume_mm3 > 0) & (a.pred_volume_mm3 > 0)].copy().reset_index(drop=True)
    ps = pd.read_csv(args.per_stone)
    if len(ps) != len(d) or not (ps["case"].to_numpy() == d["case"].to_numpy()).all():
        raise SystemExit("per-stone file does not align with the table (row order); stop")
    d["miss"] = ~ps["covered|volume_mm3|0.90|U2"].astype(bool).to_numpy()
    d["r"] = np.log(d.ref_volume_mm3 / d.pred_volume_mm3)
    d["band"] = pd.cut(d.pred_max_diameter_3d_mm, [-np.inf, *EDGES, np.inf], right=False, labels=["<5", "5-<10", ">=10"])
    d["hu_tertile"] = pd.qcut(d.pred_mean_hu, 3, labels=["low", "mid", "high"])
    d["slice_group"] = pd.cut(d.slice_mm, [0, 1.0, 2.0, np.inf], labels=["<=1 mm", "1-2 mm", ">2 mm"])
    d["elongation_tertile"] = pd.qcut(d.pred_elongation, 3, labels=["low", "mid", "high"]) if "pred_elongation" in d else "n/a"
    out = {"status": "DEVELOPMENT_FAILURE_ANALYSIS", "definitions": __doc__, "stones": len(d), "misses": int(d.miss.sum()),
           "per_stone_U2_volume_0.90": {c: rate(d, c) for c in ("band", "hu_tertile", "slice_group", "elongation_tertile", "fold")},
           "miss_direction": {"under_estimate_ref_above_interval": int((d.miss & (d.r > 0)).sum()),
                              "over_estimate_ref_below_interval": int((d.miss & (d.r < 0)).sum())},
           "miss_log_error_abs": {"median": float(d.loc[d.miss, "r"].abs().median()), "max": float(d.loc[d.miss, "r"].abs().max())}}  # fmt: skip
    s = pd.read_csv(args.table_folder / "scans.csv")
    missed = a[a.group == "missed"].groupby("case")["ref_volume_mm3"].sum()
    p = pd.read_csv(args.table_folder / "predicted.csv")
    fp = p[(p.group == "false_positive") & ~p.beyond_30mm.astype(bool)].groupby("case")["volume_mm3"].sum()
    s["missed_mm3"], s["fp_mm3"] = s.case.map(missed).fillna(0.0), s.case.map(fp).fillna(0.0)
    with_stone = s[s.annotated_burden_mm3 > 0]
    out["burden_drivers"] = {
        "scans": len(s), "stone_free": int((s.annotated_burden_mm3 == 0).sum()),
        "stone_free_with_fp_volume_after_filter": int(((s.annotated_burden_mm3 == 0) & (s.fp_mm3 > 0)).sum()),
        "missed_share_of_burden_median": float((with_stone.missed_mm3 / with_stone.annotated_burden_mm3).median()),
        "missed_share_of_burden_p90": float((with_stone.missed_mm3 / with_stone.annotated_burden_mm3).quantile(0.9)),
        "fp_share_of_predicted_median": float((s.fp_mm3 / s.predicted_burden_filtered_mm3.replace(0, np.nan)).median()),
        "total_missed_over_total_burden": float(s.missed_mm3.sum() / s.annotated_burden_mm3.sum()),
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "failure_analysis_development")
    (run / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k not in ("definitions",)}, indent=1, default=float)[:3500])
    print("Saved", run)


if __name__ == "__main__":
    main()

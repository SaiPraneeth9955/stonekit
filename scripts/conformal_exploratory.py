"""Exploratory per-stone volume intervals by split conformal prediction, leave-one-scan-out.

Question: can StoneKit attach a per-stone interval to a measured volume that contains the reference
(annotated) volume at a stated rate? This is a prediction interval for the reference value of one
detected stone, not a confidence interval for a mean.

Data: one-to-one matched pieces of the 100-epoch model on the 32 fold-0 validation scans (saved
rows; reads no image). The reference is the Li annotation, not physical truth. Missed stones get no
interval, so coverage is conditional on detection, and false positives have no reference.

Method (established; Vovk et al.; Lei et al.): nonconformity s = |log(annotated / predicted)|.
Variant "global": one quantile. Variant "mondrian": separate quantiles by predicted 3D diameter
(<5, 5-<10, >=10 mm), because relative error depends strongly on size. Calibration is
leave-one-scan-out: each scan's stones get the quantile ceil((n+1)(1-alpha))/n of all OTHER scans'
scores, so no stone calibrates its own interval. Pieces within a scan are not exchangeable with
independence, so the finite-sample guarantee is only approximate; coverage is measured, not assumed.
Interval: [predicted * exp(-q), predicted * exp(q)] in mm3.

Status: EXPLORATORY. Calibration and evaluation scans come from the same 32 development scans
(cross-fitted, never the same scan). The confirmatory version calibrates on the reserved calibration
scans and is evaluated on the locked test set, and needs owner approval.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

SEED = 20261105
LEVELS = [0.8, 0.9]
SIZE_BINS = [(0, 5, "<5 mm"), (5, 10, "5-<10 mm"), (10, np.inf, ">=10 mm")]


def size_bin(d: float) -> str:
    return next(name for lo, hi, name in SIZE_BINS if lo <= d < hi)


def quantile(scores: np.ndarray, level: float) -> float:
    n = len(scores)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * level))
    return float(np.sort(scores)[k - 1]) if k <= n else np.inf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=Path, required=True)
    parser.add_argument("--method", default="model_within_30mm_of_kidney")
    args = parser.parse_args()
    with (args.detail / "pairs.csv").open(newline="", encoding="utf-8") as f:
        pairs = [r for r in csv.DictReader(f) if r["method"] == args.method]
    for r in pairs:
        r["a"], r["p"], r["d"] = float(r["annotated_mm3"]), float(r["predicted_mm3"]), float(r["predicted_d3_mm"])
        r["score"] = abs(np.log(r["a"] / r["p"]))
        r["bin"] = size_bin(r["d"])
    cases = sorted({r["case"] for r in pairs})
    results = {"status": "EXPLORATORY_DEVELOPMENT_NOT_CALIBRATED_CONFIRMATORY", "definitions": __doc__, "pairs": len(pairs), "scans": len(cases), "variants": {}}
    rows = []
    for variant in ("global", "mondrian"):
        out = {}
        for level in LEVELS:
            covered, widths, per_bin = [], [], {name: [] for _, _, name in SIZE_BINS}
            for case in cases:
                calib = [r for r in pairs if r["case"] != case]
                for r in (r for r in pairs if r["case"] == case):
                    pool = calib if variant == "global" else [c for c in calib if c["bin"] == r["bin"]]
                    q = quantile(np.array([c["score"] for c in pool]), level)
                    low, high = r["p"] * np.exp(-q), r["p"] * np.exp(q)
                    hit = low <= r["a"] <= high
                    covered.append(hit)
                    widths.append(np.exp(q) if np.isfinite(q) else np.inf)
                    per_bin[r["bin"]].append((hit, np.exp(q) if np.isfinite(q) else np.inf))
                    rows.append({"variant": variant, "level": level, "case": case, "bin": r["bin"], "covered": hit, "factor": float(np.exp(q)) if np.isfinite(q) else None})
            # Scan-level bootstrap interval for empirical coverage (stones within a scan are clustered).
            by_case = {c: [x["covered"] for x in rows if x["variant"] == variant and x["level"] == level and x["case"] == c] for c in cases}
            rng = np.random.default_rng(SEED)
            boot = []
            for _ in range(10000):
                sample = [by_case[cases[i]] for i in rng.integers(0, len(cases), len(cases))]
                flat = [v for s in sample for v in s]
                boot.append(np.mean(flat))
            finite = np.array([w for w in widths if np.isfinite(w)])
            out[f"{int(level * 100)}%"] = {
                "nominal": level, "empirical_coverage": round(float(np.mean(covered)), 3),
                "coverage_interval_95_scan_bootstrap": [round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)],
                "median_multiplicative_half_width": round(float(np.median(finite)), 2) if len(finite) else None,
                "median_interval_as_percent_of_predicted": [round(float(100 / np.median(finite)), 1), round(float(100 * np.median(finite)), 1)] if len(finite) else None,
                "infinite_intervals": int(sum(not np.isfinite(w) for w in widths)),
                "by_predicted_size": {name: {"n": len(v), "coverage": round(float(np.mean([h for h, _ in v])), 3) if v else None,
                                             "median_factor": round(float(np.median([w for _, w in v if np.isfinite(w)])), 2) if any(np.isfinite(w) for _, w in v) else None}
                                      for name, v in per_bin.items()},
            }  # fmt: skip
        results["variants"][variant] = out
    run = create_run_folder(Path("results/runs"), "conformal_exploratory")
    (run / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["variants"], indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

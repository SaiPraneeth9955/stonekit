"""Pre-specified power simulation that fixes the P1 non-inferiority margin delta (amendment 6, 11 Nov 2026).

P1 primary: marginal per-stone coverage on the locked test (53 scans; 50 only if D-REV is approved); success if the scan-cluster bootstrap
95% lower bound >= nominal - delta. delta = the smallest value in {0.05, 0.075, 0.10} giving >= 80% power
when the true coverage equals nominal.

Simulation (development data only; no calibration or test data):
- the cluster structure (matched stones per scan, and the within-scan correlation of coverage) is taken
  from the development OOF coverage indicators of the method selected by E6;
- to impose "true coverage = nominal", each simulated test set resamples TEST_SCANS development scans with
  replacement and keeps their within-scan covered/not-covered pattern, after the scans are reweighted so
  the expected stone-level coverage equals nominal exactly (exponential tilting of scan weights by their
  coverage share; the tilt is solved numerically);
- each simulated set gets the same scan-cluster percentile bootstrap (2,000 resamples) as the real test.
Reported: power for each delta, the chosen delta, and the expected number of matched test stones.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from stonekit.config import create_run_folder

SEED = 20261105
TEST_SCANS = 53  # authoritative locked test design; 50 only if the owner approves D-REV (use --test-scans)
SIMS = 2000
BOOT = 2000
DELTAS = (0.05, 0.075, 0.10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_stone_csv", type=Path, help="data/interim/e6_per_stone_<run>.csv")
    parser.add_argument("--column", required=True, help="covered|<target>|<nominal>|<method> column of the selected method")
    parser.add_argument("--nominal", type=float, default=0.90)
    parser.add_argument("--test-scans", type=int, default=TEST_SCANS)
    args = parser.parse_args()
    df = pd.read_csv(args.per_stone_csv)
    g = df.groupby("case")[args.column].agg(["sum", "size"])
    hits, sizes = g["sum"].to_numpy(float), g["size"].to_numpy(float)
    share = hits / sizes

    def expected(t: float) -> float:
        w = np.exp(t * share)
        w /= w.sum()
        return float((w * hits).sum() / (w * sizes).sum())

    tilt = brentq(lambda t: expected(t) - args.nominal, -50, 50) if 0 < share.mean() < 1 else 0.0
    weights = np.exp(tilt * share)
    weights /= weights.sum()
    rng = np.random.default_rng(SEED)
    lower = np.empty(SIMS)
    for k in range(SIMS):
        pick = rng.choice(len(hits), size=args.test_scans, replace=True, p=weights)
        h, n = hits[pick], sizes[pick]
        b = rng.integers(0, args.test_scans, size=(BOOT, args.test_scans))
        est = h[b].sum(1) / n[b].sum(1)
        lower[k] = np.quantile(est, 0.025)
    power = {str(d): float((lower >= args.nominal - d).mean()) for d in DELTAS}
    chosen = next((d for d in DELTAS if power[str(d)] >= 0.80), None)
    out = {"definitions": __doc__, "source": str(args.per_stone_csv), "column": args.column, "nominal": args.nominal,
           "development_scans": len(hits), "development_matched_stones": int(sizes.sum()),
           "development_coverage": float(hits.sum() / sizes.sum()), "tilt": float(tilt),
           "test_scans": args.test_scans, "expected_test_matched_stones": float(args.test_scans * (weights * sizes).sum()),
           "power_by_delta": power, "chosen_delta": chosen,
           "note": "chosen_delta None means even 0.10 gives < 80% power; P1 is then reported as underpowered."}  # fmt: skip
    debug = "DEBUG" in args.per_stone_csv.name
    out["status"] = "DEBUG_NOT_A_RESULT" if debug else "PRE_FREEZE_DESIGN"
    run = create_run_folder(Path("results/runs"), "p1_power_simulation" + ("_DEBUG" if debug else ""))
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "definitions"}, indent=1))
    print("Saved", run)


if __name__ == "__main__":
    main()

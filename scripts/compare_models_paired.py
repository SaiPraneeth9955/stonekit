"""Paired comparison of two methods on the same validation scans, from saved stone-analysis rows.

Reads the per-piece CSVs that scripts/analyse_stone_validation.py writes to
data/interim/stone_analysis/<stamp>/ (annotated.csv, pairs.csv, scans.csv). Reruns nothing.

Written on 7 November 2026, before the 250-epoch model's or the kidney-distance filter's validation
results existed. Endpoints and rules are those in docs/decisions.md (7 November 2026):
- pooled annotated-piece detection rate (any voxel overlap);
- false-positive pieces per scan;
- mean per-scan Dice on scans with annotated stones;
- detection in the <3 mm and 3-<5 mm bands;
- one-to-one volume bias: mean of 200*(predicted-annotated)/(predicted+annotated) over pairs, and
  its limits of agreement.
Paired intervals: percentile bootstrap over scans, both methods resampled with the same scan
indices, seed 20261105, 10000 resamples. Differences are B minus A.

Rules (point estimates, as pre-specified):
- "schedule": keep B (the longer schedule) only if detection(B) >= detection(A), false positives per
  scan(B) - (A) <= 0.5, and |volume bias(B)| - |volume bias(A)| <= 5 percentage points.
- "filter": adopt B (A plus the kidney-distance filter) only if B removes false positives
  (FP per scan lower), detection is identical (same count of detected annotated pieces), and
  one-to-one agreement is unchanged, operationalised as the same number of one-to-one pairs and a
  volume-bias change smaller than 0.1 percentage points. This operational definition was written
  before the filtered results were seen.
These scans are development scans: a choice made here is model selection, not confirmatory evidence.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

SEED = 20261105
RESAMPLES = 10000
SMALL_BANDS = {"<3 mm": (0, 3), "3-<5 mm": (3, 5)}


def read(folder: Path, name: str, method: str) -> list[dict]:
    with (folder / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["method"] == method]


def per_scan(folder: Path, method: str) -> dict[str, dict]:
    """Sufficient statistics per scan, so any resample can be scored by summing."""
    scans = {}
    for r in read(folder, "scans", method):
        dice = r["dice"]
        scans[r["case"]] = {
            "has_stone": r["has_stone"] == "True",
            "dice": float(dice) if dice not in ("", "None") else np.nan,
            "fp": int(r["false_positive_pieces"]),
            "annotated": 0, "detected": 0,
            **{f"annotated {b}": 0 for b in SMALL_BANDS}, **{f"detected {b}": 0 for b in SMALL_BANDS},
            "pair_volume_pct": [],
        }  # fmt: skip
    for r in read(folder, "annotated", method):
        s = scans[r["case"]]
        hit = r["kind"] != "missed"
        s["annotated"] += 1
        s["detected"] += hit
        d3 = float(r["d3_mm"])
        for b, (low, high) in SMALL_BANDS.items():
            if low <= d3 < high:
                s[f"annotated {b}"] += 1
                s[f"detected {b}"] += hit
    for r in read(folder, "pairs", method):
        p, a = float(r["predicted_mm3"]), float(r["annotated_mm3"])
        scans[r["case"]]["pair_volume_pct"].append(200 * (p - a) / (p + a))
    return scans


def score(scans: dict[str, dict], cases: list[str]) -> dict:
    rows = [scans[c] for c in cases]
    out = {
        "detection": sum(r["detected"] for r in rows) / max(1, sum(r["annotated"] for r in rows)),
        "fp_per_scan": sum(r["fp"] for r in rows) / len(rows),
        "dice_scans_with_stone": float(np.nanmean([r["dice"] for r in rows if r["has_stone"]] or [np.nan])),
    }
    for b in SMALL_BANDS:
        denominator = sum(r[f"annotated {b}"] for r in rows)
        out[f"detection {b}"] = sum(r[f"detected {b}"] for r in rows) / denominator if denominator else np.nan
    volumes = np.array([v for r in rows for v in r["pair_volume_pct"]])
    out["volume_bias_pct"] = float(volumes.mean()) if len(volumes) else np.nan
    out["abs_volume_bias_pct"] = abs(out["volume_bias_pct"])
    return out


def counts(scans: dict[str, dict], cases: list[str]) -> dict:
    rows = [scans[c] for c in cases]
    volumes = np.array([v for r in rows for v in r["pair_volume_pct"]])
    sd = float(volumes.std(ddof=1)) if len(volumes) > 1 else np.nan
    return {
        "detected": sum(r["detected"] for r in rows), "annotated": sum(r["annotated"] for r in rows),
        "false_positive_pieces": sum(r["fp"] for r in rows), "scans": len(rows),
        "one_to_one_pairs": len(volumes),
        "volume_limits_of_agreement_pct": [round(float(volumes.mean() - 1.96 * sd), 2), round(float(volumes.mean() + 1.96 * sd), 2)] if len(volumes) > 1 else None,
        **{f"detected {b}": f"{sum(r[f'detected {b}'] for r in rows)}/{sum(r[f'annotated {b}'] for r in rows)}" for b in SMALL_BANDS},
    }  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="Detail folder of method A")
    parser.add_argument("--a-method", default="model")
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--b", type=Path, required=True, help="Detail folder of method B")
    parser.add_argument("--b-method", default="model")
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--rule", choices=("schedule", "filter"), required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    a, b = per_scan(args.a, args.a_method), per_scan(args.b, args.b_method)
    if set(a) != set(b):
        raise ValueError(f"Methods were not run on the same scans: only A {sorted(set(a) - set(b))}, only B {sorted(set(b) - set(a))}")
    cases = sorted(a)
    point_a, point_b = score(a, cases), score(b, cases)
    rng = np.random.default_rng(SEED)
    draws = {k: [] for k in point_a}
    for _ in range(RESAMPLES):
        sample = [cases[i] for i in rng.integers(0, len(cases), len(cases))]
        sa, sb = score(a, sample), score(b, sample)
        for k, values in draws.items():
            values.append(sb[k] - sa[k])
    endpoints = {}
    for k in point_a:
        d = np.array(draws[k], dtype=float)
        d = d[~np.isnan(d)]
        endpoints[k] = {
            "A": round(point_a[k], 4), "B": round(point_b[k], 4), "B_minus_A": round(point_b[k] - point_a[k], 4),
            "paired_interval_95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)] if len(d) else None,
        }  # fmt: skip
    ca, cb = counts(a, cases), counts(b, cases)
    if args.rule == "schedule":
        checks = {
            "detection_not_lower": point_b["detection"] >= point_a["detection"],
            "fp_per_scan_not_higher_by_more_than_0.5": point_b["fp_per_scan"] - point_a["fp_per_scan"] <= 0.5,
            "abs_volume_bias_not_worse_by_more_than_5pp": point_b["abs_volume_bias_pct"] - point_a["abs_volume_bias_pct"] <= 5,
        }
    else:
        checks = {
            "removes_false_positives": point_b["fp_per_scan"] < point_a["fp_per_scan"],
            "detection_identical": cb["detected"] == ca["detected"],
            "one_to_one_pair_count_identical": cb["one_to_one_pairs"] == ca["one_to_one_pairs"],
            "volume_bias_change_below_0.1pp": abs(point_b["volume_bias_pct"] - point_a["volume_bias_pct"]) < 0.1,
        }
    summary = {
        "status": "EXPLORATORY_DEVELOPMENT_EVIDENCE_MODEL_SELECTION_NOT_CONFIRMATORY",
        "definitions": __doc__,
        "A": {"label": args.a_label, "source": str(args.a), "method": args.a_method, **ca},
        "B": {"label": args.b_label, "source": str(args.b), "method": args.b_method, **cb},
        "scans": len(cases),
        "endpoints": endpoints,
        "rule": args.rule,
        "rule_checks": checks,
        "rule_outcome": "B_selected" if all(checks.values()) else "A_kept",
    }
    run = create_run_folder(Path("results/runs"), f"paired_comparison_{args.label}")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("A", "B", "endpoints", "rule_checks", "rule_outcome")}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

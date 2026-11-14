"""Freiburg a0 run re-described with error metrics that do not depend on between-stone variance.

POST-HOC DESCRIPTION of the frozen a0 run (results/runs/a0_replicate_freiburg_20261102T133018Z/,
a0_stones.csv). It recomputes nothing from images and changes no G3 decision. The a0 report used
ICC(A,1) as its deciding metric; with reference volumes spanning about 0.01 to 20 mL, between-stone
variance dominates the ICC denominator, so ICC can be above 0.99 while individual stones are off by
tens of percent. This file reports the errors themselves.

Measured values are StoneKit's own volume and diameter code on the 272 HU masks of Freiburg's
masked phantom images (the a0 comparator), against Freiburg's physical references. Two spacing
assumptions (A: voxsiz, B: voxvol_mL) are both carried, as the stage 1 finding requires; the 80
stones whose spacing fields agree are reported separately because they do not depend on that choice.
Stones are independent objects (one per row), so the bootstrap resamples stones (seed 20261105).
"""

import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

SOURCE = Path("results/runs/a0_replicate_freiburg_20261102T133018Z/a0_stones.csv")
SEED = 20261105
RESAMPLES = 10000
SIZE_BINS = [(0, 10, "<10 mm"), (10, 20, "10-<20 mm"), (20, np.inf, ">=20 mm")]


def boot(values: np.ndarray, statistic) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(RESAMPLES)]
    return [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)]


def describe(rows: list[dict], assumption: str) -> dict:
    if len(rows) < 2:
        return {"n": len(rows)}
    ref_v = np.array([r["reference_volume_ml"] for r in rows])
    v = np.array([r[f"volume_{assumption}"] for r in rows])
    rel = 200 * (v - ref_v) / (v + ref_v)
    ref_d = np.array([r["reference_diameter_mm"] for r in rows])
    out = {"n": len(rows),
           "volume_relative_bias_pct": round(float(rel.mean()), 2), "volume_relative_bias_interval_95": boot(rel, np.mean),
           "volume_abs_relative_error_pct_median": round(float(np.median(np.abs(rel))), 2),
           "volume_abs_relative_error_pct_p90": round(float(np.percentile(np.abs(rel), 90)), 2),
           "volume_relative_limits_of_agreement_pct": [round(float(rel.mean() - 1.96 * rel.std(ddof=1)), 1), round(float(rel.mean() + 1.96 * rel.std(ddof=1)), 1)],
           "volume_mae_ml": round(float(np.abs(v - ref_v).mean()), 4)}  # fmt: skip
    for name in ("d_centre", "d_paper"):
        d = np.array([r[f"{name}_{assumption}"] for r in rows]) - ref_d
        out[f"{name}_bias_mm"] = round(float(d.mean()), 3)
        out[f"{name}_bias_interval_95"] = boot(d, np.mean)
        out[f"{name}_mae_mm"] = round(float(np.abs(d).mean()), 3)
        out[f"{name}_rmse_mm"] = round(float(np.sqrt((d**2).mean())), 3)
    d = np.array([r[f"d_centre_{assumption}"] for r in rows])
    mean, diff = (d + ref_d) / 2, d - ref_d
    pairs = np.column_stack([mean, diff])
    out["d_centre_proportional_bias_slope"] = round(float(np.polyfit(mean, diff, 1)[0]), 4)
    out["d_centre_proportional_bias_interval_95"] = boot(pairs, lambda p: np.polyfit(p[:, 0], p[:, 1], 1)[0]) if len(rows) > 5 else None
    return out


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        row = {k: (float(v) if k not in ("stone_type", "spacing_fields_disagree", "image_dtype", "mask_empty", "volume_exceeds_diameter_sphere", "volume_below_4mm_sphere") and v != "" else v) for k, v in r.items()}
        row["agree"] = r["spacing_fields_disagree"] == "False"
        rows.append(row)
    out = {"status": "POST_HOC_DESCRIPTION_OF_FROZEN_A0_RUN", "definitions": __doc__, "source": str(SOURCE), "stones": len(rows), "results": {}}
    for assumption in ("A", "B"):
        res = {"all": describe(rows, assumption), "spacing_fields_agree": describe([r for r in rows if r["agree"]], assumption)}
        res["by_reference_diameter"] = {name: describe([r for r in rows if lo <= r["reference_diameter_mm"] < hi], assumption) for lo, hi, name in SIZE_BINS}
        types = sorted({r["stone_type"] for r in rows if r["stone_type"]})
        res["by_stone_type_descriptive"] = {t: describe([r for r in rows if r["stone_type"] == t], assumption) for t in types if sum(r["stone_type"] == t for r in rows) >= 5}
        out["results"][assumption] = res
    ref = np.array([r["reference_volume_ml"] for r in rows])
    out["why_icc_misleads"] = {
        "reference_volume_ml_range": [round(float(ref.min()), 4), round(float(ref.max()), 2)],
        "between_stone_sd_ml": round(float(ref.std(ddof=1)), 3),
        "note": "a0 volume ICC(A,1) was 0.995 (A) and 0.996 (B); the relative errors above show the per-stone accuracy that ICC hides.",
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "freiburg_agreement_metrics")
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["results"]["A"]["all"], indent=1))
    print(json.dumps(out["results"]["A"]["spacing_fields_agree"], indent=1))
    print(json.dumps({k: {kk: v[kk] for kk in ("n", "volume_relative_bias_pct", "volume_abs_relative_error_pct_median", "d_centre_bias_mm")} for k, v in out["results"]["A"]["by_reference_diameter"].items()}, indent=1))
    print(json.dumps({k: {kk: v.get(kk) for kk in ("n", "volume_relative_bias_pct", "volume_abs_relative_error_pct_median")} for k, v in out["results"]["A"]["by_stone_type_descriptive"].items()}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

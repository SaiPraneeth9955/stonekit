"""Post-hoc diagnostic: volume and diameter bias of two methods on the annotated pieces both match one-to-one.

POST-HOC, written on 7 November 2026 AFTER the pre-specified schedule rule was applied (outcome: the
100-epoch model kept, because the 250-epoch model's one-to-one volume bias was 7.5 points worse). It
does NOT change that outcome. Its only question: is the difference in bias a difference in how the
same stones are measured, or a difference in which stones enter the one-to-one set (the 250-epoch
model matched more small stones, and small stones carry the largest relative bias)?

Pieces are identified by (case, annotated volume, annotated 3D diameter), which are computed from
the same Li annotation for both methods. Interval: paired percentile bootstrap over scans, seed
20261105, 10000 resamples.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

SEED = 20261105
RESAMPLES = 10000


def pairs(folder: Path, method: str) -> dict[tuple, dict]:
    with (folder / "pairs.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method]
    out = {}
    for r in rows:
        key = (r["case"], round(float(r["annotated_mm3"]), 4), round(float(r["annotated_d3_mm"]), 4))
        if key in out:
            raise ValueError(f"Piece key not unique: {key}")
        p, a = float(r["predicted_mm3"]), float(r["annotated_mm3"])
        out[key] = {
            "volume_pct": 200 * (p - a) / (p + a),
            "d3_mm": float(r["predicted_d3_mm"]) - float(r["annotated_d3_mm"]),
            "annotated_d3_mm": float(r["annotated_d3_mm"]),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--a-label", required=True)
    parser.add_argument("--b-label", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    a, b = pairs(args.a, "model"), pairs(args.b, "model")
    common = sorted(set(a) & set(b))
    only_b = sorted(set(b) - set(a))
    only_a = sorted(set(a) - set(b))
    cases = sorted({k[0] for k in common})
    rng = np.random.default_rng(SEED)
    by_case = {c: [k for k in common if k[0] == c] for c in cases}

    def stats(keys, field):
        va = np.array([a[k][field] for k in keys])
        vb = np.array([b[k][field] for k in keys])
        return va.mean(), vb.mean()

    result = {}
    for field in ("volume_pct", "d3_mm"):
        mean_a, mean_b = stats(common, field)
        draws = []
        for _ in range(RESAMPLES):
            keys = [k for i in rng.integers(0, len(cases), len(cases)) for k in by_case[cases[i]]]
            sa, sb = stats(keys, field)
            draws.append(sb - sa)
        result[field] = {
            "A_mean": round(float(mean_a), 3), "B_mean": round(float(mean_b), 3),
            "B_minus_A": round(float(mean_b - mean_a), 3),
            "paired_interval_95": [round(float(np.percentile(draws, 2.5)), 3), round(float(np.percentile(draws, 97.5)), 3)],
        }  # fmt: skip

    def describe(keys, source):
        v = np.array([source[k]["volume_pct"] for k in keys])
        d = np.array([source[k]["annotated_d3_mm"] for k in keys])
        return {"n": len(keys), "mean_volume_pct": round(float(v.mean()), 2) if len(v) else None,
                "median_annotated_d3_mm": round(float(np.median(d)), 2) if len(d) else None}  # fmt: skip

    summary = {
        "status": "POST_HOC_DIAGNOSTIC_DOES_NOT_CHANGE_THE_PRE_SPECIFIED_OUTCOME",
        "definitions": __doc__,
        "A": args.a_label, "B": args.b_label,
        "common_one_to_one_pieces": len(common), "scans_with_common_pieces": len(cases),
        "common": result,
        "one_to_one_only_in_A": describe(only_a, a),
        "one_to_one_only_in_B": describe(only_b, b),
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), f"common_pair_bias_{args.label}")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "definitions"}, indent=1))
    print("Saved to", run)


if __name__ == "__main__":
    main()

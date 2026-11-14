"""Aggregate the low-memory kidney-step equivalence runs (scripts/kidney_lowmem_equivalence.py).

Each scan ran in its own process under /usr/bin/time -v, so the peak RSS here is per scan (the
pipeline-style Python process plus TotalSegmentator). Development scans only.
"""

import argparse
import json
import re
from pathlib import Path

from stonekit.config import create_run_folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for out in sorted(args.folder.glob("*.out")):
        line = next((x for x in out.read_text().splitlines() if x.startswith("RESULT ")), None)
        err = out.with_suffix(".err").read_text()
        peak = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", err)
        row = json.loads(line[7:]) if line else {"case": out.stem, "failed": err[-300:]}
        row["peak_rss_gb"] = round(int(peak.group(1)) / 1024**2, 2) if peak else None
        row["wall_clock"] = wall.group(1) if wall else None
        rows.append(row)
    compared = [r for r in rows if "right_dice" in r]
    summary = {
        "definitions": __doc__, "source": str(args.folder), "scans": rows,
        "equivalence": {
            "scans_compared": len(compared),
            "kidney_dice_min": min(min(r["right_dice"], r["left_dice"]) for r in compared) if compared else None,
            "kidney_dice_median": sorted(d for r in compared for d in (r["right_dice"], r["left_dice"]))[len(compared)] if compared else None,
            "max_volume_difference_pct": max(abs(r[f"{s}_ml_lowmem"] - r[f"{s}_ml_normal"]) / r[f"{s}_ml_normal"] * 100 for r in compared for s in ("right", "left")) if compared else None,
            "roi_box_max_shift_mm": max(r["box_max_shift_mm"] for r in compared) if compared else None,
            "stone_voxels_outside_lowmem_box": sum(r["stone_voxels_outside_box_lowmem"] for r in rows if "stone_voxels_outside_box_lowmem" in r),
            "peak_rss_gb_range": [min(r["peak_rss_gb"] for r in rows if r["peak_rss_gb"]), max(r["peak_rss_gb"] for r in rows if r["peak_rss_gb"])],
        },
    }  # fmt: skip
    run = create_run_folder(Path("results/runs"), "kidney_lowmem_equivalence")
    (run / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["equivalence"], indent=1))
    for r in rows:
        print({k: r.get(k) for k in ("case", "slices", "slice_mm", "seconds", "peak_rss_gb", "right_dice", "left_dice", "box_max_shift_mm", "stone_voxels", "stone_voxels_outside_box_lowmem")})
    print("Saved to", run)


if __name__ == "__main__":
    main()

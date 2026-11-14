"""Computational characterisation from saved run records (reads no image voxels).

Sources:
- the kidney-step log of 5 November 2026 (data/interim/li2022_kidney_ts/run_log.csv): seconds and
  peak child RSS per development scan, with scan length from the NIfTI header;
- report.json of the 100- and 250-epoch validation runs (`--runs`): per-stage seconds, peak RSS,
  and the size of each run folder on disk.
Hardware and software are recorded in each report. GPU memory was not recorded in these runs.
Aggregates only.
"""

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from stonekit.config import create_run_folder
from stonekit.data.li2022 import LiDevelopmentData


def describe(values) -> dict:
    v = np.array(values, dtype=float)
    return {"n": len(v), "median": round(float(np.median(v)), 2), "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2)} if len(v) else {"n": 0}


def folder_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    data = LiDevelopmentData(root)
    out = {"definitions": __doc__}

    with (root / "data/interim/li2022_kidney_ts/run_log.csv").open(newline="", encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    ok = []
    for r in log:
        if r["outcome"] != "ok":
            continue
        ct, _ = data.paths(r["case"])
        shape = nib.load(ct).shape
        ok.append((shape[2], float(np.prod(shape)) / 1e6, float(r["seconds"]), int(r["children_peak_rss_kb"]) / 1024**2))
    slices, mvox, seconds, gb = (np.array(x) for x in zip(*ok, strict=True))
    out["kidney_step_totalsegmentator"] = {
        "successful_scans": len(ok), "failed_attempts": sum(r["outcome"] != "ok" for r in log),
        "seconds": describe(seconds), "slices": describe(slices),
        "seconds_vs_megavoxels_pearson_r": round(float(np.corrcoef(mvox, seconds)[0, 1]), 3),
        "largest_successful_megavoxels": round(float(mvox.max()), 1),
        "batch_peak_child_rss_gb": round(float(gb.max()), 2),
        "note": "The logged child RSS is a running maximum over one batch process (getrusage RUSAGE_CHILDREN), "
                "not a per-scan value, so only its batch maximum is reported. Per-scan peaks need separate processes.",
    }  # fmt: skip
    for folder in args.runs:
        stages, totals, rss_self, rss_child, disk = {}, [], [], [], []
        for case_dir in sorted(p for p in folder.iterdir() if (p / "report.json").exists()):
            report = json.loads((case_dir / "report.json").read_text())
            for s in report["stages"]:
                stages.setdefault(s["stage"], []).append(s["seconds"])
            totals.append(report["total_seconds"])
            rss_self.append(report["peak_memory_kb"]["this_process"] / 1024**2)
            rss_child.append(report["peak_memory_kb"]["largest_child_process"] / 1024**2)
            disk.append(folder_mb(case_dir))
        out[folder.name] = {"scans": len(totals), "total_seconds": describe(totals), "stage_seconds": {k: describe(v) for k, v in stages.items()},
                            "batch_peak_rss_gb_pipeline_process": round(max(rss_self), 2), "batch_peak_rss_gb_largest_child": round(max(rss_child), 2),
                            "disk_mb_per_case": describe(disk),
                            "note": "These runs reused the saved kidney mask, so the kidney stage is a file copy; "
                                    "peak RSS of the pipeline process is cumulative over the batch (one Python process)."}  # fmt: skip
    run = create_run_folder(root / "results/runs", "computational_profile")
    (run / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1)[:5000])
    print("Saved to", run)


if __name__ == "__main__":
    main()

"""Run TotalSegmentator on development scans, resuming where it stopped.

Development scans only, through the guarded loader. Outputs are git-ignored. Each scan's time and
any failure go in a log; a failed scan is recorded and the run goes on.
"""

import argparse
import csv
import resource
import time
from pathlib import Path

from stonekit.data.li2022 import LiDevelopmentData
from stonekit.seg.kidney_roi import run_totalsegmentator
from stonekit.seg.subprocess_guard import ExternalToolError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="gpu")
    parser.add_argument("--low-memory", action="store_true", help="One resampling and saving thread.")
    args = parser.parse_args()
    data = LiDevelopmentData(args.root)
    out = args.root / "data" / "interim" / "li2022_kidney_ts"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run_log.csv"
    new_log = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if new_log:
            writer.writerow(["case", "seconds", "outcome", "children_peak_rss_kb"])
        for case_id in data.development_ids():
            target = out / f"{case_id}_ts.nii.gz"
            if target.exists():
                continue
            start = time.perf_counter()
            try:
                run_totalsegmentator(data.paths(case_id)[0], target, args.device, args.low_memory)
                outcome = "ok_low_memory" if args.low_memory else "ok"
            except ExternalToolError as error:
                outcome = "failed: " + str(error)[:300]
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            writer.writerow([case_id, round(time.perf_counter() - start, 1), outcome, peak])
            f.flush()
            print(case_id, outcome[:60], flush=True)


if __name__ == "__main__":
    main()

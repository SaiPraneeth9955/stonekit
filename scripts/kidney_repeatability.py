"""Repeat the kidney step on the validation scans and rerun the system pipeline on each new mask.

Question: does TotalSegmentator's run-to-run variation change any final StoneKit measurement?
For each fold-0 validation scan, TotalSegmentator is run again (same command, same settings as the
pipeline) and `run_case` is run with `configs/system_pipeline_v1.yaml` on that new mask. The analysis
(scripts/analyse_kidney_repeatability.py) compares it with the saved mask and the original run.
Development scans only; reserved scans are refused. Writes new folders only.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from stonekit.config import load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.pipeline.run_case import run_case
from stonekit.seg import kidney_roi
from stonekit.seg.subprocess_guard import ExternalToolError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cases", nargs="*", help="Only these validation scans (default: all 32)")
    parser.add_argument("--stamp", help="Add to an earlier study's folders instead of starting new ones")
    args = parser.parse_args()
    root = args.root.resolve()
    split = json.loads((root / "data/nnunet/li_stone_split.json").read_text(encoding="utf-8"))
    cases = args.cases or split["validation"]
    if set(cases) - set(split["validation"]):
        raise ValueError("Validation scans only")
    data = LiDevelopmentData(root)
    require_development(cases, data.manifest)
    config_path = root / "configs/system_pipeline_v1.yaml"
    config = load_yaml(config_path)
    stamp = args.stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    masks = root / "data/interim/diagnostics" / f"kidney_repeat_{stamp}"
    runs = root / "data/interim/stonekit_runs" / f"kidney_repeat_{stamp}"
    log_file = masks / "log.json"
    log = json.loads(log_file.read_text(encoding="utf-8")) if args.stamp and log_file.exists() else []
    for case in cases:
        ct_path, annotation = data.paths(case)
        mask = masks / f"{case}_ts.nii.gz"
        entry = {"case": case}
        try:
            result = kidney_roi.run_totalsegmentator(ct_path, mask, "gpu", timeout_seconds=600,
                                                     log_path=masks / f"{case}.log")  # fmt: skip
            entry["kidney"] = result.to_dict()
            folder = run_case(ct_path, runs, config, root, case, annotation, mask, "gpu", config_path)
            entry["run"] = str(folder.relative_to(root))
        except ExternalToolError as error:
            entry["failure"] = error.to_dict()
        log.append(entry)
        print(case, entry.get("run", entry.get("failure")), flush=True)
        log_file.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("masks:", masks, "runs:", runs)


if __name__ == "__main__":
    main()

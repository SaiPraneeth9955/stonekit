"""End-to-end research demonstration on ONE non-contrast CT: pipeline -> 8-flip TTA -> calibrated ranges.

    python scripts/demo_case.py --ct data/external/mswal/imagesTr/MSWAL_0075_0000.nii.gz --name MSWAL_0075

Steps (the frozen system, unchanged):
1. `run_case`: input checks, TotalSegmentator kidneys, 40 mm ROI, nnU-Net stones, 30 mm kidney-distance filter,
   3D measurement, report.json/report.html and meshes, written under data/interim/stonekit_runs/ so that
   `stonekit app` lists the result.
2. 8-flip test-time augmentation probabilities (scripts/oof_tta.py), saved inside the run folder.
3. Calibrated ranges from the frozen locked-study calibration (stonekit.t1.case_ranges): per-stone volume and
   3D diameter ranges and the patient burden interval at nominal 90%, saved as calibrated_ranges.json.
Nothing is fitted. Locked Li scans and the owner's six demonstration cases are refused.
Research use only. Not a diagnostic medical device.
"""

import argparse
import importlib.util
from pathlib import Path

from stonekit.config import load_yaml
from stonekit.data import li_split
from stonekit.pipeline.run_case import run_case
from stonekit.t1.access import GatedLiData
from stonekit.t1.case_ranges import write_case_ranges

ROOT = Path(".")
CONFIG = Path("configs/system_pipeline_v1.yaml")
OUT = Path("data/interim/stonekit_runs")
DEMO_CASES = {"wd0063", "wd0231", "wd0128", "wd0042", "wd0022", "wd0149"}  # owner: never rerun


def _tta_module():
    spec = importlib.util.spec_from_file_location("oof_tta", Path("scripts") / "oof_tta.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ct", type=Path, help="a non-contrast CT in NIfTI format")
    src.add_argument("--case", help="a Li DEVELOPMENT scan ID (locked scans are refused)")
    ap.add_argument("--name", help="label for the run (default: the case ID or the file name)")
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    args = ap.parse_args()
    if args.case:
        if args.case in DEMO_CASES:
            raise SystemExit(f"{args.case} is one of the owner's demonstration cases, which must not be rerun")
        data = GatedLiData(ROOT)  # no gates: locked scans raise here
        li_split.require_development([args.case], data.manifest)
        ct, _mask = data.paths(args.case)
        name = args.name or args.case
    else:
        ct = args.ct
        if not ct.exists():
            raise SystemExit(f"CT not found: {ct}")
        name = args.name or ct.name.split(".")[0]
    config = load_yaml(ROOT / CONFIG)
    print(f"[1/3] Pipeline on {name} (kidneys, ROI, stone segmentation, filter, measurement) ...", flush=True)
    folder = run_case(ct, OUT, config, ROOT, name, None, None, args.device, ROOT / CONFIG)
    print(f"      run folder: {folder}", flush=True)
    print("[2/3] 8-flip test-time augmentation ...", flush=True)
    tta_out = folder / "tta"
    tta_out.mkdir(exist_ok=False)
    _tta_module().process_case(folder, dict(config["model"]), int(config["model"]["fold"]), tta_out, ROOT)
    print("[3/3] Calibrated ranges from the frozen locked-study calibration ...", flush=True)
    try:
        out = write_case_ranges(folder, tta_out / "probabilities.npz")
    except ValueError as error:
        raise SystemExit(f"No calibrated ranges: {error}") from None
    print("\nRESEARCH DEMONSTRATION - not for clinical use. Nominal 90% ranges.")
    for s in out["stones"]:
        print(f"  Stone {s['stone_id']}: volume {s['volume_mm3']:.0f} mm3 (range {s['volume_range_mm3'][0]:.0f}-{s['volume_range_mm3'][1]:.0f}); "
              f"3D diameter {s['max_diameter_3d_mm']:.1f} mm (range {s['diameter_range_mm'][0]:.1f}-{s['diameter_range_mm'][1]:.1f}); "
              f"false-detection probability {s['false_detection_probability']:.2f}")  # fmt: skip
    b = out["burden"]
    print(f"  Total stone burden: {b['predicted_mm3']:.0f} mm3 (detection-aware interval {b['interval_mm3'][0]:.0f}-{b['interval_mm3'][1]:.0f} mm3)")
    print(f"\nSaved {folder / 'calibrated_ranges.json'}. View it with: stonekit app")


if __name__ == "__main__":
    main()

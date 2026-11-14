"""Guarded T1 wrapper (docs/T1_RUNNER_DESIGN.md): gates -> calibration inference + TTA + tables -> analysis stage
(calibrate, hash) -> ONLY THEN test inference + TTA + tables -> evaluation.

Modes:
- `T1`: every gate in stonekit.t1.gates must pass (frozen protocol SHA logged, clean git at the frozen commit,
  owner unlock file, split unchanged, no completed T1 run, frozen config and checkpoint hashes). Cohorts come
  from the frozen-protocol cohort files. Only this mode can write COMPLETED. NOT runnable until the owner freezes,
  commits and unlocks.
- `SMOKE_DEVELOPMENT`: an engineering smoke test of the full chain on a handful of DEVELOPMENT scans given on the
  command line (reserved IDs are refused by the gated loader). Never a result; the analysis runs in mode
  DRESS_REHEARSAL.
The fit set is the development out-of-fold table (all eligible development scans).
"""

import argparse
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd

from stonekit.config import create_run_folder, load_yaml
from stonekit.data import li_split
from stonekit.pipeline.run_case import run_case
from stonekit.t1 import gates
from stonekit.t1.access import GatedLiData
from stonekit.t1.analysis import Exclusion, run_analysis
from stonekit.uq import confirmatory as cfm

CONFIG = Path("configs/system_pipeline_v1.yaml")
CONFIG_SHA = "64bf721da9d46bc885b331b8a0ecf7a5608c17064c5778d5402cc3a12702b5ee"
CHECKPOINT = Path("data/nnunet/nnUNet_results/Dataset501_StoneNCCT/nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres/fold_0/checkpoint_final.pth")
CHECKPOINT_SHA = "2100ec09d2e2eadcfd11d928e9db7dbe487a1f1fd10bc17924cfff108a6d410c"
SCREENING_OUTCOME = Path("data/processed/li2022_review_outcome_v1.csv")
SCREENING_OUTCOME_SHA = "0d7bf638b5c38eccded8353d0b89b0f5728e2822140adbb3427e86f81b4feaeb"
SPLIT = Path("data/processed/li2022_working_split_v1.csv")
DECISIONS = Path("docs/decisions.md")
ITEM5_UNAVAILABLE = "no qualified radiology-trained reader; D-REV not performed (owner decision OD-19)"
# The owner's six demonstration cases must never be rerun (owner instruction); smoke mode refuses them.
DEMO_CASES = {"wd0063", "wd0231", "wd0128", "wd0042", "wd0022", "wd0149"}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, Path("scripts") / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_tables(cases: list[str], data: GatedLiData, work: Path, fold: int) -> cfm.Tables:
    """Frozen pipeline -> 8-flip TTA -> stone tables, for the given cases (the segmentation mask is unchanged)."""
    tta_mod, table_mod = _script("oof_tta"), _script("build_oof_stone_table")
    root = Path.cwd()
    config = load_yaml(root / CONFIG)
    model = dict(config["model"])
    ann, pred, scans = [], [], []
    for case in cases:
        ct, _mask = data.paths(case)  # the guard runs here
        folder = run_case(ct, work / "cases", config, root, case, None, None, "gpu", root / CONFIG)
        if json.loads((folder / "report.json").read_text()).get("result_status") != "complete":
            scans.append({"case": case, "failed": True})  # the protocol keeps failed scans; their pieces count as missed
            continue
        tta_out = work / "tta" / case
        tta_out.mkdir(parents=True, exist_ok=False)
        tta_mod.process_case(folder, model, fold, tta_out, root)
        a, p, s = table_mod.process(case, folder, fold, work / "tta", data)
        ann += a
        pred += p
        scans.append(s)
    return cfm.Tables(pd.DataFrame(ann), pd.DataFrame(pred), pd.DataFrame(scans))


def t1_sensitivity(manifest: dict[str, str]) -> dict[str, Exclusion]:
    """Protocol v3 §5.1 cohorts from FROZEN sources (OD-19). Called only after every T1 gate has passed.
    S1: scans the owner's screening rule would exclude (action = exclude), removed from fit, calibration and test.
    S2: every screened-flagged scan (exclude, or kept with a sensitivity mark), removed from all three sets.
    S3: reserved scans in a suspected repeat pair that crosses split groups (the 7 Nov audit computation, from the
        pre-split bone projections; no image is opened), removed from calibration and test.
    Item 4 (device subgroup) has no reserved scan (the only device-coded scan, wd0149, is development)."""
    with open(SCREENING_OUTCOME, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    s1 = frozenset(r["case"] for r in rows if r["action"] == "exclude")
    s2 = s1 | frozenset(r["case"] for r in rows if r["reason"] == "sensitivity")
    pairs = _script("make_cv_folds").suspected_pairs()
    s3 = frozenset(c for a, b in pairs if manifest.get(a) != manifest.get(b) for c in (a, b) if manifest.get(c) in li_split.RESERVED)
    return {"S1_screening_exclusions": Exclusion(s1, reserved_only=False), "S2_all_screened_flags": Exclusion(s2, reserved_only=False),
            "S3_cross_split_repeat_pairs": Exclusion(s3, reserved_only=True)}  # fmt: skip


def run_t1(args, root: Path, fold: int) -> None:
    """The single confirmatory run. Every gate must pass before any reserved scan can be reached."""
    logged = gates.frozen_protocol_sha(DECISIONS) or ""
    results = [gates.gate_protocol(args.protocol, DECISIONS), gates.gate_git_clean(root, args.commit),
               gates.gate_unlock(args.unlock, logged, args.commit), gates.gate_split(SPLIT), gates.gate_single_run(Path("results/runs")),
               gates.gate_file_sha("config_frozen", CONFIG, CONFIG_SHA), gates.gate_file_sha("checkpoint_frozen", CHECKPOINT, CHECKPOINT_SHA),
               gates.gate_file_sha("screening_outcome_frozen", SCREENING_OUTCOME, SCREENING_OUTCOME_SHA)]  # fmt: skip
    for r in results:
        print(r)
    if not gates.all_passed(results):
        raise SystemExit("T1 gates not all passed; nothing loaded, nothing written")
    data = GatedLiData(root, results)
    cal_cases = sorted(c for c, g in data.manifest.items() if g == "reserved_calibration")
    test_cases = sorted(c for c, g in data.manifest.items() if g == "reserved_test")
    if (len(cal_cases), len(test_cases)) != (35, 53):
        raise SystemExit(f"cohort sizes {len(cal_cases)}/{len(test_cases)} differ from the frozen 35/53")
    fit_dir = args.fit_table
    fit = cfm.Tables(pd.read_csv(fit_dir / "annotated.csv"), pd.read_csv(fit_dir / "predicted.csv"), pd.read_csv(fit_dir / "scans.csv"))
    if set(fit.scans.case) != {c for c, g in data.manifest.items() if g == li_split.DEVELOPMENT}:
        raise SystemExit("fit table is not exactly the 165 development scans")
    sensitivity = t1_sensitivity(data.manifest)
    run = create_run_folder(Path("results/runs"), "T1")
    (run / "inputs.json").write_text(json.dumps({"mode": "T1", "commit": args.commit, "protocol_sha256": logged, "fit_table": str(fit_dir),
                                                 "gates": [r.__dict__ for r in results], "calibration": cal_cases, "test": test_cases,
                                                 "sensitivity": {k: {"scans": sorted(v.scans), "reserved_only": v.reserved_only} for k, v in sensitivity.items()},
                                                 "item5": ITEM5_UNAVAILABLE}, indent=2), encoding="utf-8")  # fmt: skip
    cal = build_tables(cal_cases, data, run / "cal_work", fold)
    ev = run_analysis(fit, cal, lambda: build_tables(test_cases, data, run / "test_work", fold), run / "analysis", "T1",
                      sensitivity=sensitivity, item5_unavailable=ITEM5_UNAVAILABLE)  # fmt: skip
    (run / "COMPLETED").write_text(ev["calibration_quantities_sha256"] + "\n", encoding="utf-8")
    print("T1 COMPLETED | formal P2:", ev["formal_P2"])
    print("Saved", run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["T1", "SMOKE_DEVELOPMENT"], required=True)
    parser.add_argument("--fit-table", type=Path, required=True, help="development out-of-fold table folder")
    parser.add_argument("--cal-cases", nargs="*", default=[])
    parser.add_argument("--test-cases", nargs="*", default=[])
    parser.add_argument("--protocol", type=Path, default=Path("docs/CONFIRMATORY_PROTOCOL_v3_DRAFT.md"))
    parser.add_argument("--commit", default="")
    parser.add_argument("--unlock", type=Path, default=Path("data/processed/T1_UNLOCK.json"))
    args = parser.parse_args()
    root = Path.cwd()
    fold = int(load_yaml(root / CONFIG)["model"]["fold"])
    if args.mode == "T1":
        run_t1(args, root, fold)
        return
    data = GatedLiData(root)  # no gates: reserved scans are refused
    li_split.require_development(args.cal_cases + args.test_cases, data.manifest)
    if set(args.cal_cases) & set(args.test_cases):
        raise SystemExit("calibration and test cases overlap")
    if DEMO_CASES & set(args.cal_cases + args.test_cases):
        raise SystemExit(f"demo cases must not be rerun: {sorted(DEMO_CASES & set(args.cal_cases + args.test_cases))}")
    fit_dir = args.fit_table
    fit = cfm.Tables(pd.read_csv(fit_dir / "annotated.csv"), pd.read_csv(fit_dir / "predicted.csv"), pd.read_csv(fit_dir / "scans.csv"))
    fit = fit.subset(set(fit.scans.case) - set(args.cal_cases) - set(args.test_cases))
    run = create_run_folder(Path("results/runs"), "t1_smoke_development")
    (run / "inputs.json").write_text(json.dumps({"mode": args.mode, "cal": args.cal_cases, "test": args.test_cases, "fit_table": str(fit_dir),
                                                 "git_head": subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True).stdout.strip()}, indent=2))  # fmt: skip
    cal = build_tables(args.cal_cases, data, run / "cal_work", fold)
    ev = run_analysis(fit, cal, lambda: build_tables(args.test_cases, data, run / "test_work", fold), run / "analysis", "DRESS_REHEARSAL")
    print("SMOKE OK: calibration hashed", ev["calibration_quantities_sha256"][:12], "| test scans", ev["test_scans"], "| formal P2 keys", sorted(ev["formal_P2"]))
    print("Saved", run)


if __name__ == "__main__":
    main()

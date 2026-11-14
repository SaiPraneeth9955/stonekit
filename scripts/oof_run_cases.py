"""E-INF step 1: run the full pipeline on each development scan with a model that did NOT train on it.

Assignment (from data/nnunet/li_stone_cv_split.json):
- folds 1-4: each fold's validation scans, predicted by that fold's model;
- fold 0: its 32 validation scans are NOT rerun here. The existing run
  data/interim/stonekit_runs/validation_20261106T003707Z (same model, same config) is reused, which also
  respects the rule not to rerun the six demo cases;
- held-back development scans (not in any fold: 10 awaiting the Slicer review, plus wd0031): fold 0's
  model (it never trained on them).

The pipeline config is configs/system_pipeline_v1.yaml with only `model.fold` changed; run_case records
the effective config and its hash in every report. Resumable: a case with a finished report.json in
the output folder is skipped (an unfinished case folder is renamed *_incomplete_<time> and redone).
Development scans only; reserved scans are refused.
"""

import argparse
import copy
import json
import time
from pathlib import Path

from stonekit.config import load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.pipeline.run_case import run_case
from stonekit.seg.nnunet_infer import checkpoint_path

BASE_CONFIG = Path("configs/system_pipeline_v1.yaml")


def assignments(root: Path, data: LiDevelopmentData) -> dict[int, list[str]]:
    cv = json.loads((root / "data/nnunet/li_stone_cv_split.json").read_text())
    trainable = {c for f in cv["folds"] for c in f["val"]}
    development = sorted(c for c, g in data.manifest.items() if g == "development")
    held_back = [c for c in development if c not in trainable]
    out = {k: sorted(cv["folds"][k]["val"]) for k in range(1, 5)}
    out[0] = held_back
    for k, cases in out.items():
        if set(cases) & set(cv["folds"][k]["train"]):
            raise SystemExit(f"fold {k} would predict a scan it trained on")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output root, e.g. data/interim/stonekit_runs/oof_<stamp>")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these cases (smoke tests)")
    args = parser.parse_args()
    root = Path.cwd()
    data = LiDevelopmentData(root)
    base = load_yaml(root / BASE_CONFIG)
    plan = assignments(root, data)
    log = args.out / "progress.jsonl"
    args.out.mkdir(parents=True, exist_ok=True)
    for fold in args.folds:
        config = copy.deepcopy(base)
        config["model"]["fold"] = fold
        if not checkpoint_path(root, config["model"]).is_file():
            print(f"fold {fold}: checkpoint missing; skipped", flush=True)
            continue
        cases = [c for c in plan[fold] if args.cases is None or c in args.cases]
        require_development(cases, data.manifest)
        fold_out = args.out / f"fold{fold}"
        fold_out.mkdir(exist_ok=True)
        for n, case in enumerate(cases, start=1):
            done = [p for p in fold_out.glob(f"{case}_*") if (p / "report.json").exists() and not p.name.endswith("incomplete")]
            if done:
                continue
            for partial in fold_out.glob(f"{case}_*"):
                if not (partial / "report.json").exists() and "_incomplete_" not in partial.name:
                    partial.rename(partial.with_name(f"{partial.name}_incomplete_{int(time.time())}"))
            ct_path, annotation = data.paths(case)
            saved = root / "data" / "interim" / "li2022_kidney_ts" / f"{case}_ts.nii.gz"
            start = time.perf_counter()
            folder = run_case(ct_path, fold_out, config, root, case, annotation, saved if saved.exists() else None, "gpu", root / BASE_CONFIG)
            report = json.loads((folder / "report.json").read_text())
            entry = {"fold": fold, "case": case, "folder": str(folder), "seconds": round(time.perf_counter() - start, 1),
                     "result_status": report.get("result_status"), "failure": report.get("failure")}  # fmt: skip
            with log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            print(f"fold {fold} {n}/{len(cases)} {case} {entry['result_status']} {entry['seconds']} s", flush=True)


if __name__ == "__main__":
    main()

"""Write the T1 freeze manifest: SHA-256 of every artefact the single T1 run depends on (13 November 2026).

Hashes files only; it never loads an image or a reserved scan. Output: docs/T1_FREEZE_MANIFEST.json (committed).
"""

import hashlib
import json
from pathlib import Path

FILES = [
    "docs/CONFIRMATORY_PROTOCOL_v3_DRAFT.md",
    "configs/system_pipeline_v1.yaml",
    "configs/baseline_pipeline.yaml",
    "data/nnunet/nnUNet_results/Dataset501_StoneNCCT/nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres/fold_0/checkpoint_final.pth",
    "data/nnunet/nnUNet_results/Dataset501_StoneNCCT/nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres/plans.json",
    "data/processed/li2022_working_split_v1.csv",
    "data/nnunet/li_stone_split.json",
    "data/nnunet/li_stone_cv_split.json",
    "data/processed/li2022_review_outcome_v1.csv",
    "data/interim/oof_tables_20261112T082351Z/annotated.csv",
    "data/interim/oof_tables_20261112T082351Z/predicted.csv",
    "data/interim/oof_tables_20261112T082351Z/scans.csv",
    "src/stonekit/uq/confirmatory.py",
    "src/stonekit/uq/conformal.py",
    "src/stonekit/t1/analysis.py",
    "src/stonekit/t1/gates.py",
    "src/stonekit/t1/access.py",
    "scripts/t1_run.py",
    "scripts/oof_tta.py",
    "scripts/build_oof_stone_table.py",
    "scripts/make_cv_folds.py",
    "scripts/leakage_repeat_pairs.py",
    "environment.lock.yml",
]


def main() -> None:
    out = {}
    for f in FILES:
        p = Path(f)
        out[f] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "MISSING"
    missing = [f for f, h in out.items() if h == "MISSING"]
    Path("docs/T1_FREEZE_MANIFEST.json").write_text(json.dumps({"created": "13 November 2026", "purpose": __doc__.strip().splitlines()[0], "sha256": out}, indent=2) + "\n", encoding="utf-8")
    for f, h in out.items():
        print(h[:16], f)
    if missing:
        raise SystemExit(f"missing: {missing}")


if __name__ == "__main__":
    main()

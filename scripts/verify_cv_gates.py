"""Engineering gates for E-CV training (folds 1-4): split correctness, reserved-data firewall,
output isolation and provenance. Reads split files, the cohort CSV and nnU-Net folder listings only;
no image is opened and no reserved scan is loaded. Prints PASS/FAIL per check; exit 1 on any FAIL.
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NN = ROOT / "data" / "nnunet"
DS = "Dataset501_StoneNCCT"
FOLD0_CHECKPOINT_SHA_PREFIX = "2100ec09"  # 100-epoch fold-0 checkpoint_final, recorded 5-6 November
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def case_ids(folder: Path, suffix: str) -> set[str]:
    return {p.name[: -len(suffix)] for p in folder.iterdir() if p.name.endswith(suffix)}


def main() -> None:
    with open(ROOT / "data/processed/li2022_working_split_v1.csv", encoding="utf-8") as handle:
        rows = {r["case"]: r["assignment"] for r in csv.DictReader(handle)}
    development = {c for c, a in rows.items() if a == "development"}
    reserved = {c for c, a in rows.items() if a.startswith("reserved")}
    fold0 = json.loads((NN / "li_stone_split.json").read_text())
    cv = json.loads((NN / "li_stone_cv_split.json").read_text())
    trainable = set(fold0["train"]) | set(fold0["validation"])
    folds = cv["folds"]

    # 1. Fold correctness
    check("fold 0 identical to the frozen split", sorted(folds[0]["train"]) == sorted(fold0["train"]) and sorted(folds[0]["val"]) == sorted(fold0["validation"]))
    check("cv split records the frozen fold-0 file hash", cv["fold0_source_sha256"] == hashlib.sha256((NN / "li_stone_split.json").read_bytes()).hexdigest())
    for i, f in enumerate(folds):
        check(f"fold {i}: train and val disjoint", not set(f["train"]) & set(f["val"]), f"{len(f['train'])}/{len(f['val'])}")
        check(f"fold {i}: train + val = the 154 trainable scans", set(f["train"]) | set(f["val"]) == trainable)
    check("every trainable scan validated exactly once", sorted(c for f in folds for c in f["val"]) == sorted(trainable))
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_cv_folds

    pairs = [(a, b) for a, b in make_cv_folds.suspected_pairs() if a in trainable and b in trainable]
    fold_of = {c: i for i, f in enumerate(folds) for c in f["val"]}
    together = [(a, b) for a, b in pairs if a in fold0["train"] and b in fold0["train"]]
    check("repeat pairs inside fold-0 training are in the same new fold", all(fold_of[a] == fold_of[b] for a, b in together), f"{len(together)} pairs")
    check("pair counts match the split record", len(pairs) == cv["within_development_repeat_pairs"] and len(together) == cv["pairs_kept_together_in_folds_1_4"])
    splits_final = json.loads((NN / "nnUNet_preprocessed" / DS / "splits_final.json").read_text())
    check("splits_final.json equals the CV split (5 folds)", len(splits_final) == 5 and all(
        sorted(s["train"]) == sorted(f["train"]) and sorted(s["val"]) == sorted(f["val"]) for s, f in zip(splits_final, folds, strict=True)))  # fmt: skip

    # 2. Reserved-data firewall
    check("no reserved scan in any fold", not reserved & {c for f in folds for c in f["train"] + f["val"]})
    check("all fold scans are development scans", {c for f in folds for c in f["train"] + f["val"]} <= development)
    raw_images = case_ids(NN / "nnUNet_raw" / DS / "imagesTr", "_0000.nii.gz")
    raw_labels = case_ids(NN / "nnUNet_raw" / DS / "labelsTr", ".nii.gz")
    pre = case_ids(NN / "nnUNet_preprocessed" / DS / "nnUNetPlans_4GB_3d_fullres", ".pkl")
    gt = case_ids(NN / "nnUNet_preprocessed" / DS / "gt_segmentations", ".nii.gz")
    for name, ids in (("raw images", raw_images), ("raw labels", raw_labels), ("preprocessed cases", pre), ("gt_segmentations", gt)):
        check(f"{name}: exactly the 154 trainable scans, no reserved", ids == trainable and not ids & reserved, f"{len(ids)} cases")
    held_back = development - trainable
    check("held-back development scans are in no fold and no nnU-Net cache", not held_back & (raw_images | pre | gt | {c for f in folds for c in f["train"] + f["val"]}), f"{len(held_back)} scans held back (OOF by fold 0 later)")

    # 6. Output isolation
    results = NN / "nnUNet_results" / DS / "nnUNetTrainer_100epochs__nnUNetPlans_4GB__3d_fullres"
    fold0_ckpt = results / "fold_0" / "checkpoint_final.pth"
    sha = hashlib.sha256(fold0_ckpt.read_bytes()).hexdigest()
    check("fold-0 100-epoch checkpoint unchanged", sha.startswith(FOLD0_CHECKPOINT_SHA_PREFIX), sha[:16])
    launcher = (ROOT / "scripts/train_cv_folds.sh").read_text()
    check("launcher trains only folds 1-4", "for FOLD in 1 2 3 4" in launcher and "--fold 0" not in launcher)
    check("launcher is flock-guarded (no two runs at once)", "flock -n" in launcher)
    print("RESULT:", "ALL PASS" if not failures else f"{len(failures)} FAIL")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

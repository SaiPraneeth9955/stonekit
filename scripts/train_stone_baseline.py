"""Plan, preprocess and train the stone baseline on this machine. No data leaves the computer.

Reads the saved development split, and refuses any scan that is not a development scan. Seeds are
set, but nnU-Net training is not fully deterministic. Records settings, IDs' hash, hardware, time
and GPU memory in a new run folder. Use --plan-only to stop after planning.
"""

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.seg.nnunet_infer import nnunet_environment


def write_splits(root: Path, config: dict, split: dict, cv: dict | None = None) -> None:
    """Write nnU-Net's splits_final.json. With the CV split (E-CV), all five folds are written;
    fold 0 must equal the frozen split exactly, so fold-0 training is unaffected."""
    folder = root / "data" / "nnunet" / "nnUNet_preprocessed" / config["dataset_name"]
    fold = {"train": split["train"], "val": split["validation"]}
    folds = [fold]
    if cv is not None:
        if sorted(cv["folds"][0]["train"]) != sorted(split["train"]) or sorted(cv["folds"][0]["val"]) != sorted(split["validation"]):
            raise ValueError("CV split fold 0 differs from the frozen fold-0 split")
        folds = [{"train": f["train"], "val": f["val"]} for f in cv["folds"]]
    (folder / "splits_final.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def code_provenance(root: Path) -> dict:
    """Git commit, whether the tree is dirty, and SHA-256 of the files that define training."""

    def git(*a):
        r = subprocess.run(["git", *a], cwd=root, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else None

    files = ["scripts/train_stone_baseline.py", "scripts/train_cv_folds.sh", "configs/li_stone_baseline.yaml",
             "data/nnunet/li_stone_split.json", "data/nnunet/li_stone_cv_split.json"]  # fmt: skip
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files if (root / f).exists()},
    }


def watch_gpu(samples: list[int], stop: threading.Event) -> None:
    while not stop.wait(10):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if result.returncode == 0 and result.stdout.strip():
            samples.append(int(result.stdout.strip().splitlines()[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--trainer", required=True, help="An nnU-Net trainer, such as nnUNetTrainer_100epochs")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--save-every", type=int, default=None,
        help="Save checkpoint_latest every N epochs (nnU-Net default 50). Engineering only: it does not change training.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Fold to train; folds 1-4 need data/nnunet/li_stone_cv_split.json")
    args = parser.parse_args()
    root = args.root.resolve()
    config = load_yaml(root / "configs" / "li_stone_baseline.yaml")
    split = json.loads((root / "data" / "nnunet" / "li_stone_split.json").read_text(encoding="utf-8"))
    data = LiDevelopmentData(root)
    require_development([*split["train"], *split["validation"]], data.manifest)
    if set(split["train"]) & set(split["validation"]):
        raise ValueError("A scan is in both training and validation")
    fold_index = config["fold"] if args.fold is None else args.fold
    cv_path = root / "data" / "nnunet" / "li_stone_cv_split.json"
    cv = json.loads(cv_path.read_text(encoding="utf-8")) if cv_path.exists() else None
    if fold_index != 0 and cv is None:
        raise ValueError("folds other than 0 need data/nnunet/li_stone_cv_split.json")
    if cv is not None:
        for f in cv["folds"]:
            require_development([*f["train"], *f["val"]], data.manifest)
            if set(f["train"]) & set(f["val"]):
                raise ValueError("A scan is in both training and validation (CV split)")

    env = nnunet_environment(root)
    os.environ.update(env)
    os.environ["nnUNet_n_proc_DA"] = str(config["augmentation_processes"])
    plans_file = (
        root / "data" / "nnunet" / "nnUNet_preprocessed" / config["dataset_name"] / f"{config['plans']}.json"
    )
    if not plans_file.exists():
        command = [
            str(Path(sys.executable).parent / "nnUNetv2_plan_and_preprocess"),
            "-d", str(config["dataset_id"]), "-c", config["configuration"],
            "--verify_dataset_integrity",
            "-gpu_memory_target", str(config["gpu_memory_target_gb"]),
            "-overwrite_plans_name", config["plans"],
            "-npfp", str(config["preprocessing_processes"]),
            "-np", str(config["preprocessing_processes"]),
            "--no_pbar",
        ]  # fmt: skip
        subprocess.run(command, check=True, env=os.environ)
    write_splits(root, config, split, cv)
    plans = json.loads(plans_file.read_text(encoding="utf-8"))
    print(json.dumps({k: plans["configurations"][config["configuration"]].get(k) for k in ("batch_size", "patch_size", "spacing")}))
    if args.plan_only:
        return

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    import torch

    torch.manual_seed(config["seed"])
    from nnunetv2.run.run_training import run_training

    if args.save_every is not None:
        # nnU-Net sets save_every in the trainer's __init__; wrap it so the trainer class (and therefore
        # its output folder name) stays exactly the same. Checkpoint frequency does not affect training.
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        original_init = nnUNetTrainer.__init__

        def init_with_save_every(self, *a, **k):
            original_init(self, *a, **k)
            self.save_every = args.save_every

        nnUNetTrainer.__init__ = init_with_save_every

    run = create_run_folder(root / "results" / "runs", f"stone_baseline_training_{args.trainer}_fold{fold_index}")
    record = {
        "status": "DEVELOPMENT_ONLY_NOT_FINAL",
        "seed": config["seed"],
        "seed_note": "Seeds set, but nnU-Net training is not fully deterministic.",
        "trainer": args.trainer,
        "resumed": args.resume,
        "plans_sha256": hashlib.sha256(plans_file.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256((root / "data" / "nnunet" / "li_stone_split.json").read_bytes()).hexdigest(),
        "fold": fold_index,
        "cv_split_sha256": hashlib.sha256(cv_path.read_bytes()).hexdigest() if cv is not None else None,
        "train_scans": len(cv["folds"][fold_index]["train"]) if cv is not None else len(split["train"]),
        "validation_scans": len(cv["folds"][fold_index]["val"]) if cv is not None else len(split["validation"]),
        "plan": {k: plans["configurations"][config["configuration"]].get(k) for k in ("batch_size", "patch_size", "spacing")},
        "augmentation_processes": config["augmentation_processes"],
        "save_every": args.save_every if args.save_every is not None else 50,
        "code": code_provenance(root),
        "nnunetv2_version": package_version("nnunetv2"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    (run / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    samples: list[int] = []
    stop = threading.Event()
    watcher = threading.Thread(target=watch_gpu, args=(samples, stop), daemon=True)
    watcher.start()
    start = time.perf_counter()
    try:
        run_training(
            config["dataset_name"],
            config["configuration"],
            fold_index,
            trainer_class_name=args.trainer,
            plans_identifier=config["plans"],
            continue_training=args.resume,
        )
        record["outcome"] = "finished"
    finally:
        stop.set()
        record.setdefault("outcome", "stopped_or_failed")
        record["seconds"] = round(time.perf_counter() - start, 1)
        record["gpu_memory_used_mib_peak"] = max(samples, default=None)
        record["gpu_memory_used_mib_median"] = int(np.median(samples)) if samples else None
        (run / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Saved to", run)


if __name__ == "__main__":
    main()

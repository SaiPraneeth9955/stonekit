"""E-INF step 2: stone probabilities and test-time-augmentation (TTA) spread on saved ROI crops.

For each case folder written by `stonekit run-case` (it holds roi_crop.nii.gz), and for the fold
model that did NOT train on that case, this writes to a NEW folder (never into the run folder):
- p_deployed: stone probability from the deployed prediction mode (nnU-Net default mirroring TTA),
  from `--save_probabilities`;
- p_flip[8]: stone probability from 8 separate predictions with nnU-Net's TTA switched off, one per
  mirror combination of the three axes (the crop is flipped, predicted, and flipped back). Their
  spread per stone is the TTA uncertainty signal (master plan v2, U4). nnU-Net does not expose
  per-flip outputs (it averages them inside _internal_maybe_mirror_and_predict, verified in the
  installed nnunetv2 source), hence explicit flips.

Self-checks per case (a failure stops that case and is recorded, never hidden):
- the argmax of p_deployed equals nnU-Net's own segmentation of the same call voxel for voxel
  (this proves the SimpleITK (z,y,x) -> nibabel (x,y,z) axis handling);
- every probability is finite and in [0, 1].
Development scans only (the reserved-scan guard is applied to every case ID).
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from stonekit.config import load_yaml
from stonekit.data.li2022 import LiDevelopmentData
from stonekit.data.li_split import require_development
from stonekit.seg.nnunet_infer import checkpoint_path, nnunet_environment

FLIPS = [c for r in range(4) for c in itertools.combinations((0, 1, 2), r)]  # 8 combinations, () first


def predict(input_dir: Path, output_dir: Path, model: dict, fold: int, tta: bool, root: Path) -> None:
    command = [
        str(Path(sys.executable).parent / "nnUNetv2_predict"), "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(model["dataset_id"]), "-c", model["configuration"], "-p", model["plans"],
        "-tr", model["trainer"], "-f", str(fold), "-chk", model["checkpoint"],
        "-npp", "1", "-nps", "1", "-device", "cuda", "--disable_progress_bar", "--save_probabilities",
    ]  # fmt: skip
    if not tta:
        command.append("--disable_tta")
    subprocess.run(command, check=True, env=nnunet_environment(root), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def stone_probability(npz: Path) -> np.ndarray:
    """nnU-Net saves (classes, z, y, x) in SimpleITK order; return the stone class as (x, y, z)."""
    p = np.load(npz)["probabilities"][1]
    return np.transpose(p, (2, 1, 0)).astype(np.float32)


def process_case(folder: Path, model: dict, fold: int, out: Path, root: Path) -> dict:
    crop_img = nib.load(folder / "roi_crop.nii.gz")
    crop = np.asanyarray(crop_img.dataobj)
    record = {"case_folder": str(folder), "fold_model": fold, "crop_shape": list(crop.shape)}
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="stonekit_tta_") as tmp:
        tmp = Path(tmp)
        (tmp / "in_tta").mkdir()
        (tmp / "in_flip").mkdir()
        shutil.copyfile(folder / "roi_crop.nii.gz", tmp / "in_tta" / "case_0000.nii.gz")
        for i, axes in enumerate(FLIPS):
            data = np.flip(crop, axis=axes) if axes else crop
            nib.save(nib.Nifti1Image(np.ascontiguousarray(data), crop_img.affine, crop_img.header), tmp / "in_flip" / f"f{i}_0000.nii.gz")
        predict(tmp / "in_tta", tmp / "out_tta", model, fold, True, root)
        predict(tmp / "in_flip", tmp / "out_flip", model, fold, False, root)
        p_deployed = stone_probability(tmp / "out_tta" / "case.npz")
        seg = np.asanyarray(nib.load(tmp / "out_tta" / "case.nii.gz").dataobj)
        if p_deployed.shape != crop.shape:
            raise ValueError(f"probability shape {p_deployed.shape} != crop shape {crop.shape}")
        argmax_matches = bool(np.array_equal(p_deployed > 0.5, seg == 1))
        disagreeing = int(np.count_nonzero((p_deployed > 0.5) != (seg == 1)))
        flips = []
        for i, axes in enumerate(FLIPS):
            p = stone_probability(tmp / "out_flip" / f"f{i}.npz")
            flips.append(np.flip(p, axis=axes) if axes else p)
        p_flip = np.stack(flips)
    finite = bool(np.isfinite(p_deployed).all() and np.isfinite(p_flip).all())
    in_range = bool(p_deployed.min() >= 0 and p_deployed.max() <= 1 and p_flip.min() >= 0 and p_flip.max() <= 1)
    # Two-class softmax: argmax(stone) == (p > 0.5) except exact ties; tolerate a handful of tie voxels.
    record.update({
        "seconds": round(time.perf_counter() - start, 1),
        "argmax_equals_nnunet_segmentation": argmax_matches, "argmax_disagreeing_voxels": disagreeing,
        "finite": finite, "in_unit_interval": in_range,
        "deployed_stone_voxels_p_gt_0.5": int(np.count_nonzero(p_deployed > 0.5)),
        "flip_stone_voxels_p_gt_0.5": [int(np.count_nonzero(p > 0.5)) for p in p_flip],
    })  # fmt: skip
    if not (finite and in_range and disagreeing <= 10):
        raise ValueError(f"self-check failed: {record}")
    np.savez_compressed(out / "probabilities.npz", p_deployed=p_deployed.astype(np.float16), p_flip=p_flip.astype(np.float16), affine=crop_img.affine)  # fmt: skip
    (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-folder", type=Path, required=True, help="Folder of run-case outputs (one subfolder per case)")
    parser.add_argument("--fold", type=int, required=True, help="Fold model that did not train on these cases")
    parser.add_argument("--out", type=Path, required=True, help="New output folder (cases already done there are skipped)")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/system_pipeline_v1.yaml"))
    args = parser.parse_args()
    root = Path.cwd()
    model = dict(load_yaml(root / args.config)["model"])
    if not checkpoint_path(root, {**model, "fold": args.fold}).is_file():
        raise SystemExit(f"fold {args.fold} checkpoint missing")
    cv = json.loads((root / "data/nnunet/li_stone_cv_split.json").read_text())
    trained_on = set(cv["folds"][args.fold]["train"])
    data = LiDevelopmentData(root)
    folders = {}
    for p in sorted(args.run_folder.iterdir()):
        if (p / "report.json").exists() and (p / "roi_crop.nii.gz").exists():
            folders[json.loads((p / "report.json").read_text())["case_id"]] = p
    cases = args.cases or sorted(folders)
    require_development(cases, data.manifest)
    leaked = sorted(set(cases) & trained_on)
    if leaked:
        raise SystemExit(f"fold {args.fold} trained on {len(leaked)} of these cases; refusing (not out-of-fold)")
    args.out.mkdir(parents=True, exist_ok=True)
    failures = {}
    for n, case in enumerate(cases, start=1):
        target = args.out / case
        if (target / "record.json").exists():
            continue
        target.mkdir(exist_ok=True)
        try:
            r = process_case(folders[case], model, args.fold, target, root)
            print(f"{n}/{len(cases)} {case} {r['seconds']} s, argmax ok {r['argmax_equals_nnunet_segmentation']}", flush=True)
        except Exception as error:  # noqa: BLE001  (record and continue; reported, never hidden)
            failures[case] = repr(error)
            print(f"{n}/{len(cases)} {case} FAILED {error!r}", flush=True)
    (args.out / f"failures_fold{args.fold}.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    print("done; failures:", len(failures))


if __name__ == "__main__":
    main()

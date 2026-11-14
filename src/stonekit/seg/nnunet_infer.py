"""Stone prediction on one kidney ROI crop, with a trained nnU-Net model on this machine.

The prediction runs through the same guard as TotalSegmentator (stonekit.seg.subprocess_guard): a
timeout, output to a log file, and every helper process stopped afterwards. The returned mask is
checked against the crop's grid and label values, so a wrong or corrupt output is never measured.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from stonekit.config import load_yaml
from stonekit.seg.subprocess_guard import check_result, run_guarded

# Engineering safety limit, not a method setting. The longest whole-pipeline run on the 32 validation
# scans was 219.1 s, inference included (docs/results.md, 7 November 2026). 900 s is over four
# times that. A config may override it with model_inference.timeout_seconds.
DEFAULT_PREDICT_TIMEOUT_SECONDS = 900
# nnU-Net writes the dataset's labels: 0 background, 1 stone (Dataset501_StoneNCCT).
ALLOWED_PREDICTION_LABELS = frozenset({0, 1})


def nnunet_environment(root: Path) -> dict[str, str]:
    paths = load_yaml(Path(root) / "configs" / "paths.yaml")["nnunet"]
    env = dict(os.environ)
    env.update({name: str((Path(root) / value).resolve()) for name, value in paths.items()})
    return env


def model_folder(root: Path, model: dict) -> Path:
    name = f"{model['trainer']}__{model['plans']}__{model['configuration']}"
    return Path(root) / "data" / "nnunet" / "nnUNet_results" / model["dataset_name"] / name


def checkpoint_path(root: Path, model: dict) -> Path:
    return model_folder(root, model) / f"fold_{model['fold']}" / model["checkpoint"]


def model_available(root: Path, model: dict) -> bool:
    """True only if a real trained checkpoint exists on disk."""
    return checkpoint_path(root, model).is_file()


def predict_command(input_dir: Path, output_dir: Path, model: dict, device: str) -> list[str]:
    binary = Path(sys.executable).parent / "nnUNetv2_predict"
    command = [
        str(binary), "-i", str(input_dir), "-o", str(output_dir),
        "-d", str(model["dataset_id"]), "-c", model["configuration"],
        "-p", model["plans"], "-tr", model["trainer"], "-f", str(model["fold"]),
        "-chk", model["checkpoint"], "-npp", "1", "-nps", "1", "-device", device,
        "--disable_progress_bar",
    ]  # fmt: skip
    if model.get("disable_tta"):
        command.append("--disable_tta")
    return command


def check_prediction(prediction: nib.Nifti1Image, crop_path: Path) -> np.ndarray:
    """Refuse a prediction that is not on the crop's grid or holds labels other than 0 and 1."""
    crop = nib.load(crop_path)
    if prediction.shape != crop.shape:
        raise ValueError(f"Model prediction shape {prediction.shape} is not the crop shape {crop.shape}")
    if not np.allclose(prediction.affine, crop.affine, atol=1e-3):
        raise ValueError("Model prediction is not on the crop grid (affines differ)")
    data = np.asanyarray(prediction.dataobj)
    if not np.issubdtype(data.dtype, np.integer) and not np.array_equal(data, np.round(data)):
        raise ValueError("Model prediction holds non-integer values")
    unknown = {int(v) for v in np.unique(data)} - ALLOWED_PREDICTION_LABELS
    if unknown:
        raise ValueError(f"Model prediction holds unexpected labels {sorted(unknown)}")
    return data == 1


def predict_crop(
    root: Path,
    model: dict,
    crop_path: Path,
    device: str = "cuda",
    timeout_seconds: float = DEFAULT_PREDICT_TIMEOUT_SECONDS,
    log_path: Path | None = None,
) -> np.ndarray:
    """Run the trained model on a crop saved as NIfTI. Returns a boolean stone mask on that grid.

    Raises ExternalToolError on timeout, a non-zero exit, a kill signal or a missing output, and
    ValueError when the output is not on the crop grid or holds unexpected labels.
    """
    if not model_available(root, model):
        raise FileNotFoundError(f"No trained checkpoint at {checkpoint_path(root, model)}")
    with tempfile.TemporaryDirectory(prefix="stonekit_predict_") as temporary:
        input_dir = Path(temporary) / "in"
        output_dir = Path(temporary) / "out"
        input_dir.mkdir()
        shutil.copyfile(crop_path, input_dir / "case_0000.nii.gz")
        log = Path(log_path) if log_path else Path(temporary) / "nnunet_predict.log"
        command = predict_command(input_dir, output_dir, model, device)
        result = run_guarded(command, log, timeout_seconds, env=nnunet_environment(root))
        output = output_dir / "case.nii.gz"
        check_result("nnUNetv2_predict", result, output, timeout_seconds)
        return check_prediction(nib.load(output), crop_path)

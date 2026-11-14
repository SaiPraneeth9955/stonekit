"""One CT to masks, individual stones, measurements, QC, meshes and reports.

Each stage passes real output to the next. A stage that fails is recorded in the report. No result is
ever filled in with an example value, and a missing model is reported as missing.
"""

import hashlib
import json
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import ndimage

from stonekit.io.nifti import Volume, load_volume, save_on_original_grid
from stonekit.measure.mesh import mask_mesh, write_obj
from stonekit.measure.stones import measure_stones, separate_stones
from stonekit.pipeline import qc as qc_rules
from stonekit.pipeline.config_check import check_config
from stonekit.pipeline.input_checks import check_ct
from stonekit.pipeline.report_html import write_html_report
from stonekit.seg import kidney_roi
from stonekit.seg.nnunet_convert import crop_affine, save_crop
from stonekit.seg.nnunet_infer import (
    DEFAULT_PREDICT_TIMEOUT_SECONDS,
    checkpoint_path,
    model_available,
    predict_crop,
)
from stonekit.seg.subprocess_guard import ExternalToolError
from stonekit.seg.threshold import threshold_near_kidney

DISCLAIMER = "Research use only. Not a diagnostic medical device."
STATUS_LINE = "Development demonstration — not final validated performance."
UNCERTAINTY_STATUS = "Uncertainty calibration: later phase. No calibrated ranges are reported."
MODEL = "model"
THRESHOLD = "threshold_272hu"
# TotalSegmentator labels for the ROI subset (checked on the installed 2.18.0 and on saved outputs).
KIDNEY_OUTPUT_LABELS = frozenset({0, kidney_roi.KIDNEY_RIGHT, kidney_roi.KIDNEY_LEFT, kidney_roi.AORTA})
# Result status: "complete" (the trained model's result exists), "no_result" (the run ended without
# it, for example no kidney found or no checkpoint; the threshold comparator may still be shown) or
# "failed" (a processing failure).
COMPLETE, NO_RESULT, FAILED = "complete", "no_result", "failed"


class Stages:
    """Records each stage's status and time."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, name: str, status: str, start: float, detail: str = "") -> None:
        self.records.append(
            {
                "stage": name,
                "status": status,
                "seconds": round(time.perf_counter() - start, 2),
                "detail": detail,
            }
        )


def _versions() -> dict:
    from importlib.metadata import version

    import nibabel
    import nnunetv2  # noqa: F401
    import scipy
    import skimage
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_image": skimage.__version__,
        "nibabel": nibabel.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nnunetv2": version("nnunetv2"),
        "totalsegmentator": version("TotalSegmentator"),
    }


def _code_version(root: Path) -> dict:
    """StoneKit's own code: git commit and whether the working tree had uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, capture_output=True, text=True,
                                timeout=10, check=True).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "uncommitted_changes": None, "note": "Not a git checkout, or git unavailable."}
    return {
        "git_commit": commit,
        "uncommitted_changes": bool(status.strip()),
        "note": "With uncommitted changes, the commit alone does not identify the code that ran.",
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plain(value):
    """Turn NaN into null, so the JSON stays standard."""
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


def _measure_method(
    binary_crop: np.ndarray,
    ct: Volume,
    box: kidney_roi.Box,
    distance: np.ndarray,
    config: dict,
    method: str,
    out_dir: Path,
    max_distance_mm: float | None = None,
) -> dict:
    """Separate, measure, flag and mesh one method's mask, then save it on the original grid.

    With `max_distance_mm`, pieces whose nearest voxel lies farther than that from the kidney mask
    are removed from the stones, the meshes and the saved label map, and listed in the result.
    """
    affine = crop_affine(ct.affine, box)
    spacing = ct.spacing_mm
    separation = separate_stones(binary_crop, config["measurement"]["min_component_voxels"])
    stones = measure_stones(separation, ct.data[box.slices], affine, spacing)
    qc = config["qc"]
    for stone in stones:
        window = tuple(slice(a, b) for a, b in zip(stone["voxel_start"], stone["voxel_stop"], strict=True))
        inside = separation.labels[window] == stone["stone_id"]
        stone["distance_to_kidney_mm"] = float(distance[window][inside].min())
    removed = []
    if max_distance_mm is not None:
        removed = [s for s in stones if s["distance_to_kidney_mm"] > max_distance_mm]
        stones = [s for s in stones if s["distance_to_kidney_mm"] <= max_distance_mm]
        if removed:
            removed_ids = [s["stone_id"] for s in removed]
            in_removed = np.isin(separation.labels, removed_ids)
            # Removed pieces stay traceable: their own label map, original IDs, original grid.
            kept_aside = np.where(in_removed, separation.labels, 0).astype(np.uint16)
            full_removed = kidney_roi.paste_into_full(kept_aside, box, ct.data.shape)
            save_on_original_grid(full_removed, ct, out_dir / f"stone_labels_{method}_removed_by_kidney_distance_filter.nii.gz")
            separation.labels[in_removed] = 0
    for stone in stones:
        low = stone["voxel_start"]
        high = stone["voxel_stop"]
        window = tuple(slice(a, b) for a, b in zip(low, high, strict=True))
        inside = separation.labels[window] == stone["stone_id"]
        stone["touches_roi_box"] = any(a == 0 for a in low) or any(
            b == n for b, n in zip(high, box.shape, strict=True)
        )
        stone["voxel_start"] = [a + s for a, s in zip(low, box.start, strict=True)]
        stone["voxel_stop"] = [b + s for b, s in zip(high, box.start, strict=True)]
        stone["qc_flags"] = qc_rules.stone_flags(stone, qc, spacing)
        name = f"{method}_stone_{stone['stone_id']:03d}"
        vertices, faces = mask_mesh(inside, _window_affine(affine, low))
        write_obj(out_dir / "meshes" / f"{name}.obj", vertices, faces, name)
        stone["mesh_file"] = f"meshes/{name}.obj"

    full = kidney_roi.paste_into_full(separation.labels.astype(np.uint16), box, ct.data.shape)
    save_on_original_grid(full, ct, out_dir / f"stone_labels_{method}.nii.gz")
    result = {}
    if max_distance_mm is not None:
        result["kidney_distance_filter"] = {
            "max_distance_mm": max_distance_mm,
            "removed_label_file": f"stone_labels_{method}_removed_by_kidney_distance_filter.nii.gz" if removed else None,
            "removed_pieces": [
                {"stone_id": s["stone_id"], "volume_mm3": s["volume_mm3"], "distance_to_kidney_mm": s["distance_to_kidney_mm"]}
                for s in removed
            ],
        }
    return result | {
        "available": True,
        "stones": stones,
        "total_volume_mm3": sum(s["volume_mm3"] for s in stones),
        "total_volume_ml": sum(s["volume_ml"] for s in stones),
        "dropped_tiny_pieces": separation.dropped_pieces,
        "dropped_tiny_voxels": separation.dropped_voxels,
        "qc_flags": qc_rules.method_flags(stones, separation.dropped_pieces, qc),
        "label_file": f"stone_labels_{method}.nii.gz",
    }


LOW_MEMORY_SLICE_MM = 1.5


def _memory_like(error: ExternalToolError) -> bool:
    """A failure that the low-memory kidney path can plausibly avoid: killed, timed out, or out of memory."""
    return error.reason in ("killed_by_signal", "timeout") or (error.reason == "nonzero_exit" and "memory" in error.detail.lower())


def _window_affine(affine: np.ndarray, start: list[int]) -> np.ndarray:
    moved = affine.copy()
    moved[:3, 3] = (affine @ np.array([*start, 1.0]))[:3]
    return moved


def _annotation_comparison(report_method: dict, label_path: Path, reference: np.ndarray) -> dict:
    """Compare with the Li annotation. A reference only: never physical truth, never an input."""
    prediction = load_volume(label_path).data > 0
    both = int((prediction & reference).sum())
    total = int(prediction.sum()) + int(reference.sum())
    union = int((prediction | reference).sum())
    return {
        "dice": 2 * both / total if total else None,
        "iou": both / union if union else None,
        "predicted_voxels": int(prediction.sum()),
        "annotation_voxels": int(reference.sum()),
        "predicted_stones": len(report_method["stones"]),
        "annotation_components": int(ndimage.label(reference, structure=np.ones((3, 3, 3)))[1]),
    }


def run_case(
    ct_path: Path,
    out_root: Path,
    config: dict,
    root: Path,
    case_id: str,
    annotation_path: Path | None = None,
    kidney_labels_path: Path | None = None,
    device: str = "gpu",
    config_path: Path | None = None,
) -> Path:
    """Process one CT from start to finish, and return the case's new output folder.

    `config_path` names the file `config` was loaded from, for provenance only; it defaults to
    configs/baseline_pipeline.yaml, as before.
    """
    check_config(config)  # Before any output exists, so a bad config leaves nothing behind.
    config_file = Path(config_path) if config_path else Path(root) / "configs" / "baseline_pipeline.yaml"
    wall_start = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_root) / f"{case_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    stages = Stages()
    report: dict = {
        "disclaimer": DISCLAIMER,
        "status_line": STATUS_LINE,
        "uncertainty_status": UNCERTAINTY_STATUS,
        "case_id": case_id,
        "ct_path": str(Path(ct_path).resolve()),
        "created_utc": stamp,
        "qc_note": "QC flags are research heuristics, not validated rules or calibrated uncertainty.",
        "software": _versions(),
        "code_version": _code_version(Path(root)),
        "config_file": config_file.name,
        "config_sha256": _file_sha256(config_file),
        "config_sha256_note": "Hash of config_file on disk. The configuration actually used is `config`, hashed in "
        "`effective_config_sha256`; they can differ when a script overrides a setting.",
        "effective_config_sha256": hashlib.sha256(json.dumps(_plain(config), sort_keys=True).encode()).hexdigest(),
        "config": config,
        "methods": {},
        "case_qc_flags": [],
        "preprocessing": [
            "Axes reordered to RAS without interpolation.",
            "No HU values changed. Measurements use original voxels and spacing.",
            "The model's own resampling and normalisation apply to its input only.",
        ],
    }
    limit = (config.get("postprocessing") or {}).get("model_max_distance_to_kidney_mm")
    report["postprocessing"] = {
        "model_kidney_distance_filter_mm": limit,
        "applied_to": "model method only" if limit is not None else "not applied",
        "note": "Development-selected filter (docs/decisions.md, 7 November 2026). It can remove real stones "
        "farther than the limit from the kidney mask, such as some ureteric stones; removed pieces are listed "
        "under each method's kidney_distance_filter. Not validated for general use.",
    }
    try:
        _process(report, stages, Path(ct_path), out_dir, config, Path(root), annotation_path, kidney_labels_path, device)
    except (OSError, ValueError, MemoryError, EOFError, subprocess.SubprocessError, ExternalToolError) as error:
        report["case_qc_flags"].append("processing_failure")
        report["failure"] = f"{type(error).__name__}: {error}"[:500]
    except Exception as error:  # noqa: BLE001 - any other error must still end in a report, never a bare folder
        report["case_qc_flags"] += ["processing_failure", "unexpected_error"]
        report["failure"] = f"{type(error).__name__}: {error}"[:500]
    if report.get("failure"):
        report["result_status"] = FAILED
    elif report.get("primary_method"):
        report["result_status"] = COMPLETE
    else:
        report["result_status"] = NO_RESULT
    report["stages"] = stages.records
    report["total_seconds"] = round(time.perf_counter() - wall_start, 2)
    report["peak_memory_kb"] = {
        "this_process": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "largest_child_process": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    report = _plain(report)
    with (out_dir / "report.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    write_html_report(report, out_dir / "report.html", out_dir)
    return out_dir


def _process(report, stages, ct_path, out_dir, config, root, annotation_path, kidney_labels_path, device) -> None:
    start = time.perf_counter()
    ct = load_volume(ct_path)
    stages.add("load_ct", "ok", start)

    start = time.perf_counter()
    input_report = check_ct(ct, config["input_checks"])
    report["input_checks"] = input_report.to_dict()
    report["voxel_volume_mm3"] = ct.voxel_volume_mm3
    if not input_report.can_process:
        stages.add("input_checks", "failed", start, "Scan cannot be measured safely")
        report["case_qc_flags"].append("processing_failure")
        return
    stages.add("input_checks", "ok", start)

    start = time.perf_counter()
    kidney_path = out_dir / "kidney_totalsegmentator.nii.gz"
    if kidney_labels_path is None:
        tool_device = "gpu" if device in ("gpu", "cuda") else device
        timeout = config.get("kidney_segmentation", {}).get("timeout_seconds", kidney_roi.DEFAULT_TIMEOUT_SECONDS)
        fallback = (config.get("kidney_segmentation") or {}).get("low_memory_fallback", True)
        try:
            result = kidney_roi.run_totalsegmentator(
                ct_path, kidney_path, tool_device, timeout_seconds=timeout,
                log_path=out_dir / "totalsegmentator.log",
            )  # fmt: skip
            detail = "TotalSegmentator 3 mm model, run for this case"
        except ExternalToolError as error:
            if not (fallback and _memory_like(error)):
                stages.add("kidney_segmentation", "failed", start, str(error)[:300])
                report["kidney_segmentation_failure"] = error.to_dict()
                report["case_qc_flags"].append("kidney_segmentation_failed")
                raise
            report["kidney_segmentation_first_attempt_failure"] = error.to_dict()
            try:
                result, info = kidney_roi.run_totalsegmentator_z_resampled(
                    ct_path, kidney_path, tool_device, LOW_MEMORY_SLICE_MM, timeout_seconds=timeout,
                    log_path=out_dir / "totalsegmentator_low_memory.log",
                )  # fmt: skip
            except (ExternalToolError, ValueError) as second:
                stages.add("kidney_segmentation", "failed", start, f"normal: {error.reason}; low-memory: {second}"[:300])
                report["kidney_segmentation_failure"] = error.to_dict()  # the primary failure stays the reported one
                report["kidney_segmentation_fallback_failure"] = (
                    second.to_dict() if isinstance(second, ExternalToolError) else {"reason": "low_memory_not_applicable", "detail": str(second)}
                )
                report["case_qc_flags"].append("kidney_segmentation_failed")
                raise error from second
            report["kidney_segmentation_fallback"] = {
                **info, "trigger": error.reason,
                "note": "Normal kidney step failed in a memory-like way, so TotalSegmentator ran on a copy with slices linearly "
                "resampled to 1.5 mm (its --fast model works at 3 mm); labels were mapped back to every original slice by nearest "
                "neighbour. Stone segmentation and measurement still use the original voxels. On 6 development scans the fallback "
                "agreed with the normal kidney step at Dice 0.948 to 0.988 and kept every stone inside the ROI "
                "(results/runs/kidney_lowmem_equivalence_20261108T010544Z/).",
            }  # fmt: skip
            report["case_qc_flags"].append("kidney_segmentation_low_memory_fallback")
            detail = "TotalSegmentator 3 mm model, low-memory fallback (slices resampled to 1.5 mm for the kidney step only)"
        report["kidney_segmentation_run"] = result.to_dict()
    else:
        kidney_path.write_bytes(Path(kidney_labels_path).read_bytes())
        detail = "TotalSegmentator 3 mm output, reused from an earlier run of this scan"
    labels = load_volume(kidney_path)
    if labels.data.shape != ct.data.shape or not np.allclose(labels.affine, ct.affine, atol=1e-3):
        raise ValueError("Kidney segmentation is not on the CT grid")
    found = {float(v) for v in np.unique(labels.data)}
    if not found <= {float(v) for v in KIDNEY_OUTPUT_LABELS}:
        unexpected = sorted(found - {float(v) for v in KIDNEY_OUTPUT_LABELS})[:5]
        raise ValueError(f"Kidney segmentation holds unexpected label values {unexpected}; it is not used")
    stages.add("kidney_segmentation", "ok", start, detail)

    start = time.perf_counter()
    kidneys = kidney_roi.kidney_mask(labels.data)
    right = int((labels.data == kidney_roi.KIDNEY_RIGHT).sum())
    left = int((labels.data == kidney_roi.KIDNEY_LEFT).sum())
    aorta = labels.data == kidney_roi.AORTA
    aorta_hu = float(np.median(ct.data[aorta])) if aorta.any() else None
    report["contrast_check"] = {
        "aorta_median_hu": aorta_hu,
        "threshold_hu": config["qc"]["contrast_aorta_median_hu"],
        "note": "Heuristic warning only. It cannot confirm that a scan is non-contrast.",
    }
    report["case_qc_flags"] += qc_rules.contrast_flags(aorta_hu, config["qc"])
    box = kidney_roi.mask_box(kidneys, ct.spacing_mm, config["roi"]["box_margin_mm"])
    at_edge = box is not None and (
        any(a == 0 for a in box.start) or any(b == n for b, n in zip(box.stop, ct.data.shape, strict=True))
    )
    report["case_qc_flags"] += qc_rules.kidney_flags(right, left, at_edge)
    voxel_ml = ct.voxel_volume_mm3 / 1000
    report["kidney"] = {
        "right_volume_ml": right * voxel_ml,
        "left_volume_ml": left * voxel_ml,
    }
    if box is None:
        stages.add("kidney_roi", "failed", start, "No kidney found, so no ROI")
        return
    report["roi"] = {
        "box_start_voxel": list(box.start),
        "box_stop_voxel": list(box.stop),
        "box_shape_voxels": list(box.shape),
        "box_size_mm": [n * s for n, s in zip(box.shape, ct.spacing_mm, strict=True)],
        "margin_mm": config["roi"]["box_margin_mm"],
        "reaches_image_edge": at_edge,
    }
    crop_path = out_dir / "roi_crop.nii.gz"
    save_crop(ct.data, ct.affine, box, crop_path, np.float32)
    distance = ndimage.distance_transform_edt(~kidneys[box.slices], sampling=ct.spacing_mm)
    stages.add("kidney_roi", "ok", start)

    model = config["model"]
    start = time.perf_counter()
    if model_available(root, model):
        timeout = (config.get("model_inference") or {}).get("timeout_seconds", DEFAULT_PREDICT_TIMEOUT_SECONDS)
        try:
            prediction = predict_crop(
                root, model, crop_path, "cuda" if device in ("gpu", "cuda") else device,
                timeout_seconds=timeout, log_path=out_dir / "nnunet_predict.log",
            )  # fmt: skip
        except (ExternalToolError, ValueError) as error:
            stages.add("stone_segmentation_model", "failed", start, str(error)[:300])
            report["stone_segmentation_failure"] = (
                error.to_dict() if isinstance(error, ExternalToolError) else {"reason": "invalid_output", "detail": str(error)}
            )
            report["case_qc_flags"].append("stone_segmentation_failed")
            raise
        if prediction.shape != box.shape:
            raise ValueError("Model prediction is not on the crop grid")
        report["model_info"] = {
            "name": "nnU-Net v2 3D full resolution, one fold, smaller 4 GB plan, trained on Li development crops",
            "checkpoint": str(checkpoint_path(root, model)),
            "checkpoint_sha256": _file_sha256(checkpoint_path(root, model)),
        }
        stages.add("stone_segmentation_model", "ok", start)
        start = time.perf_counter()
        limit = (config.get("postprocessing") or {}).get("model_max_distance_to_kidney_mm")
        report["methods"][MODEL] = _measure_method(prediction, ct, box, distance, config, MODEL, out_dir, limit)
        if (report["methods"][MODEL].get("kidney_distance_filter") or {}).get("removed_pieces"):
            # Never silent: the filter can remove real stones far from the kidney (for example ureteric).
            report["case_qc_flags"].append("pieces_removed_by_kidney_distance_filter")
        stages.add("stones_measurements_qc_meshes_model", "ok", start)
    else:
        report["methods"][MODEL] = {
            "available": False,
            "stones": [],
            "qc_flags": ["trained_model_not_available"],
        }
        stages.add("stone_segmentation_model", "not_available", start, "No trained checkpoint on disk")

    start = time.perf_counter()
    baseline = config["threshold_baseline"]
    threshold = threshold_near_kidney(
        ct.data[box.slices], distance, baseline["threshold_hu"], baseline["kidney_distance_mm"]
    )
    stages.add("stone_segmentation_threshold", "ok", start)
    start = time.perf_counter()
    report["methods"][THRESHOLD] = _measure_method(threshold, ct, box, distance, config, THRESHOLD, out_dir)
    stages.add("stones_measurements_qc_meshes_threshold", "ok", start)

    start = time.perf_counter()
    kidney_vertices, kidney_faces = mask_mesh(kidneys[box.slices], crop_affine(ct.affine, box), step_size=2)
    write_obj(out_dir / "meshes" / "kidneys.obj", kidney_vertices, kidney_faces, "kidneys")
    report["kidney_mesh_file"] = "meshes/kidneys.obj"
    # The threshold is a locked comparator. It never becomes the primary result, even without a model.
    report["primary_method"] = MODEL if report["methods"][MODEL]["available"] else None
    if report["primary_method"] is None:
        report["case_qc_flags"].append("trained_model_not_available")
    stages.add("kidney_mesh", "ok", start)

    if annotation_path is not None:
        start = time.perf_counter()
        annotation = load_volume(annotation_path).data == 1
        report["annotation_comparison"] = {
            "note": "Li annotation masks are references, not physical truth. Development scan only. Not an input to any stage.",
            "methods": {
                name: _annotation_comparison(method, out_dir / method["label_file"], annotation)
                for name, method in report["methods"].items()
                if method.get("available")
            },
        }
        stages.add("annotation_comparison", "ok", start)

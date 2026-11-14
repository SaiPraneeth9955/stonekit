"""Kidney segmentation with TotalSegmentator, and the kidney region of interest (ROI).

The ROI is one box around both kidneys plus a margin in millimetres. Crops keep original voxels,
so a crop's result pastes back exactly onto the CT grid. Nothing here resamples.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from stonekit.seg.subprocess_guard import (
    ExternalToolError,
    GuardedResult,
    check_result,
    run_guarded,
)

# Label numbers checked against totalsegmentator.map_to_binary.class_map["total"], version 2.18.0.
KIDNEY_RIGHT = 2
KIDNEY_LEFT = 3
AORTA = 52
ROI_SUBSET = ("kidney_right", "kidney_left", "aorta")


# Longest successful run on the 164 development scans was 112.9 s (median 40 s). 600 s is over five
# times that, and caps the worst case at 10 minutes instead of forever.
DEFAULT_TIMEOUT_SECONDS = 600


def totalsegmentator_command(ct_path: Path, output_path: Path, device: str, low_memory: bool) -> list[str]:
    binary = Path(sys.executable).parent / "TotalSegmentator"
    command = [
        str(binary), "-i", str(ct_path), "-o", str(output_path), "--ml", "--fast",
        "--roi_subset", *ROI_SUBSET, "-d", device, "-q",
    ]  # fmt: skip
    if low_memory:
        command += ["--nr_thr_resamp", "1", "--nr_thr_saving", "1", "--force_split"]
    return command


def run_totalsegmentator(
    ct_path: Path,
    output_path: Path,
    device: str = "gpu",
    low_memory: bool = False,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    log_path: Path | None = None,
) -> GuardedResult:
    """Segment kidneys and aorta with the 3 mm model. An existing output is never replaced.

    Raises ExternalToolError on timeout, a non-zero exit, a kill signal or a missing output; the
    tool and all its helpers are stopped first (see stonekit.seg.subprocess_guard). The output
    appears at `output_path` only after a checked success.

    `low_memory` is for very long thin-slice scans: one resampling and one saving thread, and the
    scan processed in three chunks. Same model and resolution, but chunk edges can differ slightly.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name[:-7]}.partial.nii.gz")
    temporary.unlink(missing_ok=True)
    if log_path is None:
        log_path = output_path.with_name(f"{output_path.name[:-7]}.totalsegmentator.log")
    command = totalsegmentator_command(ct_path, temporary, device, low_memory)
    result = run_guarded(command, log_path, timeout_seconds)
    try:
        check_result("TotalSegmentator", result, temporary, timeout_seconds)
    except ExternalToolError:
        temporary.unlink(missing_ok=True)
        raise
    temporary.rename(output_path)
    return result


def slice_axis(affine: np.ndarray) -> int:
    """The file's array axis that runs head-to-foot (largest superior-inferior component)."""
    return int(np.argmax(np.abs(affine[2, :3])))


def write_z_resampled(ct_path: Path, output_path: Path, target_mm: float) -> dict:
    """Copy of the CT with its slices linearly resampled to `target_mm`, built one slab at a time.

    Only the head-to-foot axis changes; in-plane voxels are untouched. Used solely as the input to
    TotalSegmentator's 3 mm model when the full-resolution scan does not fit in memory.
    """
    image = nib.load(ct_path)
    axis = slice_axis(image.affine)
    spacing = float(np.linalg.norm(image.affine[:3, axis]))
    if target_mm <= spacing:
        raise ValueError(f"Target {target_mm} mm is not coarser than the slice spacing {spacing:.3f} mm")
    n = image.shape[axis]
    positions = np.arange(0, (n - 1) * spacing + 1e-6, target_mm) / spacing  # in old slice-index units
    shape = list(image.shape)
    shape[axis] = len(positions)
    out = np.empty(shape, dtype=np.float32)
    # The stored integers are read once (no float copy of the whole scan); scaling is applied per slab.
    raw = image.dataobj.get_unscaled()
    slope = float(image.dataobj.slope) if np.isfinite(image.dataobj.slope) else 1.0
    intercept = float(image.dataobj.inter) if np.isfinite(image.dataobj.inter) else 0.0
    def plane(index: int) -> np.ndarray:
        # Basic indexing: np.take along the last axis of a Fortran-ordered array took ~9 s per slice.
        take = [slice(None)] * 3
        take[axis] = index
        return raw[tuple(take)].astype(np.float32) * slope + intercept

    for k, position in enumerate(positions):
        low = int(np.floor(position))
        high = min(low + 1, n - 1)
        weight = position - low
        a = plane(low)
        b = plane(high) if weight else a
        put = [slice(None)] * 3
        put[axis] = k
        out[tuple(put)] = (1 - weight) * a + weight * b
    affine = image.affine.copy()
    affine[:3, axis] *= target_mm / spacing
    nib.save(nib.Nifti1Image(out, affine), output_path)
    return {"axis": axis, "original_slices": n, "resampled_slices": len(positions), "original_spacing_mm": spacing,
            "resampled_spacing_mm": target_mm}  # fmt: skip


def map_labels_back(labels_path: Path, ct_path: Path, info: dict, output_path: Path) -> None:
    """Nearest-neighbour return of a resampled-grid label map to every original slice."""
    labels = np.asanyarray(nib.load(labels_path).dataobj)
    image = nib.load(ct_path)
    axis = info["axis"]
    in_plane = [n for i, n in enumerate(labels.shape) if i != axis]
    if in_plane != [n for i, n in enumerate(image.shape) if i != axis] or image.shape[axis] != info["original_slices"]:
        raise ValueError(f"Labels {labels.shape} do not fit the CT {image.shape} along the resampled axis {axis}")
    index = np.rint(np.arange(info["original_slices"]) * info["original_spacing_mm"] / info["resampled_spacing_mm"]).astype(int)
    index = np.clip(index, 0, labels.shape[axis] - 1)
    full = np.empty(image.shape, dtype=np.uint8)
    for k, source in enumerate(index):
        put = [slice(None)] * 3
        put[axis] = k
        take = [slice(None)] * 3
        take[axis] = int(source)
        full[tuple(put)] = labels[tuple(take)]
    nib.save(nib.Nifti1Image(full, image.affine), output_path)


def run_totalsegmentator_z_resampled(
    ct_path: Path,
    output_path: Path,
    device: str = "gpu",
    target_mm: float = 1.5,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    log_path: Path | None = None,
) -> tuple[GuardedResult, dict]:
    """TotalSegmentator on a slice-resampled copy, labels returned to the original grid.

    For scans too long for memory at full resolution. TotalSegmentator's --fast model works at 3 mm,
    so input slices of `target_mm` (finer than 3 mm) carry what it uses. Equivalence with the normal
    path is checked on development scans (scripts/kidney_lowmem_equivalence.py) before use.
    """
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"{output_path} already exists")
    stem = output_path.name[:-7]
    resampled = output_path.with_name(f"{stem}.zresampled_ct.nii.gz")
    labels = output_path.with_name(f"{stem}.zresampled_labels.nii.gz")
    for path in (resampled, labels):
        path.unlink(missing_ok=True)
    try:
        info = write_z_resampled(ct_path, resampled, target_mm)
        result = run_totalsegmentator(resampled, labels, device, timeout_seconds=timeout_seconds, log_path=log_path)
        map_labels_back(labels, ct_path, info, output_path)
    finally:
        resampled.unlink(missing_ok=True)
        labels.unlink(missing_ok=True)
    return result, info


def kidney_mask(labels: np.ndarray) -> np.ndarray:
    return np.isin(labels, [KIDNEY_RIGHT, KIDNEY_LEFT])


@dataclass(frozen=True)
class Box:
    """A half-open voxel box in RAS array order."""

    start: tuple[int, int, int]
    stop: tuple[int, int, int]

    @property
    def slices(self) -> tuple[slice, slice, slice]:
        return tuple(slice(a, b) for a, b in zip(self.start, self.stop, strict=True))

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(b - a for a, b in zip(self.start, self.stop, strict=True))


def mask_box(mask: np.ndarray, spacing_mm: tuple[float, ...], margin_mm: float) -> Box | None:
    """Bounding box of a mask plus a margin in mm, clipped to the image. None if the mask is empty."""
    found = np.argwhere(mask)
    if found.size == 0:
        return None
    pad = [int(np.ceil(margin_mm / s)) for s in spacing_mm]
    start = tuple(max(int(found[:, a].min()) - pad[a], 0) for a in range(3))
    stop = tuple(min(int(found[:, a].max()) + pad[a] + 1, mask.shape[a]) for a in range(3))
    return Box(start, stop)


def paste_into_full(crop: np.ndarray, box: Box, shape: tuple[int, ...]) -> np.ndarray:
    """Place a crop's result back on the full grid. Everything outside the box is zero."""
    if crop.shape != box.shape:
        raise ValueError(f"Crop shape {crop.shape} does not match box shape {box.shape}")
    full = np.zeros(shape, dtype=crop.dtype)
    full[box.slices] = crop
    return full


def within_mm(mask: np.ndarray, spacing_mm: tuple[float, ...], margin_mm: float) -> np.ndarray:
    """Voxels within a true millimetre distance of a mask. Empty if the mask is empty."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    return ndimage.distance_transform_edt(~mask, sampling=spacing_mm) <= margin_mm

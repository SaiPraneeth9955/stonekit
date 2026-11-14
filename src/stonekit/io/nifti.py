"""NIfTI load and save, with RAS orientation for internal use (Section 11.1).

Reorienting only flips and reorders axes. It never interpolates, so voxel values and physical
spacing are unchanged, and any result can be mapped exactly back to the original grid.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation, ornt_transform

RAS_ORNT = np.array([[0, 1], [1, 1], [2, 1]])


@dataclass(frozen=True)
class Volume:
    """A 3D volume in RAS array order. Keeps what is needed to return to the original grid."""

    data: np.ndarray
    affine: np.ndarray
    original_affine: np.ndarray
    original_shape: tuple[int, int, int]
    original_codes: tuple[str, str, str]
    original_header_zooms: tuple[float, float, float]
    path: Path

    @property
    def spacing_mm(self) -> tuple[float, float, float]:
        """Voxel size along the RAS array axes (x, y, z), from the affine."""
        return tuple(float(v) for v in np.linalg.norm(self.affine[:3, :3], axis=0))

    @property
    def voxel_volume_mm3(self) -> float:
        return float(abs(np.linalg.det(self.affine[:3, :3])))


def load_volume(path: Path) -> Volume:
    """Load a 3D NIfTI file into RAS array order. The file is only read."""
    path = Path(path)
    image = nib.load(path)
    if image.ndim != 3:
        raise ValueError(f"{path.name} has {image.ndim} dimensions, not 3")
    data = np.asanyarray(image.dataobj)
    affine = image.affine.copy()
    ornt = io_orientation(affine)
    return Volume(
        data=apply_orientation(data, ornt),
        affine=affine @ inv_ornt_aff(ornt, data.shape),
        original_affine=affine,
        original_shape=tuple(int(v) for v in data.shape),
        original_codes=tuple(nib.aff2axcodes(affine)),
        original_header_zooms=tuple(float(v) for v in image.header.get_zooms()[:3]),
        path=path,
    )


def to_original_grid(array: np.ndarray, volume: Volume) -> np.ndarray:
    """Map an array in the volume's RAS order exactly back to the file's array order."""
    if array.shape != volume.data.shape:
        raise ValueError(f"Array shape {array.shape} does not match {volume.data.shape}")
    transform = ornt_transform(RAS_ORNT, io_orientation(volume.original_affine))
    return apply_orientation(array, transform)


def world_to_voxel(volume: Volume, ras_mm: tuple[float, float, float]) -> tuple[int, int, int]:
    """Nearest RAS-order voxel for a position in RAS millimetres."""
    point = np.linalg.inv(volume.affine) @ np.array([*ras_mm, 1.0])
    return tuple(round(float(v)) for v in point[:3])


def save_on_original_grid(array: np.ndarray, volume: Volume, path: Path) -> None:
    """Save a RAS-order array on the file's original grid. An existing file is never replaced."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(to_original_grid(array, volume), volume.original_affine), path)

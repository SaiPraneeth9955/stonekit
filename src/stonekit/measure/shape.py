"""Marching cubes, area, sphericity (Section 11.3)."""

import math

import numpy as np
from skimage.measure import marching_cubes, mesh_surface_area

MIN_VOXELS_FOR_SPHERICITY = 100


def surface_area_mm2(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    padded = np.pad(mask.astype(float), 1)
    vertices, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing_mm)
    return float(mesh_surface_area(vertices, faces))


def sphericity(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float | None:
    """Return pi^(1/3) (6V)^(2/3) / A, or None below the minimum voxel count."""
    if mask.sum() < MIN_VOXELS_FOR_SPHERICITY:
        return None
    volume = float(mask.sum()) * float(np.prod(spacing_mm))
    return math.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface_area_mm2(mask, spacing_mm)


def volume_equivalent_diameter_mm(volume_mm3: float) -> float:
    """Diameter of the sphere with the same volume: (6V/pi)^(1/3)."""
    return (6.0 * volume_mm3 / math.pi) ** (1.0 / 3.0)


def principal_axes(mask: np.ndarray, spacing_mm: tuple[float, float, float], z_axis: int = 2) -> dict:
    """Moment-based principal axes of a voxel mask (experiment E3, 2D/3D study).

    Each voxel is treated as a uniform box, so its own variance (spacing^2 / 12 per axis) is added
    to the covariance of the voxel centres; this keeps one-voxel-thick stones from getting zero width.
    A solid ellipsoid with semi-axis a has variance a^2 / 5 along that axis, so each full axis length
    is reported as 2 * sqrt(5 * eigenvalue): exact for ellipsoids, a moment-equivalent length otherwise.

    Returns the three lengths (major >= middle >= minor, mm), elongation = minor / major (1 = round,
    near 0 = needle-like), flatness = minor / middle, and the angle in degrees (0-90) between the
    major axis and the scanner z axis (0 = along z, through the slices; 90 = in the axial plane).
    """
    spacing = np.asarray(spacing_mm, dtype=float)
    points = np.argwhere(mask) * spacing
    if len(points) == 0:
        return {"axis_major_mm": 0.0, "axis_middle_mm": 0.0, "axis_minor_mm": 0.0, "elongation": None, "flatness": None, "major_axis_angle_to_z_deg": None}  # fmt: skip
    centred = points - points.mean(axis=0)
    covariance = centred.T @ centred / len(points) + np.diag(spacing**2 / 12.0)
    values, vectors = np.linalg.eigh(covariance)  # ascending
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    lengths = 2.0 * np.sqrt(5.0 * np.clip(values, 0.0, None))
    angle = math.degrees(math.acos(min(1.0, abs(float(vectors[z_axis, 0])))))
    return {
        "axis_major_mm": float(lengths[0]),
        "axis_middle_mm": float(lengths[1]),
        "axis_minor_mm": float(lengths[2]),
        "elongation": float(lengths[2] / lengths[0]) if lengths[0] > 0 else None,
        "flatness": float(lengths[2] / lengths[1]) if lengths[1] > 0 else None,
        "major_axis_angle_to_z_deg": angle,
    }


def max_axial_area_mm2(mask: np.ndarray, spacing_mm: tuple[float, float, float], z_axis: int = 2) -> float:
    """Largest cross-section area in one axial slice (voxel count x in-plane pixel area), mm^2."""
    inplane = [a for a in range(3) if a != z_axis]
    pixel_area = float(spacing_mm[inplane[0]] * spacing_mm[inplane[1]])
    counts = mask.sum(axis=tuple(inplane))
    return float(counts.max()) * pixel_area if counts.size else 0.0

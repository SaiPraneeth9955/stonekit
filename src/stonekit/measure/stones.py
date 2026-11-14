"""Separate a stone mask into individual stones and measure each one (Sections 11.2, 11.3).

Same rules and measures as the Freiburg code: 26-connectivity, a minimum size, holes filled per
component, then the existing volume, diameter, shape and density functions. "Stone" here means one
connected piece of a mask. Touching stones merge, and one stone can split. Dropped pieces are counted,
never hidden. Everything is on the original voxel grid, in RAS array order, with no resampling.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

from stonekit.measure.components import CONNECTIVITY_26, DEFAULT_MIN_VOXELS
from stonekit.measure.density import hu_statistics
from stonekit.measure.diameter import axial_max_distance_mm, max_distance_mm
from stonekit.measure.shape import (
    max_axial_area_mm2,
    principal_axes,
    sphericity,
    volume_equivalent_diameter_mm,
)
from stonekit.measure.volume import MM3_PER_ML, volume_mm3

Z_AXIS_RAS = 2


@dataclass(frozen=True)
class Separation:
    labels: np.ndarray
    n_stones: int
    dropped_pieces: int
    dropped_voxels: int


def separate_stones(binary: np.ndarray, min_voxels: int = DEFAULT_MIN_VOXELS) -> Separation:
    """Label connected pieces, largest first. Pieces under the minimum are dropped and counted."""
    pieces, count = ndimage.label(binary, structure=CONNECTIVITY_26)
    sizes = np.bincount(pieces.ravel(), minlength=count + 1)[1:]
    kept = sorted(
        (index for index, size in enumerate(sizes, start=1) if size >= min_voxels),
        key=lambda index: (-int(sizes[index - 1]), index),
    )
    labels = np.zeros(binary.shape, dtype=np.int32)
    locations = ndimage.find_objects(pieces)
    for stone_id, index in enumerate(kept, start=1):
        window = locations[index - 1]
        component = pieces[window] == index
        filled = ndimage.binary_fill_holes(component)
        # A hole may hold another piece. That piece keeps its own voxels.
        free = (labels[window] == 0) & ((pieces[window] == 0) | component)
        labels[window][filled & free] = stone_id
    dropped = [int(size) for size in sizes if size < min_voxels]
    return Separation(labels, len(kept), len(dropped), sum(dropped))


def _hull_fill(points_mm: np.ndarray, volume: float) -> float | None:
    """Stone volume as a fraction of its convex hull. None when the hull is flat or too small."""
    if len(points_mm) < 10:
        return None
    try:
        return float(min(volume / ConvexHull(points_mm).volume, 1.0))
    except QhullError:
        return None


def measure_stones(
    separation: Separation,
    image: np.ndarray,
    affine: np.ndarray,
    spacing_mm: tuple[float, float, float],
) -> list[dict]:
    """One dictionary per stone. Diameters are centre-to-centre, because voxels are anisotropic."""
    stones = []
    locations = ndimage.find_objects(separation.labels)
    for stone_id, window in enumerate(locations, start=1):
        if window is None:
            continue
        mask = separation.labels[window] == stone_id
        volume = volume_mm3(mask, spacing_mm)
        indices = np.argwhere(mask)
        start = np.array([s.start for s in window])
        centroid_voxel = indices.mean(axis=0) + start
        centroid_ras = (affine @ np.array([*centroid_voxel, 1.0]))[:3]
        extent_mm = [int(np.ptp(indices[:, a]) + 1) * spacing_mm[a] for a in range(3)]
        shape_value = sphericity(mask, spacing_mm)
        stones.append(
            {
                "stone_id": stone_id,
                "voxel_count": int(mask.sum()),
                "volume_mm3": volume,
                "volume_ml": volume / MM3_PER_ML,
                "max_diameter_3d_mm": max_distance_mm(mask, spacing_mm),
                "max_diameter_axial_mm": axial_max_distance_mm(mask, spacing_mm, Z_AXIS_RAS),
                "volume_equivalent_diameter_mm": volume_equivalent_diameter_mm(volume),
                "max_axial_area_mm2": max_axial_area_mm2(mask, spacing_mm, Z_AXIS_RAS),
                **principal_axes(mask, spacing_mm, Z_AXIS_RAS),
                "extent_x_mm": extent_mm[0],
                "extent_y_mm": extent_mm[1],
                "extent_z_mm": extent_mm[2],
                "slices_spanned": int(np.ptp(indices[:, Z_AXIS_RAS]) + 1),
                "centroid_ras_mm": [float(v) for v in centroid_ras],
                "voxel_start": [int(s.start) for s in window],
                "voxel_stop": [int(s.stop) for s in window],
                **hu_statistics(image[window], mask),
                "min_hu": float(np.min(image[window][mask])),
                "max_hu": float(np.max(image[window][mask])),
                "sphericity": shape_value,
                "hull_fill": _hull_fill(indices * np.asarray(spacing_mm), volume),
            }
        )
    return stones

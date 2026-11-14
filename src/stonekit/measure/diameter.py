"""3D max diameter, axial max diameter (Section 11.3)."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

AXIS_ASSIGNMENTS = {"Z2": 2, "Z1": 1, "Z0": 0}


def spacing_for_assignment(
    inplane_mm: float, z_mm: float, assignment: str
) -> tuple[float, float, float]:
    """Return per-axis spacing with `z_mm` on the axis the assignment names."""
    spacing = [inplane_mm, inplane_mm, inplane_mm]
    spacing[AXIS_ASSIGNMENTS[assignment]] = z_mm
    return tuple(spacing)


def _largest_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except QhullError:  # Too few points, or points that are flat or in a line.
        pass
    return float(pdist(points).max())


def max_distance_mm(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    """Largest centre-to-centre distance between mask voxels, in mm. Empty or single: 0."""
    return _largest_distance(np.argwhere(mask) * np.asarray(spacing_mm))


def axial_max_distance_mm(
    mask: np.ndarray, spacing_mm: tuple[float, float, float], z_axis: int
) -> float:
    """Largest in-plane centre-to-centre distance over slices along the stated z axis."""
    inplane = [axis for axis in range(3) if axis != z_axis]
    spacing = np.asarray(spacing_mm)[inplane]
    largest = 0.0
    for position in range(mask.shape[z_axis]):
        points = np.argwhere(np.take(mask, position, axis=z_axis))
        largest = max(largest, _largest_distance(points * spacing))
    return largest

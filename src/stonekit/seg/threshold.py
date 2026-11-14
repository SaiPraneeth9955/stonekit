"""HU threshold baseline. A locked comparator, not a trained model.

Freiburg patches hold one stone, so the threshold applies directly. Patient scans also hold bone
and bowel, so only voxels near a kidney count there.
"""

import numpy as np


def threshold_mask(image: np.ndarray, threshold_hu: float) -> np.ndarray:
    """A voxel is included if its value is strictly greater than the threshold. NaN is not."""
    return image > threshold_hu


def threshold_near_kidney(
    image: np.ndarray, distance_to_kidney_mm: np.ndarray, threshold_hu: float, max_distance_mm: float
) -> np.ndarray:
    """The threshold mask, limited to voxels within a distance of a kidney."""
    return threshold_mask(image, threshold_hu) & (distance_to_kidney_mm <= max_distance_mm)

"""Stone volume (Section 11.3)."""

import numpy as np

MM3_PER_ML = 1000.0


def volume_mm3(mask: np.ndarray, spacing_mm: tuple[float, float, float]) -> float:
    return float(mask.sum()) * float(np.prod(spacing_mm))


def volume_ml(mask: np.ndarray, voxel_volume_ml: float) -> float:
    return float(mask.sum()) * voxel_volume_ml

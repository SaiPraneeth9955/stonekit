"""HU statistics (Section 11.3)."""

import numpy as np


def hu_statistics(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Mean and median HU inside the mask. NaN voxels are ignored."""
    values = image[mask]
    return {"mean_hu": float(np.nanmean(values)), "median_hu": float(np.nanmedian(values))}

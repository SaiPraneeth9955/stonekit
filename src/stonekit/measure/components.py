"""Connected components, min size, per-component hole filling (Section 11.2)."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
DEFAULT_MIN_VOXELS = 3


@dataclass(frozen=True)
class StoneMask:
    mask: np.ndarray
    n_components: int
    filled_voxels: int


def stone_mask(
    binary: np.ndarray, min_voxels: int = DEFAULT_MIN_VOXELS, largest_only: bool = False
) -> StoneMask:
    """Keep components of at least `min_voxels`, fill each one's holes, and combine them."""
    labels, count = ndimage.label(binary, structure=CONNECTIVITY_26)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    kept = [label for label, size in enumerate(sizes, start=1) if size >= min_voxels]
    if largest_only and kept:
        kept = [max(kept, key=lambda label: sizes[label - 1])]

    combined = np.zeros(binary.shape, dtype=bool)
    unfilled = np.zeros(binary.shape, dtype=bool)
    for label in kept:
        component = labels == label
        unfilled |= component
        combined |= ndimage.binary_fill_holes(component)
    return StoneMask(combined, len(kept), int(combined.sum() - unfilled.sum()))

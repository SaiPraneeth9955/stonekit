"""Shared pytest fixtures.

Tests use tiny hand-made arrays and fake one-line tables only. These are code
fixtures, not datasets, and are never used for any result (Section 6).
"""

import numpy as np
import pytest


def _ellipsoid(radii_mm, spacing_mm):
    half_widths = [int(np.ceil(r / s)) + 2 for r, s in zip(radii_mm, spacing_mm)]
    axes = [np.arange(-h, h + 1) * s for h, s in zip(half_widths, spacing_mm)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return (x / radii_mm[0]) ** 2 + (y / radii_mm[1]) ** 2 + (z / radii_mm[2]) ** 2 <= 1


@pytest.fixture
def voxel_ellipsoid():
    """Return a function that builds a voxelised ellipsoid from radii and spacing in mm."""
    return _ellipsoid

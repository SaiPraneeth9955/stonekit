"""Sphere ~1.0, cube ~0.806 (tolerances in Section 11.3), and HU statistics."""

import numpy as np
import pytest

from stonekit.measure.density import hu_statistics
from stonekit.measure.shape import sphericity

SPACING = (0.5, 0.5, 0.5)


@pytest.mark.parametrize("radius", [4, 6, 10])
def test_sphere_sphericity_is_in_range(voxel_ellipsoid, radius):
    value = sphericity(voxel_ellipsoid((radius, radius, radius), SPACING), SPACING)
    assert 0.9 <= value <= 1.05


def test_cube_sphericity_is_near_0806():
    cube = np.zeros((42, 42, 42), bool)
    cube[1:41, 1:41, 1:41] = True
    assert sphericity(cube, SPACING) == pytest.approx(0.806, abs=0.05)


def test_elongated_ellipsoid_is_clearly_lower_than_sphere(voxel_ellipsoid):
    sphere = sphericity(voxel_ellipsoid((6, 6, 6), SPACING), SPACING)
    elongated = sphericity(voxel_ellipsoid((2, 2, 12), SPACING), SPACING)
    assert elongated < sphere - 0.1


def test_sphericity_is_not_reported_under_100_voxels():
    assert sphericity(np.ones((4, 4, 4), bool), SPACING) is None


def test_hu_statistics_ignore_nan():
    image = np.array([[[100.0, 200.0, np.nan, 900.0]]])
    mask = np.array([[[True, True, True, False]]])
    assert hu_statistics(image, mask) == {"mean_hu": 150.0, "median_hu": 150.0}

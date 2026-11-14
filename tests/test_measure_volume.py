"""Spheres/ellipsoids with known volume, and components (Sections 11.2 and 11.3)."""

import math

import numpy as np
import pytest

from stonekit.measure.components import stone_mask
from stonekit.measure.volume import volume_ml, volume_mm3

SPACINGS = [(0.5, 0.5, 0.5), (0.7, 0.7, 0.7), (0.7, 0.7, 2.5)]


# Section 11.3 asks for 5% at every listed radius and spacing. This one case cannot meet it: a
# 3 mm sphere centred on a 2.5 mm slice grid has only three slices, and the voxel count is 11.6%
# high whatever the code does. Kept visible, and it fails the suite if it ever starts to pass.
LATTICE_LIMIT = pytest.mark.xfail(strict=True, reason="Lattice limit, logged in docs/decisions.md")


@pytest.mark.parametrize(
    "radius, spacing",
    [
        pytest.param(
            radius,
            spacing,
            marks=[LATTICE_LIMIT] if (radius, spacing) == (3, (0.7, 0.7, 2.5)) else [],
        )
        for radius in (3, 4, 6, 8, 10)
        for spacing in SPACINGS
    ],
)
def test_sphere_volume_is_within_5_percent(voxel_ellipsoid, radius, spacing):
    mask = voxel_ellipsoid((radius, radius, radius), spacing)
    assert volume_mm3(mask, spacing) == pytest.approx(4 / 3 * math.pi * radius**3, rel=0.05)


def test_small_sphere_on_thick_slices_is_unbiased_over_grid_positions():
    spacing = np.array((0.7, 0.7, 2.5))
    rng = np.random.default_rng(0)
    volumes = []
    for _ in range(300):
        axes = [(np.arange(-12, 13) + shift) * step for shift, step in zip(rng.random(3), spacing)]
        x, y, z = np.meshgrid(*axes, indexing="ij")
        volumes.append(volume_mm3(x**2 + y**2 + z**2 <= 9, tuple(spacing)))
    assert np.mean(volumes) == pytest.approx(4 / 3 * math.pi * 27, rel=0.05)


def test_ellipsoid_volume_is_within_5_percent(voxel_ellipsoid):
    mask = voxel_ellipsoid((4, 6, 8), (0.5, 0.5, 0.5))
    assert volume_mm3(mask, (0.5, 0.5, 0.5)) == pytest.approx(4 / 3 * math.pi * 4 * 6 * 8, rel=0.05)


def test_volume_ml_uses_the_stated_voxel_volume():
    assert volume_ml(np.ones((2, 2, 2), bool), 0.001) == pytest.approx(0.008)


def test_volume_does_not_depend_on_spacing_order():
    mask = np.ones((3, 4, 5), bool)
    assert volume_mm3(mask, (0.7363, 0.7363, 0.7)) == pytest.approx(
        volume_mm3(mask, (0.7, 0.7363, 0.7363))
    )


def test_diagonal_voxels_are_one_component_with_26_connectivity():
    binary = np.zeros((5, 5, 5), bool)
    binary[1, 1, 1] = binary[2, 2, 2] = binary[3, 3, 3] = True
    assert stone_mask(binary).n_components == 1


def test_small_components_are_dropped():
    binary = np.zeros((9, 9, 9), bool)
    binary[1:3, 1:3, 1:3] = True
    binary[7, 7, 7] = True
    result = stone_mask(binary, min_voxels=3)
    assert result.n_components == 1
    assert result.mask.sum() == 8


def test_holes_are_filled_and_counted():
    binary = np.zeros((7, 7, 7), bool)
    binary[1:6, 1:6, 1:6] = True
    binary[3, 3, 3] = False
    result = stone_mask(binary)
    assert result.filled_voxels == 1
    assert result.mask[3, 3, 3]


def test_union_keeps_every_component_and_largest_only_keeps_one():
    binary = np.zeros((12, 12, 12), bool)
    binary[1:4, 1:4, 1:4] = True
    binary[8:10, 8:10, 8:10] = True
    assert stone_mask(binary).mask.sum() == 27 + 8
    assert stone_mask(binary, largest_only=True).mask.sum() == 27


def test_empty_mask_is_handled():
    result = stone_mask(np.zeros((3, 3, 3), bool))
    assert result.n_components == 0
    assert not result.mask.any()

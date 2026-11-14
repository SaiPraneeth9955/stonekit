"""E3 feature tests on geometric test objects with known answers (fixtures, not patient data)."""

import math

import numpy as np
import pytest

from stonekit.measure.shape import max_axial_area_mm2, principal_axes, volume_equivalent_diameter_mm


def ellipsoid(semi_mm, spacing_mm, shape=(81, 81, 81)):
    grids = np.meshgrid(*[(np.arange(n) - n // 2) * s for n, s in zip(shape, spacing_mm, strict=True)], indexing="ij")
    return sum((g / a) ** 2 for g, a in zip(grids, semi_mm, strict=True)) <= 1.0


def test_volume_equivalent_diameter_of_a_sphere():
    assert volume_equivalent_diameter_mm(4 / 3 * math.pi * 5.0**3) == pytest.approx(10.0)


def test_ellipsoid_axis_lengths_and_order():
    axes = principal_axes(ellipsoid((12.0, 6.0, 3.0), (0.5, 0.5, 0.5)), (0.5, 0.5, 0.5))
    assert axes["axis_major_mm"] == pytest.approx(24.0, rel=0.03)
    assert axes["axis_middle_mm"] == pytest.approx(12.0, rel=0.03)
    assert axes["axis_minor_mm"] == pytest.approx(6.0, rel=0.05)
    assert axes["elongation"] == pytest.approx(0.25, rel=0.06)


def test_angle_to_z():
    along_z = principal_axes(ellipsoid((3.0, 3.0, 10.0), (0.5, 0.5, 0.5)), (0.5, 0.5, 0.5))
    in_plane = principal_axes(ellipsoid((10.0, 3.0, 3.0), (0.5, 0.5, 0.5)), (0.5, 0.5, 0.5))
    assert along_z["major_axis_angle_to_z_deg"] == pytest.approx(0.0, abs=2.0)
    assert in_plane["major_axis_angle_to_z_deg"] == pytest.approx(90.0, abs=2.0)


def test_anisotropic_spacing_gives_the_same_lengths():
    iso = principal_axes(ellipsoid((8.0, 5.0, 4.0), (0.5, 0.5, 0.5)), (0.5, 0.5, 0.5))
    aniso = principal_axes(ellipsoid((8.0, 5.0, 4.0), (0.5, 0.5, 1.0), (81, 81, 41)), (0.5, 0.5, 1.0))
    for key in ("axis_major_mm", "axis_middle_mm", "axis_minor_mm"):
        assert aniso[key] == pytest.approx(iso[key], rel=0.05)


def test_single_voxel_has_voxel_sized_axes_and_empty_mask_is_safe():
    one = np.zeros((3, 3, 3), bool)
    one[1, 1, 1] = True
    axes = principal_axes(one, (1.0, 1.0, 2.0))
    assert axes["axis_minor_mm"] > 0 and axes["axis_major_mm"] == pytest.approx(2 * math.sqrt(5 * 4 / 12))
    assert principal_axes(np.zeros((3, 3, 3), bool), (1.0, 1.0, 1.0))["elongation"] is None


def test_max_axial_area():
    mask = np.zeros((10, 10, 4), bool)
    mask[2:6, 2:5, 1] = True  # 12 voxels in one slice
    mask[2:4, 2:4, 2] = True
    assert max_axial_area_mm2(mask, (0.5, 0.5, 2.0)) == pytest.approx(12 * 0.25)

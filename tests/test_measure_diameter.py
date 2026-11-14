"""3D and axial maximum diameter, and per-axis spacing."""

import math

import numpy as np
import pytest

from stonekit.measure.diameter import (
    axial_max_distance_mm,
    max_distance_mm,
    spacing_for_assignment,
)


@pytest.mark.parametrize("spacing", [(0.5, 0.5, 0.5), (0.7, 0.7, 0.7), (0.7, 0.7, 2.5)])
@pytest.mark.parametrize("radius", [2, 3, 5, 10])
def test_sphere_diameter_is_within_one_voxel(voxel_ellipsoid, spacing, radius):
    mask = voxel_ellipsoid((radius, radius, radius), spacing)
    assert abs(max_distance_mm(mask, spacing) - 2 * radius) <= max(spacing)


@pytest.mark.parametrize("axis, expected", [(0, 2.8), (1, 3.5), (2, 4.2)])
def test_each_axis_uses_its_own_spacing(axis, expected):
    mask = np.zeros((9, 9, 9), bool)
    index = [4, 4, 4]
    for position in (2, 6):
        index[axis] = position
        mask[tuple(index)] = True
    assert max_distance_mm(mask, (0.7, 0.875, 1.05)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "assignment, expected",
    [
        ("Z2", (0.7363, 0.7363, 0.7)),
        ("Z1", (0.7363, 0.7, 0.7363)),
        ("Z0", (0.7, 0.7363, 0.7363)),
    ],
)
def test_axis_assignment_places_z_spacing(assignment, expected):
    assert spacing_for_assignment(0.7363, 0.7, assignment) == expected


def test_axis_assignment_changes_diameter_along_the_named_axis():
    mask = np.zeros((11, 3, 3), bool)
    mask[0, 1, 1] = mask[10, 1, 1] = True
    assert max_distance_mm(mask, spacing_for_assignment(0.7363, 0.7, "Z2")) == pytest.approx(7.363)
    assert max_distance_mm(mask, spacing_for_assignment(0.7363, 0.7, "Z0")) == pytest.approx(7.0)


def test_empty_and_single_voxel_masks_have_zero_distance():
    mask = np.zeros((3, 3, 3), bool)
    assert max_distance_mm(mask, (1, 1, 1)) == 0.0
    mask[1, 1, 1] = True
    assert max_distance_mm(mask, (1, 1, 1)) == 0.0


def test_flat_and_line_masks_fall_back_to_pairwise_distances():
    line = np.zeros((5, 5, 5), bool)
    line[0:5, 2, 2] = True
    assert max_distance_mm(line, (0.5, 1, 1)) == pytest.approx(2.0)
    flat = np.zeros((5, 5, 5), bool)
    flat[0:4, 0:4, 2] = True
    assert max_distance_mm(flat, (1, 1, 1)) == pytest.approx(math.hypot(3, 3))


def test_hull_result_matches_all_pairwise_distances(voxel_ellipsoid):
    from scipy.spatial.distance import pdist

    spacing = (0.7, 0.7, 0.9)
    mask = voxel_ellipsoid((3, 2, 4), spacing)
    expected = pdist(np.argwhere(mask) * np.array(spacing)).max()
    assert max_distance_mm(mask, spacing) == pytest.approx(expected)


def test_axial_diameter_uses_the_stated_z_axis():
    mask = np.zeros((9, 3, 5), bool)
    mask[0:9, 1, 2] = True
    assert axial_max_distance_mm(mask, (0.5, 1, 1), z_axis=2) == pytest.approx(4.0)
    assert axial_max_distance_mm(mask, (0.5, 1, 1), z_axis=0) == 0.0

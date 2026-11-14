"""Stone separation, physical measurements and meshes on shapes with known geometry.

Hand-made arrays only: code fixtures, not data.
"""

import math

import numpy as np
import pytest

from stonekit.measure import mesh, stones

SPACING = (0.5, 0.7, 2.5)
AFFINE = np.diag([*SPACING, 1.0])


def ball(shape, centre, radius_mm, spacing=SPACING):
    grids = np.meshgrid(*[np.arange(n) * s for n, s in zip(shape, spacing, strict=True)], indexing="ij")
    return sum((g - c * s) ** 2 for g, c, s in zip(grids, centre, spacing, strict=True)) <= radius_mm**2


def test_volume_is_in_physical_units():
    mask = ball((100, 100, 40), (50, 50, 20), 10)
    result = stones.measure_stones(stones.separate_stones(mask), np.ones(mask.shape), AFFINE, SPACING)[0]
    expected = 4 / 3 * math.pi * 10**3
    assert result["volume_mm3"] == pytest.approx(expected, rel=0.05)
    assert result["volume_ml"] == pytest.approx(result["volume_mm3"] / 1000)
    assert result["voxel_count"] * 0.5 * 0.7 * 2.5 == pytest.approx(result["volume_mm3"])


def test_3d_diameter_exceeds_axial_for_a_tall_shape():
    mask = np.zeros((10, 10, 40), dtype=bool)
    mask[5, 5, 5:25] = True
    mask[4:7, 5, 15] = True
    result = stones.measure_stones(stones.separate_stones(mask), np.ones(mask.shape), AFFINE, SPACING)[0]
    assert result["max_diameter_axial_mm"] == pytest.approx(1.0)
    assert result["max_diameter_3d_mm"] == pytest.approx(19 * 2.5)
    assert result["extent_z_mm"] == pytest.approx(20 * 2.5)
    assert result["slices_spanned"] == 20


def test_diagonal_diameter_uses_each_axis_spacing():
    mask = np.zeros((6, 6, 6), dtype=bool)
    for i in range(4):
        mask[i, i, i] = True
    result = stones.measure_stones(stones.separate_stones(mask), np.ones(mask.shape), AFFINE, SPACING)[0]
    assert result["max_diameter_3d_mm"] == pytest.approx(3 * math.sqrt(0.5**2 + 0.7**2 + 2.5**2))


def test_two_pieces_are_separate_stones_largest_first():
    mask = np.zeros((30, 30, 30), dtype=bool)
    mask[2:4, 2:4, 2:4] = True
    mask[20:25, 20:25, 20:25] = True
    separation = stones.separate_stones(mask)
    assert separation.n_stones == 2
    measured = stones.measure_stones(separation, np.ones(mask.shape), AFFINE, SPACING)
    assert [s["voxel_count"] for s in measured] == [125, 8]
    assert [s["stone_id"] for s in measured] == [1, 2]


def test_diagonal_touch_is_one_stone():
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    mask[3:5, 3:5, 3:5] = True
    assert stones.separate_stones(mask).n_stones == 1


def test_small_pieces_are_counted_not_hidden():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[1:5, 1:5, 1:5] = True
    mask[10, 10, 10] = True
    mask[15, 15, 15:17] = True
    separation = stones.separate_stones(mask, min_voxels=3)
    assert separation.n_stones == 1
    assert separation.dropped_pieces == 2
    assert separation.dropped_voxels == 3


def test_no_stones():
    mask = np.zeros((5, 5, 5), dtype=bool)
    separation = stones.separate_stones(mask)
    assert separation.n_stones == 0
    assert stones.measure_stones(separation, np.ones(mask.shape), AFFINE, SPACING) == []


def test_holes_are_filled():
    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    mask[4, 4, 4] = False
    result = stones.measure_stones(stones.separate_stones(mask), np.ones(mask.shape), AFFINE, SPACING)[0]
    assert result["voxel_count"] == 125


def test_piece_inside_a_hole_stays_its_own_stone():
    mask = np.zeros((11, 11, 11), dtype=bool)
    mask[1:10, 1:10, 1:10] = True
    mask[3:8, 3:8, 3:8] = False
    mask[4:7, 4:7, 4:7] = True
    separation = stones.separate_stones(mask)
    assert separation.n_stones == 2
    assert (separation.labels == 2).sum() == 27


def test_hu_statistics_and_centroid_use_the_right_voxels():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:4, 2:4, 2:4] = True
    image = np.full(mask.shape, -100.0)
    image[mask] = 500.0
    image[2, 2, 2] = 700.0
    affine = np.diag([-0.5, 0.7, 2.5, 1.0])
    affine[:3, 3] = [10.0, 20.0, 30.0]
    result = stones.measure_stones(stones.separate_stones(mask), image, affine, SPACING)[0]
    assert result["min_hu"] == 500
    assert result["max_hu"] == 700
    assert result["median_hu"] == 500
    assert result["mean_hu"] == pytest.approx(525)
    assert result["centroid_ras_mm"] == pytest.approx([10 - 2.5 * 0.5, 20 + 2.5 * 0.7, 30 + 2.5 * 2.5])


def test_mesh_is_in_world_position_and_encloses_the_mask_volume():
    mask = ball((100, 100, 40), (50, 50, 20), 10)
    affine = AFFINE.copy()
    affine[:3, 3] = [100.0, -50.0, 7.0]
    vertices, faces = mesh.mask_mesh(mask, affine)
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    assert centre == pytest.approx([100 + 25, -50 + 35, 7 + 50], abs=0.5)
    assert mesh.mesh_volume_mm3(vertices, faces) == pytest.approx(mask.sum() * 0.5 * 0.7 * 2.5, rel=0.05)


def test_obj_round_trip_and_no_overwrite(tmp_path):
    mask = ball((30, 30, 12), (15, 15, 6), 4)
    vertices, faces = mesh.mask_mesh(mask, AFFINE)
    path = tmp_path / "stone.obj"
    mesh.write_obj(path, vertices, faces, "stone_1")
    back_vertices, back_faces = mesh.read_obj(path)
    np.testing.assert_allclose(back_vertices, vertices, atol=1e-3)
    np.testing.assert_array_equal(back_faces, faces)
    with pytest.raises(FileExistsError):
        mesh.write_obj(path, vertices, faces, "stone_1")


def test_mesh_refuses_empty_mask():
    with pytest.raises(ValueError):
        mesh.mask_mesh(np.zeros((3, 3, 3), dtype=bool), AFFINE)

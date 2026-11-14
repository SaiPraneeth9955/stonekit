"""NIfTI loading, RAS orientation and exact mapping back. Tiny hand-made arrays only."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from stonekit.io import nifti

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AFFINES = {
    "RAS": np.diag([0.5, 0.7, 2.5, 1.0]),
    "LPS": np.diag([-0.5, -0.7, 2.5, 1.0]),
    "LAI": np.diag([-0.5, 0.7, -2.5, 1.0]),
    "permuted": np.array(
        [[0, 0, 2.5, 0], [-0.5, 0, 0, 0], [0, 0.7, 0, 0], [0, 0, 0, 1.0]], dtype=float
    ),
}


def write_file(path, affine):
    data = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    nib.save(nib.Nifti1Image(data, affine), path)
    return data


@pytest.mark.parametrize("name", AFFINES)
def test_each_voxel_keeps_its_world_position(tmp_path, name):
    path = tmp_path / "scan.nii.gz"
    data = write_file(path, AFFINES[name])
    volume = nifti.load_volume(path)
    assert nib.aff2axcodes(volume.affine) == ("R", "A", "S")
    for index in [(0, 0, 0), (3, 4, 5), (1, 2, 3)]:
        world = (AFFINES[name] @ np.array([*index, 1.0]))[:3]
        assert volume.data[nifti.world_to_voxel(volume, tuple(world))] == data[index]


@pytest.mark.parametrize("name", AFFINES)
def test_mapping_back_is_exact(tmp_path, name):
    path = tmp_path / "scan.nii.gz"
    data = write_file(path, AFFINES[name])
    volume = nifti.load_volume(path)
    np.testing.assert_array_equal(nifti.to_original_grid(volume.data, volume), data)


def test_spacing_follows_ras_axes(tmp_path):
    path = tmp_path / "scan.nii.gz"
    write_file(path, AFFINES["permuted"])
    volume = nifti.load_volume(path)
    # File axis 2 runs along R at 2.5 mm, so RAS x is 2.5 mm and z (from file axis 1) is 0.7 mm.
    assert volume.spacing_mm == pytest.approx((2.5, 0.5, 0.7))
    assert volume.original_shape == (4, 5, 6)
    assert volume.data.shape == (6, 4, 5)
    assert volume.voxel_volume_mm3 == pytest.approx(0.5 * 0.7 * 2.5)


def test_saved_mask_matches_original_grid_and_refuses_overwrite(tmp_path):
    path = tmp_path / "scan.nii.gz"
    write_file(path, AFFINES["LPS"])
    volume = nifti.load_volume(path)
    mask = np.zeros(volume.data.shape, dtype=np.uint8)
    mask[0, 1, 2] = 1
    out = tmp_path / "out" / "mask.nii.gz"
    nifti.save_on_original_grid(mask, volume, out)
    saved = nib.load(out)
    np.testing.assert_allclose(saved.affine, AFFINES["LPS"])
    assert saved.shape == volume.original_shape
    assert nifti.load_volume(out).data[0, 1, 2] == 1
    with pytest.raises(FileExistsError):
        nifti.save_on_original_grid(mask, volume, out)


def test_to_original_grid_refuses_wrong_shape(tmp_path):
    path = tmp_path / "scan.nii.gz"
    write_file(path, AFFINES["LPS"])
    with pytest.raises(ValueError):
        nifti.to_original_grid(np.zeros((1, 1, 1)), nifti.load_volume(path))


def test_refuses_four_dimensions(tmp_path):
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 2), dtype=np.float32), np.eye(4)), path)
    with pytest.raises(ValueError):
        nifti.load_volume(path)


# Values the owner read in 3D Slicer at these positions, in RAS millimetres, on 5 November 2026.
# Slicer is an independent reader, so this checks orientation against a separate tool.
SLICER_READINGS = [
    ("1", "wd0045", (86.2, -11.5, -212.8), 173.0, 1),
    ("3", "wd0149", (-48.1, -9.9, -274.6), 338.0, 0),
    ("5", "wd0215", (-53.9, 3.0, -210.3), 1151.0, 1),
    ("4", "wd0173", (30.1, 2.7, -109.4), 598.0, 0),
]


@pytest.mark.parametrize(("part", "case", "position", "hu", "label"), SLICER_READINGS)
def test_loader_agrees_with_slicer(part, case, position, hu, label):
    folder = PROJECT_ROOT / "data" / "interim" / f"li2022_part{part}"
    image_path = folder / f"{case}_ps_image.nii.gz"
    if not image_path.exists():
        pytest.skip("Li files are not present")
    image = nifti.load_volume(image_path)
    mask = nifti.load_volume(folder / f"{case}_ps_mask.nii.gz")
    voxel = nifti.world_to_voxel(image, position)
    assert image.data[voxel] == pytest.approx(hu)
    assert mask.data[voxel] == label

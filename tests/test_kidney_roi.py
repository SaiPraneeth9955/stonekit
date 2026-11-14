"""Kidney ROI geometry. Tiny hand-made arrays only."""

import numpy as np
import pytest

from stonekit.seg import kidney_roi


def test_kidney_mask_uses_both_kidney_labels_only():
    labels = np.array([0, kidney_roi.KIDNEY_RIGHT, kidney_roi.KIDNEY_LEFT, kidney_roi.AORTA])
    assert kidney_roi.kidney_mask(labels).tolist() == [False, True, True, False]


def test_totalsegmentator_label_numbers_match_the_installed_version():
    class_map = pytest.importorskip("totalsegmentator.map_to_binary").class_map["total"]
    assert class_map[kidney_roi.KIDNEY_RIGHT] == "kidney_right"
    assert class_map[kidney_roi.KIDNEY_LEFT] == "kidney_left"
    assert class_map[kidney_roi.AORTA] == "aorta"


def test_box_margin_is_in_mm_on_anisotropic_spacing():
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[20, 20, 20] = True
    box = kidney_roi.mask_box(mask, (0.5, 1.0, 2.5), 5)
    assert box.start == (10, 15, 18)
    assert box.stop == (31, 26, 23)
    assert box.shape == (21, 11, 5)


def test_box_is_clipped_to_the_image():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0, 9, 5] = True
    box = kidney_roi.mask_box(mask, (1.0, 1.0, 1.0), 3)
    assert box.start == (0, 6, 2)
    assert box.stop == (4, 10, 9)


def test_empty_mask_has_no_box():
    assert kidney_roi.mask_box(np.zeros((4, 4, 4), dtype=bool), (1, 1, 1), 5) is None


def test_crop_and_paste_returns_every_voxel_to_its_place():
    full = np.random.default_rng(0).random((20, 30, 40)) > 0.5
    box = kidney_roi.Box((2, 3, 4), (12, 23, 34))
    pasted = kidney_roi.paste_into_full(full[box.slices], box, full.shape)
    assert pasted.shape == full.shape
    np.testing.assert_array_equal(pasted[box.slices], full[box.slices])
    assert pasted.sum() == full[box.slices].sum()


def test_paste_refuses_wrong_shape():
    with pytest.raises(ValueError):
        kidney_roi.paste_into_full(np.zeros((2, 2, 2)), kidney_roi.Box((0, 0, 0), (3, 3, 3)), (5, 5, 5))


def test_within_mm_uses_physical_distance():
    mask = np.zeros((1, 1, 21), dtype=bool)
    mask[0, 0, 10] = True
    near = kidney_roi.within_mm(mask, (1.0, 1.0, 2.5), 5.0)
    assert near[0, 0].nonzero()[0].tolist() == [8, 9, 10, 11, 12]


def test_within_mm_on_empty_mask_is_empty():
    assert not kidney_roi.within_mm(np.zeros((3, 3, 3), dtype=bool), (1, 1, 1), 5).any()


def test_slice_axis_follows_the_head_foot_direction():
    assert kidney_roi.slice_axis(np.diag([-0.7, -0.7, 0.5, 1.0])) == 2
    permuted = np.array([[0, 0, -0.7, 0], [0, -0.7, 0, 0], [0.5, 0, 0, 0], [0, 0, 0, 1.0]])
    assert kidney_roi.slice_axis(permuted) == 0


def test_z_resampling_interpolates_slices_only_and_keeps_geometry(tmp_path):
    import nibabel as nib

    # A ramp of 10 HU per slice on 0.5 mm slices: resampled to 1.5 mm, slice k must read 30 * k.
    data = np.broadcast_to(np.arange(13, dtype=np.int16) * 10, (4, 5, 13)).copy()
    affine = np.diag([-0.7, -0.7, 0.5, 1.0])
    affine[:3, 3] = [10, 20, -30]
    nib.save(nib.Nifti1Image(data, affine), tmp_path / "ct.nii.gz")
    info = kidney_roi.write_z_resampled(tmp_path / "ct.nii.gz", tmp_path / "z.nii.gz", 1.5)
    out = nib.load(tmp_path / "z.nii.gz")
    assert info["resampled_slices"] == 5 and out.shape == (4, 5, 5)
    np.testing.assert_allclose(np.asanyarray(out.dataobj)[0, 0], [0, 30, 60, 90, 120])
    np.testing.assert_allclose(out.affine[:3, 2], [0, 0, 1.5])
    np.testing.assert_allclose(out.affine[:3, :2], affine[:3, :2])
    np.testing.assert_allclose(out.affine[:3, 3], affine[:3, 3])
    # Every original slice gets the label of the nearest resampled slice, on the original grid.
    labels = np.zeros((4, 5, 5), np.uint8)
    labels[..., 2] = 2  # resampled slice 2 lies at 3.0 mm, original slices 5, 6 and 7 (2.5, 3.0, 3.5 mm)
    nib.save(nib.Nifti1Image(labels, out.affine), tmp_path / "labels_z.nii.gz")
    kidney_roi.map_labels_back(tmp_path / "labels_z.nii.gz", tmp_path / "ct.nii.gz", info, tmp_path / "labels.nii.gz")
    back = nib.load(tmp_path / "labels.nii.gz")
    np.testing.assert_allclose(back.affine, affine)
    assert np.nonzero(np.asanyarray(back.dataobj)[0, 0])[0].tolist() == [5, 6, 7]


def test_z_resampling_refuses_a_finer_target(tmp_path):
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.zeros((2, 2, 4), np.int16), np.diag([1, 1, 2.5, 1])), tmp_path / "ct.nii.gz")
    with pytest.raises(ValueError, match="not coarser"):
        kidney_roi.write_z_resampled(tmp_path / "ct.nii.gz", tmp_path / "z.nii.gz", 1.5)

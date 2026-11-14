"""Input and geometry checks. Tiny hand-made arrays only; they are test fixtures, not data."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from stonekit.config import load_yaml
from stonekit.io.nifti import load_volume
from stonekit.pipeline.input_checks import FAIL, PASS, WARN, check_ct

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIMITS = load_yaml(PROJECT_ROOT / "configs" / "baseline_pipeline.yaml")["input_checks"]
GOOD_AFFINE = np.diag([-0.7, -0.7, 2.5, 1.0])


def make_ct(tmp_path, affine=GOOD_AFFINE, shape=(8, 8, 24), low=-1000.0, high=1500.0):
    data = np.full(shape, 40.0, dtype=np.float32)
    data[0, 0, 0] = low
    data[1, 1, 1] = high
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), path)
    return load_volume(path)


def status(report, name):
    return next(check.status for check in report.checks if check.name == name)


def test_good_scan_passes_everything(tmp_path):
    report = check_ct(make_ct(tmp_path), LIMITS)
    assert report.can_process
    assert {check.status for check in report.checks} == {PASS}
    assert report.original_orientation == "LPS"
    assert report.spacing_mm == pytest.approx((0.7, 0.7, 2.5))


def test_nan_fails(tmp_path):
    volume = make_ct(tmp_path)
    data = volume.data.copy()
    data[2, 2, 2] = np.nan
    path = tmp_path / "nan.nii.gz"
    nib.save(nib.Nifti1Image(data, volume.affine), path)
    report = check_ct(load_volume(path), LIMITS)
    assert status(report, "values_finite") == FAIL
    assert not report.can_process


def test_too_few_slices_fails(tmp_path):
    report = check_ct(make_ct(tmp_path, shape=(8, 8, 5)), LIMITS)
    assert status(report, "enough_slices") == FAIL
    assert not report.can_process


def test_sheared_axes_fail(tmp_path):
    affine = GOOD_AFFINE.copy()
    affine[0, 1] = 0.3
    report = check_ct(make_ct(tmp_path, affine=affine), LIMITS)
    assert status(report, "axes_orthogonal") == FAIL


def test_tilted_axes_warn_but_can_process(tmp_path):
    angle = np.radians(5)
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.7, 0.7, 2.5])
    report = check_ct(make_ct(tmp_path, affine=affine), LIMITS)
    assert status(report, "axes_aligned") == WARN
    assert report.can_process


def test_unusual_spacing_warns(tmp_path):
    report = check_ct(make_ct(tmp_path, affine=np.diag([2.0, 2.0, 5.0, 1.0])), LIMITS)
    assert status(report, "in_plane_spacing_in_range") == WARN
    assert status(report, "slice_spacing_in_range") == WARN
    assert report.can_process


def test_values_that_do_not_look_like_hu_warn(tmp_path):
    report = check_ct(make_ct(tmp_path, low=0.0, high=1.0), LIMITS)
    assert status(report, "values_look_like_hu") == WARN


def test_padding_is_reported_and_values_are_not_changed(tmp_path):
    volume = make_ct(tmp_path, low=-3024.0)
    before = volume.data.copy()
    report = check_ct(volume, LIMITS)
    assert report.padding_fraction == pytest.approx(1 / volume.data.size)
    np.testing.assert_array_equal(volume.data, before)


def test_report_converts_to_plain_values(tmp_path):
    result = check_ct(make_ct(tmp_path), LIMITS).to_dict()
    assert result["can_process"] is True
    assert result["checks"][0]["name"] == "axes_orthogonal"

"""The guarded Li loader. A fake project for the guard, plus a few real development scans."""

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import yaml

from stonekit.data import li2022, li_split
from stonekit.io.nifti import load_volume

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AFFINE = np.diag([-0.7, -0.7, 2.5, 1.0])


def write_pair(folder, case, labels=(0, 1, 2), mask_shape=(4, 4, 4), mask_type=np.int16):
    folder.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), AFFINE),
        folder / f"{case}_ps_image.nii.gz",
    )
    mask = np.zeros(mask_shape, dtype=mask_type)
    for i, label in enumerate(labels):
        mask[i, 0, 0] = label
    nib.save(nib.Nifti1Image(mask, AFFINE), folder / f"{case}_ps_mask.nii.gz")


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs").mkdir()
    manifest = tmp_path / "data" / "processed" / "split.csv"
    rows = [
        {"case": "dev1", "part": "1", "spacing_mm": "2.5", "spacing_group": "2.5", "volume_group": "none", "assignment": "development", "status": li_split.STATUS},
        {"case": "dev2", "part": "1", "spacing_mm": "2.5", "spacing_group": "2.5", "volume_group": "none", "assignment": "development", "status": li_split.STATUS},
        {"case": "cal1", "part": "1", "spacing_mm": "2.5", "spacing_group": "2.5", "volume_group": "none", "assignment": "reserved_calibration", "status": li_split.STATUS},
        {"case": "test1", "part": "1", "spacing_mm": "2.5", "spacing_group": "2.5", "volume_group": "none", "assignment": "reserved_test", "status": li_split.STATUS},
    ]
    li_split.write_manifest(rows, manifest, {})
    sheet = tmp_path / "sheet.csv"
    sheet.write_text("case,finding_code\ndev2,\n", encoding="utf-8")
    config = {"manifest": "data/processed/split.csv", "review_sheet": "sheet.csv"}
    (tmp_path / "configs" / "li_working_split.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    folder = tmp_path / "data" / "interim" / "li2022_part1"
    for case in ("dev1", "dev2", "cal1", "test1"):
        write_pair(folder, case)
    return tmp_path


def test_loads_development_scan(project):
    case = li2022.LiDevelopmentData(project).load("dev1")
    assert case.image.data.shape == (4, 4, 4)
    assert case.stone_reference.sum() == 1
    assert case.kidney_reference.sum() == 1


@pytest.mark.parametrize("case", ["cal1", "test1", "never_listed"])
def test_refuses_reserved_and_unknown_scans(project, case):
    data = li2022.LiDevelopmentData(project)
    with pytest.raises(li_split.ProtectedCaseError):
        data.load(case)
    with pytest.raises(li_split.ProtectedCaseError):
        data.paths(case)


def test_id_lists_hold_only_allowed_scans(project):
    data = li2022.LiDevelopmentData(project)
    assert data.development_ids() == ["dev1", "dev2"]
    assert data.training_ids() == ["dev1"]


def test_changed_manifest_is_refused(project):
    with (project / "data" / "processed" / "split.csv").open("a", encoding="utf-8") as f:
        f.write("test1,1,2.5,2.5,none,development,x\n")
    with pytest.raises(ValueError):
        li2022.LiDevelopmentData(project)


def _pair(tmp_path, **kwargs):
    write_pair(tmp_path, "c", **kwargs)
    return load_volume(tmp_path / "c_ps_image.nii.gz"), load_volume(tmp_path / "c_ps_mask.nii.gz")


def test_validate_refuses_shape_mismatch(tmp_path):
    with pytest.raises(li2022.LiDataError):
        li2022.validate_pair(*_pair(tmp_path, mask_shape=(4, 4, 5)))


def test_validate_refuses_unknown_label(tmp_path):
    with pytest.raises(li2022.LiDataError):
        li2022.validate_pair(*_pair(tmp_path, labels=(0, 1, 3)))


def test_validate_refuses_non_integer_mask(tmp_path):
    with pytest.raises(li2022.LiDataError):
        li2022.validate_pair(*_pair(tmp_path, mask_type=np.float32))


def test_files_are_not_changed_by_loading(project):
    data = li2022.LiDevelopmentData(project)
    paths = data.paths("dev1")
    before = [li_split.sha256_file(path) for path in paths]
    data.load("dev1")
    assert before == [li_split.sha256_file(path) for path in paths]


def _real_cases():
    manifest = PROJECT_ROOT / "data" / "processed" / "li2022_working_split_v1.csv"
    if not manifest.exists():
        return []
    data = li2022.LiDevelopmentData(PROJECT_ROOT)
    allowed = set(data.training_ids())
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["case"] in allowed]
    # One scan from each of three slice-spacing groups, the smallest on disk, for speed.
    chosen = []
    for group in ("1.25", "2.0", "2.5"):
        members = [row["case"] for row in rows if row["spacing_group"] == group]
        chosen.append(min(members, key=lambda c: data.paths(c)[0].stat().st_size))
    return chosen


@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_scan_matches_the_audit(index):
    cases = _real_cases()
    if not cases:
        pytest.skip("Working split is not present")
    case_id = cases[index]
    data = li2022.LiDevelopmentData(PROJECT_ROOT)
    case = data.load(case_id)
    table = PROJECT_ROOT / "data" / "interim" / "li2022_full_audit" / "li_full_cases.csv"
    with table.open(newline="", encoding="utf-8") as f:
        audit = next(row for row in csv.DictReader(f) if row["case"] == case_id)
    assert case.image.spacing_mm[2] == pytest.approx(float(audit["dz"]), abs=1e-3)
    assert case.image.data.shape[2] == int(audit["slices"])
    voxel_ml = case.image.voxel_volume_mm3 / 1000
    assert case.stone_reference.sum() * voxel_ml == pytest.approx(float(audit["stone_ml"]), abs=1e-3)
    assert case.kidney_reference.sum() * voxel_ml == pytest.approx(float(audit["kidney_ml"]), abs=1e-3)
    assert case.image.data.min() == pytest.approx(float(audit["hu_min"]), abs=0.5)
    assert case.image.data.max() == pytest.approx(float(audit["hu_max"]), abs=0.5)

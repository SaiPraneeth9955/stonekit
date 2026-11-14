"""The whole pipeline on a tiny hand-made volume, with a saved kidney segmentation.

A code fixture, NOT a dataset, and never used for any result. No model exists in the temporary
project, which also checks that a missing model is reported as missing and never filled in.
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from typer.testing import CliRunner

from stonekit.cli import app
from stonekit.config import load_yaml
from stonekit.io.nifti import load_volume
from stonekit.measure.mesh import mesh_volume_mm3, read_obj
from stonekit.pipeline.report_html import stone_table
from stonekit.pipeline.run_case import THRESHOLD, run_case
from stonekit.seg import kidney_roi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AFFINE = np.diag([-0.8, -0.8, 2.5, 1.0])
SPACING = (0.8, 0.8, 2.5)
VOXEL_MM3 = 0.8 * 0.8 * 2.5


def make_case(tmp_path, with_stones=True, shape=(80, 80, 40)):
    """Files use LPS order, like Li. Two boxes are stones, sitting inside the right kidney box."""
    ct = np.full(shape, 40.0, dtype=np.float32)
    ct[:5, :5, :5] = -1000
    ct[6, 6, 6] = 1000
    labels = np.zeros(shape, dtype=np.uint8)
    labels[20:40, 30:50, 10:30] = kidney_roi.KIDNEY_RIGHT
    labels[50:70, 30:50, 10:30] = kidney_roi.KIDNEY_LEFT
    labels[35:45, 5:15, 5:35] = kidney_roi.AORTA
    if with_stones:
        ct[25:29, 35:39, 15:19] = 700
        ct[32:34, 42:44, 20:22] = 500
    ct_path = tmp_path / "scan.nii.gz"
    label_path = tmp_path / "labels.nii.gz"
    nib.save(nib.Nifti1Image(ct, AFFINE), ct_path)
    nib.save(nib.Nifti1Image(labels, AFFINE), label_path)
    return ct_path, label_path


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / "configs" / "baseline_pipeline.yaml", tmp_path / "configs")
    return tmp_path


def run(project, tmp_path, config_change=None, **case_options):
    ct_path, label_path = make_case(tmp_path, **case_options)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    if config_change:
        config_change(config)
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu")
    return folder, json.loads((folder / "report.json").read_text(encoding="utf-8"))


def test_known_stones_measured_in_physical_units(project, tmp_path):
    _folder, report = run(project, tmp_path)
    method = report["methods"][THRESHOLD]
    assert report.get("failure") is None
    assert [s["voxel_count"] for s in method["stones"]] == [64, 8]
    assert method["stones"][0]["volume_mm3"] == pytest.approx(64 * VOXEL_MM3)
    assert method["stones"][0]["volume_ml"] == pytest.approx(64 * VOXEL_MM3 / 1000)
    assert method["stones"][0]["max_hu"] == 700
    assert method["stones"][0]["max_diameter_axial_mm"] == pytest.approx(np.hypot(3 * 0.8, 3 * 0.8))
    assert method["stones"][0]["max_diameter_3d_mm"] == pytest.approx(
        np.sqrt(2 * (3 * 0.8) ** 2 + (3 * 2.5) ** 2)
    )


@pytest.mark.parametrize("limit", [None, 30.0])
def test_kidney_distance_filter_removes_only_far_pieces(project, tmp_path, limit):
    """A hand-made mask with one piece near the kidney and one far away; the filter is off by default."""
    from stonekit.pipeline.run_case import _measure_method

    ct_path, _ = make_case(tmp_path, with_stones=False)
    ct = load_volume(ct_path)
    box = kidney_roi.Box((0, 0, 0), ct.data.shape)
    kidney = np.zeros(ct.data.shape, bool)
    kidney[10:20, 10:20, 10:20] = True
    from scipy import ndimage

    distance = ndimage.distance_transform_edt(~kidney, sampling=ct.spacing_mm)
    mask = np.zeros(ct.data.shape, bool)
    mask[12:14, 12:14, 12:14] = True  # inside the kidney
    mask[70:73, 70:73, 35:38] = True  # far from it
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    (tmp_path / "out" / "meshes").mkdir(parents=True)
    result = _measure_method(mask, ct, box, distance, config, "model", tmp_path / "out", limit)
    saved = load_volume(tmp_path / "out" / "stone_labels_model.nii.gz").data
    if limit is None:
        assert len(result["stones"]) == 2
        assert "kidney_distance_filter" not in result
        assert saved[71, 71, 36] > 0
    else:
        assert len(result["stones"]) == 1
        assert result["stones"][0]["distance_to_kidney_mm"] == 0
        removed = result["kidney_distance_filter"]["removed_pieces"]
        assert len(removed) == 1 and removed[0]["distance_to_kidney_mm"] > limit
        assert saved[71, 71, 36] == 0 and saved[12, 12, 12] > 0
        assert not list((tmp_path / "out" / "meshes").glob(f"model_stone_{removed[0]['stone_id']:03d}.obj"))


def test_system_config_differs_from_baseline_only_by_the_filter():
    baseline = load_yaml(PROJECT_ROOT / "configs" / "baseline_pipeline.yaml")
    system = load_yaml(PROJECT_ROOT / "configs" / "system_pipeline_v1.yaml")
    assert system.pop("postprocessing") == {"model_max_distance_to_kidney_mm": 30}
    assert system.pop("id") == "system_pipeline_v1" and baseline.pop("id") == "baseline_pipeline"
    assert system == baseline
    assert baseline["model"]["trainer"] == "nnUNetTrainer_100epochs"


def test_missing_model_is_reported_and_never_filled_in(project, tmp_path):
    _, report = run(project, tmp_path)
    assert report["methods"]["model"]["available"] is False
    assert report["methods"]["model"]["stones"] == []
    # The threshold is a comparator: it never becomes the primary result.
    assert report["primary_method"] is None
    assert "trained_model_not_available" in report["case_qc_flags"]
    assert report["result_status"] == "no_result"
    assert report["methods"][THRESHOLD]["available"] is True
    assert "model_info" not in report


def test_saved_labels_are_on_the_original_grid_at_the_right_voxels(project, tmp_path):
    folder, _ = run(project, tmp_path)
    saved = nib.load(folder / f"stone_labels_{THRESHOLD}.nii.gz")
    original = nib.load(tmp_path / "scan.nii.gz")
    np.testing.assert_allclose(saved.affine, original.affine)
    assert saved.shape == original.shape
    raw = np.asanyarray(saved.dataobj)
    assert (raw[25:29, 35:39, 15:19] == 1).all()
    assert (raw[32:34, 42:44, 20:22] == 2).all()
    assert (raw > 0).sum() == 72


def test_reported_position_matches_the_world_position(project, tmp_path):
    _, report = run(project, tmp_path)
    expected = AFFINE @ np.array([26.5, 36.5, 16.5, 1.0])
    assert report["methods"][THRESHOLD]["stones"][0]["centroid_ras_mm"] == pytest.approx(expected[:3])


def test_mesh_matches_its_stone(project, tmp_path):
    folder, report = run(project, tmp_path)
    stone = report["methods"][THRESHOLD]["stones"][0]
    vertices, faces = read_obj(folder / stone["mesh_file"])
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    assert centre == pytest.approx(stone["centroid_ras_mm"], abs=0.01)
    assert mesh_volume_mm3(vertices, faces) == pytest.approx(stone["volume_mm3"], rel=0.5)


def test_no_stone_scan_reports_no_stones(project, tmp_path):
    folder, report = run(project, tmp_path, with_stones=False)
    method = report["methods"][THRESHOLD]
    assert method["stones"] == []
    assert "no_stone_detected" in method["qc_flags"]
    assert method["total_volume_mm3"] == 0
    assert (folder / "report.html").exists()


def test_html_shows_the_same_values_as_json(project, tmp_path):
    folder, report = run(project, tmp_path)
    page = (folder / "report.html").read_text(encoding="utf-8")
    for row in stone_table(report["methods"][THRESHOLD]):
        for key in ("Volume (mm3)", "Volume (mL)", "3D diameter (mm)", "Axial diameter (mm)", "Peak HU", "Voxels"):
            assert f"<td>{row[key]}</td>" in page
    assert report["disclaimer"] in page
    assert "Development demonstration" in page
    assert "calibration: later phase" in page


def test_unusable_scan_is_reported_as_a_failure_not_a_result(project, tmp_path):
    folder, report = run(project, tmp_path, shape=(80, 80, 40), config_change=lambda c: c["input_checks"].update(min_slices=1000))
    assert "processing_failure" in report["case_qc_flags"]
    assert report["methods"] == {}
    assert (folder / "report.html").exists()


def test_no_kidney_found_gives_no_roi_and_no_stones(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    nib.save(nib.Nifti1Image(np.zeros((80, 80, 40), dtype=np.uint8), AFFINE), label_path)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert "no_kidney_found" in report["case_qc_flags"]
    assert "contrast_check_not_possible_no_aorta_found" in report["case_qc_flags"]
    assert report["methods"] == {}


def test_bright_aorta_raises_the_contrast_flag(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    data = load_volume(ct_path)
    raw = np.asanyarray(nib.load(ct_path).dataobj).copy()
    raw[35:45, 5:15, 5:35] = 300
    nib.save(nib.Nifti1Image(raw, data.original_affine), ct_path)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert "contrast_suspected_bright_aorta" in report["case_qc_flags"]


def test_rerun_gives_identical_measurements(project, tmp_path):
    _, first = run(project, tmp_path)
    ct_path = tmp_path / "scan.nii.gz"
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    second_folder = run_case(ct_path, tmp_path / "again", config, project, "fixture", None, tmp_path / "labels.nii.gz", "cpu")
    second = json.loads((second_folder / "report.json").read_text(encoding="utf-8"))
    assert first["methods"] == second["methods"]


def test_failed_fresh_kidney_step_is_a_reported_failure_not_a_result(project, tmp_path, monkeypatch):
    import sys

    def failing(ct_path, output_path, device, low_memory):
        return [sys.executable, "-c", "import sys; print('stand-in failure'); sys.exit(1)"]

    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", failing)
    ct_path, _ = make_case(tmp_path)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, None, "cpu")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert "processing_failure" in report["case_qc_flags"]
    assert "kidney_segmentation_failed" in report["case_qc_flags"]
    assert report["kidney_segmentation_failure"]["reason"] == "nonzero_exit"
    assert report["methods"] == {}
    assert [s["status"] for s in report["stages"] if s["stage"] == "kidney_segmentation"] == ["failed"]
    assert not (folder / "kidney_totalsegmentator.nii.gz").exists()
    assert "stand-in failure" in (folder / "totalsegmentator.log").read_text()
    assert (folder / "report.html").exists()


def test_timed_out_fresh_kidney_step_returns_promptly(project, tmp_path, monkeypatch):
    import sys
    import time

    def hanging(ct_path, output_path, device, low_memory):
        return [sys.executable, "-c", "import time; time.sleep(120)"]

    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", hanging)
    ct_path, _ = make_case(tmp_path)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    config["kidney_segmentation"]["timeout_seconds"] = 1
    start = time.monotonic()
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, None, "cpu")
    assert time.monotonic() - start < 30
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["kidney_segmentation_failure"]["reason"] == "timeout"
    assert report["methods"] == {}


def test_fresh_kidney_step_success_runs_the_rest(project, tmp_path, monkeypatch):
    import sys

    _, label_path = make_case(tmp_path)

    def copying(ct_path, output_path, device, low_memory):
        return [sys.executable, "-c", f"import shutil; shutil.copy({str(label_path)!r}, {str(output_path)!r})"]

    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", copying)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(tmp_path / "scan.nii.gz", tmp_path / "runs", config, project, "fixture", None, None, "cpu")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report.get("failure") is None
    assert report["kidney_segmentation_run"]["returncode"] == 0
    assert [s["voxel_count"] for s in report["methods"][THRESHOLD]["stones"]] == [64, 8]


def test_cli_refuses_reserved_scan_by_id_and_by_path():
    manifest = PROJECT_ROOT / "data" / "processed" / "li2022_working_split_v1.csv"
    if not manifest.exists():
        pytest.skip("Working split is not present")
    from stonekit.data.li_split import load_manifest

    reserved = next(c for c, group in load_manifest(manifest).items() if group == "reserved_test")
    runner = CliRunner()
    by_id = runner.invoke(app, ["run-case", "--case", reserved])
    assert by_id.exit_code == 2
    by_path = runner.invoke(app, ["run-case", "--ct", f"/nowhere/{reserved}_ps_image.nii.gz"])
    assert by_path.exit_code == 2
    assert "Refused" in by_id.output


def test_cli_needs_exactly_one_input():
    runner = CliRunner()
    assert runner.invoke(app, ["run-case"]).exit_code == 1
    assert runner.invoke(app, ["run-case", "--case", "a", "--ct", "b"]).exit_code == 1

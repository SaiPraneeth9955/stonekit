"""The viewer on unusual runs: success, no stones, failure, incomplete and corrupted runs.

Runs come from the hand-made fixture in test_run_case_smoke.py, not from any dataset.
"""

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from test_run_case_smoke import PROJECT_ROOT, make_case

from stonekit.app import streamlit_app
from stonekit.config import load_yaml
from stonekit.pipeline.run_case import run_case

APP = str(Path(streamlit_app.__file__))


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    shutil.copy(PROJECT_ROOT / "configs" / "baseline_pipeline.yaml", tmp_path / "configs")
    monkeypatch.setenv("STONEKIT_ROOT", str(tmp_path))
    return tmp_path


def runs(project: Path) -> Path:
    return project / "data" / "interim" / "stonekit_runs"


def make_run(project, tmp_path, name, **options):
    case_dir = tmp_path / name
    case_dir.mkdir()
    ct, labels = make_case(case_dir, **options)
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    return run_case(ct, runs(project), config, project, name, None, labels, "cpu")


def start(timeout=60):
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def test_no_runs_shows_guidance(project):
    app = start()
    assert not app.exception
    assert any("No processed runs" in c.value for c in app.sidebar.caption)


def test_successful_run_renders_every_tab(project, tmp_path):
    make_run(project, tmp_path, "stones")
    app = start()
    assert not app.exception
    assert len(app.tabs) == 5
    options = [str(o) for box in app.selectbox for o in box.options]
    assert any(o.startswith("Stone 1 ·") for o in options)
    assert any("Kidney stone measurement research workstation" in m.value for m in app.markdown)


def test_no_stone_run_renders(project, tmp_path):
    make_run(project, tmp_path, "nostone", with_stones=False)
    app = start()
    assert not app.exception
    assert any("nothing to render" in i.value for i in app.info)


def test_failed_run_is_shown_as_a_failure(project, tmp_path):
    make_run(project, tmp_path, "nokidney")
    folder = next(runs(project).glob("nokidney_*"))
    report = json.loads((folder / "report.json").read_text())
    report["failure"] = "ExternalToolError: TotalSegmentator timeout"
    report["methods"] = {}
    (folder / "report.json").write_text(json.dumps(report))
    app = start()
    assert not app.exception
    assert any("Processing failure" in e.value for e in app.error)


def test_incomplete_and_corrupted_runs_are_listed_as_not_results(project, tmp_path):
    make_run(project, tmp_path, "stones")
    (runs(project) / "wd0063_20261106T023244Z").mkdir()
    broken = runs(project) / "broken_20261107T000000Z"
    broken.mkdir()
    (broken / "report.json").write_text("{not json")
    app = start()
    assert not app.exception
    captions = " ".join(c.value for c in app.sidebar.caption)
    assert "Incomplete (no report.json): wd0063_20261106T023244Z" in captions
    assert "Unreadable report" in captions
    assert all("wd0063" not in str(o) and "broken" not in str(o) for o in app.sidebar.selectbox[0].options)


def test_missing_ct_file_does_not_crash(project, tmp_path):
    folder = make_run(project, tmp_path, "stones")
    Path(json.loads((folder / "report.json").read_text())["ct_path"]).unlink()
    app = start()
    assert not app.exception
    assert any("No images to show" in i.value for i in app.info)


def test_discover_separates_valid_incomplete_and_corrupted(tmp_path):
    root = tmp_path / "runs"
    (root / "validation_x" / "wd0001_a").mkdir(parents=True)
    (root / "validation_x" / "wd0001_a" / "report.json").write_text("{}")
    (root / "validation_x" / "wd0002_b").mkdir()
    (root / "wd0063_c").mkdir()
    (root / "wd0004_d" / "meshes").mkdir(parents=True)
    (root / "wd0004_d" / "report.json").write_text("[")
    valid, incomplete, corrupted = streamlit_app.discover(root)
    assert [p.name for p in valid] == ["wd0001_a"]
    assert sorted(p.name for p in incomplete) == ["wd0002_b", "wd0063_c"]
    assert [p.name for p, _ in corrupted] == ["wd0004_d"]


def test_analysis_runs_newest_first_and_labels_readable(tmp_path):
    runs_dir = tmp_path / "results" / "runs"
    old = runs_dir / "stone_analysis_a_20261107T000000Z"
    new = runs_dir / "stone_analysis_b_20261108T000000Z"
    bad = runs_dir / "stone_analysis_c_20261109T000000Z"
    for folder in (old, new, bad):
        folder.mkdir(parents=True)
    (old / "summary.json").write_text(json.dumps({"methods": {}}))
    (new / "summary.json").write_text(json.dumps({"methods": {}, "method_labels": {"model": "nnU-Net (250 epochs)"}}))
    (bad / "summary.json").write_text("{")
    found = streamlit_app.analysis_runs(tmp_path)
    assert [run.name for run, _ in found] == [new.name, old.name]
    assert streamlit_app.analysis_method_label("model", found[0][1]) == "nnU-Net (250 epochs)"
    assert streamlit_app.analysis_method_label("model", {"model_label": "nnU-Net baseline (100 epochs)"}) == "nnU-Net baseline (100 epochs)"
    assert streamlit_app.analysis_method_label("threshold_272hu", {}) == "272 HU threshold"
    assert streamlit_app.analysis_method_label("other", {}) == "other"


def test_axial_image_orientation_markers_and_size():
    class Fake:
        def __init__(self, data, spacing=(0.8, 0.8, 2.5)):
            self.data = data
            self.spacing_mm = spacing

    ct = Fake(np.zeros((40, 30, 5), dtype=np.float32))
    kidneys = Fake(np.zeros((40, 30, 5), dtype=np.uint8))
    labels = Fake(np.zeros((40, 30, 5), dtype=np.uint16))
    labels.data[5:8, 5:8, 2] = 1
    image = streamlit_app.compose_axial(ct, kidneys, labels, 2, 500, 2000, True, True, None, None, out_width=400)
    # Image width follows RAS x (40 voxels), height RAS y (30 voxels), at 10 display pixels per voxel.
    assert image.size == (400, 300)
    # RAS+ x grows toward the patient's right, so a stone at low x (patient's left) is on the image right.
    pixels = np.asarray(image)
    stone_columns = np.nonzero((pixels[..., 0] > 150) & (pixels[..., 1] > 100) & (pixels[..., 2] < 90))[1]
    assert stone_columns.size and stone_columns.min() >= (40 - 8) * 10 - 1


def test_crop_to_roi_shows_only_the_box_region():
    class Fake:
        def __init__(self, data, spacing=(1.0, 1.0, 2.5)):
            self.data = data
            self.spacing_mm = spacing

    ct = Fake(np.zeros((100, 80, 5), dtype=np.float32))
    zeros = Fake(np.zeros((100, 80, 5), dtype=np.uint8))
    box = {"box_start_voxel": [40, 30, 0], "box_stop_voxel": [60, 50, 5]}
    image = streamlit_app.compose_axial(ct, zeros, zeros, 2, 40, 400, True, True, box, None, box, out_width=410)
    # 20 voxels plus a 10-voxel border on each side gives 41 voxels across, 10 pixels each.
    assert image.size == (410, 410)


def test_runs_are_listed_newest_first_not_by_scan_id(tmp_path):
    root = tmp_path / "runs"
    for name in ("wd0256_20261107T000000Z", "wd0001_20261108T000000Z", "wd0100_20261106T000000Z"):
        (root / name).mkdir(parents=True)
        (root / name / "report.json").write_text("{}")
    valid, _, _ = streamlit_app.discover(root)
    assert [p.name for p in valid] == ["wd0001_20261108T000000Z", "wd0256_20261107T000000Z", "wd0100_20261106T000000Z"]


def test_run_without_model_says_the_threshold_is_only_a_comparator(project, tmp_path):
    make_run(project, tmp_path, "stones")
    app = start()
    assert not app.exception
    warnings = " ".join(w.value for w in app.warning)
    assert "comparator only" in warnings
    assert "without a stone result" in warnings
    text = " ".join(m.value for m in app.markdown)
    assert "Result status" in text and "no_result" in text
    assert "Configuration used SHA-256" in text


def test_model_failure_details_are_shown(project, tmp_path):
    folder = make_run(project, tmp_path, "stones")
    report = json.loads((folder / "report.json").read_text())
    report.update(failure="ExternalToolError: nnUNetv2_predict timeout", methods={}, result_status="failed",
                  stone_segmentation_failure={"tool": "nnUNetv2_predict", "reason": "timeout", "detail": "no exit"})  # fmt: skip
    (folder / "report.json").write_text(json.dumps(report))
    app = start()
    assert not app.exception
    assert any("Processing failure" in e.value for e in app.error)
    assert "nnUNetv2_predict" in " ".join(m.value for m in app.markdown)


def test_kidney_distance_filter_removals_are_listed(project, tmp_path):
    folder = make_run(project, tmp_path, "stones")
    report = json.loads((folder / "report.json").read_text())
    report["methods"]["threshold_272hu"]["kidney_distance_filter"] = {
        "max_distance_mm": 30, "removed_pieces": [{"stone_id": 9, "volume_mm3": 12.5, "distance_to_kidney_mm": 41.2}]
    }  # fmt: skip
    (folder / "report.json").write_text(json.dumps(report))
    app = start()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Kidney-distance filter (30 mm)" in text
    assert any("ureteric" in c.value for c in app.caption)
    assert any(41.2 in frame.value["Distance to kidney (mm)"].tolist() for frame in app.dataframe if "Distance to kidney (mm)" in frame.value)


def test_ras_image_is_loaded_like_the_pipeline(tmp_path):
    affine = np.diag([-0.8, -0.8, 2.5, 1.0])
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.float32), affine), tmp_path / "x.nii.gz")
    assert streamlit_app.load_volume(tmp_path / "x.nii.gz").data.shape == (4, 4, 4)


def test_size_bins_and_qc_status():
    assert streamlit_app.size_category(4.99) == "<5 mm"
    assert streamlit_app.size_category(5.0) == "5-<10 mm"
    assert streamlit_app.size_category(10.0) == ">=10 mm"
    assert streamlit_app.qc_status({"failure": "x"}, []) == "FAILED"
    assert streamlit_app.qc_status({"result_status": "no_result"}, []) == "FAILED"
    assert streamlit_app.qc_status({"case_qc_flags": ["many_components"]}, []) == "REVIEW"
    assert streamlit_app.qc_status({}, [{"qc_flags": ["very_small_component"]}]) == "REVIEW"
    assert streamlit_app.qc_status({"result_status": "complete"}, [{"qc_flags": []}]) == "PASS"


def test_uncertainty_is_never_labelled_calibrated_before_validation():
    assert "calibrated" in streamlit_app.UNCERTAINTY_TEXT and "No interval is shown" in streamlit_app.UNCERTAINTY_TEXT

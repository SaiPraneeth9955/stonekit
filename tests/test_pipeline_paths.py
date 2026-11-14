"""Failure paths and the trained-model path of run_case, on tiny hand-made volumes.

Code fixtures, NOT datasets, and never used for any result. The "model" here is a stand-in command
that writes a mask by thresholding the crop, so the real wiring (guarded subprocess, output checks,
measurement, kidney-distance filter, report) is exercised without a GPU or a checkpoint.
"""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from test_run_case_smoke import AFFINE, PROJECT_ROOT, make_case
from typer.testing import CliRunner

from stonekit.cli import app
from stonekit.config import load_yaml
from stonekit.pipeline import run_case as run_case_module
from stonekit.pipeline.config_check import ConfigError, config_problems
from stonekit.pipeline.run_case import MODEL, THRESHOLD, run_case
from stonekit.seg import nnunet_infer

SYSTEM = PROJECT_ROOT / "configs" / "system_pipeline_v1.yaml"
BASELINE = PROJECT_ROOT / "configs" / "baseline_pipeline.yaml"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "configs").mkdir()
    for name in ("baseline_pipeline.yaml", "system_pipeline_v1.yaml", "paths.yaml"):
        shutil.copy(PROJECT_ROOT / "configs" / name, tmp_path / "configs")
    return tmp_path


def read(folder: Path) -> dict:
    return json.loads((folder / "report.json").read_text(encoding="utf-8"))


def go(project, tmp_path, ct_path, label_path, config=None):
    config = config or load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu")
    return folder, read(folder)


# ---------------------------------------------------------------- corrupt inputs end in a report


@pytest.mark.parametrize(
    "content",
    [b"not a nifti at all" * 10, b"\x1f\x8b\x08\x00" + b"\x00" * 50],
    ids=["garbage_bytes", "broken_gzip"],
)
def test_corrupt_ct_is_a_reported_failure(project, tmp_path, content):
    _, label_path = make_case(tmp_path)
    bad = tmp_path / "bad.nii.gz"
    bad.write_bytes(content)
    folder, report = go(project, tmp_path, bad, label_path)
    assert "processing_failure" in report["case_qc_flags"]
    assert report["result_status"] == "failed"
    assert report["methods"] == {}
    assert (folder / "report.html").exists()


def test_corrupt_kidney_output_is_a_reported_failure(project, tmp_path):
    ct_path, _ = make_case(tmp_path)
    bad = tmp_path / "labels_bad.nii.gz"
    bad.write_bytes(b"junk")
    _, report = go(project, tmp_path, ct_path, bad)
    assert report["result_status"] == "failed"
    assert report["methods"] == {}


def test_kidney_output_with_unexpected_labels_is_refused_not_read_as_no_kidney(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    odd = tmp_path / "labels_odd.nii.gz"
    data = np.asanyarray(nib.load(label_path).dataobj).astype(np.float32) * 0.5
    nib.save(nib.Nifti1Image(data, AFFINE), odd)
    _, report = go(project, tmp_path, ct_path, odd)
    assert report["result_status"] == "failed"
    assert "unexpected label values" in report["failure"]
    assert "no_kidney_found" not in report["case_qc_flags"]


def test_kidney_output_on_another_grid_is_refused(project, tmp_path):
    ct_path, _ = make_case(tmp_path)
    other = tmp_path / "labels_other.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((40, 40, 20), np.uint8), AFFINE), other)
    _, report = go(project, tmp_path, ct_path, other)
    assert "not on the CT grid" in report["failure"]


def test_missing_ct_is_a_reported_failure(project, tmp_path):
    _, label_path = make_case(tmp_path)
    _, report = go(project, tmp_path, tmp_path / "absent.nii.gz", label_path)
    assert report["result_status"] == "failed"
    assert "FileNotFoundError" in report["failure"]


def test_unexpected_error_still_ends_in_a_report(project, tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("stand-in bug")

    monkeypatch.setattr(run_case_module, "separate_stones", broken)
    ct_path, label_path = make_case(tmp_path)
    folder, report = go(project, tmp_path, ct_path, label_path)
    assert {"processing_failure", "unexpected_error"} <= set(report["case_qc_flags"])
    assert report["failure"].startswith("RuntimeError: stand-in bug")
    assert (folder / "report.html").exists()


def test_html_figure_failure_does_not_lose_the_page(project, tmp_path, monkeypatch):
    from stonekit.pipeline import report_html

    def broken(report, out_dir):
        raise OSError("stand-in")

    monkeypatch.setattr(report_html, "_largest_stone_figure", broken)
    ct_path, label_path = make_case(tmp_path)
    folder, _report = go(project, tmp_path, ct_path, label_path)
    assert "Figure not drawn (OSError)" in (folder / "report.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- configuration


def test_invalid_config_fails_before_any_output(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    config = load_yaml(BASELINE)
    del config["roi"]
    config["measurement"]["min_component_voxels"] = 0
    with pytest.raises(ConfigError, match="missing section roi"):
        run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu")
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("path", [BASELINE, SYSTEM])
def test_shipped_configs_are_valid(path):
    assert config_problems(load_yaml(path)) == []


@pytest.mark.parametrize("value", [0, -5, "30", True])
def test_bad_filter_value_is_refused(value):
    config = load_yaml(SYSTEM)
    config["postprocessing"]["model_max_distance_to_kidney_mm"] = value
    assert config_problems(config)


def test_effective_config_hash_is_the_hash_of_the_config_used(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    config = load_yaml(project / "configs" / "system_pipeline_v1.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, label_path, "cpu",
                      project / "configs" / "system_pipeline_v1.yaml")  # fmt: skip
    report = read(folder)
    assert report["config"] == load_yaml(SYSTEM)
    assert report["effective_config_sha256"] == hashlib.sha256(json.dumps(report["config"], sort_keys=True).encode()).hexdigest()
    assert report["config_file"] == "system_pipeline_v1.yaml"
    assert report["config_sha256"] == hashlib.sha256(SYSTEM.read_bytes()).hexdigest()
    assert report["postprocessing"]["model_kidney_distance_filter_mm"] == 30
    changed = load_yaml(SYSTEM)
    changed["model"]["trainer"] = "nnUNetTrainer_250epochs"
    other = read(run_case(ct_path, tmp_path / "runs2", changed, project, "fixture", None, label_path, "cpu"))
    assert other["effective_config_sha256"] != report["effective_config_sha256"]


def test_code_version_is_recorded(project, tmp_path):
    ct_path, label_path = make_case(tmp_path)
    _, report = go(project, tmp_path, ct_path, label_path)
    assert report["code_version"]["git_commit"] is None  # the temporary project is not a git checkout
    real = run_case_module._code_version(PROJECT_ROOT)
    if real["git_commit"] is None:
        pytest.skip("Not a git checkout")
    assert len(real["git_commit"]) == 40 and isinstance(real["uncommitted_changes"], bool)


@pytest.mark.skipif(not (PROJECT_ROOT / "data/processed/li2022_working_split_v1.json").exists(),
                    reason="needs the Li 2022 split manifest (restricted data; not distributed)")
def test_cli_refuses_invalid_config_and_uppercase_reserved_name(tmp_path):
    from stonekit.data.li_split import load_manifest

    bad = tmp_path / "bad.yaml"
    bad.write_text("id: broken\n", encoding="utf-8")
    ct = tmp_path / "scan.nii.gz"
    ct.write_bytes(b"x")
    result = CliRunner().invoke(app, ["run-case", "--ct", str(ct), "--config", str(bad)])
    assert result.exit_code == 1
    assert "Invalid pipeline config" in result.output
    manifest = PROJECT_ROOT / "data" / "processed" / "li2022_working_split_v1.csv"
    if not manifest.exists():
        pytest.skip("Working split is not present")
    reserved = next(c for c, group in load_manifest(manifest).items() if group == "reserved_test")
    upper = CliRunner().invoke(app, ["run-case", "--ct", f"/nowhere/{reserved.upper()}_ps_image.nii.gz"])
    assert upper.exit_code == 2


# ---------------------------------------------------------------- low-memory kidney fallback


def thin_case(tmp_path):
    """0.5 mm slices, so the 1.5 mm low-memory path applies. Same layout idea as make_case."""
    affine = np.diag([-0.8, -0.8, 0.5, 1.0])
    ct = np.full((80, 80, 120), 40.0, dtype=np.float32)
    ct[:5, :5, :5] = -1000
    ct[6, 6, 6] = 1000
    labels = np.zeros(ct.shape, dtype=np.uint8)
    labels[20:40, 30:50, 30:90] = 2
    labels[50:70, 30:50, 30:90] = 3
    labels[35:45, 5:15, 10:110] = 52
    ct[25:29, 35:39, 50:62] = 700
    nib.save(nib.Nifti1Image(ct, affine), tmp_path / "thin.nii.gz")
    nib.save(nib.Nifti1Image(labels, affine), tmp_path / "thin_labels.nii.gz")
    return tmp_path / "thin.nii.gz", tmp_path / "thin_labels.nii.gz"


def killed_then_resampled(labels_path: Path, marker: Path):
    """First call: the tool is SIGKILLed (as by the OOM killer). Later calls: labels on the input's grid."""
    script = (
        "import os, signal, sys, pathlib, nibabel as nib, numpy as np\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        "    marker.write_text('first call')\n"
        "    os.kill(os.getpid(), signal.SIGKILL)\n"
        "ct = nib.load(sys.argv[1]); out = sys.argv[2]\n"
        f"orig = nib.load({str(labels_path)!r})\n"
        "o = np.asanyarray(orig.dataobj); dz_o = abs(orig.affine[2, 2]); dz_n = abs(ct.affine[2, 2])\n"
        "idx = np.clip(np.rint(np.arange(ct.shape[2]) * dz_n / dz_o).astype(int), 0, o.shape[2] - 1)\n"
        "nib.save(nib.Nifti1Image(o[:, :, idx], ct.affine), out)\n"
    )

    def command(ct_path, output_path, device, low_memory):
        return [sys.executable, "-c", script, str(ct_path), str(output_path)]

    return command


def test_memory_failure_falls_back_to_the_low_memory_kidney_step(project, tmp_path, monkeypatch):
    from stonekit.seg import kidney_roi

    ct_path, labels_path = thin_case(tmp_path)
    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", killed_then_resampled(labels_path, tmp_path / "marker"))
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    folder = run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, None, "cpu")
    report = read(folder)
    assert report.get("failure") is None
    assert report["kidney_segmentation_first_attempt_failure"]["reason"] == "killed_by_signal"
    assert "kidney_segmentation_low_memory_fallback" in report["case_qc_flags"]
    fallback = report["kidney_segmentation_fallback"]
    assert fallback["trigger"] == "killed_by_signal" and fallback["original_slices"] == 120 and fallback["resampled_slices"] == 40
    mask = nib.load(folder / "kidney_totalsegmentator.nii.gz")
    assert mask.shape == (80, 80, 120)
    np.testing.assert_allclose(mask.affine, np.diag([-0.8, -0.8, 0.5, 1.0]))
    # Labels come back within one original slice of the truth (nearest-neighbour both ways).
    original = np.asanyarray(nib.load(labels_path).dataobj)
    differ = np.nonzero((np.asanyarray(mask.dataobj) != original).any(axis=(0, 1)))[0]
    assert set(differ) <= {29, 30, 31, 88, 89, 90, 9, 10, 11, 108, 109, 110}
    stones = report["methods"][THRESHOLD]["stones"]
    assert [s["voxel_count"] for s in stones] == [4 * 4 * 12]  # measured on original voxels
    assert not list(folder.glob("*.zresampled*"))  # temporary resampled files are removed


def test_fallback_can_be_switched_off(project, tmp_path, monkeypatch):
    from stonekit.seg import kidney_roi

    ct_path, labels_path = thin_case(tmp_path)
    monkeypatch.setattr(kidney_roi, "totalsegmentator_command", killed_then_resampled(labels_path, tmp_path / "marker"))
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    config["kidney_segmentation"]["low_memory_fallback"] = False
    report = read(run_case(ct_path, tmp_path / "runs", config, project, "fixture", None, None, "cpu"))
    assert report["result_status"] == "failed"
    assert report["kidney_segmentation_failure"]["reason"] == "killed_by_signal"
    assert "kidney_segmentation_fallback" not in report


# ---------------------------------------------------------------- the trained-model path


def fake_checkpoint(project: Path, config: dict) -> None:
    checkpoint = nnunet_infer.checkpoint_path(project, config["model"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"stand-in checkpoint, not a model")


def stand_in(script: str):
    """A predict_command replacement that runs `script` with the input and output folders."""

    def command(input_dir, output_dir, model, device):
        return [sys.executable, "-c", script, str(input_dir), str(output_dir)]

    return command


THRESHOLD_MODEL = (
    "import sys, pathlib, nibabel as nib, numpy as np\n"
    "i, o = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); o.mkdir()\n"
    "img = nib.load(i / 'case_0000.nii.gz')\n"
    "mask = (np.asanyarray(img.dataobj) >= 450).astype(np.uint8)\n"
    "nib.save(nib.Nifti1Image(mask, img.affine), o / 'case.nii.gz')\n"
)


def model_case(tmp_path):
    """Stones near the kidney plus one bright piece far from it, which the 30 mm filter must remove."""
    ct_path, label_path = make_case(tmp_path, shape=(80, 160, 40))
    raw = np.asanyarray(nib.load(ct_path).dataobj).copy()
    raw[60:63, 140:143, 20:23] = 900  # about 72 mm from the nearest kidney voxel
    nib.save(nib.Nifti1Image(raw, AFFINE), ct_path)
    return ct_path, label_path


def test_model_path_measures_filters_and_records_provenance(project, tmp_path, monkeypatch):
    config = load_yaml(project / "configs" / "system_pipeline_v1.yaml")
    config["roi"]["box_margin_mm"] = 80  # so the far piece is inside the ROI and reaches the filter
    fake_checkpoint(project, config)
    monkeypatch.setattr(nnunet_infer, "predict_command", stand_in(THRESHOLD_MODEL))
    ct_path, label_path = model_case(tmp_path)
    folder, report = go(project, tmp_path, ct_path, label_path, config)
    assert report["result_status"] == "complete", report.get("failure")
    assert report["primary_method"] == MODEL
    model = report["methods"][MODEL]
    assert [s["voxel_count"] for s in model["stones"]] == [64, 8]
    removed = model["kidney_distance_filter"]["removed_pieces"]
    assert len(removed) == 1 and removed[0]["distance_to_kidney_mm"] > 30
    saved = nib.load(folder / "stone_labels_model.nii.gz")
    assert saved.shape == (80, 160, 40)
    assert np.asanyarray(saved.dataobj)[61, 141, 21] == 0
    # The removed piece is never silent: a case flag, and its own label map on the original grid.
    assert "pieces_removed_by_kidney_distance_filter" in report["case_qc_flags"]
    aside = nib.load(folder / model["kidney_distance_filter"]["removed_label_file"])
    assert aside.shape == (80, 160, 40)
    assert np.asanyarray(aside.dataobj)[61, 141, 21] == removed[0]["stone_id"]
    assert int((np.asanyarray(aside.dataobj) > 0).sum()) == 27
    # The comparator is unfiltered and independent of the model.
    assert "kidney_distance_filter" not in report["methods"][THRESHOLD]
    assert report["model_info"]["checkpoint_sha256"] == hashlib.sha256(b"stand-in checkpoint, not a model").hexdigest()
    assert (folder / "nnunet_predict.log").exists()
    page = (folder / "report.html").read_text(encoding="utf-8")
    assert "nnU-Net (100 epochs, fold 0)" in page
    assert "Pieces removed by the kidney-distance filter: 1" in page


def test_model_output_on_wrong_grid_is_refused(project, tmp_path, monkeypatch):
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    fake_checkpoint(project, config)
    shifted = THRESHOLD_MODEL.replace("nib.Nifti1Image(mask, img.affine)", "nib.Nifti1Image(mask, img.affine + SHIFT)").replace(
        "img = nib.load", "SHIFT = np.zeros((4, 4)); SHIFT[0, 3] = 5.0\nimg = nib.load"
    )
    monkeypatch.setattr(nnunet_infer, "predict_command", stand_in(shifted))
    ct_path, label_path = make_case(tmp_path)
    _, report = go(project, tmp_path, ct_path, label_path, config)
    assert report["result_status"] == "failed"
    assert "stone_segmentation_failed" in report["case_qc_flags"]
    assert report["stone_segmentation_failure"]["reason"] == "invalid_output"


def test_model_output_with_unexpected_labels_is_refused(project, tmp_path, monkeypatch):
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    fake_checkpoint(project, config)
    monkeypatch.setattr(nnunet_infer, "predict_command", stand_in(THRESHOLD_MODEL.replace("astype(np.uint8)", "astype(np.uint8) * 7")))
    ct_path, label_path = make_case(tmp_path)
    _, report = go(project, tmp_path, ct_path, label_path, config)
    assert "unexpected labels" in report["failure"]


@pytest.mark.parametrize(
    ("script", "reason"),
    [("import sys; sys.exit(3)", "nonzero_exit"), ("pass", "missing_output"), ("import os, signal; os.kill(os.getpid(), signal.SIGKILL)", "killed_by_signal")],
)
def test_model_failures_are_reported(project, tmp_path, monkeypatch, script, reason):
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    fake_checkpoint(project, config)
    monkeypatch.setattr(nnunet_infer, "predict_command", stand_in(script))
    ct_path, label_path = make_case(tmp_path)
    _, report = go(project, tmp_path, ct_path, label_path, config)
    assert report["result_status"] == "failed"
    assert report["stone_segmentation_failure"]["reason"] == reason
    assert [s["status"] for s in report["stages"] if s["stage"] == "stone_segmentation_model"] == ["failed"]


def test_model_timeout_returns_promptly_and_leaves_no_process(project, tmp_path, monkeypatch):
    config = load_yaml(project / "configs" / "baseline_pipeline.yaml")
    config["model_inference"] = {"timeout_seconds": 1}
    fake_checkpoint(project, config)
    marker = "stonekit_predict_hang_marker_7731"
    hang = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)  # {marker}'])\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setattr(nnunet_infer, "predict_command", stand_in(hang))
    ct_path, label_path = make_case(tmp_path)
    start = time.monotonic()
    _, report = go(project, tmp_path, ct_path, label_path, config)
    assert time.monotonic() - start < 40
    failure = report["stone_segmentation_failure"]
    assert failure["reason"] == "timeout"
    assert failure["processes_still_alive_after_cleanup"] == 0
    assert failure["leftover_processes_stopped"] >= 1
    assert not processes_with(marker)


def processes_with(text: str) -> list[int]:
    """Live processes whose command line contains `text`, read from /proc."""
    found = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if text in (entry / "cmdline").read_bytes().decode(errors="replace"):
                    found.append(int(entry.name))
            except OSError:
                continue
    return found

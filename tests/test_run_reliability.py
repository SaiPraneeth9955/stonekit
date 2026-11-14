"""The runner on tiny hand-made patches. A code fixture only, never a result or dataset."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from stonekit.config import load_yaml
from stonekit.data.freiburg import CSV_NAME, MAT_NAME, build_table, write_table
from stonekit.reliability.provenance import (
    file_sha256,
    folder_hashes,
    reconciliation_mismatches,
)

REPO = Path(__file__).resolve().parents[1]
N_STONES = 45
GRID = [130, 150, 200, 250, 272, 300, 350, 400]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch(index):
    """A core cube inside a shell, with core size and values that vary between stones."""
    size = 3 + index % 7
    image = np.zeros((18, 18, 18))
    image[2 : 6 + size, 2 : 6 + size, 2 : 6 + size] = 160.0 + 25 * (index % 9)
    image[3 : 5 + size, 3 : 5 + size, 3 : 5 + size] = 280.0 + 11 * (index % 5)
    image[4 : 4 + size, 4 : 4 + size, 4 : 4 + size] = 500.0 + 45 * (index % 11)
    return image


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    raw = root / "data" / "raw" / "freiburg"
    raw.mkdir(parents=True)
    names = ("gt_diam", "gt_vol", "imgdata", "voxsiz", "voxvol_mL")
    stones = np.zeros((1, N_STONES), dtype=[(name, object) for name in names])
    lines = ["ID;Volume;Diameter;StoneType"]
    for i in range(N_STONES):
        voxvol = 0.000344149 if i == 0 else 0.000379525
        volume = (3 + i % 7 + 2) ** 3 * voxvol * (1 + 0.25 * np.sin(1.7 * i)) + 0.002 * i
        diameter = 3.0 + 0.3 * i
        stones[0, i] = (diameter, volume, patch(i), np.array([0.7363, 0.7363, 0.7]), voxvol)
        lines.append(f"{i + 1};{round(volume, 2)};{round(diameter, 2)};")
    (raw / CSV_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    savemat(raw / MAT_NAME, {"stones": stones})
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "data" / "registry.yaml").write_text("", encoding="utf-8")
    write_table(
        build_table(raw, expected_stones=N_STONES),
        root / "data" / "processed" / "freiburg_stones.parquet",
    )
    (root / "docs").mkdir()
    (root / "docs" / "prereg.md").write_text("fixture", encoding="utf-8")
    a0_config = {
        "id": "a0_replicate_freiburg",
        "seed": 1,
        "raw_folder": "data/raw/freiburg",
        "table": "data/processed/freiburg_stones.parquet",
        "runs_folder": "results/runs",
        "verify_source_checksums": False,
        "primary_threshold_hu": 272,
        "exploratory_thresholds_hu": [101, *GRID],
        "min_component_voxels": 3,
        "assumed_z_spacing_mm": 0.7,
        "primary_axis_assignment": "Z2",
        "sensitivity_axis_assignments": ["Z1", "Z0"],
        "paper_voxel_edge_mm": 0.7,
        "bootstrap_resamples": 10,
        "g3": {"icc_above": 0.99, "borderline_low": 0.985, "borderline_high": 0.995},
        "sensitivity_exclusions": {"S1": [1]},
    }
    a0_folder = load("run_experiment").run_a0(a0_config, root / "config.yaml", root)
    return root, a0_folder


def study_config(project_tuple, runs_folder="results/study"):
    root, a0_folder = project_tuple
    return {
        "id": "r1_reliability_freiburg",
        "seed": 7,
        "preregistration_path": "docs/prereg.md",
        "preregistration_sha256": file_sha256(root / "docs" / "prereg.md"),
        "a0_run": str(a0_folder.relative_to(root)),
        "a0_stones_sha256": file_sha256(a0_folder / "a0_stones.csv"),
        "raw_folder": "data/raw/freiburg",
        "table": "data/processed/freiburg_stones.parquet",
        "runs_folder": runs_folder,
        "verify_source_checksums": False,
        "primary_threshold_hu": 272,
        "min_component_voxels": 3,
        "assumed_z_spacing_mm": 0.7,
        "paper_voxel_edge_mm": 0.7,
        "threshold_grid_hu": GRID,
        "exploratory_local_pairs_hu": [[250, 300], [200, 350]],
        "flagged_per_fold": [1, 2, 3],
        "primary_repeats": 3,
        "bootstrap_resamples": 6,
        "bootstrap_partitions": 1,
        "integrity_permutations": 4,
        "integrity_partitions": 1,
        "range_repeats": 2,
        "range_bootstrap_resamples": 3,
        "range_bootstrap_partitions": 1,
        "exploratory_bootstrap_resamples": 2,
        "exploratory_bootstrap_partitions": 1,
        "training_threshold_burdens": [0.2],
        "risk_curve_flagged_per_fold": [0, 1, 2],
        "expected_stones": N_STONES,
        "expected_spacing_agree": N_STONES - 1,
        "coarsest_reference_ml": 99.0,
        "expected_coarsest": 0,
        "unreachable_reference_ids": [2, 3],
        "flagged_reference_ids": [4],
        "float32_ids": [5],
        "failure_top_fraction": 0.1,
        "failure_report_flagged_per_fold": 2,
    }


def quiet_diagnostic(module, monkeypatch):
    """Tiny fixtures can trigger by chance, so the trigger itself is tested separately."""
    real = module.integrity_diagnostic

    def wrapped(*args, **kwargs):
        frame = real(*args, **kwargs)
        frame["triggered"] = False
        return frame

    monkeypatch.setattr(module, "integrity_diagnostic", wrapped)


@pytest.fixture(scope="module")
def completed(project):
    module = load("run_reliability")
    monkeypatch = pytest.MonkeyPatch()
    quiet_diagnostic(module, monkeypatch)
    root, a0_folder = project
    before = folder_hashes(a0_folder)
    config = study_config(project)
    run_folder, stop = module.run_study(config, root / "study.yaml", root)
    monkeypatch.undo()
    return run_folder, stop, before, a0_folder


def test_fixture_run_finishes_and_writes_all_outputs(completed):
    run_folder, stop, _, _ = completed
    assert stop is None
    for name in (
        "r1_reconciliation_mismatches.csv",
        "r1_integrity_diagnostic.csv",
        "r1_primary.csv",
        "r1_secondary_estimates.csv",
        "r1_flag_differences.csv",
        "r1_ranges_summary.csv",
        "r1_ranges_subgroups_exploratory.csv",
        "r1_ranges_interval_score.csv",
        "r1_exploratory_psi.csv",
        "r1_exploratory_training_threshold.csv",
        "r1_exploratory_risk_curve.csv",
        "r1_exploratory_coefficient_spread.csv",
        "r1_exploratory_half_attenuation.csv",
        "r1_exploratory_composition.csv",
        "r1_stones.csv",
        "r1_failure_analysis.csv",
        "run_manifest.json",
    ):
        assert (run_folder / name).is_file()


def test_primary_has_one_estimand_under_both_assumptions(completed):
    primary = pd.read_csv(completed[0] / "r1_primary.csv")
    assert primary.assumption.tolist() == ["A", "B"]
    assert primary.reading.notna().all()
    assert "passes" not in " ".join(primary.columns)


def test_a0_stays_unchanged_and_masks_are_not_saved(completed):
    run_folder, _, before, a0_folder = completed
    assert folder_hashes(a0_folder) == before
    assert not [p for p in run_folder.rglob("*") if p.suffix in {".npy", ".npz", ".nii", ".gz"}]


def test_manifest_records_provenance(completed):
    import json

    manifest = json.loads((completed[0] / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["a0_unchanged"]
    assert manifest["stop_reason"] is None
    hashes = manifest["source_hashes"]
    assert "preregistration" in hashes
    assert "processed_table" in hashes
    assert "scripts/run_reliability.py" in hashes
    assert any(name.startswith("tests/") for name in hashes)
    assert manifest["stages_completed"][0] == "provenance_and_reconciliation"


def test_outcomes_use_saved_a0_volumes(completed):
    run_folder, _, _, a0_folder = completed
    stones = pd.read_csv(run_folder / "r1_stones.csv")
    saved = pd.read_csv(a0_folder / "a0_stones.csv")
    assert stones.volume_A.to_numpy() == pytest.approx(saved.volume_A.to_numpy())
    expected = np.abs(np.log(saved.volume_B / saved.reference_volume_ml))
    assert stones.E_B.to_numpy() == pytest.approx(expected.to_numpy())


def test_changed_preregistration_stops_before_any_analysis(project):
    module = load("run_reliability")
    root, _ = project
    config = study_config(project, "results/stop_prereg")
    config["preregistration_sha256"] = "0" * 64
    run_folder, stop = module.run_study(config, root / "study.yaml", root)
    assert "frozen hash" in stop
    assert not (run_folder / "r1_primary.csv").exists()
    assert not (run_folder / "r1_integrity_diagnostic.csv").exists()


def test_diagnostic_trigger_stops_before_psi(project, monkeypatch):
    module = load("run_reliability")
    real = module.integrity_diagnostic

    def triggered(*args, **kwargs):
        frame = real(*args, **kwargs)
        frame["triggered"] = True
        return frame

    monkeypatch.setattr(module, "integrity_diagnostic", triggered)
    root, _ = project
    run_folder, stop = module.run_study(
        study_config(project, "results/stop_diagnostic"), root / "study.yaml", root
    )
    assert "integrity" in stop
    assert (run_folder / "r1_integrity_diagnostic.csv").exists()
    assert not (run_folder / "r1_primary.csv").exists()


def test_diagnostic_is_labelled_as_integrity_only(completed):
    frame = pd.read_csv(completed[0] / "r1_integrity_diagnostic.csv")
    assert set(frame.role) == {"pipeline-integrity diagnostic only"}


def test_coarsest_count_mismatch_stops_the_run(project):
    module = load("run_reliability")
    root, _ = project
    config = study_config(project, "results/stop_coarsest")
    config["expected_coarsest"] = 12
    run_folder, stop = module.run_study(config, root / "study.yaml", root)
    assert "coarsest" in stop
    assert not (run_folder / "r1_integrity_diagnostic.csv").exists()


def test_reconciliation_detects_each_kind_of_mismatch():
    saved = pd.DataFrame(
        {
            "stone_id": [1, 2],
            "mask_voxels": [10, 20],
            "n_components": [1, 1],
            "filled_voxels": [0, 0],
            "volume_A": [1.0, 2.0],
            "volume_B": [1.0, 2.0],
            "d_paper_A": [5.0, 6.0],
            "d_paper_B": [5.0, 6.0],
        }
    )
    assert reconciliation_mismatches(saved, saved.copy()) == []
    tiny = saved.copy()
    tiny.loc[0, "volume_A"] = 1.0 + 1e-12
    assert reconciliation_mismatches(saved, tiny) == []
    voxels = saved.copy()
    voxels.loc[1, "mask_voxels"] = 21
    assert reconciliation_mismatches(saved, voxels) == ["mask_voxels: stone 2"]
    volume = saved.copy()
    volume.loc[0, "volume_B"] = 1.0001
    assert reconciliation_mismatches(saved, volume) == ["volume_B: stone 1"]
    assert reconciliation_mismatches(saved, saved.iloc[:1]) == ["stone IDs differ"]


def test_a_second_run_is_refused_without_the_flag(project, tmp_path):
    module = load("run_reliability")
    root, _ = project
    folder = root / "configs_guard"
    folder.mkdir(exist_ok=True)
    (root / "results" / "guard").mkdir(parents=True, exist_ok=True)
    (root / "results" / "guard" / "r1_reliability_freiburg_old").mkdir(exist_ok=True)
    config = folder / "study.yaml"
    config.write_text(
        "id: r1_reliability_freiburg\nruns_folder: results/guard\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="A run already exists"):
        module.main([str(config)])


def test_real_config_matches_the_frozen_design():
    config = load_yaml(REPO / "configs" / "r1_reliability_freiburg.yaml")
    assert file_sha256(REPO / "docs" / "preregistration.md") == config["preregistration_sha256"]
    assert config["flagged_per_fold"] == [2, 3, 4, 5, 6]
    assert config["primary_repeats"] == 200
    assert config["bootstrap_resamples"] == 2000
    assert config["bootstrap_partitions"] == 20
    assert config["threshold_grid_hu"] == GRID
    assert config["integrity_permutations"] == 100
    assert config["integrity_partitions"] == 20
    assert config["range_repeats"] == 200
    assert config["primary_threshold_hu"] == 272

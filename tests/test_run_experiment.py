"""End-to-end a0 runner on tiny hand-made patches. A code fixture only, never a result."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from stonekit.data.freiburg import CSV_NAME, MAT_NAME, build_table, write_table

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
SIZES = [3, 4, 5, 6, 7, 8, 9, 10]


def load_script():
    spec = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def block_image(size):
    image = np.zeros((16, 16, 16))
    image[2 : 2 + size, 2 : 2 + size, 2 : 2 + size] = 900.0
    return image


@pytest.fixture
def project(tmp_path):
    raw = tmp_path / "data" / "raw" / "freiburg"
    raw.mkdir(parents=True)
    names = ("gt_diam", "gt_vol", "imgdata", "voxsiz", "voxvol_mL")
    stones = np.zeros((1, len(SIZES)), dtype=[(name, object) for name in names])
    lines = ["ID;Volume;Diameter;StoneType"]
    for i, size in enumerate(SIZES):
        voxvol = 0.000344149 if i == 0 else 0.000379525
        diameter = 0.72 * size * 3**0.5 + 0.1 * i
        volume = size**3 * voxvol + 0.001 * i
        stones[0, i] = (
            diameter,
            volume,
            block_image(size),
            np.array([0.7363, 0.7363, 0.7]),
            voxvol,
        )
        lines.append(f"{i + 1};{round(volume, 2)};{round(diameter, 2)};")
    (raw / CSV_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    savemat(raw / MAT_NAME, {"stones": stones})
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    write_table(
        build_table(raw, expected_stones=len(SIZES)),
        tmp_path / "data" / "processed" / "freiburg_stones.parquet",
    )
    return tmp_path


def config():
    return {
        "id": "a0_replicate_freiburg",
        "seed": 1,
        "raw_folder": "data/raw/freiburg",
        "table": "data/processed/freiburg_stones.parquet",
        "runs_folder": "results/runs",
        "verify_source_checksums": False,
        "primary_threshold_hu": 272,
        "exploratory_thresholds_hu": [101, 272, 400],
        "min_component_voxels": 3,
        "assumed_z_spacing_mm": 0.7,
        "primary_axis_assignment": "Z2",
        "sensitivity_axis_assignments": ["Z1", "Z0"],
        "paper_voxel_edge_mm": 0.7,
        "bootstrap_resamples": 50,
        "g3": {"icc_above": 0.99, "borderline_low": 0.985, "borderline_high": 0.995},
        "sensitivity_exclusions": {"S1_reference_flagged": [1], "S2_largest_outliers": [8]},
    }


@pytest.fixture
def run(project):
    return load_script().run_a0(config(), project / "config.yaml", project)


def test_all_outputs_are_written_separately(run):
    for name in (
        "a0_primary.csv",
        "a0_sweep_exploratory.csv",
        "a0_sensitivity.csv",
        "a0_axis_sensitivity.csv",
        "a0_stones.csv",
        "a0_bootstrap_summary.csv",
        "a0_stop_rules.csv",
        "run_manifest.json",
        "a0_bland_altman_volume_A.pdf",
        "a0_bland_altman_d_paper_B.png",
    ):
        assert (run / name).is_file()


def test_primary_reports_both_assumptions_and_all_metrics(run):
    primary = pd.read_csv(run / "a0_primary.csv")
    assert set(primary.assumption) == {"A", "B"}
    assert set(primary.metric) == {"volume", "d_paper", "d_centre"}
    assert set(primary[primary.decides_g3].metric) == {"volume", "d_paper"}
    assert set(primary.n) == {len(SIZES)}


def test_sweep_reports_every_threshold_and_is_labelled_exploratory(run):
    sweep = pd.read_csv(run / "a0_sweep_exploratory.csv")
    assert set(sweep.threshold_hu) == {101, 272, 400}
    assert set(sweep.analysis) == {"exploratory"}


def test_per_stone_table_keeps_every_stone(run):
    assert len(pd.read_csv(run / "a0_stones.csv")) == len(SIZES)


def test_d_paper_is_d_centre_plus_the_fixed_edge(run):
    stones = pd.read_csv(run / "a0_stones.csv")
    assert (stones.d_paper_A - stones.d_centre_A).to_numpy() == pytest.approx(0.7)
    assert (stones.d_paper_B - stones.d_centre_B).to_numpy() == pytest.approx(0.7)


def test_spacing_assumptions_differ_only_where_the_fields_disagree(run):
    stones = pd.read_csv(run / "a0_stones.csv")
    assert stones.volume_A[0] != pytest.approx(stones.volume_B[0], rel=0.01)
    assert stones.volume_A[1] == pytest.approx(stones.volume_B[1], rel=0.001)


def test_axis_sensitivity_has_every_combination_and_no_volume(run):
    axis = pd.read_csv(run / "a0_axis_sensitivity.csv")
    assert len(axis) == 12
    assert set(axis.assignment) == {"Z2", "Z1", "Z0"}
    assert set(axis.metric) == {"d_paper", "d_centre"}


def test_stop_rules_cover_all_six(run):
    assert pd.read_csv(run / "a0_stop_rules.csv").rule.tolist() == [1, 2, 3, 4, 5, 6]


def test_source_files_are_not_changed(project):
    path = project / "data" / "raw" / "freiburg" / MAT_NAME
    before = path.read_bytes()
    load_script().run_a0(config(), project / "config.yaml", project)
    assert path.read_bytes() == before


def test_gate_requires_strictly_above_and_flags_borderline():
    module = load_script()
    g3 = config()["g3"]
    at = module.gate_outcome({"icc": 0.99, "icc_ci_low": 0.98, "icc_ci_high": 0.995}, g3)
    assert at == {"passes": False, "borderline": True}
    clear = module.gate_outcome({"icc": 0.998, "icc_ci_low": 0.996, "icc_ci_high": 0.999}, g3)
    assert clear == {"passes": True, "borderline": False}
    wide = module.gate_outcome({"icc": 0.998, "icc_ci_low": 0.98, "icc_ci_high": 0.999}, g3)
    assert wide == {"passes": True, "borderline": True}

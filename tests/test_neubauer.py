"""Neubauer 2023 replication methods (protocol rep1). Hand-made arrays only; never a dataset."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from stonekit.replication import neubauer as nb

ROOT = Path(__file__).resolve().parents[1]


def cube(size=3, pad=2, value=700.0):
    image = np.zeros((size + 2 * pad,) * 3)
    image[pad : pad + size, pad : pad + size, pad : pad + size] = value
    return image


def test_surface_counts_by_hand():
    mask = cube() > 0
    assert nb.surface_voxels(mask, 6) == 26 and nb.surface_voxels(mask, 26) == 26
    big = cube(5) > 0
    assert nb.surface_voxels(big, 6) == 125 - 27
    # Remove one corner: the centre now touches the outside only diagonally.
    mask[2, 2, 2] = False
    assert nb.surface_voxels(mask, 6) == 25
    assert nb.surface_voxels(mask, 26) == 26
    assert nb.surface_voxels(np.zeros((3, 3, 3), bool)) == 0


def test_patch_border_counts_as_outside():
    full = np.ones((3, 3, 3), bool)
    assert nb.surface_voxels(full, 6) == 26


def test_rough_median_uses_voxels_above_20_hu():
    assert nb.rough_median_hu(np.array([0.0, 10.0, 30.0, 50.0, 100.0])) == 50.0
    assert math.isnan(nb.rough_median_hu(np.zeros(4)))


def test_stone_profile_matches_hand_values():
    geometries = {"iso": nb.Geometry((0.7, 0.7, 0.7), 2), "thick": nb.Geometry((0.7, 0.7, 1.4), 2)}
    profile = nb.stone_profile(cube(), [100, 699, 700, 800], geometries, edge_mm=0.7)
    assert profile.voxels.tolist() == [27, 27, 0, 0]  # strictly above the threshold, as a0
    assert profile.surface_6.tolist() == [26, 26, 0, 0]
    assert profile.diameter_3d_mm["iso"][0] == pytest.approx(math.sqrt(3) * 1.4 + 0.7)
    assert profile.diameter_3d_mm["thick"][0] == pytest.approx(math.sqrt(1.4**2 + 1.4**2 + 2.8**2) + 0.7)
    assert profile.diameter_axial_mm["iso"][0] == pytest.approx(math.sqrt(2) * 1.4 + 0.7)
    assert profile.diameter_3d_mm["iso"][2] == 0.0  # empty mask: no voxel edge added


def test_corrected_volume_and_cost():
    assert nb.corrected_volume_ml(27, 26, 0.5, 0.343e-3) == pytest.approx((27 - 13) * 0.343e-3)
    assert nb.cost([1.0, 2.0], [10.0, 20.0], [1.5, 2.0], [9.0, 21.0]) == pytest.approx(4 * 0.5 + 1 + 1)


def test_batch_optimiser_equals_per_split_optimiser():
    rng = np.random.default_rng(0)
    volume = rng.uniform(0, 2, size=(12, 7, 5))
    diameter = rng.uniform(5, 15, size=(12, 7, 5))
    ref_v, ref_d = rng.uniform(0, 2, 12), rng.uniform(5, 15, 12)
    splits = nb.random_splits(12, 9, 25, seed=3)
    batch = nb.optimise_splits(nb.per_stone_cost(volume, diameter, ref_v, ref_d), splits)
    single = [nb.optimise(volume, diameter, ref_v, ref_d, train) for train, _ in splits]
    assert batch.tolist() == single


def test_optimiser_finds_the_known_threshold():
    # Volume equals the reference exactly at grid index 3, so the cost is lowest there.
    grid = np.arange(6, dtype=float)
    ref_v = np.array([1.0, 2.0, 3.0])
    volume = ref_v[:, None] * (1 + 0.1 * np.abs(grid - 3))[None, :]
    diameter = np.zeros_like(volume)
    assert nb.optimise(volume, diameter, ref_v, np.zeros(3), [0, 1, 2]) == 3


def test_splits_are_reproducible_disjoint_and_sized():
    a = nb.random_splits(101, 80, 5, seed=20261111)
    b = nb.random_splits(101, 80, 5, seed=20261111)
    for (train, validation), (train_b, _) in zip(a, b, strict=True):
        assert np.array_equal(train, train_b)
        assert len(train) == 80 and len(validation) == 21
        assert not set(train) & set(validation)


def test_validation_stats_sign_and_sd():
    out = nb.validation_stats([2.0, 4.0, 6.0], [1.0, 4.0, 8.0], lambda e, r: 0.5)
    assert out["difference_estimate_minus_reference"] == pytest.approx(-1 / 3)
    assert out["sd"] == pytest.approx(np.std([1.0, 0.0, -2.0], ddof=1))


def load_runner():
    spec = importlib.util.spec_from_file_location("run_neubauer_replication", ROOT / "scripts" / "run_neubauer_replication.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_refuses_an_unapproved_protocol_before_reading_data(tmp_path):
    runner = load_runner()
    config = tmp_path / "rep1.yaml"
    config.write_text("id: rep1\nstatus: draft\nraw_folder: /nowhere\n", encoding="utf-8")
    with pytest.raises(runner.NotApproved):
        runner.main([str(config)])


def test_runner_end_to_end_on_hand_made_cubes(tmp_path, monkeypatch):
    """Executes the whole runner on 12 cube 'stones' (code fixtures, not data) with tiny grids."""
    import yaml

    from stonekit.data.freiburg import MatStone

    runner = load_runner()
    stones = {}
    for i in range(12):
        size = 2 + i % 4
        image = cube(size, pad=2, value=400.0 + 50 * i)
        voxel_ml = 0.7**3 / 1000
        stones[i + 1] = MatStone(i + 1, size * 0.7 + 0.7, size**3 * voxel_ml, (0.7, 0.7, 0.7), voxel_ml, image)
    monkeypatch.setattr(runner, "load_table", lambda path: [])
    monkeypatch.setattr(runner, "load_mat_stones", lambda path: [])
    monkeypatch.setattr(runner, "match_stones", lambda rows, mats: stones)
    monkeypatch.setattr(runner, "find_project_root", lambda: tmp_path)
    config = yaml.safe_load((ROOT / "configs" / "rep1_neubauer_replication.yaml").read_text())
    config.update(status="approved", n_splits=5, n_train=9, bootstrap_resamples=50,
                  m1_threshold_hu={"start": 100, "stop": 500, "step": 100},
                  m3_factor={"start": 0.3, "stop": 0.7, "step": 0.2}, pv_factor={"start": 0.0, "stop": 0.5, "step": 0.25})  # fmt: skip
    path = tmp_path / "rep1.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert runner.main([str(path)]) == 0
    run = next((tmp_path / "results" / "runs").glob("rep1_neubauer_replication_*"))
    import json

    summary = json.loads((run / "summary.json").read_text())
    assert "P_Z2_surface6_3d" in summary["configurations"]
    assert "bartlett_volume_p" in summary["primary_all_stone"]
    assert (run / "table1_comparison_P_Z2_surface6_3d.csv").exists()
    # One factor at a time: geometry, surface and diameter variants, never two at once.
    assert "A_Z2_surface26_3d" not in summary["configurations"]


def test_shipped_config_matches_the_approved_protocol():
    # Approved by the owner on 11 November 2026 (docs/decisions.md); the grids must still be the protocol's.
    runner = load_runner()
    from stonekit.config import load_yaml

    config = load_yaml(ROOT / "configs" / "rep1_neubauer_replication.yaml")
    assert config["status"] == "approved"
    assert runner.grid(config["m1_threshold_hu"])[[0, -1]].tolist() == [100, 400]
    assert set(config["paper_table1"]) == {"M1", "M2", "M3", "M4"}

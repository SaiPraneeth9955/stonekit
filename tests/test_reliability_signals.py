"""Signals on hand-made arrays. Code fixtures only, never results."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stonekit.reliability.signals import (
    SignalSettings,
    assumption_geometry,
    attenuation_signal,
    local_elasticity,
    stone_record,
    threshold_sensitivity,
)

SETTINGS = SignalSettings(
    primary_threshold_hu=272,
    grid_hu=(130, 250, 272, 300, 1000),
    local_pairs=((250, 300),),
    min_component_voxels=3,
    assumed_z_spacing_mm=0.7,
    paper_voxel_edge_mm=0.7,
)


def shell_stone(core_hu=900.0, shell_hu=260.0, voxvol=0.000379525):
    """A 5-voxel cube at core_hu inside a 7-voxel cube at shell_hu, on a zero background."""
    image = np.zeros((11, 11, 11))
    image[2:9, 2:9, 2:9] = shell_hu
    image[3:8, 3:8, 3:8] = core_hu
    return SimpleNamespace(image=image, voxsiz_mm=(0.7363, 0.7363, 0.7), voxvol_ml=voxvol)


def test_threshold_sensitivity_recovers_a_known_slope():
    counts = {t: 1e8 * t**-2.0 - 1 for t in (130, 200, 272, 400)}
    assert threshold_sensitivity(counts) == pytest.approx(2.0)


def test_threshold_sensitivity_stays_finite_with_a_zero_count():
    assert math.isfinite(threshold_sensitivity({130: 50, 272: 20, 400: 0}))


def test_local_elasticity_uses_the_stated_form():
    expected = abs(math.log(101 / 51)) / math.log(300 / 250)
    assert local_elasticity(100, 50, 250, 300) == pytest.approx(expected)


def test_attenuation_signal_is_distance_from_half_attenuation():
    assert attenuation_signal(544.0, 272) == pytest.approx(0.0)
    assert attenuation_signal(1088.0, 272) == pytest.approx(math.log(2))
    assert attenuation_signal(272.0, 272) == pytest.approx(math.log(2))


def test_stone_record_counts_follow_the_frozen_pipeline():
    record = stone_record(shell_stone(), SETTINGS)
    assert record["N_130"] == 7**3
    assert record["N_250"] == 7**3
    assert record["N_272"] == 5**3
    assert record["mask_voxels"] == 5**3
    assert record["N_1000"] == 0
    assert record["any_zero_count"]
    assert not record["non_monotonic_counts"]
    assert math.isfinite(record["S_G"])


def test_threshold_is_strictly_greater_than():
    assert stone_record(shell_stone(shell_hu=272.0), SETTINGS)["N_272"] == 5**3


def test_attenuation_values_are_separate_quantities():
    record = stone_record(shell_stone(core_hu=1000.0), SETTINGS)
    assert record["H"] == 1000.0
    assert record["T_HA"] == 500.0
    assert record["M"] == pytest.approx(math.log(500 / 272))
    assert record["ln_H"] == pytest.approx(math.log(1000))


def test_half_attenuation_volume_uses_the_stone_threshold():
    record = stone_record(shell_stone(core_hu=1000.0), SETTINGS)
    assert record["volume_HA_B"] == pytest.approx(5**3 * 0.000379525)


def test_every_axis_and_assumption_has_a_diameter():
    record = stone_record(shell_stone(), SETTINGS)
    for name in ("Z2", "Z1", "Z0"):
        for assumption in ("A", "B"):
            assert record[f"d_paper_{name}_{assumption}"] > 0


def test_geometry_matches_the_a0_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stone = shell_stone(voxvol=0.000344149)
    for assumption in ("A", "B"):
        assert assumption_geometry(stone, assumption, 0.7) == module.assumption_geometry(
            stone, assumption, 0.7
        )

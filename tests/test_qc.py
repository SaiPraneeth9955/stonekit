"""QC heuristics on hand-written stone records. Heuristics, not validated rules."""

from pathlib import Path

import pytest

from stonekit.config import load_yaml
from stonekit.pipeline import qc

RULES = load_yaml(Path(__file__).resolve().parents[1] / "configs" / "baseline_pipeline.yaml")["qc"]
SPACING = (0.7, 0.7, 2.5)


def stone(**changes):
    base = {
        "voxel_count": 500,
        "volume_mm3": 600.0,
        "volume_ml": 0.6,
        "slices_spanned": 6,
        "max_diameter_3d_mm": 12.0,
        "max_hu": 900.0,
        "hull_fill": 0.8,
        "touches_roi_box": False,
        "distance_to_kidney_mm": 0.0,
    }
    return {**base, **changes}


def test_ordinary_stone_has_no_flags():
    assert qc.stone_flags(stone(), RULES, SPACING) == []


@pytest.mark.parametrize(
    ("changes", "flag"),
    [
        ({"voxel_count": 5}, "very_small_component"),
        ({"volume_mm3": 5.0}, "very_small_component"),
        ({"volume_ml": 50.0}, "unusually_large_component"),
        ({"slices_spanned": 1}, "measurement_warning_one_slice_only"),
        ({"max_diameter_3d_mm": 5.0}, "measurement_warning_size_near_voxel_size"),
        ({"touches_roi_box": True}, "touches_roi_boundary"),
        ({"distance_to_kidney_mm": 12.0}, "outside_kidney_segmentation"),
        ({"max_hu": 3000.0}, "suspicious_very_high_density"),
        ({"hull_fill": 0.2}, "irregular_shape_possible_merged_stones"),
        ({"max_diameter_3d_mm": 150.0, "volume_mm3": 300.0}, "suspicious_long_thin_shape_possible_device"),
    ],
)
def test_each_flag_fires(changes, flag):
    assert flag in qc.stone_flags(stone(**changes), RULES, SPACING)


def test_small_stone_is_not_called_merged():
    assert "irregular_shape_possible_merged_stones" not in qc.stone_flags(
        stone(voxel_count=50, hull_fill=0.1), RULES, SPACING
    )


def test_method_flags():
    assert "no_stone_detected" in qc.method_flags([], 0, RULES)
    assert "tiny_components_dropped" in qc.method_flags([stone()], 2, RULES)
    assert "many_components" in qc.method_flags([stone()] * 50, 0, RULES)
    assert qc.method_flags([stone()], 0, RULES) == []


def test_kidney_flags():
    assert qc.kidney_flags(0, 0, False) == ["no_kidney_found"]
    assert qc.kidney_flags(10, 0, False) == ["only_one_kidney_found"]
    assert qc.kidney_flags(10, 10, True) == ["roi_reaches_image_edge"]
    assert qc.kidney_flags(10, 10, False) == []


def test_contrast_flags():
    assert qc.contrast_flags(None, RULES) == ["contrast_check_not_possible_no_aorta_found"]
    assert qc.contrast_flags(40.0, RULES) == []
    assert qc.contrast_flags(250.0, RULES) == ["contrast_suspected_bright_aorta"]

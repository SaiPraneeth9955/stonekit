"""QC flags for stones and cases.

These are research heuristics with limits in configs/baseline_pipeline.yaml. They are not validated
quality rules, not diagnostic statements and not calibrated uncertainty. A flag says "look at this".
"""

import math


def stone_flags(stone: dict, qc: dict, spacing_mm: tuple[float, float, float]) -> list[str]:
    """Flags for one measured stone. Needs `distance_to_kidney_mm` and `touches_roi_box`."""
    flags = []
    if stone["voxel_count"] < qc["very_small_voxels"] or stone["volume_mm3"] < qc["very_small_mm3"]:
        flags.append("very_small_component")
    if stone["volume_ml"] > qc["unusually_large_ml"]:
        flags.append("unusually_large_component")
    if stone["slices_spanned"] == 1:
        flags.append("measurement_warning_one_slice_only")
    if stone["max_diameter_3d_mm"] < 3 * max(spacing_mm):
        flags.append("measurement_warning_size_near_voxel_size")
    if stone["touches_roi_box"]:
        flags.append("touches_roi_boundary")
    if stone["distance_to_kidney_mm"] > qc["outside_kidney_mm"]:
        flags.append("outside_kidney_segmentation")
    diameter = stone["max_diameter_3d_mm"]
    if diameter >= qc["long_thin_min_diameter_mm"]:
        sphere = math.pi / 6 * diameter**3
        if stone["volume_mm3"] / sphere < qc["long_thin_max_sphere_fraction"]:
            flags.append("suspicious_long_thin_shape_possible_device")
    if stone["max_hu"] >= qc["metal_like_hu"]:
        flags.append("suspicious_very_high_density")
    fill = stone.get("hull_fill")
    enough_voxels = stone["voxel_count"] >= qc["merged_min_voxels"]
    if enough_voxels and fill is not None and fill < qc["merged_max_hull_fill"]:
        flags.append("irregular_shape_possible_merged_stones")
    return flags


def method_flags(stones: list[dict], dropped_pieces: int, qc: dict) -> list[str]:
    """Flags for one segmentation method on one scan."""
    flags = []
    if not stones:
        flags.append("no_stone_detected")
    if len(stones) > qc["many_stones"]:
        flags.append("many_components")
    if dropped_pieces:
        flags.append("tiny_components_dropped")
    return flags


def kidney_flags(right_voxels: int, left_voxels: int, roi_at_image_edge: bool) -> list[str]:
    flags = []
    if not right_voxels and not left_voxels:
        flags.append("no_kidney_found")
    elif not right_voxels or not left_voxels:
        flags.append("only_one_kidney_found")
    if roi_at_image_edge:
        flags.append("roi_reaches_image_edge")
    return flags


def contrast_flags(aorta_median_hu: float | None, qc: dict) -> list[str]:
    """Heuristic only. Blood is brighter in the aorta if contrast is circulating."""
    if aorta_median_hu is None:
        return ["contrast_check_not_possible_no_aorta_found"]
    if aorta_median_hu >= qc["contrast_aorta_median_hu"]:
        return ["contrast_suspected_bright_aorta"]
    return []

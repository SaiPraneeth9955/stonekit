"""Check a pipeline config before any output is written, so a bad config fails clearly and early.

It checks that every setting `run_case` reads is present and has a usable value. It does not judge
whether a value is scientifically right; that is recorded in docs/decisions.md.
"""

from numbers import Real

REQUIRED = {
    "input_checks": (
        "in_plane_mm", "slice_mm", "min_slices", "max_axis_dot", "max_tilt_degrees", "spacing_agreement_mm",
        "air_hu_at_most", "bone_hu_at_least", "saturation_hu", "max_saturated_fraction", "padding_values_hu",
    ),
    "roi": ("box_margin_mm",),
    "threshold_baseline": ("threshold_hu", "kidney_distance_mm"),
    "measurement": ("min_component_voxels",),
    "model": ("dataset_id", "dataset_name", "configuration", "plans", "trainer", "fold", "checkpoint"),
    "qc": (
        "very_small_voxels", "very_small_mm3", "unusually_large_ml", "outside_kidney_mm", "long_thin_min_diameter_mm",
        "long_thin_max_sphere_fraction", "metal_like_hu", "merged_min_voxels", "merged_max_hull_fill", "many_stones",
        "contrast_aorta_median_hu",
    ),
}  # fmt: skip
POSITIVE = {
    ("roi", "box_margin_mm"),
    ("threshold_baseline", "kidney_distance_mm"),
    ("measurement", "min_component_voxels"),
}


class ConfigError(ValueError):
    """The pipeline config cannot be used."""


def _number(value) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _optional_positive(config: dict, section: str, key: str) -> list[str]:
    value = (config.get(section) or {}).get(key)
    if value is not None and not (_number(value) and value > 0):
        return [f"{section}.{key} must be a positive number or absent, not {value!r}"]
    return []


def config_problems(config) -> list[str]:
    """Every problem found; an empty list means the config is usable."""
    if not isinstance(config, dict):
        return ["config is not a mapping"]
    problems = []
    for section, keys in REQUIRED.items():
        values = config.get(section)
        if not isinstance(values, dict):
            problems.append(f"missing section {section}")
            continue
        problems += [f"missing {section}.{key}" for key in keys if key not in values]
    for section, key in POSITIVE:
        value = (config.get(section) or {}).get(key)
        if value is not None and not (_number(value) and value > 0):
            problems.append(f"{section}.{key} must be a positive number, not {value!r}")
    problems += _optional_positive(config, "postprocessing", "model_max_distance_to_kidney_mm")
    problems += _optional_positive(config, "kidney_segmentation", "timeout_seconds")
    problems += _optional_positive(config, "model_inference", "timeout_seconds")
    return problems


def check_config(config) -> None:
    problems = config_problems(config)
    if problems:
        raise ConfigError("Invalid pipeline config: " + "; ".join(problems))

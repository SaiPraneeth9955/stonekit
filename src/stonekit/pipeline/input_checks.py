"""Input and geometry checks for one CT. Research heuristics, not validated quality rules.

The limits come from configs/baseline_pipeline.yaml. They describe what the Li audit found, so a
warning means "unlike the data this was built on", not that a scan is wrong. A failure means the
scan cannot be measured safely. Nothing here changes voxel values.
"""

from dataclasses import asdict, dataclass

import numpy as np

from stonekit.io.nifti import Volume

PASS = "pass"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str


@dataclass(frozen=True)
class InputReport:
    checks: tuple[Check, ...]
    shape: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    original_orientation: str
    hu_min: float
    hu_max: float
    padding_fraction: float

    @property
    def can_process(self) -> bool:
        return all(check.status != FAIL for check in self.checks)

    def to_dict(self) -> dict:
        return {**asdict(self), "can_process": self.can_process}


def _axis_checks(volume: Volume, limits: dict) -> list[Check]:
    directions = volume.affine[:3, :3] / np.linalg.norm(volume.affine[:3, :3], axis=0)
    off_axis = np.abs(directions.T @ directions - np.eye(3)).max()
    tilt = float(np.degrees(np.arccos(np.clip(np.abs(np.diag(directions)).min(), 0, 1))))
    checks = []
    if off_axis > limits["max_axis_dot"]:
        checks.append(Check("axes_orthogonal", FAIL, f"Axes are sheared (dot {off_axis:.4f})"))
    else:
        checks.append(Check("axes_orthogonal", PASS, "Axes are at right angles"))
    status = WARN if tilt > limits["max_tilt_degrees"] else PASS
    checks.append(Check("axes_aligned", status, f"Largest tilt from scanner axes {tilt:.2f} degrees"))
    return checks


def _spacing_checks(volume: Volume, limits: dict) -> list[Check]:
    spacing = volume.spacing_mm
    if min(spacing) <= 0:
        return [Check("spacing_positive", FAIL, f"Spacing {spacing} is not positive")]
    checks = [Check("spacing_positive", PASS, f"Spacing {tuple(round(s, 4) for s in spacing)} mm")]

    # The header is stored in the file's axis order, and the affine here is in RAS order.
    header = sorted(volume.original_header_zooms)
    agrees = np.allclose(header, sorted(spacing), atol=limits["spacing_agreement_mm"])
    checks.append(
        Check(
            "spacing_header_matches_affine",
            PASS if agrees else WARN,
            f"Header {volume.original_header_zooms}",
        )
    )
    low, high = limits["in_plane_mm"]
    in_plane = low <= min(spacing[:2]) and max(spacing[:2]) <= high
    checks.append(
        Check("in_plane_spacing_in_range", PASS if in_plane else WARN, f"Expected {low} to {high} mm")
    )
    low, high = limits["slice_mm"]
    in_range = low <= spacing[2] <= high
    checks.append(
        Check("slice_spacing_in_range", PASS if in_range else WARN, f"Expected {low} to {high} mm")
    )
    return checks


def check_ct(volume: Volume, limits: dict) -> InputReport:
    """Check one CT that is already loaded."""
    data = volume.data
    checks = _axis_checks(volume, limits) + _spacing_checks(volume, limits)

    finite = bool(np.isfinite(data).all())
    checks.append(
        Check("values_finite", PASS if finite else FAIL, "No NaN or infinite voxels" if finite else "NaN or infinite voxels")
    )
    slices = data.shape[2]
    checks.append(
        Check(
            "enough_slices",
            PASS if slices >= limits["min_slices"] else FAIL,
            f"{slices} slices; at least {limits['min_slices']} needed",
        )
    )

    hu_min = float(np.nanmin(data)) if finite else float("nan")
    hu_max = float(np.nanmax(data)) if finite else float("nan")
    looks_like_hu = finite and hu_min <= limits["air_hu_at_most"] and hu_max >= limits["bone_hu_at_least"]
    checks.append(
        Check(
            "values_look_like_hu",
            PASS if looks_like_hu else WARN,
            f"Range {hu_min:.0f} to {hu_max:.0f}. Needs air-like and bone-like values",
        )
    )
    saturated = float(np.mean(data >= limits["saturation_hu"]))
    checks.append(
        Check(
            "saturation",
            WARN if saturated > limits["max_saturated_fraction"] else PASS,
            f"{saturated:.6f} of voxels at or above {limits['saturation_hu']} HU",
        )
    )
    padding = float(np.isin(data, limits["padding_values_hu"]).mean())
    return InputReport(
        checks=tuple(checks),
        shape=tuple(int(v) for v in data.shape),
        spacing_mm=volume.spacing_mm,
        original_orientation="".join(volume.original_codes),
        hu_min=hu_min,
        hu_max=hu_max,
        padding_fraction=padding,
    )

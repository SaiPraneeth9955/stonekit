"""Per-stone signals (preregistration, section 7). Masks stay in memory and are never saved."""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from stonekit.measure.components import StoneMask, stone_mask
from stonekit.measure.density import hu_statistics
from stonekit.measure.diameter import max_distance_mm, spacing_for_assignment
from stonekit.measure.volume import volume_ml
from stonekit.seg.threshold import threshold_mask

ASSUMPTIONS = ("A", "B")
AXES = ("Z2", "Z1", "Z0")


def segment(image: np.ndarray, threshold_hu: float, min_voxels: int) -> StoneMask:
    """The frozen pipeline in docs/STAGE2_PROTOCOL.md, section 4."""
    return stone_mask(threshold_mask(image, threshold_hu), min_voxels)


def threshold_sensitivity(counts: dict[int, int]) -> float:
    """S_G: the negative least-squares slope of ln(N_t + 1) on ln(t) over the grid."""
    thresholds = np.log(np.array(sorted(counts), float))
    values = np.log(np.array([counts[t] for t in sorted(counts)], float) + 1)
    centred = thresholds - thresholds.mean()
    return float(-(centred * (values - values.mean())).sum() / (centred**2).sum())


def local_elasticity(low_count: int, high_count: int, low_hu: float, high_hu: float) -> float:
    """Exploratory and post-hoc local elasticity between two thresholds (section 12)."""
    return abs(math.log((low_count + 1) / (high_count + 1))) / math.log(high_hu / low_hu)


def half_attenuation_threshold(median_hu: float) -> float:
    """T_HA, a measurement parameter used only in the half-attenuation comparator."""
    return 0.5 * median_hu


def attenuation_signal(median_hu: float, threshold_hu: float) -> float:
    """M, the attenuation-derived signal. It is not a direct physical density measurement."""
    return abs(math.log(half_attenuation_threshold(median_hu) / threshold_hu))


def assumption_geometry(stone, assumption: str, z_mm: float) -> tuple[float, float, float]:
    """In-plane spacing, z spacing and voxel volume (mL), as in the a0 run."""
    if assumption == "A":
        x_mm, _, voxsiz_z_mm = stone.voxsiz_mm
        return x_mm, voxsiz_z_mm, float(np.prod(stone.voxsiz_mm)) / 1000
    return math.sqrt(stone.voxvol_ml * 1000 / z_mm), z_mm, stone.voxvol_ml


@dataclass(frozen=True)
class SignalSettings:
    primary_threshold_hu: int
    grid_hu: tuple[int, ...]
    local_pairs: tuple[tuple[int, int], ...]
    min_component_voxels: int
    assumed_z_spacing_mm: float
    paper_voxel_edge_mm: float


def stone_record(stone, settings: SignalSettings) -> dict:
    """Recompute one stone in memory: reconciliation values, signals and comparators."""
    image = stone.image
    primary = segment(image, settings.primary_threshold_hu, settings.min_component_voxels)
    empty = not primary.mask.any()
    counts = {
        threshold: int(segment(image, threshold, settings.min_component_voxels).mask.sum())
        for threshold in {*settings.grid_hu, *(hu for pair in settings.local_pairs for hu in pair)}
    }
    grid_counts = {threshold: counts[threshold] for threshold in settings.grid_hu}
    ordered = [grid_counts[threshold] for threshold in sorted(grid_counts)]
    median_hu = float("nan") if empty else hu_statistics(image, primary.mask)["median_hu"]

    record = {
        "mask_voxels": int(primary.mask.sum()),
        "n_components": primary.n_components,
        "filled_voxels": primary.filled_voxels,
        "S_G": threshold_sensitivity(grid_counts),
        "any_zero_count": any(count == 0 for count in ordered),
        "non_monotonic_counts": any(b > a for a, b in pairwise(ordered)),
        "H": median_hu,
        "ln_H": math.log(median_hu) if median_hu > 0 else float("nan"),
        "T_HA": half_attenuation_threshold(median_hu),
        "M": (
            float("nan") if empty else attenuation_signal(median_hu, settings.primary_threshold_hu)
        ),
    }
    for threshold, count in sorted(grid_counts.items()):
        record[f"N_{threshold}"] = count
    for low, high in settings.local_pairs:
        record[f"S_{low}_{high}"] = local_elasticity(counts[low], counts[high], low, high)

    half_mask = None
    if not empty:
        half_mask = segment(image, record["T_HA"], settings.min_component_voxels).mask
    for assumption in ASSUMPTIONS:
        inplane, z_mm, voxel_ml = assumption_geometry(
            stone, assumption, settings.assumed_z_spacing_mm
        )
        record[f"volume_{assumption}"] = volume_ml(primary.mask, voxel_ml)
        record[f"volume_HA_{assumption}"] = (
            float("nan") if half_mask is None else volume_ml(half_mask, voxel_ml)
        )
        for axis in AXES:
            spacing = spacing_for_assignment(inplane, z_mm, axis)
            edge = 0.0 if empty else settings.paper_voxel_edge_mm
            record[f"d_paper_{axis}_{assumption}"] = max_distance_mm(primary.mask, spacing) + edge
    return record

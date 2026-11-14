"""The four segmentation methods of Neubauer et al. 2023 and their split-based optimisation.

Protocol: docs/FREIBURG_REPLICATION_PROTOCOL.md (rep1). Interpretations of details the paper does not
state are fixed there; this module implements them and nothing else.

- M1: fixed threshold P1 (HU).
- M2: M1 plus surface partial-volume correction: volume = (N - P2 * S) * voxel volume.
- M3: adaptive threshold P1 * median HU of a rough (> 20 HU) segmentation.
- M4: M3 plus the same surface correction.
Masks follow a0 exactly: voxels strictly above the threshold, 26-connected components of at least
`min_voxels`, union of components, holes filled. Diameter is the largest centre-to-centre distance
plus one voxel edge (the paper's convention); the correction changes volume only.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from stonekit.measure.components import stone_mask
from stonekit.measure.diameter import axial_max_distance_mm, max_distance_mm
from stonekit.seg.threshold import threshold_mask

ROUGH_THRESHOLD_HU = 20.0
COST_VOLUME_WEIGHT = 4.0
STRUCTURES = {6: ndimage.generate_binary_structure(3, 1), 26: np.ones((3, 3, 3), dtype=bool)}


def surface_voxels(mask: np.ndarray, connectivity: int = 6) -> int:
    """Mask voxels with at least one neighbour outside the mask (the patch border counts as outside)."""
    if not mask.any():
        return 0
    eroded = ndimage.binary_erosion(mask, structure=STRUCTURES[connectivity], border_value=0)
    return int((mask & ~eroded).sum())


def rough_median_hu(image: np.ndarray) -> float:
    """Median HU of the rough segmentation (voxels above 20 HU). NaN if nothing is above 20 HU."""
    rough = np.asarray(image) > ROUGH_THRESHOLD_HU
    return float(np.median(np.asarray(image)[rough])) if rough.any() else float("nan")


@dataclass(frozen=True)
class Geometry:
    """One spacing interpretation: per-axis spacing (mm) and the axis treated as z."""

    spacing_mm: tuple[float, float, float]
    z_axis: int


@dataclass(frozen=True)
class Profile:
    """Per-threshold measurements of one stone. Diameters (mm, with the voxel edge) per geometry name."""

    thresholds: np.ndarray
    voxels: np.ndarray
    surface_6: np.ndarray
    surface_26: np.ndarray
    diameter_3d_mm: dict
    diameter_axial_mm: dict


def stone_profile(
    image: np.ndarray, thresholds, geometries: dict[str, Geometry], edge_mm: float, min_voxels: int = 3
) -> Profile:
    """Measure one stone at each threshold with the a0 mask rules, once per mask for every geometry.

    Voxel and surface counts do not depend on spacing. An empty mask gives zeros (no voxel edge added).
    """
    thresholds = np.asarray(thresholds, dtype=float)
    n = len(thresholds)
    voxels, s6, s26 = np.zeros(n), np.zeros(n), np.zeros(n)
    d3 = {name: np.zeros(n) for name in geometries}
    d2 = {name: np.zeros(n) for name in geometries}
    for i, threshold in enumerate(thresholds):
        mask = stone_mask(threshold_mask(image, threshold), min_voxels).mask
        if not mask.any():
            continue
        voxels[i] = mask.sum()
        s6[i] = surface_voxels(mask, 6)
        s26[i] = surface_voxels(mask, 26)
        for name, geometry in geometries.items():
            d3[name][i] = max_distance_mm(mask, geometry.spacing_mm) + edge_mm
            d2[name][i] = axial_max_distance_mm(mask, geometry.spacing_mm, geometry.z_axis) + edge_mm
    return Profile(thresholds, voxels, s6, s26, d3, d2)


def corrected_volume_ml(voxels, surface, factor, voxel_ml: float):
    """Surface partial-volume correction: (N - factor * S) * voxel volume. Broadcasts over arrays."""
    return (np.asarray(voxels) - np.asarray(factor) * np.asarray(surface)) * voxel_ml


def cost(volume_ml, diameter_mm, reference_volume_ml, reference_diameter_mm, axis: int = 0):
    """The paper's cost, 4 |Ve - Vr| + |De - Dr|, summed over stones (along `axis`)."""
    v = np.abs(np.asarray(volume_ml) - np.asarray(reference_volume_ml))
    d = np.abs(np.asarray(diameter_mm) - np.asarray(reference_diameter_mm))
    return (COST_VOLUME_WEIGHT * v + d).sum(axis=axis)


def optimise(volume_ml: np.ndarray, diameter_mm: np.ndarray, reference_volume_ml, reference_diameter_mm, train) -> int:
    """Index (flattened over the parameter axes) of the lowest training cost; ties go to the first.

    `volume_ml` and `diameter_mm` have shape (stones, *parameter grid*).
    """
    train = np.asarray(train)
    shape = (len(train),) + (1,) * (volume_ml.ndim - 1)
    total = cost(volume_ml[train], diameter_mm[train], np.reshape(np.asarray(reference_volume_ml)[train], shape),
                 np.reshape(np.asarray(reference_diameter_mm)[train], shape))  # fmt: skip
    return int(np.argmin(total.ravel()))


def per_stone_cost(volume_ml: np.ndarray, diameter_mm: np.ndarray, reference_volume_ml, reference_diameter_mm) -> np.ndarray:
    """Each stone's cost at every grid point, flattened to (stones, grid points)."""
    n = volume_ml.shape[0]
    shape = (n,) + (1,) * (volume_ml.ndim - 1)
    c = COST_VOLUME_WEIGHT * np.abs(volume_ml - np.reshape(reference_volume_ml, shape)) + np.abs(
        diameter_mm - np.reshape(reference_diameter_mm, shape)
    )
    return c.reshape(n, -1)


def optimise_splits(costs: np.ndarray, splits) -> np.ndarray:
    """Grid index of the lowest training cost for every split at once (ties go to the first index).

    Equivalent to `optimise` per split: the training cost is the sum of per-stone costs.
    """
    indicator = np.zeros((len(splits), costs.shape[0]))
    for k, (train, _) in enumerate(splits):
        indicator[k, train] = 1.0
    return np.argmin(indicator @ costs, axis=1)


def random_splits(n: int, n_train: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """`n_splits` random train/validation partitions of n items, reproducible from `seed`."""
    rng = np.random.default_rng(seed)
    splits = []
    for _ in range(n_splits):
        order = rng.permutation(n)
        splits.append((np.sort(order[:n_train]), np.sort(order[n_train:])))
    return splits


def validation_stats(estimate, reference, icc_function) -> dict:
    """ICC, mean difference (estimate - reference) and SD of the difference on one validation set."""
    estimate = np.asarray(estimate, float)
    reference = np.asarray(reference, float)
    difference = estimate - reference
    return {
        "icc": float(icc_function(estimate, reference)),
        "difference_estimate_minus_reference": float(difference.mean()),
        "sd": float(difference.std(ddof=1)),
    }

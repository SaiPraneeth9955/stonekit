"""Primary evaluation: repeated stratified folds and the estimand psi (sections 9 to 11).

A row is one stone, or one copy of a stone in a bootstrap resample. Folds are built on unique
stones, so all copies of a stone stay together. Nothing here uses a held-out outcome to fit,
flag or assign folds.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import rankdata

from stonekit.reliability.model import ScaleFit, fit_scale_model, predict_scale

N_FOLDS = 5


def trapezoid_weights(ks: tuple[int, ...]) -> np.ndarray:
    """Trapezoid area weights over equally spaced k. For k = 2..6: 1/8, 1/4, 1/4, 1/4, 1/8."""
    weights = np.ones(len(ks))
    weights[[0, -1]] = 0.5
    return weights / weights.sum()


def stratified_folds(
    x1: np.ndarray, stone_id: np.ndarray, rng: np.random.Generator, n_folds: int = N_FOLDS
) -> np.ndarray:
    """Assign unique stones to folds, one stone per consecutive block of sorted size.

    Sort by ascending x1 with ties by stone_id. Remaining largest stones go to random folds.
    """
    order = np.lexsort((stone_id, x1))
    folds = np.empty(len(order), dtype=int)
    n_blocks = len(order) // n_folds
    for block in range(n_blocks):
        folds[order[block * n_folds : (block + 1) * n_folds]] = rng.permutation(n_folds)
    leftover = order[n_blocks * n_folds :]
    folds[leftover] = rng.permutation(n_folds)[: len(leftover)]
    return folds


def fold_ranks(
    priority: np.ndarray, stone_id: np.ndarray, copy: np.ndarray, fold: np.ndarray
) -> np.ndarray:
    """Rank rows within each fold. Rank 0 is flagged first, and a larger priority goes earlier.

    Ties break by ascending stone_id, then copy number.
    """
    ranks = np.empty(len(priority), dtype=int)
    for current in np.unique(fold):
        rows = np.flatnonzero(fold == current)
        order = np.lexsort((copy[rows], stone_id[rows], -priority[rows]))
        ranks[rows[order]] = np.arange(len(rows))
    return ranks


def accepted_mean(errors: np.ndarray, ranks: np.ndarray, k: int) -> float:
    """Mean error over stones not among the k flagged in their fold."""
    return float(errors[ranks >= k].mean())


def delta_by_k(
    errors: np.ndarray, baseline: np.ndarray, proposed: np.ndarray, ks: tuple[int, ...]
) -> np.ndarray:
    """Baseline rule's accepted mean minus the proposed rule's, at each k."""
    return np.array(
        [accepted_mean(errors, baseline, k) - accepted_mean(errors, proposed, k) for k in ks]
    )


@dataclass(frozen=True)
class ModelSpec:
    """A fitted rule: its predictor columns and the outcome column it is fitted on."""

    predictors: tuple[str, ...]
    outcome: str


@dataclass(frozen=True)
class Comparison:
    """A difference in accepted mean of `errors`, baseline rule minus proposed rule."""

    key: str
    errors: str
    baseline: str
    proposed: str


@dataclass
class Rows:
    stone_id: np.ndarray
    copy: np.ndarray
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    def take(self, indices: np.ndarray) -> "Rows":
        return Rows(
            self.stone_id[indices],
            copy_numbers(self.stone_id[indices]),
            {name: values[indices] for name, values in self.columns.items()},
        )

    def __len__(self) -> int:
        return len(self.stone_id)


def copy_numbers(stone_id: np.ndarray) -> np.ndarray:
    """Number repeated stones 0, 1, 2 ... in row order."""
    seen: dict[int, int] = {}
    numbers = np.empty(len(stone_id), dtype=int)
    for position, stone in enumerate(stone_id.tolist()):
        numbers[position] = seen.get(stone, 0)
        seen[stone] = numbers[position] + 1
    return numbers


def row_folds(rows: Rows, rng: np.random.Generator, n_folds: int = N_FOLDS) -> np.ndarray:
    """Stratified folds on unique stones, mapped back to every copy."""
    unique, first = np.unique(rows.stone_id, return_index=True)
    folds = stratified_folds(rows.columns["x1"][first], unique, rng, n_folds)
    return folds[np.searchsorted(unique, rows.stone_id)]


def design_matrix(rows: Rows, predictors: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([rows.columns[name] for name in predictors])


def model_ranks(
    rows: Rows, fold: np.ndarray, spec: ModelSpec
) -> tuple[np.ndarray | None, list[ScaleFit]]:
    """Fit on stones outside each fold and rank the held-out stones. None if any fit fails."""
    predictors = design_matrix(rows, spec.predictors)
    errors = rows.columns[spec.outcome]
    scores = np.empty(len(rows))
    fits = []
    for current in np.unique(fold):
        held_out = fold == current
        fit = fit_scale_model(predictors[~held_out], errors[~held_out])
        fits.append(fit)
        if not fit.converged:
            return None, fits
        scores[held_out] = predict_scale(fit, predictors[held_out])
    return fold_ranks(scores, rows.stone_id, rows.copy, fold), fits


def size_only_ranks(rows: Rows, fold: np.ndarray) -> np.ndarray:
    """The unfitted comparator: smallest measured size is flagged first."""
    return fold_ranks(-rows.columns["x1"], rows.stone_id, rows.copy, fold)


def rank_score_ranks(rows: Rows, fold: np.ndarray) -> np.ndarray:
    """Exploratory zero-parameter score: equal-weight within-fold ranks, with assumed signs."""
    priority = np.empty(len(rows))
    for current in np.unique(fold):
        held_out = fold == current
        priority[held_out] = (
            rankdata(-rows.columns["x1"][held_out])
            + rankdata(rows.columns["S_G"][held_out])
            + rankdata(rows.columns["M"][held_out])
        )
    return fold_ranks(priority, rows.stone_id, rows.copy, fold)


@dataclass
class PartitionResult:
    ranks: dict[str, np.ndarray | None]
    fits: dict[str, list[ScaleFit]]
    deltas: dict[str, np.ndarray]
    fold: np.ndarray


def evaluate_partition(
    rows: Rows,
    fold: np.ndarray,
    models: dict[str, ModelSpec],
    comparisons: list[Comparison],
    ks: tuple[int, ...],
) -> PartitionResult:
    """All rules on one fold assignment. A comparison with a failed rule is NaN at every k."""
    ranks: dict[str, np.ndarray | None] = {"size": size_only_ranks(rows, fold)}
    if any(comparison.proposed == "rank_score" for comparison in comparisons):
        ranks["rank_score"] = rank_score_ranks(rows, fold)
    fits: dict[str, list[ScaleFit]] = {}
    for name, spec in models.items():
        ranks[name], fits[name] = model_ranks(rows, fold, spec)

    too_small = np.bincount(fold).min() <= max(ks)
    deltas = {}
    for comparison in comparisons:
        baseline = ranks[comparison.baseline]
        proposed = ranks[comparison.proposed]
        if too_small or baseline is None or proposed is None:
            deltas[comparison.key] = np.full(len(ks), np.nan)
        else:
            deltas[comparison.key] = delta_by_k(
                rows.columns[comparison.errors], baseline, proposed, ks
            )
    return PartitionResult(ranks, fits, deltas, fold)


def mean_deltas(
    rows: Rows,
    n_partitions: int,
    rng: np.random.Generator,
    models: dict[str, ModelSpec],
    comparisons: list[Comparison],
    ks: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Average each comparison's differences by k over fold partitions."""
    totals = {comparison.key: np.zeros(len(ks)) for comparison in comparisons}
    for _ in range(n_partitions):
        result = evaluate_partition(rows, row_folds(rows, rng), models, comparisons, ks)
        for key, values in result.deltas.items():
            totals[key] += values
    return {key: values / n_partitions for key, values in totals.items()}


def psi(deltas: np.ndarray, weights: np.ndarray) -> float:
    """The primary summary: the trapezoid area of the differences over k."""
    return float(np.dot(deltas, weights))


def section_11_reading(lower: float, upper: float) -> str:
    """The pre-written reading of an interval. No result counts as passing or failing."""
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "Interval unavailable"
    if lower > 0:
        return (
            "The signals added measurable ranking information in this dataset, on supplied "
            "references. The size of the effect is the estimate shown"
        )
    if upper < 0:
        return "Size-only performed better in this dataset"
    return (
        "The data are compatible with no added information and with effects up to the "
        "interval's limits. This does not show the signals add nothing"
    )


def spacing_dependent(psi_a: float, psi_b: float, a: tuple, b: tuple) -> bool:
    """True if estimates differ in sign, or only one 95% interval excludes zero."""
    excludes_a = a[0] > 0 or a[1] < 0
    excludes_b = b[0] > 0 or b[1] < 0
    return bool(np.sign(psi_a) != np.sign(psi_b) or excludes_a != excludes_b)


def scale_dependent(psi_log: float, psi_percent: float) -> bool:
    return bool(np.sign(psi_log) != np.sign(psi_percent))


def training_threshold_summary(
    rows: Rows,
    fold: np.ndarray,
    spec: ModelSpec,
    errors: np.ndarray,
    burden: float,
) -> dict:
    """Exploratory single-stone check: flag held-out stones using a training-fold threshold."""
    x1 = rows.columns["x1"]
    predictors = design_matrix(rows, spec.predictors)
    outcome = rows.columns[spec.outcome]
    flagged_model = np.zeros(len(rows), dtype=bool)
    flagged_size = np.zeros(len(rows), dtype=bool)
    for current in np.unique(fold):
        held_out = fold == current
        fit = fit_scale_model(predictors[~held_out], outcome[~held_out])
        if not fit.converged:
            return {"model_accepted_mean": np.nan, "size_accepted_mean": np.nan,
                    "model_realized_burden": np.nan, "size_realized_burden": np.nan}
        limit = np.quantile(predict_scale(fit, predictors[~held_out]), 1 - burden)
        flagged_model[held_out] = predict_scale(fit, predictors[held_out]) >= limit
        flagged_size[held_out] = x1[held_out] <= np.quantile(x1[~held_out], burden)
    return {
        "model_accepted_mean": float(errors[~flagged_model].mean()),
        "size_accepted_mean": float(errors[~flagged_size].mean()),
        "model_realized_burden": float(flagged_model.mean()),
        "size_realized_burden": float(flagged_size.mean()),
    }

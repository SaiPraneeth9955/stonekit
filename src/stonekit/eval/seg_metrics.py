"""Overlap and piece-level detection metrics between a prediction and a reference mask.

The reference here is an annotation, which is not physical truth. A piece counts as detected when it
shares at least one voxel with the other mask: a lenient rule, stated wherever results are reported.
"""

import numpy as np


def dice(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    """None when both masks are empty, because Dice is then undefined."""
    total = int(prediction.sum()) + int(reference.sum())
    return 2 * int((prediction & reference).sum()) / total if total else None


def iou(prediction: np.ndarray, reference: np.ndarray) -> float | None:
    union = int((prediction | reference).sum())
    return int((prediction & reference).sum()) / union if union else None


def overlap_pairs(predicted_labels: np.ndarray, reference_labels: np.ndarray) -> dict[tuple[int, int], int]:
    """Voxels shared by each overlapping (predicted piece, reference piece) pair."""
    # Stones are sparse: index the reference's voxels once instead of building full-volume masks.
    reference_flat = np.asarray(reference_labels).reshape(-1)
    index = np.flatnonzero(reference_flat)
    predicted = np.asarray(predicted_labels).reshape(-1)[index]
    keep = predicted > 0
    if not keep.any():
        return {}
    pairs, counts = np.unique(
        np.stack([predicted[keep], reference_flat[index][keep]], axis=1), axis=0, return_counts=True
    )
    return {(int(p), int(r)): int(c) for (p, r), c in zip(pairs, counts, strict=True)}


def detection_counts(predicted_labels: np.ndarray, reference_labels: np.ndarray) -> dict[str, int]:
    pairs = overlap_pairs(predicted_labels, reference_labels)
    predicted_ids = set(np.unique(predicted_labels).tolist()) - {0}
    reference_ids = set(np.unique(reference_labels).tolist()) - {0}
    hit_reference = {r for _, r in pairs}
    hit_predicted = {p for p, _ in pairs}
    return {
        "reference_pieces": len(reference_ids),
        "reference_pieces_detected": len(reference_ids & hit_reference),
        "predicted_pieces": len(predicted_ids),
        "predicted_pieces_without_reference": len(predicted_ids - hit_predicted),
    }


def best_matches(predicted_labels: np.ndarray, reference_labels: np.ndarray) -> dict[int, int]:
    """For each reference piece, the predicted piece sharing the most voxels. Ties take the lower ID."""
    best: dict[int, tuple[int, int]] = {}
    for (predicted, reference), count in sorted(overlap_pairs(predicted_labels, reference_labels).items()):
        if reference not in best or count > best[reference][1]:
            best[reference] = (predicted, count)
    return {reference: predicted for reference, (predicted, _) in best.items()}

"""Hand-checkable tests of the statistics in scripts/analyse_stone_validation.py and
scripts/compare_models_paired.py. Tiny made-up rows, not data; never used for any result."""

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyse = load("analyse_stone_validation")
paired = load("compare_models_paired")
characterise = load("characterise_detection")
conformal = load("conformal_exploratory")


def test_logistic_fit_and_size_inverse_by_hand():
    # Exactly balanced outcomes at each x make the maximum-likelihood fit known in closed form:
    # at x = 1 (2 mm) detection 1/5, at x = 3 (8 mm) detection 4/5, so the 50% point is x = 2 (4 mm).
    x = np.array([1.0] * 5 + [3.0] * 5)
    y = np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)
    beta = characterise.logistic_fit(x, y)
    assert characterise.size_at(beta, 0.5) == pytest.approx(4.0, rel=1e-6)
    assert characterise.size_at(beta, 0.8) == pytest.approx(8.0, rel=1e-6)
    assert characterise.size_at(beta, 0.2) == pytest.approx(2.0, rel=1e-6)


@pytest.mark.parametrize(("n", "level", "rank"), [(9, 0.9, 9), (19, 0.9, 18), (4, 0.8, 4), (3, 0.9, None)])
def test_conformal_quantile_uses_the_finite_sample_rank(n, level, rank):
    scores = np.arange(1, n + 1, dtype=float)
    q = conformal.quantile(scores, level)
    assert q == (np.inf if rank is None else float(rank))


def test_category_bins_are_research_bins_with_closed_lower_edges():
    assert [characterise.category(d) for d in (4.99, 5.0, 9.99, 10.0, 19.99, 20.0)] == [
        "<5 mm", "5-<10 mm", "5-<10 mm", "10-<20 mm", "10-<20 mm", ">=20 mm"]  # fmt: skip


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, "<3 mm"), (2.999, "<3 mm"), (3.0, "3-<5 mm"), (4.999, "3-<5 mm"), (5.0, "5-<10 mm"), (10.0, ">=10 mm"), (80.0, ">=10 mm")],
)
def test_size_band_edges(value, expected):
    assert analyse.band(value, analyse.SIZE_BANDS) == expected


@pytest.mark.parametrize(("value", "expected"), [(399.9, "<400 HU"), (400.0, "400-<800 HU"), (800.0, ">=800 HU")])
def test_peak_hu_band_edges(value, expected):
    assert analyse.band(value, analyse.HU_BANDS) == expected


@pytest.mark.parametrize(("value", "expected"), [(0.625, "<1.0 mm"), (1.25, "1.0-<1.5 mm"), (2.0, "1.5-<2.2 mm"), (2.5, ">=2.2 mm")])
def test_slice_spacing_band_edges(value, expected):
    assert analyse.band(value, analyse.SPACING_BANDS) == expected


def rows(spec):
    """spec: {case: [kinds]} -> annotated rows."""
    return [{"case": c, "kind": k} for c, kinds in spec.items() for k in kinds]


def test_bootstrap_ratio_pools_pieces_and_resamples_scans():
    data = rows({"a": ["one_to_one", "missed"], "b": ["split"], "c": ["missed", "missed", "missed"]})
    result = analyse.bootstrap_ratio(data, analyse.detected, lambda r: 1)
    assert (result["numerator"], result["denominator"], result["value"], result["scans"]) == (2, 6, round(2 / 6, 4), 3)
    # Recompute the interval by hand: resample scans, keep all of each scan's pieces.
    per = np.array([[1, 2], [1, 1], [0, 3]], dtype=float)
    draws = np.random.default_rng(analyse.SEED).integers(0, 3, size=(analyse.RESAMPLES, 3))
    sums = per[draws].sum(axis=1)
    expected = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5])
    assert result["interval_95"] == [round(float(v), 4) for v in expected]
    assert "small_sample_warning" in result  # 3 scans


def test_bootstrap_mean_uses_every_resample_and_scan_clusters():
    data = [{"case": "a", "x": 1.0}, {"case": "a", "x": 3.0}, {"case": "b", "x": 10.0}] + [
        {"case": f"s{i}", "x": float(i)} for i in range(5)
    ]
    result = analyse.bootstrap_mean(data, lambda r: r["x"])
    assert result["bootstrap_resamples"] == 10000
    assert result["n"] == 8 and result["scans"] == 7
    assert result["mean"] == round(float(np.mean([r["x"] for r in data])), 3)
    cases = sorted({r["case"] for r in data})
    by_case = [[r["x"] for r in data if r["case"] == c] for c in cases]
    draws = np.random.default_rng(analyse.SEED).integers(0, len(cases), size=(10000, len(cases)))
    means = [np.mean(np.concatenate([by_case[i] for i in d])) for d in draws]
    assert result["mean_interval_95"] == [round(float(np.percentile(means, q)), 3) for q in (2.5, 97.5)]
    values = np.array([r["x"] for r in data])
    sd = values.std(ddof=1)
    assert result["limits_of_agreement"] == [round(float(values.mean() - 1.96 * sd), 3), round(float(values.mean() + 1.96 * sd), 3)]
    assert "small_sample_warning" not in result


def test_relative_volume_difference_definition():
    # 200 * (p - a) / (p + a): +/-200 at the extremes, symmetric, zero when equal.
    pct = lambda p, a: 200 * (p - a) / (p + a)
    assert pct(100, 100) == 0
    assert pct(50, 150) == -100
    assert pct(150, 50) == 100


def write(folder: Path, name: str, rows_: list[dict]) -> None:
    with (folder / f"{name}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows_[0]))
        writer.writeheader()
        writer.writerows(rows_)


def detail(folder: Path, method: str, scans: list[tuple], annotated: list[tuple], pairs: list[tuple]) -> None:
    folder.mkdir()
    write(folder, "scans", [{"case": c, "method": method, "has_stone": str(h), "dice": d, "false_positive_pieces": fp} for c, h, d, fp in scans])
    write(folder, "annotated", [{"case": c, "method": method, "kind": k, "d3_mm": d} for c, k, d in annotated])
    write(folder, "pairs", [{"case": c, "method": method, "predicted_mm3": p, "annotated_mm3": a} for c, p, a in pairs])


def test_paired_scores_and_counts_by_hand(tmp_path):
    detail(tmp_path / "a", "model",
           [("s1", True, 0.5, 1), ("s2", True, 0.7, 0), ("s3", False, "", 2)],
           [("s1", "one_to_one", 6.0), ("s1", "missed", 2.0), ("s2", "one_to_one", 4.0)],
           [("s1", 50, 100), ("s2", 100, 100)])  # fmt: skip
    scans = paired.per_scan(tmp_path / "a", "model")
    point = paired.score(scans, ["s1", "s2", "s3"])
    assert point["detection"] == pytest.approx(2 / 3)
    assert point["fp_per_scan"] == pytest.approx(1.0)
    assert point["dice_scans_with_stone"] == pytest.approx(0.6)
    assert point["detection <3 mm"] == 0 and point["detection 3-<5 mm"] == 1
    assert point["volume_bias_pct"] == pytest.approx((200 * -50 / 150 + 0) / 2)
    counts = paired.counts(scans, ["s1", "s2", "s3"])
    assert counts["detected"] == 2 and counts["annotated"] == 3 and counts["false_positive_pieces"] == 3
    assert counts["one_to_one_pairs"] == 2

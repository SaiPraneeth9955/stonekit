"""Characterise detection, measurement error, size categories and the kidney-distance filter.

Reads the saved per-piece rows of scripts/analyse_stone_validation.py (data/interim/stone_analysis/
<stamp>/). Reads no image. Development evidence on the 32 fold-0 validation scans, which were also
used for model selection: EXPLORATORY, not confirmatory. Written 8 November 2026, after the
aggregate results were known; every analysis here is descriptive characterisation, and nothing
here selects or tunes a model.

Statistics: pieces within a scan are not independent, so every interval is a percentile bootstrap
over scans (seed 20261105, 2000 resamples for the curve fits, 10000 for ratios), and the regression
uses GEE with an independence working correlation and robust standard errors clustered by scan.
Size categories (<5, 5-<10, 10-<20, >=20 mm maximum diameter) are research bins, not clinical
management thresholds.
"""

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from stonekit.config import create_run_folder

SEED = 20261105
CURVE_RESAMPLES = 2000
RATIO_RESAMPLES = 10000
CATEGORY_EDGES = [0, 5, 10, 20, np.inf]
CATEGORY_NAMES = ["<5 mm", "5-<10 mm", "10-<20 mm", ">=20 mm"]
FILTER_DISTANCES = [None, 10, 15, 20, 25, 30, 40, 50]


def read(folder: Path, name: str, method: str) -> list[dict]:
    with (folder / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["method"] == method]
    for r in rows:
        for k, v in list(r.items()):
            if k in ("case", "method", "kind", "qc_flags"):
                continue
            if k == "has_stone":
                r[k] = v == "True"
            elif v not in ("", "None"):
                r[k] = float(v)
            else:
                r[k] = None
    return rows


def scan_bootstrap(rows: list[dict], statistic, resamples: int, seed: int = SEED):
    """Percentile interval of statistic(rows) with scans resampled (all pieces of a scan kept)."""
    cases = sorted({r["case"] for r in rows})
    by_case = {c: [r for r in rows if r["case"] == c] for c in cases}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sample = [r for i in rng.integers(0, len(cases), len(cases)) for r in by_case[cases[i]]]
        try:
            v = statistic(sample)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            continue
        if v is not None and np.all(np.isfinite(v)):
            values.append(v)
    values = np.array(values, dtype=float)
    if not len(values):
        return None
    return [np.round(np.percentile(values, 2.5, axis=0), 3).tolist(), np.round(np.percentile(values, 97.5, axis=0), 3).tolist(), len(values)]


def logistic_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Maximum-likelihood logistic regression of y on [1, x] by Newton steps (no penalty)."""
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-design @ beta))
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def size_at(beta: np.ndarray, probability: float) -> float:
    """Diameter (mm) at which the fitted detection probability equals `probability`; x = log2(d)."""
    return float(2 ** ((np.log(probability / (1 - probability)) - beta[0]) / beta[1]))


def detection_curve(annotated: list[dict]) -> dict:
    def fit(rows):
        x = np.log2([r["d3_mm"] for r in rows])
        y = np.array([r["kind"] != "missed" for r in rows], dtype=float)
        beta = logistic_fit(x, y)
        if beta[1] <= 0:
            raise ValueError("non-increasing fit")
        return np.array([size_at(beta, p) for p in (0.5, 0.8, 0.9)])

    point = fit(annotated)
    interval = scan_bootstrap(annotated, fit, CURVE_RESAMPLES)
    return {
        "model": "logit P(detected) = b0 + b1 * log2(annotated 3D diameter, mm)",
        "pieces": len(annotated),
        "scans": len({r["case"] for r in annotated}),
        "d50_d80_d90_mm": np.round(point, 2).tolist(),
        "interval_95": {"lower": interval[0], "upper": interval[1], "valid_resamples": interval[2]} if interval else None,
    }


def gee(annotated: list[dict]) -> dict:
    """Detection on size, peak attenuation and slice spacing jointly, clustered by scan."""
    import statsmodels.api as sm

    rows = sorted(annotated, key=lambda r: r["case"])
    y = np.array([r["kind"] != "missed" for r in rows], dtype=float)
    x = np.column_stack([
        np.log2([r["d3_mm"] for r in rows]),
        np.array([r["peak_hu"] for r in rows]) / 100,
        np.array([r["spacing_z"] >= 1.5 for r in rows], dtype=float),
    ])  # fmt: skip
    groups = [r["case"] for r in rows]
    # Independence working correlation: the robust (sandwich) errors still account for clustering by
    # scan, and the fit is far more stable than an exchangeable one with 26 clusters and near-separation.
    model = sm.GEE(y, sm.add_constant(x), groups=groups, family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence())
    fit = model.fit(maxiter=200)
    names = ["intercept", "log2 diameter (per doubling)", "peak HU (per 100 HU)", "slice spacing >= 1.5 mm"]
    ci = np.asarray(fit.conf_int())
    if not (np.all(np.isfinite(fit.params)) and np.all(np.isfinite(ci))):
        return {"fit_failed": "GEE did not converge to finite estimates (near-separation and collinearity); no odds ratios are reported."}
    out = {
        names[i]: {"odds_ratio": round(float(np.exp(fit.params[i])), 3),
                   "interval_95": [round(float(np.exp(ci[i, 0])), 3), round(float(np.exp(ci[i, 1])), 3)]}
        for i in range(1, 4)
    }  # fmt: skip
    size = np.log2([r["d3_mm"] for r in rows])
    hu = np.array([r["peak_hu"] for r in rows])
    out["note"] = ("Odds ratios with robust (sandwich) intervals clustered by scan, GEE with independence working correlation. Size and peak HU are correlated "
                   f"(Spearman rho {spearman(size, hu):.2f}) through partial volume, so their separate effects are imprecise.")  # fmt: skip
    return out


def spearman(a, b) -> float:
    from scipy import stats

    return float(stats.spearmanr(a, b)[0])


def ratio(rows, hit) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    k = sum(1 for r in rows if hit(r))
    interval = scan_bootstrap(rows, lambda s: sum(1 for r in s if hit(r)) / len(s), RATIO_RESAMPLES) if len({r["case"] for r in rows}) > 1 else None
    out = {"detected": k, "n": n, "rate": round(k / n, 3), "scans": len({r["case"] for r in rows})}
    if interval:
        out["interval_95"] = [interval[0], interval[1]]
    if out["scans"] < 5:
        out["small_sample_warning"] = f"only {out['scans']} scan(s)"
    return out


def size_by_density(annotated: list[dict]) -> dict:
    """Peak-HU tertiles from the data, crossed with size, so density and size can be separated."""
    hu = np.array([r["peak_hu"] for r in annotated])
    cuts = np.percentile(hu, [100 / 3, 200 / 3])
    bands = [(-np.inf, cuts[0], f"<{cuts[0]:.0f} HU"), (cuts[0], cuts[1], f"{cuts[0]:.0f}-<{cuts[1]:.0f} HU"), (cuts[1], np.inf, f">={cuts[1]:.0f} HU")]
    detected = lambda r: r["kind"] != "missed"
    table = {}
    for low, high, name in bands:
        in_band = [r for r in annotated if low <= r["peak_hu"] < high]
        table[name] = {
            "all": ratio(in_band, detected),
            "<5 mm": ratio([r for r in in_band if r["d3_mm"] < 5], detected),
            ">=5 mm": ratio([r for r in in_band if r["d3_mm"] >= 5], detected),
        }
    thin = [r for r in annotated if r["spacing_z"] < 1.5]
    thick = [r for r in annotated if r["spacing_z"] >= 1.5]
    return {
        "tertile_cuts_hu": [round(float(c), 1) for c in cuts],
        "by_peak_hu_tertile": table,
        "small_stones_by_slice_spacing": {
            "<5 mm, slices <1.5 mm": ratio([r for r in thin if r["d3_mm"] < 5], detected),
            "<5 mm, slices >=1.5 mm": ratio([r for r in thick if r["d3_mm"] < 5], detected),
            ">=5 mm, slices <1.5 mm": ratio([r for r in thin if r["d3_mm"] >= 5], detected),
            ">=5 mm, slices >=1.5 mm": ratio([r for r in thick if r["d3_mm"] >= 5], detected),
        },
    }


def pct(p, a):
    return 200 * (p - a) / (p + a)


def measurement(pairs: list[dict]) -> dict:
    def summary(rows):
        if not rows:
            return {"n": 0}
        v = np.array([pct(r["predicted_mm3"], r["annotated_mm3"]) for r in rows])
        d = np.array([r["predicted_d3_mm"] - r["annotated_d3_mm"] for r in rows])
        out = {
            "n": len(rows), "scans": len({r["case"] for r in rows}),
            "volume_pct_bias": round(float(v.mean()), 2), "volume_abs_pct_error_mean": round(float(np.abs(v).mean()), 2),
            "diameter_bias_mm": round(float(d.mean()), 3), "diameter_mae_mm": round(float(np.abs(d).mean()), 3),
            "diameter_rmse_mm": round(float(np.sqrt((d**2).mean())), 3),
        }  # fmt: skip
        if out["scans"] > 1:
            ci = scan_bootstrap(rows, lambda s: np.array([np.mean([pct(r["predicted_mm3"], r["annotated_mm3"]) for r in s]),
                                                          np.mean([r["predicted_d3_mm"] - r["annotated_d3_mm"] for r in s])]), RATIO_RESAMPLES)  # fmt: skip
            out["volume_pct_bias_interval_95"] = [ci[0][0], ci[1][0]]
            out["diameter_bias_interval_95_mm"] = [ci[0][1], ci[1][1]]
        if out["scans"] < 5:
            out["small_sample_warning"] = f"only {out['scans']} scan(s)"
        return out

    def proportional(rows):
        mean_d = np.array([(r["predicted_d3_mm"] + r["annotated_d3_mm"]) / 2 for r in rows])
        diff_d = np.array([r["predicted_d3_mm"] - r["annotated_d3_mm"] for r in rows])
        return float(np.polyfit(mean_d, diff_d, 1)[0])

    hu = np.array([r["annotated_peak_hu"] for r in pairs])
    cuts = np.percentile(hu, [100 / 3, 200 / 3])
    slope_ci = scan_bootstrap(pairs, proportional, CURVE_RESAMPLES)
    return {
        "all": summary(pairs),
        "by_size": {n: summary([r for r in pairs if lo <= r["annotated_d3_mm"] < hi])
                    for lo, hi, n in zip(CATEGORY_EDGES[:-1], CATEGORY_EDGES[1:], CATEGORY_NAMES, strict=True)},
        "by_peak_hu_tertile": {
            f"<{cuts[0]:.0f} HU": summary([r for r in pairs if r["annotated_peak_hu"] < cuts[0]]),
            f"{cuts[0]:.0f}-<{cuts[1]:.0f} HU": summary([r for r in pairs if cuts[0] <= r["annotated_peak_hu"] < cuts[1]]),
            f">={cuts[1]:.0f} HU": summary([r for r in pairs if r["annotated_peak_hu"] >= cuts[1]]),
        },
        "diameter_proportional_bias": {
            "slope_of_difference_on_mean": round(proportional(pairs), 4),
            "interval_95": [slope_ci[0], slope_ci[1]] if slope_ci else None,
            "meaning": "Slope 0 means the diameter difference does not change with size.",
        },
    }


def category(d: float) -> str:
    for lo, hi, name in zip(CATEGORY_EDGES[:-1], CATEGORY_EDGES[1:], CATEGORY_NAMES, strict=True):
        if lo <= d < hi:
            return name
    return "none"


def categories(pairs: list[dict]) -> dict:
    """Agreement of size categories: 3D vs axial within the reference, and predicted vs reference."""

    def table(a_key, b_key):
        counts = {a: {b: 0 for b in CATEGORY_NAMES} for a in CATEGORY_NAMES}
        for r in pairs:
            counts[category(r[a_key])][category(r[b_key])] += 1
        agree = sum(counts[c][c] for c in CATEGORY_NAMES)
        return {"rows_" + a_key + "_cols_" + b_key: counts, "same_category": agree, "n": len(pairs), "agreement": round(agree / len(pairs), 3)}

    return {
        "reference_3d_vs_reference_axial": table("annotated_d3_mm", "annotated_axial_mm"),
        "predicted_3d_vs_reference_3d": table("annotated_d3_mm", "predicted_d3_mm"),
        "predicted_axial_vs_reference_axial": table("annotated_axial_mm", "predicted_axial_mm"),
        "note": "Pairs are one-to-one matched pieces only; missed stones have no predicted category.",
    }


def filter_sweep(predicted: list[dict], scans: list[dict]) -> list[dict]:
    no_stone = {r["case"] for r in scans if not r["has_stone"]}
    n_scans = len(scans)
    out = []
    for distance in FILTER_DISTANCES:
        kept = [r for r in predicted if distance is None or r["distance_to_kidney_mm"] <= distance]
        removed = [r for r in predicted if r not in kept]
        out.append({
            "max_distance_mm": distance,
            "false_positive_pieces": sum(r["kind"] == "false_positive" for r in kept),
            "false_positive_per_scan": round(sum(r["kind"] == "false_positive" for r in kept) / n_scans, 3),
            "true_pieces_removed": sum(r["kind"] != "false_positive" for r in removed),
            "no_stone_scans_with_a_prediction": len({r["case"] for r in kept if r["case"] in no_stone}),
            "no_stone_scans": len(no_stone),
        })  # fmt: skip
    return out


def figures(run: Path, results: dict, annotated_by_model: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    (run / "figures").mkdir()
    colours = {"100 epochs": "#1f5f8b", "250 epochs": "#b5651d"}
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    grid = np.linspace(np.log2(0.8), np.log2(60), 200)
    for label, annotated in annotated_by_model.items():
        x = np.log2([r["d3_mm"] for r in annotated])
        y = np.array([r["kind"] != "missed" for r in annotated], dtype=float)
        beta = logistic_fit(x, y)
        ax.plot(2**grid, 1 / (1 + np.exp(-(beta[0] + beta[1] * grid))), color=colours[label], label=f"{label}: logistic fit")
        edges = [0, 2, 3, 4, 5, 7, 10, 20, 100]
        for lo, hi in itertools.pairwise(edges):
            sel = [(xi, yi) for xi, yi in zip(2**x, y, strict=True) if lo <= xi < hi]
            if sel:
                ax.scatter(np.median([s[0] for s in sel]), np.mean([s[1] for s in sel]), s=12 + 2 * len(sel), color=colours[label], alpha=0.5, linewidths=0)
    ax.set_xscale("log")
    ax.set_xlabel("Annotated 3D diameter (mm, log scale)")
    ax.set_ylabel("Probability detected (any overlap)")
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(0.5, color="#999", lw=0.6, ls=":")
    ax.set_title("Detection against size, 32 development validation scans (dots: binned observed, area ~ n)", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "figures/detection_size_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for label, sweep in results["filter_sweep"].items():
        xs = [s["max_distance_mm"] if s["max_distance_mm"] is not None else 70 for s in sweep]
        ax.plot(xs, [s["false_positive_per_scan"] for s in sweep], marker="o", ms=3, color=colours[label], label=f"{label}: false pieces per scan")
        for s, x in zip(sweep, xs, strict=True):
            if s["true_pieces_removed"]:
                ax.annotate(f"-{s['true_pieces_removed']} true", (x, s["false_positive_per_scan"]), fontsize=7, xytext=(2, 4), textcoords="offset points")
    ax.set_xticks([10, 15, 20, 25, 30, 40, 50, 70], ["10", "15", "20", "25", "30", "40", "50", "off"])
    ax.set_xlabel("Kidney-distance filter limit (mm)")
    ax.set_ylabel("False-positive pieces per scan")
    ax.set_title("Filter sensitivity (labels: true pieces removed)", loc="left", fontsize=8.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "figures/filter_sensitivity.png", dpi=200)
    plt.close(fig)
    return ["figures/detection_size_curve.png", "figures/filter_sensitivity.png"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-100", type=Path, required=True)
    parser.add_argument("--detail-250", type=Path, required=True)
    args = parser.parse_args()
    results = {"status": "EXPLORATORY_DEVELOPMENT_CHARACTERISATION_NOT_CONFIRMATORY", "definitions": __doc__,
               "sources": {"100 epochs": str(args.detail_100), "250 epochs": str(args.detail_250)},
               "detection_curve": {}, "detection_gee": {}, "size_by_density": {}, "measurement": {}, "categories": {}, "filter_sweep": {}}  # fmt: skip
    annotated_by_model = {}
    for label, folder in (("100 epochs", args.detail_100), ("250 epochs", args.detail_250)):
        annotated = read(folder, "annotated", "model")
        pairs = read(folder, "pairs", "model")
        annotated_by_model[label] = annotated
        results["detection_curve"][label] = detection_curve(annotated)
        results["detection_gee"][label] = gee(annotated)
        results["size_by_density"][label] = size_by_density(annotated)
        results["measurement"][label] = measurement(pairs)
        results["categories"][label] = categories(pairs)
        results["filter_sweep"][label] = filter_sweep(read(folder, "predicted", "model"), read(folder, "scans", "model"))
        print(label, json.dumps(results["detection_curve"][label]), flush=True)
    run = create_run_folder(Path("results/runs"), "detection_characterisation")
    results["figures"] = figures(run, results, annotated_by_model)
    (run / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Saved to", run)


if __name__ == "__main__":
    main()

"""Run the reliability-aware measurement study, fixed by docs/preregistration.md.

Usage: python scripts/run_reliability.py configs/r1_reliability_freiburg.yaml

The order is fixed: provenance and A0 reconciliation, data stop rules, the pipeline-integrity
diagnostic, then the primary and other analyses. A stop ends the run before any later step. A0
outputs are read and never changed, and masks stay in memory. The study has one analysis run.
"""

import argparse
import json
import math
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.download import D1, file_md5
from stonekit.data.freiburg import MAT_NAME, load_mat_stones
from stonekit.data.registry import find_project_root
from stonekit.eval.bootstrap import percentile_interval, resample_indices
from stonekit.logging_utils import get_logger
from stonekit.reliability.evaluation import (
    Comparison,
    ModelSpec,
    Rows,
    accepted_mean,
    evaluate_partition,
    mean_deltas,
    psi,
    row_folds,
    scale_dependent,
    section_11_reading,
    spacing_dependent,
    training_threshold_summary,
    trapezoid_weights,
)
from stonekit.reliability.provenance import (
    changed_reused_sources,
    changed_settings,
    file_sha256,
    folder_hashes,
    load_manifest,
    reconciliation_mismatches,
)
from stonekit.reliability.ranges import range_partition, summarize_bounds
from stonekit.reliability.signals import ASSUMPTIONS, AXES, SignalSettings, stone_record

CODE_ROOT = Path(__file__).resolve().parents[1]
FULL = ("x1", "S_G", "M")
STREAMS = ("primary", "bootstrap", "integrity", "ranges", "range_bootstrap", "exploratory")
log = get_logger("run_reliability")


class StudyStop(Exception):
    """A stop rule in section 17 of the preregistration fired."""


def provenance_checks(config: dict, root: Path, code_root: Path) -> tuple[Path, dict]:
    """Confirm the frozen design, A0 outputs, reused code and source files before any analysis."""
    prereg = root / config["preregistration_path"]
    if file_sha256(prereg) != config["preregistration_sha256"]:
        raise StudyStop("The preregistration does not match its frozen hash")
    a0_folder = root / config["a0_run"]
    if file_sha256(a0_folder / "a0_stones.csv") != config["a0_stones_sha256"]:
        raise StudyStop("a0_stones.csv does not match its recorded hash")
    manifest = load_manifest(a0_folder)
    changed = changed_reused_sources(manifest, code_root)
    if changed:
        raise StudyStop(f"Reused source files differ from the A0 manifest: {changed}")
    settings = {
        name: config[name]
        for name in (
            "primary_threshold_hu",
            "min_component_voxels",
            "assumed_z_spacing_mm",
            "paper_voxel_edge_mm",
        )
    }
    different = changed_settings(manifest, settings)
    if different:
        raise StudyStop(f"Settings differ from the a0 run: {different}")
    documented = set(manifest["config"]["exploratory_thresholds_hu"])
    if not set(config["threshold_grid_hu"]) <= documented:
        raise StudyStop("The threshold grid contains a value outside the documented grid")
    if config.get("verify_source_checksums", True):
        for expected in D1.files:
            if file_md5(root / config["raw_folder"] / expected.name) != expected.md5:
                raise StudyStop(f"Source checksum mismatch: {expected.name}")
    return a0_folder, manifest


def recompute_stones(config: dict, root: Path) -> pd.DataFrame:
    """Recompute signals and comparators in memory. Nothing here replaces an A0 value."""
    table = pd.read_parquet(root / config["table"])
    by_position = {
        stone.mat_index: stone for stone in load_mat_stones(root / config["raw_folder"] / MAT_NAME)
    }
    settings = SignalSettings(
        primary_threshold_hu=config["primary_threshold_hu"],
        grid_hu=tuple(config["threshold_grid_hu"]),
        local_pairs=tuple(tuple(pair) for pair in config["exploratory_local_pairs_hu"]),
        min_component_voxels=config["min_component_voxels"],
        assumed_z_spacing_mm=config["assumed_z_spacing_mm"],
        paper_voxel_edge_mm=config["paper_voxel_edge_mm"],
    )
    records = []
    for row in table.itertuples():
        record = stone_record(by_position[int(row.mat_index)], settings)
        records.append({"stone_id": int(row.stone_id), **record})
    frame = pd.DataFrame(records)
    for assumption in ASSUMPTIONS:
        frame[f"d_paper_{assumption}"] = frame[f"d_paper_Z2_{assumption}"]
    return frame.sort_values("stone_id", ignore_index=True)


def reference_tiers(saved: pd.DataFrame, config: dict) -> dict[str, list[int]]:
    """Section 14's lists. The 0.1 mL tier uses reference values only."""
    coarse = np.isclose(saved.reference_volume_ml, config["coarsest_reference_ml"], atol=1e-9)
    tiers = {
        "unreachable_references": list(config["unreachable_reference_ids"]),
        "coarsest_references": [int(stone) for stone in saved.stone_id[coarse]],
        "flagged_references": list(config["flagged_reference_ids"]),
    }
    tiers["union_of_reference_tiers"] = sorted({stone for ids in tiers.values() for stone in ids})
    tiers["float32_patches"] = list(config["float32_ids"])
    tiers["spacing_fields_disagree"] = [
        int(stone) for stone in saved.stone_id[saved.spacing_fields_disagree.astype(bool)]
    ]
    return tiers


def data_stop_checks(saved: pd.DataFrame, stones: pd.DataFrame, config: dict, tiers: dict) -> None:
    """Section 17's data stop rules, checked before any error-based analysis."""
    if len(saved) != config["expected_stones"]:
        raise StudyStop(f"Expected {config['expected_stones']} stones, found {len(saved)}")
    for name in ("volume_A", "volume_B", "reference_volume_ml", "reference_diameter_mm"):
        if not (saved[name] > 0).all():
            raise StudyStop(f"Non-positive or missing values in {name}")
    if (saved.mask_voxels <= 0).any():
        raise StudyStop("A stone has an empty mask at the primary threshold")
    signals = stones[["S_G", "M"]].to_numpy(float)
    if not np.isfinite(signals).all():
        raise StudyStop("A NaN or infinite predictor exists in the original stones")
    if len(tiers["coarsest_references"]) != config["expected_coarsest"]:
        raise StudyStop(
            f"Expected {config['expected_coarsest']} stones at the coarsest reference, "
            f"found {len(tiers['coarsest_references'])}"
        )
    agree = len(saved) - len(tiers["spacing_fields_disagree"])
    if agree != config["expected_spacing_agree"]:
        raise StudyStop(f"Expected {config['expected_spacing_agree']} agreeing, found {agree}")


def build_rows(saved: pd.DataFrame, stones: pd.DataFrame) -> Rows:
    """Outcomes use saved A0 values, except Z1 and Z0 diameters, which A0 did not save per stone."""
    stones = stones.set_index("stone_id").loc[saved.stone_id]
    reference = saved.reference_volume_ml.to_numpy(float)
    diameter = saved.reference_diameter_mm.to_numpy(float)
    columns = {
        "x1": np.log(saved.mask_voxels.to_numpy(float)),
        "reference_volume_ml": reference,
    }
    for name in stones.columns:
        if name in ("S_G", "M", "ln_H") or (name.startswith("S_") and name != "S_G"):
            columns[name] = stones[name].to_numpy(float)
    for assumption in ASSUMPTIONS:
        volume = saved[f"volume_{assumption}"].to_numpy(float)
        columns[f"volume_{assumption}"] = volume
        columns[f"E_{assumption}"] = np.abs(np.log(volume / reference))
        columns[f"APE_{assumption}"] = 100 * np.abs(volume - reference) / reference
        for axis in AXES:
            measured = (
                saved[f"d_paper_{assumption}"].to_numpy(float)
                if axis == "Z2"
                else stones[f"d_paper_{axis}_{assumption}"].to_numpy(float)
            )
            columns[f"D_{axis}_{assumption}"] = np.abs(np.log(measured / diameter))
    stone_id = saved.stone_id.to_numpy(int)
    return Rows(stone_id, np.zeros(len(stone_id), dtype=int), columns)


def primary_models() -> dict[str, ModelSpec]:
    models = {}
    for assumption in ASSUMPTIONS:
        outcome = f"E_{assumption}"
        models[f"full_{assumption}"] = ModelSpec(FULL, outcome)
        models[f"no_S_{assumption}"] = ModelSpec(("x1", "M"), outcome)
        models[f"no_M_{assumption}"] = ModelSpec(("x1", "S_G"), outcome)
    return models


def primary_comparisons() -> list[Comparison]:
    comparisons = []
    for assumption in ASSUMPTIONS:
        full = f"full_{assumption}"
        error = f"E_{assumption}"
        comparisons += [
            Comparison(f"psi_E_{assumption}", error, "size", full),
            Comparison(f"psi_APE_{assumption}", f"APE_{assumption}", "size", full),
            Comparison(f"full_vs_no_S_{assumption}", error, f"no_S_{assumption}", full),
            Comparison(f"full_vs_no_M_{assumption}", error, f"no_M_{assumption}", full),
        ]
        comparisons += [
            Comparison(f"diameter_{axis}_{assumption}", f"D_{axis}_{assumption}", "size", full)
            for axis in AXES
        ]
    return comparisons


def bootstrap_deltas(
    rows: Rows,
    resamples: int,
    partitions: int,
    seed: np.random.SeedSequence,
    models: dict[str, ModelSpec],
    comparisons: list[Comparison],
    ks: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Resample stones, and repeat all fitting within each resample. Copies stay together."""
    keys = [comparison.key for comparison in comparisons]
    values = {key: np.full((resamples, len(ks)), np.nan) for key in keys}
    for position, child in enumerate(seed.spawn(resamples)):
        rng = np.random.default_rng(child)
        resampled = rows.take(rng.integers(0, len(rows), len(rows)))
        result = mean_deltas(resampled, partitions, rng, models, comparisons, ks)
        for key in keys:
            values[key][position] = result[key]
        if (position + 1) % 200 == 0:
            log.info("Bootstrap resample %d of %d", position + 1, resamples)
    return values


def integrity_diagnostic(rows: Rows, config: dict, seed: np.random.SeedSequence, ks) -> pd.DataFrame:
    """Section 17. A pipeline-integrity diagnostic only, not a test or evidence about signals."""
    models = {f"full_{a}": ModelSpec(FULL, f"E_{a}") for a in ASSUMPTIONS}
    comparisons = [Comparison(f"psi_E_{a}", f"E_{a}", "size", f"full_{a}") for a in ASSUMPTIONS]
    weights = trapezoid_weights(ks)
    n = config["integrity_permutations"]
    values = {comparison.key: np.full(n, np.nan) for comparison in comparisons}
    for position, child in enumerate(seed.spawn(n)):
        rng = np.random.default_rng(child)
        order = rng.permutation(len(rows))
        columns = dict(rows.columns)
        columns["S_G"] = rows.columns["S_G"][order]
        columns["M"] = rows.columns["M"][order]
        shuffled = Rows(rows.stone_id, rows.copy, columns)
        result = mean_deltas(
            shuffled, config["integrity_partitions"], rng, models, comparisons, ks
        )
        for key, permuted in values.items():
            permuted[position] = psi(result[key], weights)
    records = []
    for key, permuted in values.items():
        finite = permuted[np.isfinite(permuted)]
        allowance = 2 * finite.std(ddof=1) / math.sqrt(len(finite))
        records.append(
            {
                "key": key,
                "role": "pipeline-integrity diagnostic only",
                "permutations": n,
                "failed_permutations": int(n - len(finite)),
                "mean_permuted_psi": float(finite.mean()),
                "sd_permuted_psi": float(finite.std(ddof=1)),
                "trigger_above": float(allowance),
                "triggered": bool(finite.mean() > allowance),
            }
        )
    return pd.DataFrame(records)


def run_primary_repeats(rows: Rows, config: dict, seed: np.random.SeedSequence, ks: tuple) -> dict:
    """The 200 fold partitions on the original stones. A failed primary fit stops the run."""
    models = primary_models()
    comparisons = primary_comparisons()
    rng = np.random.default_rng(seed)
    repeats = config["primary_repeats"]
    deltas = {c.key: np.empty((repeats, len(ks))) for c in comparisons}
    size_loss = {a: np.empty((repeats, len(ks))) for a in ASSUMPTIONS}
    different = {a: np.empty((repeats, len(ks))) for a in ASSUMPTIONS}
    risk_ks = config["risk_curve_flagged_per_fold"]
    risk = {
        f"{rule}|{a}": np.zeros(len(risk_ks))
        for a in ASSUMPTIONS
        for rule in ("size", f"full_{a}")
    }
    flagged_counts = {
        rule: np.zeros((len(rows), len(ks)))
        for rule in ("size", *(f"full_{a}" for a in ASSUMPTIONS))
    }
    coefficient_rows = []
    failed_models: dict[str, int] = {}
    for repeat in range(repeats):
        fold = row_folds(rows, rng)
        result = evaluate_partition(rows, fold, models, comparisons, ks)
        for assumption in ASSUMPTIONS:
            if result.ranks[f"full_{assumption}"] is None:
                raise StudyStop(f"Primary model non-convergence under {assumption}, repeat {repeat}")
        for name, ranks in result.ranks.items():
            if ranks is None:
                failed_models[name] = failed_models.get(name, 0) + 1
        for key, values in result.deltas.items():
            deltas[key][repeat] = values
        size_ranks = result.ranks["size"]
        for assumption in ASSUMPTIONS:
            full_ranks = result.ranks[f"full_{assumption}"]
            errors = rows.columns[f"E_{assumption}"]
            size_loss[assumption][repeat] = [accepted_mean(errors, size_ranks, k) for k in ks]
            different[assumption][repeat] = [
                int(((size_ranks < k) & (full_ranks >= k)).sum()) for k in ks
            ]
            for position, k in enumerate(risk_ks):
                if (np.bincount(fold) > k).all():
                    risk[f"size|{assumption}"][position] += accepted_mean(errors, size_ranks, k)
                    risk[f"full_{assumption}|{assumption}"][position] += accepted_mean(
                        errors, full_ranks, k
                    )
                else:
                    risk[f"size|{assumption}"][position] = np.nan
                    risk[f"full_{assumption}|{assumption}"][position] = np.nan
            for fit_number, fit in enumerate(result.fits[f"full_{assumption}"]):
                coefficient_rows.append(
                    {
                        "assumption": assumption,
                        "repeat": repeat,
                        "fold": fit_number,
                        **dict(zip(("intercept", *FULL), fit.coefficients.tolist())),
                    }
                )
        for rule, counts in flagged_counts.items():
            for position, k in enumerate(ks):
                counts[:, position] += result.ranks[rule] < k
    return {
        "deltas": deltas,
        "size_loss": size_loss,
        "different": different,
        "risk": {key: values / repeats for key, values in risk.items()},
        "flagged": {rule: counts / repeats for rule, counts in flagged_counts.items()},
        "coefficients": pd.DataFrame(coefficient_rows),
        "failed_models": failed_models,
    }


def estimate_rows(
    point: dict[str, np.ndarray], boot: dict[str, np.ndarray], ks: tuple, label: dict
) -> list[dict]:
    """The psi summary and each k, from point estimates and bootstrap draws."""
    weights = trapezoid_weights(ks)
    rows = []
    for key, deltas in point.items():
        repeats = deltas if deltas.ndim == 2 else deltas[None, :]
        per_repeat = repeats @ weights
        interval = percentile_interval(boot[key] @ weights)
        rows.append(
            {
                **label,
                "key": key,
                "summary": "psi",
                "estimate": float(per_repeat.mean()),
                "monte_carlo_se": float(per_repeat.std(ddof=1) / math.sqrt(len(per_repeat)))
                if len(per_repeat) > 1
                else float("nan"),
                "ci_low": interval.lower,
                "ci_high": interval.upper,
                "degenerate_resamples": interval.degenerate,
                "resamples": interval.resamples,
                "interval_unreliable": interval.unreliable,
            }
        )
        for position, k in enumerate(ks):
            interval = percentile_interval(boot[key][:, position])
            rows.append(
                {
                    **label,
                    "key": key,
                    "summary": f"k={k}",
                    "estimate": float(repeats[:, position].mean()),
                    "monte_carlo_se": float(
                        repeats[:, position].std(ddof=1) / math.sqrt(len(repeats))
                    )
                    if len(repeats) > 1
                    else float("nan"),
                    "ci_low": interval.lower,
                    "ci_high": interval.upper,
                    "degenerate_resamples": interval.degenerate,
                    "resamples": interval.resamples,
                    "interval_unreliable": interval.unreliable,
                }
            )
    return rows


def primary_table(estimates: pd.DataFrame, primary: dict, ks: tuple) -> pd.DataFrame:
    """One row per spacing assumption with the pre-written readings of section 11."""
    weights = trapezoid_weights(ks)
    psis = estimates[estimates.summary == "psi"].set_index("key")
    records = []
    for assumption in ASSUMPTIONS:
        row = psis.loc[f"psi_E_{assumption}"]
        other = psis.loc[f"psi_E_{'B' if assumption == 'A' else 'A'}"]
        percent = psis.loc[f"psi_APE_{assumption}"]
        baseline = float((primary["size_loss"][assumption] @ weights).mean())
        records.append(
            {
                "assumption": assumption,
                "psi": row.estimate,
                "monte_carlo_se": row.monte_carlo_se,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "size_only_summary": baseline,
                "psi_percent_of_size_only": 100 * row.estimate / baseline,
                "reading": section_11_reading(row.ci_low, row.ci_high),
                "spacing_dependent": spacing_dependent(
                    row.estimate,
                    other.estimate,
                    (row.ci_low, row.ci_high),
                    (other.ci_low, other.ci_high),
                ),
                "scale_dependent": scale_dependent(row.estimate, percent.estimate),
                "degenerate_resamples": row.degenerate_resamples,
                "resamples": row.resamples,
                "interval_unreliable": row.interval_unreliable,
            }
        )
    return pd.DataFrame(records)


def run_ranges(rows: Rows, config: dict, seeds: dict) -> dict[str, pd.DataFrame]:
    """Section 13. Secondary, with coverage and width always reported together."""
    methods = ("full", "size_scale", "log_limits", "ml_limits")
    reference = rows.columns["reference_volume_ml"]
    x1 = rows.columns["x1"]
    cuts = np.quantile(x1, [1 / 3, 2 / 3])
    tertile = np.digitize(x1, cuts)
    rng = np.random.default_rng(seeds["ranges"])
    records = []
    subgroup = []
    failed = 0
    for repeat in range(config["range_repeats"]):
        bounds = range_partition(rows, ASSUMPTIONS, FULL, rng)
        if bounds is None:
            failed += 1
            continue
        for assumption in ASSUMPTIONS:
            for method in methods:
                name = f"{method}_{assumption}"
                records.append(
                    {
                        "repeat": repeat,
                        "assumption": assumption,
                        "method": method,
                        **summarize_bounds(bounds[name], reference, name),
                    }
                )
                for group in range(3):
                    subgroup.append(
                        {
                            "repeat": repeat,
                            "assumption": assumption,
                            "method": method,
                            "size_tertile": group + 1,
                            **summarize_bounds(bounds[name], reference, name, tertile == group),
                        }
                    )
    per_repeat = pd.DataFrame(records)
    summary = (
        per_repeat.groupby(["assumption", "method", "width_unit"])
        .agg(
            mean_coverage=("coverage", "mean"),
            coverage_repeat_p2_5=("coverage", lambda v: np.percentile(v, 2.5)),
            coverage_repeat_p97_5=("coverage", lambda v: np.percentile(v, 97.5)),
            mean_median_width=("median_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
            mean_nonpositive_lower_limits=("nonpositive_lower_limits", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "failed_repeats", failed)
    subgroups = (
        pd.DataFrame(subgroup)
        .groupby(["assumption", "method", "size_tertile", "width_unit"])
        .agg(
            mean_empirical_coverage=("coverage", "mean"),
            mean_median_width=("median_width", "mean"),
            mean_interval_score=("mean_interval_score", "mean"),
        )
        .reset_index()
    )
    subgroups.insert(0, "note", "Empirical subgroup coverage. Not guaranteed or claimed")

    resamples = config["range_bootstrap_resamples"]
    partitions = config["range_bootstrap_partitions"]
    draws = {
        f"{name}_{a}": np.full(resamples, np.nan)
        for a in ASSUMPTIONS
        for name in ("limits_minus_full", "size_scale_minus_full", "coverage_full")
    }
    for position, child in enumerate(seeds["range_bootstrap"].spawn(resamples)):
        rng = np.random.default_rng(child)
        resampled = rows.take(rng.integers(0, len(rows), len(rows)))
        target = resampled.columns["reference_volume_ml"]
        totals = {key: [] for key in draws}
        for _ in range(partitions):
            bounds = range_partition(resampled, ASSUMPTIONS, FULL, rng)
            if bounds is None:
                totals = None
                break
            for a in ASSUMPTIONS:
                full = summarize_bounds(bounds[f"full_{a}"], target, f"full_{a}")
                size = summarize_bounds(bounds[f"size_scale_{a}"], target, f"size_scale_{a}")
                limits = summarize_bounds(bounds[f"log_limits_{a}"], target, f"log_limits_{a}")
                totals[f"limits_minus_full_{a}"].append(
                    limits["mean_interval_score"] - full["mean_interval_score"]
                )
                totals[f"size_scale_minus_full_{a}"].append(
                    size["mean_interval_score"] - full["mean_interval_score"]
                )
                totals[f"coverage_full_{a}"].append(full["coverage"])
        if totals is not None:
            for key, values in totals.items():
                draws[key][position] = float(np.mean(values))
        if (position + 1) % 200 == 0:
            log.info("Range bootstrap resample %d of %d", position + 1, resamples)

    wide = per_repeat.pivot_table(
        index=["repeat", "assumption"], columns="method", values="mean_interval_score"
    ).reset_index()
    differences = []
    for assumption in ASSUMPTIONS:
        chunk = wide[wide.assumption == assumption]
        for name, values in (
            ("limits_minus_full", chunk.log_limits - chunk.full),
            ("size_scale_minus_full", chunk.size_scale - chunk.full),
        ):
            interval = percentile_interval(draws[f"{name}_{assumption}"])
            differences.append(
                {
                    "assumption": assumption,
                    "difference": name,
                    "estimate": float(values.mean()),
                    "ci_low": interval.lower,
                    "ci_high": interval.upper,
                    "degenerate_resamples": interval.degenerate,
                    "resamples": interval.resamples,
                    "interval_unreliable": interval.unreliable,
                    "note": "Approximate and descriptive. Positive favours the full-model ranges",
                }
            )
    return {
        "r1_ranges_summary.csv": summary,
        "r1_ranges_subgroups_exploratory.csv": subgroups,
        "r1_ranges_interval_score.csv": pd.DataFrame(differences),
    }


def exploratory_psi(rows: Rows, config: dict, tiers: dict, seed, ks: tuple) -> pd.DataFrame:
    """Exploratory variants and sensitivity analyses. None can replace the primary result."""
    analyses: dict[str, tuple[Rows, dict, list]] = {}

    def add(name: str, subset: Rows, make_models) -> None:
        models = {f"variant_{a}": make_models(a) for a in ASSUMPTIONS}
        comparisons = [
            Comparison(f"{name}|{a}", f"E_{a}", "size", f"variant_{a}") for a in ASSUMPTIONS
        ]
        analyses[name] = (subset, models, comparisons)

    def full_model(assumption: str) -> ModelSpec:
        return ModelSpec(FULL, f"E_{assumption}")

    for name, ids in tiers.items():
        keep = np.flatnonzero(~np.isin(rows.stone_id, ids))
        label = "only_spacing_agree" if name == "spacing_fields_disagree" else f"without_{name}"
        add(label, rows.take(keep), full_model)
    for low, high in config["exploratory_local_pairs_hu"]:
        column = f"S_{low}_{high}"
        add(
            f"local_pair_{low}_{high}_post_hoc",
            rows,
            lambda a, column=column: ModelSpec(("x1", column, "M"), f"E_{a}"),
        )
    add("ln_H_in_place_of_M", rows, lambda a: ModelSpec(("x1", "S_G", "ln_H"), f"E_{a}"))
    add("fitted_size_only", rows, lambda a: ModelSpec(("x1",), f"E_{a}"))

    rank_comparisons = [
        Comparison(f"zero_parameter_rank_score|{a}", f"E_{a}", "size", "rank_score")
        for a in ASSUMPTIONS
    ]
    analyses["zero_parameter_rank_score"] = (rows, {}, rank_comparisons)

    children = seed.spawn(len(analyses) * 2)
    output = []
    for position, (name, (subset, models, comparisons)) in enumerate(analyses.items()):
        log.info("Exploratory analysis: %s", name)
        rng = np.random.default_rng(children[position * 2])
        point = {comparison.key: [] for comparison in comparisons}
        for _ in range(config["primary_repeats"]):
            result = evaluate_partition(subset, row_folds(subset, rng), models, comparisons, ks)
            for key, values in result.deltas.items():
                point[key].append(values)
        point = {key: np.array(values) for key, values in point.items()}
        boot = bootstrap_deltas(
            subset,
            config["exploratory_bootstrap_resamples"],
            config["exploratory_bootstrap_partitions"],
            children[position * 2 + 1],
            models,
            comparisons,
            ks,
        )
        for row in estimate_rows(point, boot, ks, {"analysis": name, "n_stones": len(subset)}):
            if row["summary"] == "psi":
                row["assumption"] = row["key"].split("|")[1]
                output.append(row)
    return pd.DataFrame(output)


def training_threshold_table(rows: Rows, config: dict, rng) -> pd.DataFrame:
    """Exploratory. Realized burdens may differ between rules."""
    records = []
    for repeat in range(config["primary_repeats"]):
        fold = row_folds(rows, rng)
        for assumption in ASSUMPTIONS:
            spec = ModelSpec(FULL, f"E_{assumption}")
            for burden in config["training_threshold_burdens"]:
                errors = rows.columns[f"E_{assumption}"]
                summary = training_threshold_summary(rows, fold, spec, errors, burden)
                records.append(
                    {"repeat": repeat, "assumption": assumption, "burden": burden, **summary}
                )
    return (
        pd.DataFrame(records)
        .drop(columns="repeat")
        .groupby(["assumption", "burden"])
        .mean()
        .reset_index()
    )


def half_attenuation_table(saved, stones, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Section 15. An established comparator, with no fitted factor."""
    stones = stones.set_index("stone_id").loc[saved.stone_id]
    reference = saved.reference_volume_ml.to_numpy(float)
    per_stone = pd.DataFrame({"stone_id": saved.stone_id, "T_HA": stones.T_HA.to_numpy()})
    summaries = []
    indices = resample_indices(len(saved), config["bootstrap_resamples"], config["seed"])
    for assumption in ASSUMPTIONS:
        a0 = saved[f"volume_{assumption}"].to_numpy(float)
        half = stones[f"volume_HA_{assumption}"].to_numpy(float)
        per_stone[f"volume_HA_{assumption}"] = half
        if (half <= 0).any():
            log.warning("A half-attenuation volume is empty under %s", assumption)
        with np.errstate(divide="ignore"):
            for name, a0_error, half_error in (
                ("E", np.abs(np.log(a0 / reference)), np.abs(np.log(half / reference))),
                ("APE", 100 * np.abs(a0 - reference) / reference,
                 100 * np.abs(half - reference) / reference),
            ):
                per_stone[f"{name}_HA_{assumption}"] = half_error
                difference = a0_error - half_error
                interval = percentile_interval(difference[indices].mean(axis=1))
                summaries.append(
                    {
                        "assumption": assumption,
                        "scale": name,
                        "mean_a0": float(a0_error.mean()),
                        "mean_half_attenuation": float(half_error.mean()),
                        "median_a0": float(np.median(a0_error)),
                        "median_half_attenuation": float(np.median(half_error)),
                        "mean_a0_minus_half_attenuation": float(difference.mean()),
                        "ci_low": interval.lower,
                        "ci_high": interval.upper,
                        "note": "Exploratory comparator. Not a contribution or a predictor",
                    }
                )
    return pd.DataFrame(summaries), per_stone


def stones_table(
    saved, stones, rows: Rows, primary: dict, half: pd.DataFrame, tiers: dict, ks: tuple
) -> pd.DataFrame:
    """Every stone's signals, errors and flag frequencies, for failure analysis."""
    frame = saved[["stone_id", "reference_volume_ml", "volume_A", "volume_B", "stone_type"]].copy()
    grid_columns = [name for name in stones.columns if name.startswith("N_")]
    signal_columns = ["S_G", "H", "T_HA", "M", "ln_H", "any_zero_count", "non_monotonic_counts"]
    signal_columns += [
        name for name in stones.columns if name.startswith("S_") and name != "S_G"
    ]
    frame = frame.merge(stones[["stone_id", *signal_columns, *grid_columns]], on="stone_id")
    frame = frame.merge(half, on="stone_id").drop(columns="T_HA_y").rename(
        columns={"T_HA_x": "T_HA"}
    )
    for assumption in ASSUMPTIONS:
        frame[f"E_{assumption}"] = rows.columns[f"E_{assumption}"]
        frame[f"APE_{assumption}"] = rows.columns[f"APE_{assumption}"]
    for rule, counts in primary["flagged"].items():
        for position, k in enumerate(ks):
            frame[f"flag_rate_{rule}_k{k}"] = counts[:, position]
    for name, ids in tiers.items():
        frame[f"tier_{name}"] = frame.stone_id.isin(ids)
    return frame


def failure_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Section 16's named groups. The top and bottom fractions are descriptive labels only."""
    k = config["failure_report_flagged_per_fold"]
    fraction = config["failure_top_fraction"]
    records = []

    def add(group: str, assumption: str, chosen: pd.DataFrame) -> None:
        for row in chosen.itertuples():
            records.append(
                {
                    "group": group,
                    "assumption": assumption,
                    "stone_id": row.stone_id,
                    "E": getattr(row, f"E_{assumption}"),
                    "APE": getattr(row, f"APE_{assumption}"),
                    "flag_rate_size": getattr(row, f"flag_rate_size_k{k}"),
                    "flag_rate_full": getattr(row, f"flag_rate_full_{assumption}_k{k}"),
                }
            )

    named = {
        "named_stones_74_100_101": frame.stone_id.isin([74, 100, 101]),
        "any_zero_count": frame.any_zero_count.astype(bool),
        "non_monotonic_counts": frame.non_monotonic_counts.astype(bool),
    }
    named.update(
        {name: frame[name].astype(bool) for name in frame.columns if name.startswith("tier_")}
    )
    for assumption in ASSUMPTIONS:
        error = frame[f"E_{assumption}"]
        full_rate = frame[f"flag_rate_full_{assumption}_k{k}"]
        size_rate = frame[f"flag_rate_size_k{k}"]
        high = error >= error.quantile(1 - fraction)
        low = error <= error.quantile(fraction)
        add("worst_error", assumption, frame[high])
        add("high_error_usually_not_flagged", assumption, frame[high & (full_rate < 0.5)])
        add("low_error_usually_flagged", assumption, frame[low & (full_rate >= 0.5)])
        add("mostly_full_only", assumption, frame[(full_rate >= 0.5) & (size_rate < 0.5)])
        add("mostly_size_only", assumption, frame[(size_rate >= 0.5) & (full_rate < 0.5)])
        for name, chosen in named.items():
            add(name, assumption, frame[chosen])
    return pd.DataFrame(records)


def composition_table(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Descriptive only. Composition is entangled with attenuation, and 42 stones are blank."""
    k = config["failure_report_flagged_per_fold"]
    records = []
    labels = frame.stone_type.astype(object).where(frame.stone_type.notna(), "blank")
    for label, group in frame.groupby(labels):
        for assumption in ASSUMPTIONS:
            records.append(
                {
                    "composition": label,
                    "assumption": assumption,
                    "n": len(group),
                    "mean_E": group[f"E_{assumption}"].mean(),
                    "mean_flag_rate_full": group[f"flag_rate_full_{assumption}_k{k}"].mean(),
                    "mean_flag_rate_size": group[f"flag_rate_size_k{k}"].mean(),
                }
            )
    return pd.DataFrame(records)


def git_state(code_root: Path) -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=code_root, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = git("status", "--short")
    return {"head": git("rev-parse", "HEAD"), "uncommitted_changes": bool(status), "status": status}


def source_hashes(config_path: Path, root: Path, config: dict, code_root: Path) -> dict:
    """Hash code, tests, config, the frozen design, the protocol and the processed table."""
    hashes = {}
    paths = [
        *sorted(code_root.glob("src/stonekit/**/*.py")),
        *sorted(code_root.glob("tests/*.py")),
        Path(__file__).resolve(),
        code_root / "docs" / "STAGE2_PROTOCOL.md",
    ]
    for path in paths:
        hashes[str(path.relative_to(code_root))] = file_sha256(path)
    if config_path.is_file():
        hashes[f"config/{config_path.name}"] = file_sha256(config_path)
    hashes["preregistration"] = file_sha256(root / config["preregistration_path"])
    hashes["processed_table"] = file_sha256(root / config["table"])
    return hashes


def write_csv(frame: pd.DataFrame, folder: Path, name: str) -> None:
    frame.to_csv(folder / name, index=False)


def run_study(
    config: dict, config_path: Path, root: Path, code_root: Path = CODE_ROOT
) -> tuple[Path, str | None]:
    """Run the fixed sequence. Returns the run folder and the stop reason, if any."""
    run_folder = create_run_folder(root / config["runs_folder"], config["id"])
    log.info("Run folder: %s", run_folder)
    manifest = {
        "preregistration": config["preregistration_path"],
        "config": config,
        "git": git_state(code_root),
        "stages_completed": [],
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "scipy", "pandas")},
        },
    }
    stop_reason = None
    a0_folder = root / config["a0_run"]
    hashes_before = folder_hashes(a0_folder) if a0_folder.is_dir() else {}
    try:
        manifest["source_hashes"] = source_hashes(config_path, root, config, code_root)
        complete_study(config, root, code_root, run_folder, manifest)
    except StudyStop as stop:
        stop_reason = str(stop)
        log.error("STOP: %s", stop_reason)
    hashes_after = folder_hashes(a0_folder) if a0_folder.is_dir() else {}
    manifest["a0_hashes_before"] = hashes_before
    manifest["a0_hashes_after"] = hashes_after
    manifest["a0_unchanged"] = hashes_before == hashes_after
    if hashes_before != hashes_after:
        stop_reason = "A0 outputs changed during the run"
    manifest["stop_reason"] = stop_reason
    (run_folder / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_folder, stop_reason


def complete_study(config: dict, root: Path, code_root: Path, folder: Path, manifest: dict) -> None:
    ks = tuple(config["flagged_per_fold"])
    seeds = dict(zip(STREAMS, np.random.SeedSequence(config["seed"]).spawn(len(STREAMS))))

    # Stage 1: provenance and reconciliation with saved A0 values.
    a0_folder, _ = provenance_checks(config, root, code_root)
    saved = pd.read_csv(a0_folder / "a0_stones.csv").sort_values("stone_id", ignore_index=True)
    stones = recompute_stones(config, root)
    mismatches = reconciliation_mismatches(saved, stones)
    write_csv(pd.DataFrame({"mismatch": mismatches}), folder, "r1_reconciliation_mismatches.csv")
    if mismatches:
        raise StudyStop(f"{len(mismatches)} mismatches with saved A0 values")
    manifest["stages_completed"].append("provenance_and_reconciliation")

    # Stage 2: data stop rules, before any error-based analysis.
    tiers = reference_tiers(saved, config)
    data_stop_checks(saved, stones, config, tiers)
    rows = build_rows(saved, stones)
    manifest["reference_tiers"] = tiers
    manifest["stages_completed"].append("data_stop_rules")

    # Stage 3: the integrity diagnostic runs before psi is computed on the real signals.
    log.info("Pipeline-integrity diagnostic")
    diagnostic = integrity_diagnostic(rows, config, seeds["integrity"], ks)
    write_csv(diagnostic, folder, "r1_integrity_diagnostic.csv")
    if diagnostic.triggered.any():
        raise StudyStop("The pipeline-integrity diagnostic triggered. Investigate before any psi")
    manifest["stages_completed"].append("integrity_diagnostic")

    # Stage 4: primary and ordered secondary analyses.
    log.info("Primary repeats")
    primary = run_primary_repeats(rows, config, seeds["primary"], ks)
    log.info("Primary bootstrap")
    boot = bootstrap_deltas(
        rows,
        config["bootstrap_resamples"],
        config["bootstrap_partitions"],
        seeds["bootstrap"],
        primary_models(),
        primary_comparisons(),
        ks,
    )
    estimates = pd.DataFrame(estimate_rows(primary["deltas"], boot, ks, {}))
    write_csv(primary_table(estimates, primary, ks), folder, "r1_primary.csv")
    write_csv(estimates, folder, "r1_secondary_estimates.csv")
    differing = pd.DataFrame(
        [
            {
                "assumption": assumption,
                "flagged_per_fold": k,
                "total_flagged": k * 5,
                "mean_stones_flagged_differently": float(primary["different"][assumption][:, j].mean()),
            }
            for assumption in ASSUMPTIONS
            for j, k in enumerate(ks)
        ]
    )
    write_csv(differing, folder, "r1_flag_differences.csv")
    manifest["failed_secondary_models"] = primary["failed_models"]
    manifest["stages_completed"].append("primary_and_secondary_estimates")

    # Stage 5: secondary ranges.
    log.info("Split-conformal ranges")
    for name, frame in run_ranges(rows, config, seeds).items():
        write_csv(frame, folder, name)
    manifest["stages_completed"].append("ranges")

    # Stage 6: exploratory and failure analyses, which cannot replace the primary result.
    log.info("Exploratory analyses")
    rng = np.random.default_rng(seeds["exploratory"].spawn(1)[0])
    write_csv(
        exploratory_psi(rows, config, tiers, seeds["exploratory"].spawn(1)[0], ks),
        folder,
        "r1_exploratory_psi.csv",
    )
    write_csv(training_threshold_table(rows, config, rng), folder, "r1_exploratory_training_threshold.csv")
    risk_ks = config["risk_curve_flagged_per_fold"]
    write_csv(
        pd.DataFrame(
            [
                {"rule": key.split("|")[0], "assumption": key.split("|")[1],
                 "flagged_per_fold": k, "mean_E_accepted": value}
                for key, values in primary["risk"].items()
                for k, value in zip(risk_ks, values)
            ]
        ),
        folder,
        "r1_exploratory_risk_curve.csv",
    )
    coefficients = primary["coefficients"].drop(columns=["repeat", "fold"])
    spread = coefficients.groupby("assumption").agg(["mean", "std", "min", "max"])
    spread.columns = [f"{name}_{stat}" for name, stat in spread.columns]
    write_csv(spread.reset_index(), folder, "r1_exploratory_coefficient_spread.csv")
    half_summary, half_stones = half_attenuation_table(saved, stones, config)
    write_csv(half_summary, folder, "r1_exploratory_half_attenuation.csv")
    per_stone = stones_table(saved, stones, rows, primary, half_stones, tiers, ks)
    write_csv(per_stone, folder, "r1_stones.csv")
    write_csv(failure_table(per_stone, config), folder, "r1_failure_analysis.csv")
    write_csv(composition_table(per_stone, config), folder, "r1_exploratory_composition.csv")
    manifest["stages_completed"].append("exploratory_and_failure")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--allow-further-run",
        action="store_true",
        help="Only after a documented stop or software fault, with the owner's approval.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    if config["id"] != "r1_reliability_freiburg":
        raise SystemExit(f"This runner does not handle {config['id']}")
    existing = sorted((root / config["runs_folder"]).glob(f"{config['id']}_*"))
    if existing and not args.allow_further_run:
        raise SystemExit(f"A run already exists: {existing[-1]}")

    run_folder, stop_reason = run_study(config, config_path, root)
    print(run_folder)
    if stop_reason:
        print(f"STOP: {stop_reason}")
        return 2
    print(pd.read_csv(run_folder / "r1_primary.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

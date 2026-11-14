"""Why did the ablation without M fail to converge in the r1 run? A diagnostic, not an analysis.

Replays the r1 primary fold partitions from the saved signals and refits the volume models, using
the study's own solver. It computes no flag, rank, psi or other estimate, and changes no r1 file.
Each failed fit is compared with the unique optimum found by a separate Newton solver. The
objective is strictly convex when all errors are positive and the design has full rank.
"""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_reliability import STREAMS, primary_models

from stonekit.reliability.evaluation import Rows, row_folds
from stonekit.reliability.model import (
    MAX_HALVINGS,
    MAX_ITERATIONS,
    ROUNDING,
    TOLERANCE,
    fit_scale_model,
)

EXTENDED_ITERATIONS = 100_000


def standardized_design(predictors: np.ndarray) -> np.ndarray:
    means = predictors.mean(axis=0)
    sds = predictors.std(axis=0, ddof=1)
    return np.column_stack([np.ones(len(predictors)), (predictors - means) / sds])


def objective(design: np.ndarray, errors: np.ndarray, coefficients: np.ndarray) -> float:
    linear = design @ coefficients
    return float((linear + errors * np.exp(-linear)).sum())


def gradient(design: np.ndarray, errors: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return design.T @ (1 - errors * np.exp(-design @ coefficients))


def newton_optimum(design: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, bool]:
    """A separate solver using the observed Hessian. It shares no step rule with the study's."""
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = np.log(errors.mean())
    for _ in range(500):
        ratio = errors * np.exp(-design @ coefficients)
        slope = design.T @ (1 - ratio)
        if np.abs(slope).max() < 1e-10:
            return coefficients, True
        step = np.linalg.solve((design * ratio[:, None]).T @ design, -slope)
        current = objective(design, errors, coefficients)
        scale = 1.0
        while objective(design, errors, coefficients + scale * step) > current and scale > 1e-12:
            scale /= 2
        coefficients = coefficients + scale * step
    return coefficients, False


def study_iterations(
    design: np.ndarray, errors: np.ndarray, limit: int
) -> tuple[np.ndarray, str, int, list[float]]:
    """The study's Fisher scoring loop, with the limit raised and the step sizes kept."""
    information = design.T @ design
    coefficients = np.zeros(design.shape[1])
    coefficients[0] = np.log(errors.mean())
    current = objective(design, errors, coefficients)
    sizes = []
    for iteration in range(1, limit + 1):
        step = np.linalg.solve(information, design.T @ (errors * np.exp(-design @ coefficients) - 1))
        allowance = ROUNDING * max(abs(current), 1.0)
        for _ in range(MAX_HALVINGS):
            candidate = objective(design, errors, coefficients + step)
            if np.isfinite(candidate) and candidate <= current + allowance:
                break
            step = step / 2
        else:
            return coefficients, "no acceptable step", iteration, sizes
        coefficients = coefficients + step
        current = candidate
        sizes.append(float(np.abs(step).max()))
        if sizes[-1] < TOLERANCE:
            return coefficients, "converged", iteration, sizes
    return coefficients, "iteration limit", limit, sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("run_folder", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())

    saved = pd.read_csv(
        Path(config["a0_run"]) / "a0_stones.csv", float_precision="round_trip"
    ).sort_values("stone_id", ignore_index=True)
    stones = pd.read_csv(
        args.run_folder / "r1_stones.csv", float_precision="round_trip"
    ).sort_values("stone_id", ignore_index=True)
    if not (saved.stone_id == stones.stone_id).all():
        raise SystemExit("Stone IDs differ between a0_stones.csv and r1_stones.csv")
    columns = {"x1": np.log(saved.mask_voxels.to_numpy(float))}
    for name in ("S_G", "M", "E_A", "E_B"):
        columns[name] = stones[name].to_numpy(float)
    rows = Rows(stones.stone_id.to_numpy(int), np.zeros(len(stones), dtype=int), columns)

    seeds = dict(zip(STREAMS, np.random.SeedSequence(config["seed"]).spawn(len(STREAMS))))
    rng = np.random.default_rng(seeds["primary"])
    models = primary_models()
    top_stone = int(np.argmax(columns["S_G"]))
    highest_S_G = stones.sort_values("S_G", ascending=False).head(3)[["stone_id", "S_G", "E_A"]]

    fit_rows, failure_rows = [], []
    for repeat in range(config["primary_repeats"]):
        fold = row_folds(rows, rng)
        for name, spec in models.items():
            predictors = np.column_stack([columns[p] for p in spec.predictors])
            errors = columns[spec.outcome]
            for current in np.unique(fold):
                train = fold != current
                fit = fit_scale_model(predictors[train], errors[train])
                design = standardized_design(predictors[train])
                optimum, newton_ok = newton_optimum(design, errors[train])
                record = {
                    "model": name,
                    "repeat": repeat,
                    "fold": int(current),
                    "converged": fit.converged,
                    "reason": fit.reason,
                    "iterations": fit.iterations,
                    "newton_converged": newton_ok,
                    "newton_max_abs_gradient": float(
                        np.abs(gradient(design, errors[train], optimum)).max()
                    ),
                    "top_S_G_stone_held_out": bool(fold[top_stone] == current),
                }
                if fit.converged:
                    record["max_abs_gradient"] = float(
                        np.abs(gradient(design, errors[train], fit.coefficients)).max()
                    )
                    record["max_abs_coefficient_gap"] = float(
                        np.abs(fit.coefficients - optimum).max()
                    )
                else:
                    at_limit, _, _, sizes = study_iterations(design, errors[train], MAX_ITERATIONS)
                    longer, status, used, _ = study_iterations(
                        design, errors[train], EXTENDED_ITERATIONS
                    )
                    failure_rows.append(
                        {
                            "model": name,
                            "repeat": repeat,
                            "fold": int(current),
                            "reason": fit.reason,
                            "last_step_size": sizes[-1],
                            "first_step_size": sizes[0],
                            "gap_to_optimum_at_limit": float(np.abs(at_limit - optimum).max()),
                            "objective_gap_at_limit": objective(design, errors[train], at_limit)
                            - objective(design, errors[train], optimum),
                            "extended_status": status,
                            "extended_iterations": used,
                            "extended_gap_to_optimum": float(np.abs(longer - optimum).max()),
                            "training_max_S_G": float(predictors[train, 1].max())
                            if spec.predictors[1] == "S_G"
                            else np.nan,
                            "max_standardized_abs": float(np.abs(design[:, 1:]).max()),
                        }
                    )
                fit_rows.append(record)

    fits = pd.DataFrame(fit_rows)
    failures = pd.DataFrame(failure_rows)
    replay = (
        fits[~fits.converged].groupby("model").repeat.nunique().to_dict()
        if (~fits.converged).any()
        else {}
    )
    summary = fits.groupby("model").agg(
        fits=("converged", "size"),
        failed_fits=("converged", lambda v: int((~v).sum())),
        median_iterations=("iterations", "median"),
        max_iterations=("iterations", "max"),
        newton_failures=("newton_converged", lambda v: int((~v).sum())),
        worst_gradient_converged=("max_abs_gradient", "max"),
        worst_coefficient_gap_converged=("max_abs_coefficient_gap", "max"),
        worst_newton_gradient=("newton_max_abs_gradient", "max"),
    )
    by_held_out = (
        fits.groupby(["model", "top_S_G_stone_held_out"])
        .converged.agg(fits="size", failed=lambda v: int((~v).sum()))
        .reset_index()
    )

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder = Path(config["runs_folder"]) / f"r1_diagnostic_no_m_{stamp}"
    folder.mkdir(parents=False, exist_ok=False)
    summary.to_csv(folder / "fit_summary.csv")
    failures.to_csv(folder / "failed_fits.csv", index=False)
    by_held_out.to_csv(folder / "failures_by_top_S_G_stone.csv", index=False)
    highest_S_G.to_csv(folder / "highest_S_G_stones.csv", index=False)
    manifest = {
        "purpose": "Diagnostic of solver behaviour only. No flag, rank or estimate is computed.",
        "r1_run": str(args.run_folder),
        "failed_repeats_replayed": replay,
        "failed_repeats_expected": json.loads(
            (args.run_folder / "run_manifest.json").read_text()
        )["failed_secondary_models"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(folder)
    print(json.dumps(manifest, indent=2))
    print(summary.to_string())
    print(by_held_out.to_string())
    print(highest_S_G.to_string())
    if len(failures):
        print(failures.groupby(["model", "reason", "extended_status"]).size().to_string())
        print(failures.describe().T.to_string())
        print(failures.head(12).to_string())


if __name__ == "__main__":
    main()

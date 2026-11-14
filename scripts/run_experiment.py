"""Run any config end to end.

Usage: python scripts/run_experiment.py configs/a0_replicate_freiburg.yaml

Only a0 exists so far. Its procedure is fixed by docs/STAGE2_PROTOCOL.md. The script writes to a
new timestamped run folder, reports G3 and the stop rules, and decides nothing further.
TODO(later stages): the other experiments in Section 10.
"""

import argparse
import hashlib
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
from stonekit.eval.agreement import bland_altman, icc_a1, pingouin_icc_a1
from stonekit.eval.bootstrap import percentile_interval, resample_indices
from stonekit.eval.plots import bland_altman_figure
from stonekit.logging_utils import get_logger
from stonekit.measure.components import stone_mask
from stonekit.measure.diameter import max_distance_mm, spacing_for_assignment
from stonekit.measure.volume import volume_ml
from stonekit.seg.threshold import threshold_mask

ICC_CROSS_CHECK_TOLERANCE = 1e-6
CODE_ROOT = Path(__file__).resolve().parents[1]
ASSUMPTIONS = ("A", "B")
METRICS = {"volume": "mL", "d_paper": "mm", "d_centre": "mm"}
log = get_logger("run_experiment")


def assumption_geometry(stone, assumption: str, z_mm: float) -> tuple[float, float, float]:
    """Return in-plane spacing, z spacing and voxel volume (mL) without changing the source."""
    if assumption == "A":
        x_mm, _, voxsiz_z_mm = stone.voxsiz_mm
        return x_mm, voxsiz_z_mm, float(np.prod(stone.voxsiz_mm)) / 1000
    return math.sqrt(stone.voxvol_ml * 1000 / z_mm), z_mm, stone.voxvol_ml


def measure(stones: dict, config: dict, threshold: float, largest_only: bool, assignments) -> list:
    """Measure every stone once per assumption and axis assignment."""
    records = []
    for stone_id, stone in stones.items():
        binary = threshold_mask(stone.image, threshold)
        result = stone_mask(binary, config["min_component_voxels"], largest_only)
        empty = not result.mask.any()
        for assumption in ASSUMPTIONS:
            inplane, z_mm, voxel_ml = assumption_geometry(
                stone, assumption, config["assumed_z_spacing_mm"]
            )
            for assignment in assignments:
                spacing = spacing_for_assignment(inplane, z_mm, assignment)
                centre = max_distance_mm(result.mask, spacing)
                edge = 0.0 if empty else config["paper_voxel_edge_mm"]
                records.append(
                    {
                        "stone_id": stone_id,
                        "threshold_hu": threshold,
                        "largest_only": largest_only,
                        "assumption": assumption,
                        "assignment": assignment,
                        "volume": volume_ml(result.mask, voxel_ml),
                        "d_centre": centre,
                        "d_paper": centre + edge,
                        "mask_voxels": int(result.mask.sum()),
                        "n_components": result.n_components,
                        "filled_voxels": result.filled_voxels,
                        "mask_empty": empty,
                    }
                )
    return records


def agreement_row(measured, reference, resamples: int, seed: int) -> dict:
    """ICC, Bland-Altman and bootstrap intervals, with the same resamples for every metric."""
    measured = np.asarray(measured, float)
    reference = np.asarray(reference, float)
    indices = resample_indices(len(measured), resamples, seed)
    icc = percentile_interval(icc_a1(measured[indices], reference[indices]))
    difference = (measured - reference)[indices]
    bias = difference.mean(axis=1)
    sd = difference.std(axis=1, ddof=1)
    stats = bland_altman(measured, reference)
    row = {
        "n": len(measured),
        "icc": float(icc_a1(measured, reference)),
        "icc_ci_low": icc.lower,
        "icc_ci_high": icc.upper,
        "degenerate_resamples": icc.degenerate,
        "resamples": icc.resamples,
        "interval_unreliable": icc.unreliable,
        "bias": stats.bias,
        "sd": stats.sd,
        "loa_low": stats.lower_limit,
        "loa_high": stats.upper_limit,
    }
    for name, values in (
        ("bias", bias),
        ("loa_low", bias - 1.96 * sd),
        ("loa_high", bias + 1.96 * sd),
    ):
        interval = percentile_interval(values)
        row[f"{name}_ci_low"] = interval.lower
        row[f"{name}_ci_high"] = interval.upper
    return row


def evaluate(records: pd.DataFrame, references: pd.DataFrame, labels: dict, config: dict) -> list:
    """Return one agreement row per assumption and metric, for the stones in `records`."""
    rows = []
    merged = records.merge(references, on="stone_id")
    for assumption, group in merged.groupby("assumption", sort=True):
        for metric, unit in METRICS.items():
            reference = group[
                "reference_volume_ml" if metric == "volume" else "reference_diameter_mm"
            ]
            rows.append(
                labels
                | {"assumption": assumption, "metric": metric, "unit": unit}
                | agreement_row(
                    group[metric], reference, config["bootstrap_resamples"], config["seed"]
                )
            )
    return rows


def gate_outcome(row: dict, g3: dict) -> dict:
    icc = row["icc"]
    return {
        "passes": bool(icc > g3["icc_above"]),
        "borderline": bool(
            g3["borderline_low"] <= icc <= g3["borderline_high"]
            or row["icc_ci_low"] <= g3["icc_above"] <= row["icc_ci_high"]
        ),
    }


def stop_rules(primary: pd.DataFrame, axis: pd.DataFrame, g3: dict) -> pd.DataFrame:
    """Evaluate the protocol's stop rules. Material differences are reported for human review."""
    get = lambda assumption, metric: primary[
        (primary.assumption == assumption) & (primary.metric == metric)
    ].iloc[0]
    outcomes = {(a, m): gate_outcome(get(a, m).to_dict(), g3) for a in ASSUMPTIONS for m in METRICS}
    g3_pass = {
        a: outcomes[(a, "volume")]["passes"] and outcomes[(a, "d_paper")]["passes"]
        for a in ASSUMPTIONS
    }

    axis_changes = []
    for assumption in ASSUMPTIONS:
        base = outcomes[(assumption, "d_paper")]
        rows = axis[(axis.assumption == assumption) & (axis.metric == "d_paper")]
        for _, row in rows.iterrows():
            outcome = gate_outcome(row.to_dict(), g3)
            if outcome != base:
                axis_changes.append(f"{assumption}/{row.assignment}")

    primary_keys = [(a, m) for a in ASSUMPTIONS for m in ("volume", "d_paper")]
    rules = [
        (
            1,
            "G3 fails for a measurement under an assumption",
            not all(g3_pass.values()),
            f"A pass={g3_pass['A']}; B pass={g3_pass['B']}",
        ),
        (
            2,
            "Assumptions give different G3 outcomes",
            g3_pass["A"] != g3_pass["B"],
            "Material differences are reported for human review",
        ),
        (
            3,
            "G3 is borderline",
            any(outcomes[k]["borderline"] for k in primary_keys),
            "; ".join(f"{a}/{m}" for a, m in primary_keys if outcomes[(a, m)]["borderline"]),
        ),
        (
            4,
            "D-paper and D-centre would give different G3 outcomes",
            any(
                outcomes[(a, "d_paper")]["passes"] != outcomes[(a, "d_centre")]["passes"]
                for a in ASSUMPTIONS
            ),
            "",
        ),
        (
            5,
            "More than 1% of resamples are degenerate for a primary metric",
            bool(primary["interval_unreliable"].any()),
            f"Largest count {int(primary['degenerate_resamples'].max())}",
        ),
        (
            6,
            "An axis assignment changes a diameter G3 outcome or borderline status",
            bool(axis_changes),
            "; ".join(axis_changes),
        ),
    ]
    return pd.DataFrame(rules, columns=["rule", "description", "triggered", "detail"])


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(config_path: Path) -> dict:
    """Hash the code and protocol. HEAD alone does not identify an uncommitted tree."""
    paths = sorted(CODE_ROOT.glob("src/stonekit/**/*.py"))
    paths += [Path(__file__).resolve(), CODE_ROOT / "docs" / "STAGE2_PROTOCOL.md"]
    hashes = {str(path.relative_to(CODE_ROOT)): file_hash(path) for path in paths}
    if config_path.is_file():
        hashes[f"config/{config_path.name}"] = file_hash(config_path)
    return hashes


def git_state() -> dict:
    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=CODE_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = git("status", "--short")
    return {"head": git("rev-parse", "HEAD"), "uncommitted_changes": bool(status), "status": status}


def run_a0(config: dict, config_path: Path, root: Path) -> Path:
    raw_folder = root / config["raw_folder"]
    if config.get("verify_source_checksums", True):
        for expected in D1.files:
            if file_md5(raw_folder / expected.name) != expected.md5:
                raise SystemExit(f"Source checksum mismatch: {expected.name}")

    table = pd.read_parquet(root / config["table"])
    by_position = {stone.mat_index: stone for stone in load_mat_stones(raw_folder / MAT_NAME)}
    stones = {int(row.stone_id): by_position[int(row.mat_index)] for row in table.itertuples()}
    references = pd.DataFrame(
        {
            "stone_id": list(stones),
            "reference_volume_ml": [stone.volume_ml for stone in stones.values()],
            "reference_diameter_mm": [stone.diameter_mm for stone in stones.values()],
        }
    )
    run_folder = create_run_folder(root / config["runs_folder"], config["id"])
    log.info("Run folder: %s", run_folder)

    primary_hu = config["primary_threshold_hu"]
    primary_axis = config["primary_axis_assignment"]
    assignments = [primary_axis, *config["sensitivity_axis_assignments"]]
    all_axes = pd.DataFrame(measure(stones, config, primary_hu, False, assignments))
    records = all_axes[all_axes.assignment == primary_axis]

    # Primary analysis: all stones, locked threshold, primary axis assignment.
    primary = pd.DataFrame(evaluate(records, references, {"analysis": "primary"}, config))
    for row in primary.itertuples():
        reference = references[
            "reference_volume_ml" if row.metric == "volume" else "reference_diameter_mm"
        ].to_numpy()
        measured = records[records.assumption == row.assumption].set_index("stone_id")[row.metric]
        value, interval = pingouin_icc_a1(measured.loc[references.stone_id].to_numpy(), reference)
        if abs(value - row.icc) > ICC_CROSS_CHECK_TOLERANCE:
            raise SystemExit(f"ICC formula and pingouin disagree: {value} against {row.icc}")
        primary.loc[row.Index, "icc_pingouin"] = value
        primary.loc[row.Index, "icc_pingouin_ci95_rounded"] = interval
    g3 = config["g3"]
    primary["passes_above_threshold"] = primary["icc"] > g3["icc_above"]
    primary["decides_g3"] = primary["metric"].isin(["volume", "d_paper"])
    primary.to_csv(run_folder / "a0_primary.csv", index=False)

    # Exploratory sweep: every threshold is reported, and none is selected.
    sweep = []
    for threshold in config["exploratory_thresholds_hu"]:
        chunk = pd.DataFrame(measure(stones, config, threshold, False, [primary_axis]))
        labels = {
            "analysis": "exploratory",
            "threshold_hu": threshold,
            "empty_masks": int(chunk[chunk.assumption == "A"]["mask_empty"].sum()),
        }
        sweep.extend(evaluate(chunk, references, labels, config))
    pd.DataFrame(sweep).to_csv(run_folder / "a0_sweep_exploratory.csv", index=False)

    # Sensitivity analyses: none of these decides G3.
    sensitivity = []
    exclusions = {name: set(ids) for name, ids in config["sensitivity_exclusions"].items()}
    agree = set(table.loc[~table["spacing_fields_disagree"], "stone_id"].astype(int))
    subsets = {
        **{name: set(stones) - ids for name, ids in exclusions.items()},
        "S3_spacing_fields_agree": agree,
    }
    for name, keep in sorted(subsets.items()):
        chunk = records[records.stone_id.isin(keep)]
        sensitivity.extend(evaluate(chunk, references, {"analysis": name}, config))
    largest = pd.DataFrame(measure(stones, config, primary_hu, True, [primary_axis]))
    sensitivity.extend(evaluate(largest, references, {"analysis": "S5_largest_component"}, config))
    pd.DataFrame(sensitivity).to_csv(run_folder / "a0_sensitivity.csv", index=False)

    # Axis-assignment sensitivity: diameters only, because volume does not change.
    axis_rows = []
    for assignment in assignments:
        chunk = all_axes[all_axes.assignment == assignment]
        labels = {"analysis": "axis_sensitivity", "assignment": assignment}
        axis_rows.extend(
            row for row in evaluate(chunk, references, labels, config) if row["metric"] != "volume"
        )
    axis = pd.DataFrame(axis_rows)
    axis.to_csv(run_folder / "a0_axis_sensitivity.csv", index=False)

    # Per-stone results and flags for all stones.
    wide = records.pivot(index="stone_id", columns="assumption", values=list(METRICS))
    wide.columns = [f"{metric}_{assumption}" for metric, assumption in wide.columns]
    single = records[records.assumption == "A"].set_index("stone_id")[
        ["mask_voxels", "n_components", "filled_voxels", "mask_empty"]
    ]
    flags = table.set_index("stone_id")[
        [
            "stone_type",
            "spacing_fields_disagree",
            "image_dtype",
            "volume_exceeds_diameter_sphere",
            "volume_below_4mm_sphere",
        ]
    ]
    per_stone = references.set_index("stone_id").join([wide, single, flags])
    per_stone.to_csv(run_folder / "a0_stones.csv")

    bootstrap_columns = [
        "analysis",
        "assumption",
        "metric",
        "n",
        "resamples",
        "degenerate_resamples",
    ]
    summary = pd.concat(
        [
            frame[bootstrap_columns]
            for frame in (primary, pd.DataFrame(sweep), pd.DataFrame(sensitivity), axis)
        ],
        ignore_index=True,
    )
    summary.insert(0, "seed", config["seed"])
    summary.to_csv(run_folder / "a0_bootstrap_summary.csv", index=False)

    stops = stop_rules(primary, axis, g3)
    stops.to_csv(run_folder / "a0_stop_rules.csv", index=False)

    for assumption in ASSUMPTIONS:
        chunk = records[records.assumption == assumption].merge(references, on="stone_id")
        for metric, unit in METRICS.items():
            column = "reference_volume_ml" if metric == "volume" else "reference_diameter_mm"
            bland_altman_figure(
                chunk[metric].to_numpy(),
                chunk[column].to_numpy(),
                f"{metric}, spacing assumption {assumption}, {primary_hu} HU, {len(chunk)} stones",
                unit,
                run_folder / f"a0_bland_altman_{metric}_{assumption}",
            )

    manifest = {
        "protocol": "docs/STAGE2_PROTOCOL.md",
        "config": config,
        "git": git_state(),
        "source_hashes": source_hashes(config_path),
        "versions": {
            "python": platform.python_version(),
            **{
                name: version(name)
                for name in ("numpy", "scipy", "pandas", "pingouin", "matplotlib")
            },
        },
    }
    (run_folder / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_folder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--allow-rerun",
        action="store_true",
        help="Run again although a run exists. Every run must be reported.",
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    root = find_project_root(config_path.parent)
    config = load_yaml(config_path)
    if config["id"] != "a0_replicate_freiburg":
        raise SystemExit(f"No runner for {config['id']} yet")
    existing = sorted((root / config["runs_folder"]).glob(f"{config['id']}_*"))
    if existing and not args.allow_rerun:
        raise SystemExit(f"A run already exists: {existing[-1]}")

    run_folder = run_a0(config, config_path, root)
    print(run_folder)
    print(pd.read_csv(run_folder / "a0_stop_rules.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

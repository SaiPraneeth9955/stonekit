"""Run protocol rep1: the complete replication of Neubauer et al. 2023 on the released Freiburg data.

Usage: python scripts/run_neubauer_replication.py configs/rep1_neubauer_replication.yaml

Protocol: docs/FREIBURG_REPLICATION_PROTOCOL.md. The config must have status "approved"; otherwise the
script stops before reading any data. CPU only, Freiburg only. Writes one new timestamped folder.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from stonekit.config import create_run_folder, load_yaml
from stonekit.data.freiburg import CSV_NAME, MAT_NAME, load_mat_stones, load_table, match_stones
from stonekit.data.registry import find_project_root
from stonekit.eval.agreement import icc_a1
from stonekit.measure.diameter import AXIS_ASSIGNMENTS, spacing_for_assignment
from stonekit.replication import neubauer as nb

METHODS = ("M1", "M2", "M3", "M4")


class NotApproved(RuntimeError):
    """The protocol has not been approved, so the replication must not run."""


def grid(spec: dict) -> np.ndarray:
    return np.round(np.arange(spec["start"], spec["stop"] + spec["step"] / 2, spec["step"]), 6)


def spacing(stone, name: str, config: dict) -> tuple[float, float, float]:
    """In-plane spacing, z spacing and voxel volume (mL) for spacing interpretation A, B or P."""
    if name == "A":
        x_mm, _, z_mm = stone.voxsiz_mm
        return x_mm, z_mm, float(np.prod(stone.voxsiz_mm)) / 1000
    if name == "B":
        z_mm = config["assumed_z_spacing_mm"]
        return math.sqrt(stone.voxvol_ml * 1000 / z_mm), z_mm, stone.voxvol_ml
    p = config["paper_spacing_mm"]
    return p, p, p**3 / 1000


def geometries_for(stone, config: dict) -> tuple[dict, dict]:
    geometries, voxel_ml = {}, {}
    for name, spec in config["geometries"].items():
        inplane, z_mm, vml = spacing(stone, spec["spacing"], config)
        geometries[name] = nb.Geometry(spacing_for_assignment(inplane, z_mm, spec["axis"]), AXIS_ASSIGNMENTS[spec["axis"]])
        voxel_ml[name] = vml
    return geometries, voxel_ml


def method_arrays(profiles_abs, profiles_med, voxel_ml, geometry, surface, diameter, pv):
    """Volume and diameter arrays (stones x parameter grid) for the four methods under one configuration."""
    def pack(profiles):
        n = np.array([p.voxels for p in profiles])
        s = np.array([getattr(p, f"surface_{surface}") for p in profiles])
        d = np.array([(p.diameter_3d_mm if diameter == "3d" else p.diameter_axial_mm)[geometry] for p in profiles])
        v = np.array([voxel_ml[i][geometry] for i in range(len(profiles))])[:, None]
        return n, s, d, v

    n1, s1, d1, v1 = pack(profiles_abs)
    n3, s3, d3, v3 = pack(profiles_med)
    return {
        "M1": (n1 * v1, d1),
        "M2": (nb.corrected_volume_ml(n1[:, :, None], s1[:, :, None], pv[None, None, :], 1.0) * v1[:, :, None], np.repeat(d1[:, :, None], len(pv), axis=2)),
        "M3": (n3 * v3, d3),
        "M4": (nb.corrected_volume_ml(n3[:, :, None], s3[:, :, None], pv[None, None, :], 1.0) * v3[:, :, None], np.repeat(d3[:, :, None], len(pv), axis=2)),
    }  # fmt: skip


def parameters(method: str, index: int, p1_abs, p1_med, pv) -> dict:
    if method == "M1":
        return {"P1": float(p1_abs[index])}
    if method == "M3":
        return {"P1": float(p1_med[index])}
    p1 = p1_abs if method == "M2" else p1_med
    i, j = np.unravel_index(index, (len(p1), len(pv)))
    return {"P1": float(p1[i]), "P2": float(pv[j])}


def run_configuration(arrays, ref_v, ref_d, splits, p1_abs, p1_med, pv) -> pd.DataFrame:
    rows = []
    best = {m: nb.optimise_splits(nb.per_stone_cost(v, d, ref_v, ref_d), splits) for m, (v, d) in arrays.items()}
    for k, (_, validation) in enumerate(splits):
        for method, (volume, diameter) in arrays.items():
            index = int(best[method][k])
            v = volume.reshape(len(ref_v), -1)[validation, index]
            d = diameter.reshape(len(ref_d), -1)[validation, index]
            vs = nb.validation_stats(v, ref_v[validation], icc_a1)
            ds = nb.validation_stats(d, ref_d[validation], icc_a1)
            rows.append({"split": k, "method": method, **parameters(method, index, p1_abs, p1_med, pv),
                         "vol_icc": vs["icc"], "vol_diff": vs["difference_estimate_minus_reference"], "vol_sd": vs["sd"],
                         "diam_icc": ds["icc"], "diam_diff": ds["difference_estimate_minus_reference"], "diam_sd": ds["sd"]})  # fmt: skip
    return pd.DataFrame(rows)


def compare_with_paper(splits_df: pd.DataFrame, paper: dict) -> pd.DataFrame:
    rows = []
    for method in METHODS:
        group = splits_df[splits_df.method == method]
        for key, (paper_mean, paper_sd) in paper[method].items():
            ours = group[key]
            mean, sd = float(ours.mean()), float(ours.std(ddof=1))
            # Difference sign: ours is estimate - reference; the paper's convention is not stated, so both are checked.
            candidates = [mean, -mean] if key.endswith("_diff") else [mean]
            gap = min(abs(c - paper_mean) for c in candidates)
            verdict = "reproduced" if gap <= sd else "close" if gap <= 2 * sd else "not reproduced"
            rows.append({"method": method, "quantity": key, "paper_mean": paper_mean, "paper_sd": paper_sd,
                         "ours_mean": round(mean, 4), "ours_sd": round(sd, 4), "abs_gap": round(gap, 4), "verdict": verdict})  # fmt: skip
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    if config.get("status") != "approved":
        raise NotApproved(f"{args.config} has status {config.get('status')!r}; the protocol must be approved first")
    root = find_project_root()
    raw = root / config["raw_folder"]
    matched = match_stones(load_table(raw / CSV_NAME), load_mat_stones(raw / MAT_NAME))
    stone_ids = sorted(matched)
    stones = [matched[i] for i in stone_ids]
    ref_v = np.array([s.volume_ml for s in stones])
    ref_d = np.array([s.diameter_mm for s in stones])
    p1_abs, p1_med, pv = grid(config["m1_threshold_hu"]), grid(config["m3_factor"]), grid(config["pv_factor"])

    profiles_abs, profiles_med, voxel_ml, medians = [], [], [], []
    for stone in stones:
        geometries, vml = geometries_for(stone, config)
        voxel_ml.append(vml)
        median = nb.rough_median_hu(stone.image)
        medians.append(median)
        profiles_abs.append(nb.stone_profile(stone.image, p1_abs, geometries, config["paper_voxel_edge_mm"], config["min_component_voxels"]))
        profiles_med.append(nb.stone_profile(stone.image, p1_med * median, geometries, config["paper_voxel_edge_mm"], config["min_component_voxels"]))
        print("profiled stone", len(profiles_abs), flush=True)

    splits = nb.random_splits(len(stones), config["n_train"], config["n_splits"], config["seed"])
    run = create_run_folder(root / config["runs_folder"], config["id"])
    configurations = [(g, s, d) for g in config["geometries"] for s in (6, 26) for d in ("3d", "axial")]
    primary = (config["primary"]["geometry"], config["primary"]["surface"], config["primary"]["diameter"])
    summary = {"protocol": "docs/FREIBURG_REPLICATION_PROTOCOL.md", "config": str(args.config), "stones": len(stones),
               "splits": len(splits), "primary": primary, "rough_median_hu": {"min": float(np.nanmin(medians)), "max": float(np.nanmax(medians))},
               "configurations": {}}  # fmt: skip
    for g, s, d in configurations:
        if sum(a != b for a, b in zip((g, s, d), primary, strict=True)) > 1:
            continue  # sensitivity analyses vary one factor at a time from the primary configuration
        arrays = method_arrays(profiles_abs, profiles_med, voxel_ml, g, s, d, pv)
        per_split = run_configuration(arrays, ref_v, ref_d, splits, p1_abs, p1_med, pv)
        name = f"{g}_surface{s}_{d}"
        per_split.to_csv(run / f"splits_{name}.csv", index=False)
        comparison = compare_with_paper(per_split, config["paper_table1"])
        comparison.to_csv(run / f"table1_comparison_{name}.csv", index=False)
        summary["configurations"][name] = {
            "verdicts": comparison["verdict"].value_counts().to_dict(),
            "table1": {
                method: {col: {"mean": round(float(group[col].mean()), 4), "sd": round(float(group[col].std(ddof=1)), 4)}
                         for col in group.columns if col not in ("split", "method")}
                for method, group in per_split.groupby("method")
            },
        }
        if (g, s, d) == primary:
            summary["primary_all_stone"] = all_stone_analysis(per_split, arrays, ref_v, ref_d, p1_abs, p1_med, pv, run, config)
        print("done", name, flush=True)
    (run / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("Saved to", run)
    return 0


def all_stone_analysis(per_split, arrays, ref_v, ref_d, p1_abs, p1_med, pv, run, config) -> dict:
    """Figures 3-4 equivalents: all 101 stones at the parameters nearest the split means (in-sample), plus Bartlett."""
    errors, out = {}, {}
    for method, (volume, diameter) in arrays.items():
        group = per_split[per_split.method == method]
        p1 = p1_abs if method in ("M1", "M2") else p1_med
        i = int(np.argmin(np.abs(p1 - group["P1"].mean())))
        if method in ("M2", "M4"):
            j = int(np.argmin(np.abs(pv - group["P2"].mean())))
            v, d = volume[:, i, j], diameter[:, i, j]
        else:
            v, d = volume[:, i], diameter[:, i]
        errors[method] = (v - ref_v, d - ref_d)
        rel = 200 * (v - ref_v) / (v + ref_v)
        out[method] = {"vol_icc_all": float(icc_a1(v, ref_v)), "diam_icc_all": float(icc_a1(d, ref_d)),
                       "vol_bias_ml": float((v - ref_v).mean()), "diam_bias_mm": float((d - ref_d).mean()),
                       "vol_median_abs_relative_error_pct": float(np.median(np.abs(rel)))}  # fmt: skip
        pd.DataFrame({"reference_volume_ml": ref_v, "estimate_volume_ml": v, "reference_diameter_mm": ref_d,
                      "estimate_diameter_mm": d}).to_csv(run / f"all_stones_{method}.csv", index=False)  # fmt: skip
    for k, metric in enumerate(("volume", "diameter")):
        groups = [errors[m][k] for m in METHODS]
        out[f"bartlett_{metric}_p"] = float(stats.bartlett(*groups).pvalue)
        rng = np.random.default_rng(config["seed"])
        draws = rng.integers(0, len(ref_v), size=(config["bootstrap_resamples"], len(ref_v)))
        sds = np.array([[np.std(g[d], ddof=1) for g in groups] for d in draws])
        out[f"paired_sd_range_{metric}_interval_95"] = [float(np.percentile(sds.max(1) - sds.min(1), q)) for q in (2.5, 97.5)]
    out["paper_bartlett_p"] = config["paper_bartlett_p"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())

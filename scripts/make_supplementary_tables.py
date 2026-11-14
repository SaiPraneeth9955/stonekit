"""Supplementary tables for the manuscript, generated ONLY from saved run folders (no recomputation of results).
Development tables are captioned DEVELOPMENT (out-of-fold; not validation). Freiburg tables use the primary
configuration (P, rep1 parameters, M1). Output: paper/supplement/tab_*.tex plus a CSV of the same numbers.

S1: per-stone interval methods (E6), coverage by predicted-size band, marginal coverage and width, at 0.90.
S2: patient-burden methods (E8 run with the A/B-CQR CI), at 0.90 and 0.80.
S3: secondary, ablation and exploratory analyses (N-A, N-B, N-C, asymmetric U2), development.
S4: Freiburg R2 volume error by caliper size band and composition (primary configuration).
"""

import json
from pathlib import Path

import pandas as pd

RUNS = Path("results/runs")
OUT = Path("paper/supplement")
E6 = RUNS / "e6_uncertainty_design_20261112T082354Z/summary.json"
E8 = RUNS / "e8_burden_uncertainty_20261112T085235Z/summary.json"
NANB = RUNS / "secondary_na_nb_development_20261112T103848Z/summary.json"
NC = RUNS / "ablation_nc_development_20261112T104213Z/summary.json"
ASYM = RUNS / "exploratory_asymmetric_u2_20261112T124900Z/summary.json"
R23 = RUNS / "freiburg_r2_r3_20261111T074847Z/summary.json"
ROLE = {"U0": "baseline", "U1": "baseline", "U2": "final (pre-specified)", "U3": "comparator", "U4": "exploratory", "U5": "comparator"}


def tex(df: pd.DataFrame, caption: str, label: str, name: str) -> None:
    body = df.to_latex(index=False, escape=True, float_format=lambda v: f"{v:.3f}")
    (OUT / f"{name}.tex").write_text(f"\\begin{{table}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\centering\\scriptsize\n{body}\\end{{table}}\n", encoding="utf-8")
    df.to_csv(OUT / f"{name}.csv", index=False)


def s1():
    rows = []
    res = json.loads(E6.read_text())["results"]
    for target, tname in (("volume_mm3", "volume"), ("max_diameter_3d_mm", "3D diameter")):
        r = res[f"{target}|0.90"]
        for m in ("U0", "U1", "U2", "U3", "U4", "U5"):
            e = r[m]
            rows.append({"target": tname, "method": m, "role": ROLE[m], "cov <5": e["pred_stratum"]["<5"]["coverage"],
                         "cov 5-<10": e["pred_stratum"]["5-<10"]["coverage"], "cov >=10": e["pred_stratum"][">=10"]["coverage"],
                         "marginal": e["marginal"]["coverage"], "median rel. width": e["marginal"]["median_rel_width"]})  # fmt: skip
    tex(pd.DataFrame(rows), "DEVELOPMENT (out-of-fold; 465 matched stones, 122 scans; not validation): per-stone interval coverage at nominal 0.90 by predicted 3D diameter band. "
        "Under the pre-specified selection rule no method qualified, so U2 is carried (OD-12); U4 is exploratory.", "tab:s1", "tab_s1_per_stone_methods")


def s2():
    res = json.loads(E8.read_text())["results"]
    rows = []
    for nominal in ("0.90", "0.80"):
        for m in ("P2-0 naive", "P2-B direct", "P2-B-CQR", "P2-A decomposition", "P2-A without missed term"):
            e = res[nominal][m]
            rows.append({"nominal": nominal, "method": m, "coverage": e["coverage"], "CP low": e["cp95"][0], "CP high": e["cp95"][1],
                         "median width mm3": e["median_width_mm3"], "stone-free coverage": e["coverage_stone_free"]})  # fmt: skip
    tex(pd.DataFrame(rows), "DEVELOPMENT (out-of-fold; 165 scans, 33 stone-free; not validation): patient-burden intervals. P2-B-CQR here is the development half-split fit; "
        "the locked-test comparator is fitted on all development scans (S-2) and is narrower, so the development width ratio overstates the expected advantage.", "tab:s2", "tab_s2_burden_methods")


def s3():
    nanb, nc, asym = json.loads(NANB.read_text())["results"], json.loads(NC.read_text()), json.loads(ASYM.read_text())["results"]
    v = nanb["volume_mm3|0.90"]
    rows = [
        {"analysis": "N-A size x HU Mondrian (robustness)", "status": "pre-specified rule NOT met", "key result": f"worst cell {v['N-A']['worst_cell']['coverage']:.3f} (n={v['N-A']['worst_cell']['n']}) vs U2 {v['U2']['worst_cell']['coverage']:.3f}; width {v['N-A']['median_rel_width']:.3f} vs {v['U2']['median_rel_width']:.3f}"},
        {"analysis": "N-B patient joint coverage (secondary)", "status": "pre-specified rule met", "key result": f"joint {v['N-B']['joint_patient']['coverage']:.3f} vs U2 {v['U2']['joint_patient']['coverage']:.3f}; width x{v['N-B_rule']['width_inflation']:.2f}"},
        {"analysis": "N-C distance-only FP model (ablation)", "status": "+-5% equivalence NOT met", "key result": f"coverage {nc['distance_only']['coverage_median']:.3f} vs {nc['full']['coverage_median']:.3f}; width change {nc['relative_width_change']['median']:+.3f}"},
        {"analysis": "Asymmetric U2 (exploratory, not pre-specified)", "status": "exploratory", "key result": f"volume cov {asym['volume_mm3|0.90']['U2-asymmetric']['marginal']:.3f}, width {asym['volume_mm3|0.90']['U2-asymmetric']['median_rel_width']:.3f} vs U2 {asym['volume_mm3|0.90']['U2']['median_rel_width']:.3f}"},
    ]  # fmt: skip
    tex(pd.DataFrame(rows), "DEVELOPMENT: secondary, ablation and exploratory analyses (volume, nominal 0.90). Rules were fixed before each run except the asymmetric interval, which is exploratory.", "tab:s3", "tab_s3_secondary")


def s4():
    r2 = json.loads(R23.read_text())["R2"]["P|rep1|M1"]
    rows = [{"group": f"size {k}", "n": e["n"], "median |rel. vol. error|": e["median_abs_rel_vol_err"], "CI low": e["ci95"][0], "CI high": e["ci95"][1]} for k, e in r2["size_bin"].items()]
    rows += [{"group": f"type {k}", "n": e["n"], "median |rel. vol. error|": e["median_abs_rel_vol_err"], "CI low": e["ci95"][0], "CI high": e["ci95"][1]} for k, e in r2["type_group"].items()]
    rows.append({"group": "all", "n": r2["all"]["n"], "median |rel. vol. error|": r2["all"]["median_abs_rel_vol_err"], "CI low": r2["all"]["ci95"][0], "CI high": r2["all"]["ci95"][1]})
    tex(pd.DataFrame(rows), "Physical reference (Freiburg, 101 stones; primary configuration P, rep1 parameters, M1): median absolute relative volume error with bootstrap 95\\% CIs.", "tab:s4", "tab_s4_freiburg_r2")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s1()
    s2()
    s3()
    s4()
    print("Saved supplementary tables to", OUT, sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()

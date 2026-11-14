"""Development-result figures for the Li 2022 patient CT analyses (development out-of-fold data only; no image
pixels; no calibration or test data). Every figure is labelled DEVELOPMENT and is regenerated from saved run
folders; rerun this script after any D-REV-driven development rerun.

- L1: per-stone volume-interval coverage at nominal 0.90 by predicted-size band, for the baseline methods U0
  and U1 and the final method U2 (E6; Clopper-Pearson 95% CI).
- L2: patient-burden intervals at nominal 0.90 (E8, run with the A/B-CQR CI): coverage with CP 95% CI (left)
  and median width on a log scale (right). Two panels, because the measures have different scales.
- L3: detection share by annotated 3D diameter (E7; scan-cluster bootstrap 95% CI).
Outputs: paper/figures/fig_li_dev_*.{pdf,png} plus the plotted numbers as CSV (the table view).
Palette: a validated 3-slot categorical set (blue, orange, aqua), with marker shapes as secondary encoding because aqua
is below 3:1 contrast.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

E6 = Path("results/runs/e6_uncertainty_design_20261112T082354Z/summary.json")
E7 = Path("results/runs/e7_detection_uncertainty_20261112T082355Z/summary.json")
E8 = Path("results/runs/e8_burden_uncertainty_20261112T085235Z/summary.json")
OUT = Path("paper/figures")
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = {"U2": ("#2a78d6", "o", "U2 Mondrian (final)"), "U1": ("#eb6834", "s", "U1 global conformal"), "U0": ("#1baf7a", "^", "U0 Bland–Altman")}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name, table: pd.DataFrame):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    table.to_csv(OUT / f"{name}_data.csv", index=False)


def l1():
    res = json.loads(E6.read_text())["results"]["volume_mm3|0.90"]
    bands = ["<5", "5-<10", ">=10"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0), facecolor=SURFACE)
    style(ax)
    ax.axhline(0.90, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate("nominal 0.90", xy=(2.45, 0.90), xytext=(0, 3), textcoords="offset points", color=INK2, fontsize=7, ha="right")
    rows = []
    for k, m in enumerate(["U0", "U1", "U2"]):
        colour, marker, label = SERIES[m]
        xs = np.arange(3) + (k - 1) * 0.18
        for x, b in zip(xs, bands, strict=True):
            s = res[m]["pred_stratum"][b]
            ax.vlines(x, s["cp95"][0], s["cp95"][1], color=colour, linewidth=1.5)
            ax.scatter(x, s["coverage"], s=30, marker=marker, color=colour, edgecolor=SURFACE, linewidth=0.8, zorder=3,
                       label=label if b == bands[0] else None)  # fmt: skip
            rows.append({"method": m, "band": b, "n": s["n"], "coverage": s["coverage"], "cp_low": s["cp95"][0], "cp_high": s["cp95"][1]})
    n = {b: res["U2"]["pred_stratum"][b]["n"] for b in bands}
    ax.set_xticks(range(3), [f"< 5 mm\n(n = {n['<5']})", f"5–< 10 mm\n(n = {n['5-<10']})", f"≥ 10 mm\n(n = {n['>=10']})"], color=INK, fontsize=8)
    ax.set_ylabel("Per-stone volume coverage", color=INK2, fontsize=8)
    ax.set_xlabel("Predicted 3D maximum diameter band", color=INK2, fontsize=8)
    ax.set_ylim(0.6, 1.01)
    ax.legend(frameon=False, fontsize=7, loc="lower right", labelcolor=INK)
    ax.set_title("DEVELOPMENT (out-of-fold, 465 stones, 122 scans)", color=INK2, fontsize=7, loc="left")
    fig.tight_layout()
    save(fig, "fig_li_dev_stone_coverage_by_size", pd.DataFrame(rows))


def l2():
    res = json.loads(E8.read_text())["results"]["0.90"]
    methods = [("P2-0 naive", "P2-0 naive sum"), ("P2-B direct", "P2-B direct"), ("P2-B-CQR", "P2-B-CQR (comparator)"), ("P2-A decomposition", "P2-A (final)")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, facecolor=SURFACE)
    for ax in (a1, a2):
        style(ax)
        ax.grid(axis="y", visible=False)
    ys = np.arange(len(methods))
    rows = []
    a1.axvline(0.90, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    for y, (key, label) in zip(ys, methods, strict=True):
        r = res[key]
        a1.hlines(y, r["cp95"][0], r["cp95"][1], color="#2a78d6", linewidth=1.5)
        a1.scatter(r["coverage"], y, s=30, color="#2a78d6", edgecolor=SURFACE, linewidth=0.8, zorder=3)
        a2.scatter(r["median_width_mm3"], y, s=30, color="#2a78d6", edgecolor=SURFACE, linewidth=0.8, zorder=3)
        rows.append({"method": key, "coverage": r["coverage"], "cp_low": r["cp95"][0], "cp_high": r["cp95"][1], "median_width_mm3": r["median_width_mm3"], "n": r["n"]})
    a1.set_yticks(ys, [lbl for _, lbl in methods], color=INK, fontsize=8)
    a1.set_xlabel("Patient-burden coverage (CP 95% CI)", color=INK2, fontsize=8)
    a2.set_xscale("log")
    a2.set_xticks([1000, 2000, 5000, 10000, 20000])
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlim(800, 25000)
    a2.set_xlabel("Median interval width (mm³, log scale)", color=INK2, fontsize=8)
    a1.set_title("DEVELOPMENT (out-of-fold, 165 scans), nominal 0.90", color=INK2, fontsize=7, loc="left")
    fig.text(0.01, 0.01, "Note: P2-B-CQR as fitted in E8 (half split). The T1 comparator is fitted on all development scans (spec S-2)\n"
             "and is narrower (development dry run median ≈ 5,300 mm³).", color=INK2, fontsize=6.5, ha="left", va="bottom")  # fmt: skip
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_li_dev_burden_intervals", pd.DataFrame(rows))


def l3():
    det = json.loads(E7.read_text())["detection_by_size_bin"]
    order = [("<3", "< 3 mm"), ("3-<5", "3–< 5 mm"), ("5-<10", "5–< 10 mm"), (">=10", "≥ 10 mm")]
    fig, ax = plt.subplots(figsize=(4.6, 2.8), facecolor=SURFACE)
    style(ax)
    rows = []
    for x, (k, label) in enumerate(order):
        d = det[k]
        ax.vlines(x, d["cluster_ci95"][0], d["cluster_ci95"][1], color="#2a78d6", linewidth=1.5)
        ax.scatter(x, d["share"], s=30, color="#2a78d6", edgecolor=SURFACE, linewidth=0.8, zorder=3)
        rows.append({"band": k, "n": d["n"], "scans": d["scans"], "detected_share": d["share"], "ci_low": d["cluster_ci95"][0], "ci_high": d["cluster_ci95"][1]})
    ax.set_xticks(range(4), [f"{label}\n(n = {det[k]['n']})" for k, label in order], color=INK, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Annotated pieces detected", color=INK2, fontsize=8)
    ax.set_xlabel("Annotated 3D maximum diameter", color=INK2, fontsize=8)
    ax.set_title("DEVELOPMENT (out-of-fold, 741 pieces, 165 scans)", color=INK2, fontsize=7, loc="left")
    fig.tight_layout()
    save(fig, "fig_li_dev_detection_by_size", pd.DataFrame(rows))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    l1()
    l2()
    l3()
    print("Saved development figures to", OUT)


if __name__ == "__main__":
    main()

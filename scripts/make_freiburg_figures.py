"""Publication figures for the Freiburg physical-reference analyses (public CC BY 4.0 data; no patient data).

Source: results/runs/freiburg_r2_r3_20261111T074847Z (protocol docs/FREIBURG_R2_R3_PROTOCOL.md), primary
configuration only: spacing P, rep1 parameters, method M1 (as pre-specified).
- Figure F1 (R3): agreement of the 3D and the 2D axial maximum diameter with the caliper maximum diameter
  (estimate - caliper), one panel each, shared y axis; bias (solid) and 95% limits of agreement (dashed).
- Figure F2 (R2): median absolute relative volume error by stone type with the bootstrap 95% CI stored
  in that run (no recomputation), plus the all-stone median.
Each figure is saved as PDF and PNG (300 dpi) in paper/figures/, with the plotted numbers as CSV.
Colours: validated two-slot categorical palette (blue = 3D, orange = 2D); neutral inks for text.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path("results/runs/freiburg_r2_r3_20261111T074847Z")
OUT = Path("paper/figures")
KEY = "P|rep1|M1"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def figure_r3(df: pd.DataFrame, summary: dict) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True, facecolor=SURFACE)
    rows = []
    for ax, (col, label, colour) in zip(axes, [("err_d3_paper_mm", "3D maximum diameter", BLUE),
                                               ("err_d2_axial_paper_mm", "2D axial maximum diameter", ORANGE)], strict=True):  # fmt: skip
        style(ax)
        x, y = df.ref_diameter_mm.to_numpy(), df[col].to_numpy()
        stats = summary["R3"][KEY][col.replace("err_", "")]
        bias, (lo, hi) = stats["bias_mm"], stats["loa95_mm"]
        ax.axhline(0, color=AXIS, linewidth=1.0)
        ax.scatter(x, y, s=22, color=colour, edgecolor=SURFACE, linewidth=0.8, zorder=3)
        ax.axhline(bias, color=colour, linewidth=1.5)
        for v in (lo, hi):
            ax.axhline(v, color=colour, linewidth=1.0, linestyle=(0, (4, 3)))
        # bias and limits of agreement go in the panel subtitle, so no label can cover a point
        ax.set_title(label, color=INK, fontsize=9, loc="left", pad=16)
        ax.text(0, 1.02, f"bias {bias:+.2f} mm (solid); 95% LoA {lo:+.2f} to {hi:+.2f} mm (dashed)", transform=ax.transAxes,
                color=INK2, fontsize=7, ha="left", va="bottom")  # fmt: skip
        ax.set_xlabel("Caliper maximum diameter (mm)", color=INK2, fontsize=8)
        rows += [{"panel": label, "stone": s, "caliper_mm": a, "error_mm": b} for s, a, b in zip(df.stone, x, y, strict=True)]
    axes[0].set_ylabel("Estimate − caliper (mm)", color=INK2, fontsize=8)
    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_freiburg_r3_diameter_agreement.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(rows)


def figure_r2(summary: dict) -> pd.DataFrame:
    r2 = summary["R2"][KEY]
    groups = [(k, v) for k, v in r2["type_group"].items() if v["n"] >= 5]
    groups.sort(key=lambda kv: kv[1]["median_abs_rel_vol_err"])
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(groups) + 1.0), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", visible=False)
    ys = np.arange(len(groups))
    med = np.array([v["median_abs_rel_vol_err"] for _, v in groups])
    lo = np.array([v["ci95"][0] for _, v in groups])
    hi = np.array([v["ci95"][1] for _, v in groups])
    ax.axvline(r2["all"]["median_abs_rel_vol_err"], color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.annotate(f"all stones, median (n = {r2['all']['n']})", xy=(r2["all"]["median_abs_rel_vol_err"], -0.55), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=7, va="center", ha="left")  # fmt: skip
    ax.set_ylim(-0.9, len(groups) - 0.5)
    ax.hlines(ys, lo, hi, color=BLUE, linewidth=1.5)
    ax.scatter(med, ys, s=30, color=BLUE, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    names = {"calciumoxalate-dihydrate": "calcium oxalate dihydrate", "other": "other types"}  # display only; data labels unchanged
    ax.set_yticks(ys, [f"{names.get(k, k)} (n = {v['n']})" for k, v in groups], color=INK, fontsize=8)
    ax.set_xlabel("Median absolute relative volume error (95% CI)", color=INK2, fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlim(left=0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_freiburg_r2_volume_error_by_type.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame([{"type": k, "n": v["n"], "median": v["median_abs_rel_vol_err"], "ci_low": v["ci95"][0], "ci_high": v["ci95"][1]} for k, v in groups])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RUN / "summary.json").read_text())
    per = pd.read_csv(RUN / "per_stone.csv")
    sp, par, meth = KEY.split("|")
    df = per[(per.spacing == sp) & (per.parameters == par) & (per.method == meth)].copy()
    if len(df) != summary["stones"]:
        raise SystemExit(f"expected {summary['stones']} stones in the primary configuration, found {len(df)}")
    figure_r3(df, summary).to_csv(OUT / "fig_freiburg_r3_diameter_agreement_data.csv", index=False)
    figure_r2(summary).to_csv(OUT / "fig_freiburg_r2_volume_error_by_type_data.csv", index=False)
    print("Saved figures to", OUT, "from", RUN, KEY)


if __name__ == "__main__":
    main()

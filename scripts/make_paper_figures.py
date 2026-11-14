"""Publication figures for paper/main.tex (Stage C, 14 November 2026).

Every plotted value is read from a stored, hashed result artefact; nothing is recomputed on locked test data.
- T1 (locked test): results/runs/T1_20261113T081425Z/analysis/evaluation.json (aggregates only).
- External check: results/runs/mswal_external_check_20261113T132418Z/external_check.json.
- Development per-stone coverage: data/interim/e6_per_stone_e6_uncertainty_design_20261112T082354Z.csv (OOF; derived
  from the restricted Li data, so not distributed). When it is absent, Fig. 4's development markers are read from the
  public paper/figures/fig4_size_bands_data.csv (the exact plotted values, SHA-256 checked), so the figure is identical.
- Physical reference: paper/figures/fig_freiburg_r2_volume_error_by_type_data.csv and
  paper/figures/fig_freiburg_r3_diameter_agreement_data.csv (written by scripts/make_freiburg_figures.py).
Each figure is written as vector PDF plus a PNG preview and a CSV of exactly the plotted values.
"""

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
T1 = ROOT / "results/runs/T1_20261113T081425Z/analysis/evaluation.json"
EXT = ROOT / "results/runs/mswal_external_check_20261113T132418Z/external_check.json"
E6 = ROOT / "data/interim/e6_per_stone_e6_uncertainty_design_20261112T082354Z.csv"
FIG4_DATA = OUT / "fig4_size_bands_data.csv"
FIG4_DATA_SHA = "cede1fb83482a8d6571a10e9e9b5f3f553ce0dd70b6800940d0f6f3143b97b02"
HASHES = {T1: "d8cf65da449a8bf8c65034b4e730c857945b904857d78ad4cc42697547468ed9",
          EXT: "6b57e1f666ad8a187b7cdfafe41342921b6ed7de63d97be7e2e50a1d398d3966"}

# Okabe-Ito colour-blind-safe palette; every method also has its own marker for grayscale reading.
OURS, COMP, BASE, INK, INK2, RULE = "#0072B2", "#D55E00", "#7F7F7F", "#1A1A1A", "#4D4D4D", "#B3B3B3"
STYLE = {"U0": (BASE, "^"), "U1": (COMP, "s"), "U2": (OURS, "o")}
mpl.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5, "axes.labelsize": 7.5,
    "axes.titlesize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
    "ytick.major.size": 2.5, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "lines.linewidth": 0.9,
})  # fmt: skip
# Figs. 2-4 are printed at column or text width; these sizes keep their smallest text near 7 pt on the page.
READABLE = {"font.size": 8.4, "axes.labelsize": 8.4, "axes.titlesize": 8.4, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8}  # fmt: skip


def check(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != HASHES[path]:
        raise SystemExit(f"{path} does not match its recorded SHA-256")
    return json.loads(path.read_text())


def cp(k: int, n: int) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
    return float(lo), float(hi)


def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name: str, table: pd.DataFrame) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=200)
    table.to_csv(OUT / f"{name}_data.csv", index=False)
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Fig. 3: robustness of the burden result
def fig_robustness(t1: dict, ext: dict) -> None:
    rows = []

    def add(label, b, n_label, kind, nominal=0.90):
        a, c, r = b["P2-A decomposition"], b["P2-B-CQR"], b["width_ratio_A_over_BCQR"]
        rows.append({"analysis": label, "n": n_label, "kind": kind, "nominal": nominal, "ratio": r["median"], "ratio_lo": r["ci95"][0], "ratio_hi": r["ci95"][1],
                     "cov_ours": a["coverage"], "cov_ours_lo": a["cp95"][0], "cov_ours_hi": a["cp95"][1], "n_scans": a["n"],
                     "cov_cqr": c["coverage"], "cov_cqr_lo": c["cp95"][0], "cov_cqr_hi": c["cp95"][1]})  # fmt: skip

    add("Primary analysis", t1["burden"]["0.90|filtered=True"], "53", "locked test")
    for key, label in (("S1_screening_exclusions", "S1: screening exclusions removed"), ("S2_all_screened_flags", "S2: all screened flags removed"),
                       ("S3_cross_split_repeat_pairs", "S3: repeat pairs removed")):  # fmt: skip
        ev = t1["sensitivity"][key]["evaluation"]
        add(label, ev["burden"]["0.90|filtered=True"], str(ev["test_scans"]), "sensitivity")
    add("Without 30 mm filter", t1["burden"]["0.90|filtered=False"], "53", "ablation")
    # nominal 0.80 is descriptive only (protocol v3 S-5); stored in the same evaluation.json
    add("Nominal 0.80", t1["burden"]["0.80|filtered=True"], "53", "descriptive", 0.80)
    add("Nominal 0.80, without filter", t1["burden"]["0.80|filtered=False"], "53", "descriptive", 0.80)
    add("External (MSWAL)", ext["burden_0.90"], "58", "external")
    df = pd.DataFrame(rows)
    y = np.arange(len(df))[::-1].astype(float)
    y[-1] -= 0.35  # visual gap before the external row
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.02, 1.95), gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    for ax in (a1, a2):
        clean(ax)
        ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
        for yy in y:
            ax.axhline(yy, color="#EEEEEE", lw=0.5, zorder=0)
    a1.axvline(1.0, color=INK2, lw=0.7, ls=(0, (3, 2)))
    ext = (df.kind == "external").to_numpy()
    for sel, light in ((~ext, False), (ext, True)):  # the external row is descriptive: lighter, open marker
        d = df[sel]
        a1.errorbar(d.ratio, y[sel], xerr=[d.ratio - d.ratio_lo, d.ratio_hi - d.ratio], fmt="o", color=OURS, ms=4, capsize=1.8, elinewidth=0.9,
                    mfc="white" if light else OURS, alpha=0.5 if light else 1.0)  # fmt: skip
    a1.set_xscale("log")
    a1.set_xticks([0.4, 0.6, 0.8, 1.0, 1.4])
    a1.get_xaxis().set_major_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
    a1.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    a1.set_xlim(0.34, 1.6)
    a1.set_xlabel("Paired median width ratio, ours / P2-B-CQR (95% CI)")
    a1.set_yticks(y)
    a1.set_yticklabels([f"{a}, descriptive ($n$={n})" if k == "external" else f"{a} ($n$={n})"
                        for a, n, k in zip(df.analysis, df.n, df.kind, strict=True)])  # fmt: skip
    for lab, k in zip(a1.get_yticklabels(), df.kind, strict=True):
        if k == "external":
            lab.set_color("#8C8C8C")
            lab.set_style("italic")
    a1.tick_params(axis="y", length=0)
    a1.text(0.345, y.max() + 0.55, "narrower", color=INK2, fontsize=7.4, va="center", ha="left", style="italic")
    a1.text(1.58, y.max() + 0.55, "wider", color=INK2, fontsize=7.4, va="center", ha="right", style="italic")
    a1.set_title("(a) Interval width relative to the strongest comparator", loc="left", pad=9)
    for yy, nom in zip(y, df.nominal, strict=True):  # each row's nominal level as a short dashed tick
        a2.plot([nom, nom], [yy - 0.42, yy + 0.42], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    off = 0.14
    for sel, light in ((~ext, False), (ext, True)):
        d, yy = df[sel], y[sel]
        a2.errorbar(d.cov_ours, yy + off, xerr=[d.cov_ours - d.cov_ours_lo, d.cov_ours_hi - d.cov_ours], fmt="o", color=OURS, ms=4, capsize=1.8,
                    elinewidth=0.9, mfc="white" if light else OURS, alpha=0.5 if light else 1.0, label=None if light else "Detection-aware")  # fmt: skip
        a2.errorbar(d.cov_cqr, yy - off, xerr=[d.cov_cqr - d.cov_cqr_lo, d.cov_cqr_hi - d.cov_cqr], fmt="s", color=COMP, mfc="white", ms=3.6,
                    capsize=1.8, elinewidth=0.9, alpha=0.5 if light else 1.0, label=None if light else "P2-B-CQR")  # fmt: skip
    a2.set_yticks(y)
    a2.set_yticklabels([])
    a2.tick_params(axis="y", length=0)
    a2.set_xlim(0.65, 1.005)
    a2.set_xlabel("Coverage (Clopper–Pearson 95% CI); dashed: nominal level")
    a2.set_title("(b) Coverage of the patient-burden interval", loc="left", pad=9)
    a2.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.02))
    save(fig, "fig3_burden_robustness", df)


# ------------------------------------------------------------------ Fig. 4: per-stone coverage by size band
def dev_band_source():
    """Development (OOF) per-band statistics: (k, n, cp_low, cp_high) for (target, method, band), plus stones per band.

    Preferred source: the restricted E6 per-stone table (counts recomputed). Fallback when it is absent: the public
    plotted-values file, verified against its recorded SHA-256 before use."""
    if E6.exists():
        e6 = pd.read_csv(E6)

        def stats(target, m, b):
            d = e6[e6.pred_stratum == b][f"covered|{target}|0.90|{m}"]
            k, n = int(d.sum()), len(d)
            return (k, n, *cp(k, n))

        return stats, lambda b: int((e6.pred_stratum == b).sum()), "restricted E6 table"
    if hashlib.sha256(FIG4_DATA.read_bytes()).hexdigest() != FIG4_DATA_SHA:
        raise SystemExit(f"{FIG4_DATA} does not match its recorded SHA-256")
    pub = pd.read_csv(FIG4_DATA)
    dev = pub[pub.data == "development (OOF)"].set_index(["target", "method", "band"])

    def stats(target, m, b):
        r = dev.loc[(target, m, b)]
        return (int(r.k), int(r.n), float(r.cp_low), float(r.cp_high))

    return stats, lambda b: int(dev.loc[("volume_mm3", "U2", b)].n), "public fig4_size_bands_data.csv"


def fig_bands(t1: dict) -> None:
    dev_stats, dev_n, source = dev_band_source()
    print("Fig. 4 development markers from:", source)
    bands, labels = ["<5", "5-<10", ">=10"], ["<5 mm", "5–10 mm", "≥10 mm"]
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (target, title) in zip(axes, (("volume_mm3", "(a) Volume"), ("max_diameter_3d_mm", "(b) 3D maximum diameter")), strict=True):
        clean(ax)
        ax.axhline(0.90, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=0)
        for j, m in enumerate(("U0", "U1", "U2")):
            colour, marker = STYLE[m]
            for i, b in enumerate(bands):
                x = i + (j - 1) * 0.24
                k, n, lo, hi = dev_stats(target, m, b)
                ax.errorbar(x - 0.055, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt=marker, mfc="white", color=colour, ms=3.0, capsize=1.2,
                            elinewidth=0.6, alpha=0.75)  # fmt: skip
                rows.append({"target": target, "method": m, "band": b, "data": "development (OOF)", "k": k, "n": n, "coverage": k / n, "cp_low": lo, "cp_high": hi})
                s = t1["per_stone"][f"{target}|0.90"][m]["by_band"][b]
                ax.errorbar(x + 0.055, s["coverage"], yerr=[[s["coverage"] - s["cp95"][0]], [s["cp95"][1] - s["coverage"]]], fmt=marker, color=colour,
                            ms=3.4, capsize=1.2, elinewidth=0.9)  # fmt: skip
                rows.append({"target": target, "method": m, "band": b, "data": "locked test", "k": round(s["coverage"] * s["n"]), "n": s["n"],
                             "coverage": s["coverage"], "cp_low": s["cp95"][0], "cp_high": s["cp95"][1]})  # fmt: skip
        ax.set_xticks(range(3))
        n_test = [t1["per_stone"][f"{target}|0.90"]["U2"]["by_band"][b]["n"] for b in bands]
        n_dev = [dev_n(b) for b in bands]
        ax.set_xticklabels([f"{lab}\n{nt} / {nd}" for lab, nt, nd in zip(labels, n_test, n_dev, strict=True)])
        ax.set_title(title, loc="left", pad=3)
        ax.tick_params(axis="x", labelsize=7.4)  # three band labels per panel must not touch
        ax.set_xlim(-0.5, 2.5)
    axes[0].set_ylim(0.40, 1.02)
    axes[0].set_ylabel("Per-stone coverage")
    axes[0].text(-0.62, 0.345, "test / dev. $n$", fontsize=7.2, color=INK2, ha="right", va="top", transform=axes[0].transData)
    handles = [mpl.lines.Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls="", ms=3.6, label=lab)
               for m, lab in (("U0", "U0 Bland–Altman"), ("U1", "U1 global conformal"), ("U2", "U2 size-stratified"))]  # fmt: skip
    handles += [mpl.lines.Line2D([], [], color=INK2, marker="o", ls="", ms=3.4, label="locked test"),
                mpl.lines.Line2D([], [], color=INK2, marker="o", mfc="white", ls="", ms=3.0, label="development (OOF)")]  # fmt: skip
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.17), columnspacing=0.9, handletextpad=0.25)
    save(fig, "fig4_size_bands", pd.DataFrame(rows))


# ------------------------------------------------------------------ Fig. 2: physical reference
def fig_phantom() -> None:
    r2 = pd.read_csv(OUT / "fig_freiburg_r2_volume_error_by_type_data.csv")
    r3 = pd.read_csv(OUT / "fig_freiburg_r3_diameter_agreement_data.csv")
    names = {"uric acid": "Uric acid", "not recorded": "Not recorded", "other": "Other types", "cystine": "Cystine",
             "calciumoxalate-dihydrate": "Ca oxalate dihydrate", "brushite": "Brushite"}  # fmt: skip
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.16, 3.05), gridspec_kw={"height_ratios": [1, 1.05], "hspace": 0.62})
    clean(a1)
    clean(a2)
    r2 = r2.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(r2))
    a1.errorbar(100 * r2["median"], y, xerr=[100 * (r2["median"] - r2.ci_low), 100 * (r2.ci_high - r2["median"])], fmt="o", color=OURS, ms=3.4,
                capsize=1.4, elinewidth=0.8)  # fmt: skip
    a1.axvline(23.6, color=INK2, lw=0.7, ls=(0, (3, 2)))
    a1.set_yticks(y)
    a1.set_yticklabels([f"{names[t]} ({n})" for t, n in zip(r2.type, r2.n, strict=True)])
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, 175)
    a1.set_xlabel("Median absolute volume error (%)")
    a1.set_title("(a) Volume error by composition (ICC 0.993)", loc="left", pad=3)
    for panel, colour, marker, mfc in (("3D maximum diameter", OURS, "o", OURS), ("2D axial maximum diameter", COMP, "^", "white")):
        d = r3[r3.panel == panel]
        a2.scatter(d.caliper_mm, d.error_mm, s=7, marker=marker, facecolors=mfc, edgecolors=colour, linewidths=0.6,
                   label="3D maximum" if panel.startswith("3D") else "2D axial maximum")  # fmt: skip
        a2.axhline(d.error_mm.mean(), color=colour, lw=0.8, ls="-" if panel.startswith("3D") else (0, (3, 2)))
    a2.axhline(0, color=RULE, lw=0.5)
    a2.set_xlabel("Caliper diameter (mm)")
    a2.set_ylabel("Error vs. caliper (mm)", labelpad=1)
    a2.set_title("(b) Diameter error against caliper", loc="left", pad=3)
    b3 = r3[r3.panel == "3D maximum diameter"].error_mm.mean()
    b2 = r3[r3.panel == "2D axial maximum diameter"].error_mm.mean()
    handles = [mpl.lines.Line2D([], [], color=OURS, marker="o", ms=3.2, lw=0.8, label=f"3D maximum (bias {b3:+.2f} mm)"),
               mpl.lines.Line2D([], [], color=COMP, marker="^", mfc="white", ms=3.4, lw=0.8, ls=(0, (3, 2)),
                                label=f"2D axial (bias {b2:+.2f} mm)".replace("-", "\u2212"))]  # fmt: skip
    a2.set_ylim(-17.5, 2.2)
    a2.legend(handles=handles, loc="lower left", frameon=False, handletextpad=0.3, borderaxespad=0.0, handlelength=1.6, fontsize=7.8)
    out = pd.concat([r2.assign(panel="a: volume error by type"), r3.assign(panel_b=r3.panel)], ignore_index=True)
    save(fig, "fig2_phantom", out)


# ------------------------------------------------------------------ Fig. 1: overview schematic
def box(ax, x, y, w, h, text, fc="#F4F6F8", ec=RULE, fs=7, weight="normal", color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", fc=fc, ec=ec, lw=0.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, color=color, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.7, color=color, shrinkA=0, shrinkB=0))


def fig_overview() -> None:
    fig = plt.figure(figsize=(7.16, 3.05))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    blue_fc, orange_fc = "#E6F0F7", "#FBEDE5"
    # (a) pipeline
    ax.text(0.005, 0.975, "(a) Measurement pipeline", fontsize=8, weight="bold", va="center", color=INK)
    steps = ["Non-contrast\nCT", "Kidney region\nTotalSegmentator\n40 mm margin", "Stone segmentation\nnnU-Net\n8-flip TTA",
             "Pieces\n26-connected\n≥3 voxels", "Kidney-distance\nfilter, 30 mm", "Per-piece measures\n$\\hat V_j$, $\\hat D_j$\ndetection features"]  # fmt: skip
    w, gap, x0, yb, h = 0.148, 0.0185, 0.005, 0.795, 0.155
    for i, st in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, yb, w, h, st, fs=7.4)
        if i:
            arrow(ax, x - gap + 0.002, yb + h / 2, x - 0.002, yb + h / 2)
    xm = x0 + 5 * (w + gap) + w / 2
    ax.plot([xm, xm], [yb, 0.765], color=INK2, lw=0.7)
    ax.plot([0.20, xm], [0.765, 0.765], color=INK2, lw=0.7)
    arrow(ax, 0.20, 0.765, 0.20, 0.692)
    arrow(ax, 0.38, 0.765, 0.38, 0.718)
    # (b) uncertainty layer
    ax.text(0.005, 0.735, "(b) Uncertainty layer", fontsize=8, weight="bold", va="center", color=INK)
    box(ax, 0.005, 0.315, 0.25, 0.375,
        "Per-stone interval (U2)\n\n$r=|\\log(V/\\hat V)|$ on matched stones\nsize band $k(\\hat D)$: <5, 5–10, ≥10 mm\n"
        "$q_k$: conformal quantile within band $k$\n$[\\hat V e^{-q_k},\\ \\hat V e^{q_k}]$", fc=blue_fc, ec=OURS, fs=7.3)  # fmt: skip
    ax.text(0.29, 0.700, "Detection-aware burden interval", fontsize=7.6, weight="bold", color=INK, va="center")
    comps = [("False detection: keep piece $j$\nwith probability $1-\\pi_j$", 0.545), ("Measurement: $\\hat V_j e^{\\varepsilon_j}$, with $\\varepsilon_j$\nfrom the same size band", 0.43),
             ("Missed burden: $M$ from the\nsame slice-thickness group", 0.315)]  # fmt: skip
    for t, yy in comps:
        box(ax, 0.29, yy, 0.19, 0.10, t, fc=orange_fc, ec=COMP, fs=7.0)
        arrow(ax, 0.482, yy + 0.05, 0.512, 0.48)
    box(ax, 0.512, 0.39, 0.19, 0.18, "Monte Carlo, 2,000 draws\n$\\tilde B=\\Sigma_j\\,\\mathbb{1}[u_j\\geq\\pi_j]\\,\\hat V_j e^{\\varepsilon_j}+M$\nraw interval $[L,U]$",
        fc=orange_fc, ec=COMP, fs=7.2)  # fmt: skip
    arrow(ax, 0.704, 0.48, 0.735, 0.48)
    box(ax, 0.735, 0.39, 0.26, 0.18, "One conformal factor $t$ (calibration scans)\n$[(L+c)e^{-t}-c,\\ (U+c)e^{t}-c]$\npatient-burden interval",
        fc=blue_fc, ec=OURS, fs=7.2)  # fmt: skip
    # (c) data roles
    ax.text(0.005, 0.245, "(c) Data roles", fontsize=8, weight="bold", va="center", color=INK)
    ax.text(0.535, 0.245, "calibration quantities hashed before the test scans are loaded", fontsize=6.9, color=INK2, ha="center", va="center", style="italic")
    roles = [("Development, 165 scans (Li 2022)\nout-of-fold fit: $\\pi$ model, residual and missed pools, CQR", 0.005, 0.34, "#EEF1F4", RULE),
             ("Calibration, 35 scans\n$q_k$, $t$, comparator quantiles", 0.355, 0.17, blue_fc, OURS),
             ("Locked test, 53 scans\nevaluated once, frozen protocol", 0.535, 0.20, blue_fc, OURS),
             ("External, MSWAL, 58 of 77 scans\nno refitting, descriptive", 0.745, 0.25, orange_fc, COMP)]  # fmt: skip
    for t, x, w2, fc, ec in roles:
        box(ax, x, 0.03, w2, 0.165, t, fc=fc, ec=ec, fs=7.1)
    for xa, xb in ((0.345, 0.355), (0.525, 0.535), (0.735, 0.745)):
        arrow(ax, xa - 0.006, 0.112, xb, 0.112)
    save(fig, "fig1_overview", pd.DataFrame({"note": ["schematic; numbers from protocol v3 and the T1/MSWAL run records"]}))


def main() -> None:
    t1, ext = check(T1), check(EXT)
    fig_overview()
    with mpl.rc_context(READABLE):
        fig_phantom()
        fig_robustness(t1, ext)
        fig_bands(t1)


if __name__ == "__main__":
    main()

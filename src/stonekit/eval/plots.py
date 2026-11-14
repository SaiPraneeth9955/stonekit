"""All figures, colour-blind-safe, vector PDF plus PNG (Section 12)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from stonekit.eval.agreement import bland_altman

MARK = "#1f6fb2"
INK = "#1f2933"
MUTED = "#616e7c"
GRID = "#e4e7eb"


def bland_altman_figure(
    measured: np.ndarray, reference: np.ndarray, title: str, unit: str, stem: Path
) -> None:
    """Save `stem`.pdf and `stem`.png. Differences are measured minus reference."""
    stats = bland_altman(measured, reference)
    mean = (np.asarray(measured) + np.asarray(reference)) / 2
    difference = np.asarray(measured) - np.asarray(reference)

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(mean, difference, s=36, color=MARK, alpha=0.8, edgecolor="white", linewidth=0.6)
    lines = [
        (stats.bias, "-", f"Bias {stats.bias:+.3f} {unit}"),
        (stats.upper_limit, "--", f"+1.96 SD {stats.upper_limit:+.3f} {unit}"),
        (stats.lower_limit, "--", f"-1.96 SD {stats.lower_limit:+.3f} {unit}"),
    ]
    for value, style, label in lines:
        ax.axhline(value, color=INK, linestyle=style, linewidth=1.2)
        ax.annotate(
            label,
            xy=(1, value),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(f"Mean of measured and reference ({unit})", color=MUTED)
    ax.set_ylabel(f"Measured minus reference ({unit})", color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    fig.subplots_adjust(left=0.1, right=0.76, top=0.9, bottom=0.13)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    plt.close(fig)

#!/usr/bin/env python3
"""Plot Figure 1 from the saved affinity and neighborhood-purity results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_heatmap(
    ax,
    affinity: np.ndarray,
    order: np.ndarray,
    title: str,
    *,
    colorbar: bool = False,
    colorbar_ax=None,
) -> None:
    ordered_affinity = affinity[np.ix_(order, order)]
    sns.heatmap(
        ordered_affinity,
        ax=ax,
        cmap="flare",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar=colorbar,
        cbar_ax=colorbar_ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.set_title(title, pad=5, fontsize=16)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure1"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    purity = pd.read_csv(args.results_dir / "fig1_neighbor_purity_raw.csv")
    with np.load(args.results_dir / "figure1_affinities.npz") as saved:
        euclidean = saved["euclidean"]
        umap_affinity = saved["umap"]
        geoforest = saved["geoforest"]
        order = saved["order"]

    sns.set_theme(style="white", context="talk")
    figure = plt.figure(figsize=(13, 8))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 0.95],
        hspace=0.18,
        wspace=0.22,
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[1, :]),
    ]

    for ax, letter in zip(axes[:3], ["A", "B", "C"]):
        ax.text(
            -0.08,
            1.05,
            letter,
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
            va="bottom",
        )
    axes[3].text(
        -0.02,
        1.03,
        "D",
        transform=axes[3].transAxes,
        fontsize=16,
        fontweight="bold",
        va="bottom",
    )

    colorbar_ax = figure.add_axes([0.92, 0.56, 0.015, 0.30])
    plot_heatmap(axes[0], euclidean, order, "Euclidean kNN")
    plot_heatmap(axes[1], umap_affinity, order, "UMAP graph")
    plot_heatmap(
        axes[2],
        geoforest,
        order,
        "Forest proximity",
        colorbar=True,
        colorbar_ax=colorbar_ax,
    )

    np.random.seed(0)
    sns.stripplot(
        data=purity,
        x="method",
        y="purity",
        order=["Euclidean kNN", "UMAP graph", "GeoForest kNN"],
        ax=axes[3],
        jitter=0.25,
        alpha=0.4,
        size=3,
    )
    axes[3].set_ylabel("Neighbor purity @ k=15", fontsize=15)
    axes[3].set_xlabel("")
    axes[3].set_ylim(0, 1.05)
    axes[3].set_xticklabels(
        ["Euclidean kNN", "UMAP graph", "Forest proximity"]
    )

    figure.tight_layout(rect=[0, 0, 0.90, 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(
            args.output_dir / f"figure1.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

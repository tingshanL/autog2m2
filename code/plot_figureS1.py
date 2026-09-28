#!/usr/bin/env python3
"""Plot Supplementary Figure S1 from the Figure 2 data generators."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from run_figure2 import Config, make_dataset


DATASETS = (
    ("gaussian_subspace", "Gaussian subspace"),
    ("heavy_tail_subspace", "Heavy-tailed subspace"),
    ("swiss_roll", "Swiss roll"),
)
DIMENSIONS = (16, 256)
COLORS = ("#0072B2", "#D55E00", "#009E73")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Config()
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.2))

    for row, (dataset, title) in enumerate(DATASETS):
        for column, dimension in enumerate(DIMENSIONS):
            X, labels = make_dataset(dataset, dimension, args.seed, config)
            coordinates = PCA(n_components=2, svd_solver="full").fit_transform(X)
            ax = axes[row, column]

            for cluster, color in enumerate(COLORS):
                selected = labels == cluster
                ax.scatter(
                    coordinates[selected, 0],
                    coordinates[selected, 1],
                    s=9,
                    alpha=0.62,
                    color=color,
                    linewidths=0,
                    label=f"Cluster {cluster + 1}",
                )

            ax.set_title(f"{title}, $d={dimension}$", fontsize=11)
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            args.output_dir / f"figureS1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

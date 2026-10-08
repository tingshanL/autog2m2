#!/usr/bin/env python3
"""Plot Figure 4 head ablations from the result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHODS = {
    "ARI_Euclid_Ward": "Euclid + Ward",
    "ARI_RBF_KPCA_Ward": "RBF-KPCA–Ward",
    "ARI_GF_Agglo_avg": "Forest average-linkage",
    "ARI_GF_Agglo_comp": "Forest complete-linkage",
    "ARI_Geo_MDS_Ward_nowhite": "Forest-MDS–Ward (unwhitened)",
    "ARI_Geo_MDS_Ward_white": "Forest-MDS–Ward (whitened)",
}
DATASETS = {
    "subspace_gaussian": "Gaussian subspace",
    "heavy_tail": "Heavy-tailed subspace",
    "swiss_roll": "Swiss roll",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/figure4/ablation1_geo_heads.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    long = frame.melt(
        id_vars=["dataset", "seed"],
        value_vars=list(METHODS),
        var_name="method",
        value_name="ari",
    )
    long["Method"] = long["method"].map(METHODS)
    order = list(METHODS.values())
    palette = ["#1BA1DE", "#3E35BF", "#E7C25B", "#E6A902", "#CBA12E", "#86660F"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        panel = long.loc[long["dataset"].eq(dataset)]
        sns.stripplot(data=panel, x="Method", y="ari", order=order, hue="Method", palette=palette,
                      jitter=0.18, alpha=0.72, size=4, legend=False, ax=ax)
        medians = panel.groupby("Method")["ari"].median().reindex(order)
        # ax.scatter(range(len(order)), medians, marker="D", s=35, color="black", zorder=5)
        ax.scatter(
            range(len(order)),
            medians,
            marker="X",
            s=110,
            facecolors="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Adjusted Rand index")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    # Create legend handles using the same colors as the plotted dots
    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
            label=method,
        )
        for method, color in zip(order, palette)
    ]

    # Shared legend below all three subplots
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=13,
        columnspacing=1.5,
        handletextpad=0.6,
    )

    # Reserve space at the bottom for the legend
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure4.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

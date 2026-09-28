#!/usr/bin/env python3
"""Plot both Figure 6 panels from their result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LEFT_METHODS = (
    "autogmm_euclidean_ward",
    "autogmm_mahalanobis_ward",
    "autogmm_full",
    "mclust",
)
LEFT_LABELS = {
    "autogmm_euclidean_ward": "AutoGMM\n(Euc-Ward only)",
    "autogmm_mahalanobis_ward": "AutoGMM\n(Mah-Ward only)",
    "autogmm_full": "AutoGMM\n(Full)",
    "mclust": "mclust",
}
RIGHT_METHODS = ("autogmm_full", "mclust")
RIGHT_LABELS = {"autogmm_full": "AutoGMM\n(Full)", "mclust": "mclust"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure6"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    initialization = pd.read_csv(
        args.results_dir / "figure6_anisotropic_initialization_results.csv"
    )
    initialization = initialization.loc[initialization["status"].eq("ok")].copy()
    initialization["Method"] = initialization["method"].map(LEFT_LABELS)
    order = [LEFT_LABELS[method] for method in LEFT_METHODS]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=initialization,
        x="Method",
        y="ari",
        order=order,
        hue="Method",
        hue_order=order,
        palette=["#FA8072", "#B22222", "#8B0000", "#808000"],
        jitter=0.25,
        size=4,
        alpha=0.85,
        linewidth=0.3,
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("ARI", fontsize=20)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_yticks([0, 0.5, 1])
    sns.despine(ax=ax)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"aniso_plus.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    dimensions = pd.read_csv(args.results_dir / "figure6_anisotropic_dimension_results.csv")
    dimensions = dimensions.loc[dimensions["status"].eq("ok")]
    summary = (
        dimensions.groupby(["dimension", "method"], as_index=False)["ari"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["method", "dimension"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"autogmm_full": "#8B0000", "mclust": "#808000"}
    for method in RIGHT_METHODS:
        values = summary.loc[summary["method"].eq(method)]
        ax.plot(
            values["dimension"],
            values["median"],
            color=colors[method],
            marker="o",
            linewidth=2,
            label=RIGHT_LABELS[method],
        )
        ax.fill_between(
            values["dimension"],
            values["q25"],
            values["q75"],
            color=colors[method],
            alpha=0.18,
        )
    ax.set_ylabel("ARI", fontsize=20)
    ax.set_xlabel("Dimension", fontsize=20)
    ax.set_xticks(sorted(dimensions["dimension"].unique()))
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=16)
    sns.despine(ax=ax)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"high_dim.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

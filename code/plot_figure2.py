#!/usr/bin/env python3
"""Plot Figure 2 from the fixed-K seed-level results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    "autog2m2",
    "autogmm_spectral",
    "mclust",
    "pca",
    "umap",
    "ae_tanh",
)
LABELS = {
    "autog2m2": "AutoG2M2",
    "autogmm_spectral": "AutoGMM-Spectral",
    "mclust": "mclust",
    "pca": "PCA+AutoGMM",
    "umap": "UMAP+AutoGMM",
    "ae_tanh": "tanh AE+AutoGMM",
}
COLORS = {
    "autog2m2": "#F7BA14",
    "autogmm_spectral": "#3E35BF",
    "mclust": "#808000",
    "pca": "#56B4E9",
    "umap": "#841386",
    "ae_tanh": "#009E73",
}
MARKERS = {
    "autog2m2": "o",
    "autogmm_spectral": "s",
    "mclust": "D",
    "pca": "^",
    "umap": "v",
    "ae_tanh": "P",
}
DATASETS = (
    ("gaussian_subspace", "Gaussian subspace"),
    ("heavy_tail_subspace", "Heavy-tailed subspace"),
    ("swiss_roll", "Swiss roll"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/figure2/figure2_fixed_k_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame.loc[
        frame["method"].isin(METHODS)
        & frame["status"].eq("ok")
        & frame["ari"].notna()
    ].copy()
    counts = usable.groupby(["dataset", "dimension", "method"])["seed"].nunique()
    if counts.empty or not counts.eq(20).all():
        bad = counts.loc[~counts.eq(20)]
        raise ValueError(f"Expected 20 seeds per dataset/dimension/method; found:\n{bad}")
    return (
        usable.groupby(["dataset", "dimension", "method"], as_index=False)["ari"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .sort_values(["dataset", "dimension", "method"])
    )


def main() -> int:
    args = parse_args()
    results = pd.read_csv(args.results)
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS):
        panel = summary.loc[summary["dataset"].eq(dataset)]
        for method in METHODS:
            values = panel.loc[panel["method"].eq(method)].sort_values("dimension")
            x = values["dimension"].to_numpy(dtype=float)
            med = values["median"].to_numpy(dtype=float)
            q25 = values["q25"].to_numpy(dtype=float)
            q75 = values["q75"].to_numpy(dtype=float)
            ax.plot(
                x,
                med,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.5,
                linewidth=1.8,
                label=LABELS[method],
            )
            ax.fill_between(x, q25, q75, color=COLORS[method], alpha=0.12)
        dims = sorted(panel["dimension"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(dims)
        ax.set_xticklabels([str(int(value)) for value in dims], rotation=45)
        ax.set_title(title)
        ax.set_xlabel("Ambient dimension")
        ax.grid(False)
        ax.set_ylim(-0.03, 1.03)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Adjusted Rand index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure2.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot the automatic-K Figure 2 experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        default=Path("results/figure2/figure2_auto_k_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["method"].isin(METHODS)].copy()

    attempted = selected.groupby(["dataset", "dimension", "method"])["seed"].nunique()
    if attempted.empty or not attempted.eq(20).all():
        raise ValueError("Every reported dataset/dimension/method group must contain 20 attempted seeds")

    usable = selected.loc[selected["status"].eq("ok") & selected["ari"].notna()].copy()
    if usable.empty:
        raise ValueError("No successful fits are available to plot")

    return (
        usable.groupby(["dataset", "dimension", "method"], as_index=False)["ari"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["dataset", "dimension", "method"])
    )


def main() -> int:
    args = parse_args()
    results = pd.read_csv(args.results)
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.3), sharey=True)
    legend = {}
    for ax, (dataset, title) in zip(axes, DATASETS):
        panel = summary.loc[summary["dataset"].eq(dataset)]
        for method in METHODS:
            values = panel.loc[panel["method"].eq(method)].sort_values("dimension")
            if values.empty:
                continue
            line = ax.plot(
                values["dimension"],
                values["median"],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=4.5,
                linewidth=1.8,
                label=LABELS[method],
            )[0]
            ax.fill_between(
                values["dimension"],
                values["q25"],
                values["q75"],
                color=COLORS[method],
                alpha=0.12,
            )
            legend[method] = line
        dimensions = sorted(panel["dimension"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(dimensions)
        ax.set_xticklabels([str(int(value)) for value in dimensions], rotation=45)
        ax.set_title(title)
        ax.set_xlabel("Ambient dimension")
        ax.set_ylim(-0.03, 1.03)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Adjusted Rand index")
    ordered = [method for method in METHODS if method in legend]
    fig.legend(
        [legend[method] for method in ordered],
        [LABELS[method] for method in ordered],
        loc="lower center",
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure2_auto_k.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot TCGA Figure 7 from the observed-gene fixed-K=5 results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DIMENSIONS = (256, 512, 1024, 2048, 4096)
SEEDS = tuple(range(20))
METHODS = ("autog2m2", "pca", "umap")

METHOD_LABELS = {
    "autog2m2": "AutoG2M2",
    "pca": "PCA + AutoGMM",
    "umap": "UMAP + AutoGMM",
}
COLORS = {
    "autog2m2": "#F7BA14",
    "pca": "#56B4E9",
    "umap": "#841386",
}
MARKERS = {"autog2m2": "o", "pca": "s", "umap": "^"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paired-results",
        type=Path,
        default=Path("results/figure7/figure7_fixed_k5_results.csv"),
        help="Seed-level fixed-K=5 result table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
    )
    return parser.parse_args()


def read_observed_fixed_k(path: Path) -> pd.DataFrame:
    """Read and validate the complete observed-gene, fixed-K=5 grid."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    source = pd.read_csv(path)
    required = {
        "data_regime",
        "dimension",
        "seed",
        "method",
        "status",
        "fixed_k",
        "fixed_k_ari",
        "fixed_k_selected_components",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = source.loc[source["data_regime"].eq("observed_genes")].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no observed_genes rows")
    if not frame["status"].eq("ok").all():
        bad = frame.loc[~frame["status"].eq("ok")]
        raise ValueError(f"Observed-gene results contain {len(bad)} non-ok rows")
    if not frame["fixed_k"].astype(int).eq(5).all():
        raise ValueError("Not every observed-gene row records fixed_k=5")
    if not frame["fixed_k_selected_components"].astype(int).eq(5).all():
        raise ValueError("Not every fixed-K fit selected five components")
    if not np.isfinite(frame["fixed_k_ari"].to_numpy(dtype=float)).all():
        raise ValueError("fixed_k_ari contains non-finite values")

    keys = ["dimension", "seed", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Observed-gene results contain duplicate condition rows")
    expected = {
        (dimension, seed, method)
        for dimension in DIMENSIONS
        for seed in SEEDS
        for method in METHODS
    }
    observed = {
        (int(dimension), int(seed), str(method))
        for dimension, seed, method in frame[keys].itertuples(index=False, name=None)
    }
    if observed != expected:
        raise ValueError(
            "Incomplete observed-gene grid; "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )

    frame["dimension"] = frame["dimension"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame["ari"] = frame["fixed_k_ari"].astype(float)
    return frame.sort_values(keys).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Return median and interquartile-range summaries for plotting."""
    summary = (
        frame.groupby(["dimension", "method"], as_index=False)["ari"]
        .agg(
            n_fits="size",
            ari_median="median",
            ari_q25=lambda values: values.quantile(0.25),
            ari_q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["dimension", "method"])
    )
    summary["ari_iqr"] = summary["ari_q75"] - summary["ari_q25"]
    return summary


def draw_figure(summary: pd.DataFrame) -> plt.Figure:
    """Draw the single observed-gene, fixed-K=5 paper panel."""
    fig, ax = plt.subplots(figsize=(5.4, 3.7))
    for method in METHODS:
        values = summary.loc[summary["method"].eq(method)].sort_values("dimension")
        x = values["dimension"].to_numpy(dtype=float)
        median = values["ari_median"].to_numpy(dtype=float)
        q25 = values["ari_q25"].to_numpy(dtype=float)
        q75 = values["ari_q75"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            q25,
            q75,
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            median,
            color=COLORS[method],
            marker=MARKERS[method],
            markersize=5,
            linewidth=1.8,
            label=METHOD_LABELS[method],
            zorder=2,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMENSIONS, [str(value) for value in DIMENSIONS])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Number of top-variance genes (d)")
    ax.set_ylabel("Adjusted Rand index")
    # ax.set_title("Observed TCGA genes")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    return fig


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    frame = read_observed_fixed_k(args.paired_results)
    summary = summarize(frame)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            # "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    figure = draw_figure(summary)
    save_figure(figure, output_dir / "figure7_main_real_gene_budget")
    print(f"Figure written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

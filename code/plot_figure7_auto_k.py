#!/usr/bin/env python3
"""Plot the automatic-K TCGA sensitivity analysis for Figure 7."""

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
MARKERS = {
    "autog2m2": "o",
    "pca": "s",
    "umap": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/figure7/figure7_fixed_k5_results.csv"),
        help="Seed-level Figure 7 result table",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
    )
    return parser.parse_args()


def read_results(path: Path) -> pd.DataFrame:
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
        "automatic_ari",
        "automatic_selected_components",
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    frame = source.loc[
        source["data_regime"].eq("observed_genes")
        & source["method"].isin(METHODS)
    ].copy()
    if frame.empty:
        raise ValueError(f"{path} contains no observed_genes rows")
    if not frame["status"].eq("ok").all():
        raise ValueError("Observed-gene results contain unsuccessful fits")

    frame["dimension"] = frame["dimension"].astype(int)
    frame["seed"] = frame["seed"].astype(int)
    frame["ari"] = pd.to_numeric(frame["automatic_ari"], errors="coerce")
    frame["selected_k"] = pd.to_numeric(
        frame["automatic_selected_components"], errors="coerce"
    )
    if not np.isfinite(frame["ari"]).all():
        raise ValueError("automatic_ari contains non-finite values")
    if not frame["selected_k"].between(1, 10).all():
        raise ValueError("automatic_selected_components must be between 1 and 10")

    keys = ["dimension", "seed", "method"]
    if frame.duplicated(keys).any():
        raise ValueError("Observed-gene results contain duplicate condition rows")

    expected = {
        (dimension, seed, method)
        for dimension in DIMENSIONS
        for seed in SEEDS
        for method in METHODS
    }
    observed = set(frame[keys].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError(
            "Incomplete observed-gene grid; "
            f"missing={sorted(expected - observed)[:5]}, "
            f"extra={sorted(observed - expected)[:5]}"
        )

    return frame.sort_values(keys).reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
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
    frame = read_results(args.results)
    summary = summarize(frame)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
        }
    )
    figure = draw_figure(summary)
    save_figure(figure, output_dir / "figure7_auto_k")
    print(f"Figure written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

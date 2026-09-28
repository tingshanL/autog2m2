#!/usr/bin/env python3
"""Plot the matched Figure 9 runtime experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


METHODS = ("autog2m2", "umap", "pca")
LABELS = {
    "autog2m2": "AutoG2M2",
    "umap": "UMAP+AutoGMM",
    "pca": "PCA+AutoGMM",
}
COLORS = {
    "autog2m2": "#F7BA14",
    "umap": "#841386",
    "pca": "#56B4E9",
}
MARKERS = {
    "autog2m2": "o",
    "umap": "v",
    "pca": "^",
}
EXPECTED_SEEDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/figure9/figure9_runtime_results.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["method"].isin(METHODS)].copy()
    groups = ["method", "n_samples", "dimension"]

    attempted = selected.groupby(groups)["seed"].nunique()
    if attempted.empty or not attempted.eq(EXPECTED_SEEDS).all():
        raise ValueError("Every reported method/sample-size group must contain five attempted seeds")

    usable = selected.loc[
        selected["status"].eq("ok") & selected["total_seconds"].notna()
    ].copy()
    summary = (
        usable.groupby(groups, as_index=False)
        .agg(
            n_fits=("seed", "nunique"),
            median=("total_seconds", "median"),
            q25=("total_seconds", lambda values: values.quantile(0.25)),
            q75=("total_seconds", lambda values: values.quantile(0.75)),
        )
        .sort_values(["method", "n_samples"])
    )
    if len(summary) != len(attempted):
        raise ValueError("At least one reported group has no successful fit")
    return summary


def main() -> int:
    args = parse_args()
    results = pd.read_csv(args.results)
    summary = summarize(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    handles = []
    for method in METHODS:
        values = summary.loc[summary["method"].eq(method)].sort_values("n_samples")
        if values.empty:
            continue
        line = ax.plot(
            values["n_samples"],
            values["median"],
            color=COLORS[method],
            marker=MARKERS[method],
            # marker='o',
            markersize=3,
            linewidth=2.0,
            label=LABELS[method],
        )[0]
        ax.fill_between(
            values["n_samples"],
            values["q25"],
            values["q75"],
            color=COLORS[method],
            alpha=0.14,
        )

        # incomplete = values.loc[values["n_fits"].lt(EXPECTED_SEEDS)]
        # if not incomplete.empty:
        #     ax.scatter(
        #         incomplete["n_samples"],
        #         incomplete["median"],
        #         marker=MARKERS[method],
        #         s=38,
        #         facecolors="white",
        #         edgecolors=COLORS[method],
        #         linewidths=1.5,
        #         zorder=4,
        #     )
        # handles.append(line)

    # if summary["n_fits"].lt(EXPECTED_SEEDS).any():
    #     handles.append(
    #         Line2D(
    #             [0],
    #             [0],
    #             color="0.35",
    #             marker="o",
    #             markerfacecolor="white",
    #             markeredgewidth=1.3,
    #             linewidth=0,
    #             label="4/5 successful fits",
    #         )
    #     )

    dimensions = sorted(summary["n_samples"].unique())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(dimensions)
    ax.set_xticklabels([str(int(value)) for value in dimensions])
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Total runtime (seconds)")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure9.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

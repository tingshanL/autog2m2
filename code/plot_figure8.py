#!/usr/bin/env python3
"""Plot Figure 8 from the saved Drosophila assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


CLASS_NAMES = {1: "KC", 2: "MBIN", 3: "MBON", 4: "PN"}
MARKERS = {1: "o", 2: "^", 3: "s", 4: "*"}
COLORS = {1: "#00C853", 2: "#008AA2", 3: "#FF8F00", 4: "#8D6E63"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure8"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()

    assignments = pd.read_csv(args.results_dir / "figure8_assignments.csv")
    scores = pd.read_csv(args.results_dir / "figure8_results.csv").set_index("method")["ari"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    panels = (
        ("AutoGMM-Euclidean", "autogmm_raw", "autogmm_mapped"),
        ("mclust", "mclust_raw", "mclust_mapped"),
    )
    for ax, (method, raw_column, mapped_column) in zip(axes, panels):
        for reference in sorted(assignments["reference"].unique()):
            for predicted in sorted(assignments[raw_column].unique()):
                subset = assignments.loc[
                    assignments["reference"].eq(reference)
                    & assignments[raw_column].eq(predicted)
                ]
                if subset.empty:
                    continue
                mapped = int(subset[mapped_column].mode().iloc[0])
                edge = "none"
                width = 0.0
                if method == "mclust" and predicted in (1, 2):
                    edge = "#7F32A0" if predicted == 2 else "#615F1C"
                    width = 2.0
                marker = MARKERS[int(reference)]
                size = {"o": 30, "*": 90, "s": 40}.get(marker, 60)
                ax.scatter(
                    subset["ase_dimension_1"],
                    subset["ase_dimension_2"],
                    c=[COLORS[mapped]],
                    marker=marker,
                    s=size,
                    alpha=0.9,
                    edgecolors=edge,
                    linewidth=width,
                )
        ax.set_title(f"{method.replace('-Euclidean', '')} (ARI={scores[method]:.2f})", fontsize=17)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Dimension 1", fontsize=15)
    axes[0].set_ylabel("Dimension 2", fontsize=15)

    handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=MARKERS[value],
            linestyle="None",
            markersize=12 if MARKERS[value] in {"^", "*"} else 8,
        )
        for value in CLASS_NAMES
    ]
    fig.legend(
        handles,
        [CLASS_NAMES[value] for value in CLASS_NAMES],
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
        title="True classes",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"real_droso.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

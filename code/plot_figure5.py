#!/usr/bin/env python3
"""Plot Figure 5 axis-aligned and oblique forest comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASETS = {
    "subspace_gaussian": "Axis-aligned subspace",
    "subspace_gaussian_rotated": "Rotated subspace",
    "swiss_roll_highdim": "Swiss roll",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/figure5/ablation2_urf_vs_ourf.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    frame = pd.read_csv(args.results)
    long = frame.melt(
        id_vars=["dataset", "d", "seed"],
        value_vars=["ARI_Geo_URF", "ARI_Geo_OURF"],
        var_name="forest",
        value_name="ari",
    )
    long["Forest"] = long["forest"].map(
        {"ARI_Geo_URF": "Axis-aligned", "ARI_Geo_OURF": "Oblique"}
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4), sharey=True)
    for ax, (dataset, title) in zip(axes, DATASETS.items()):
        panel = long.loc[long["dataset"].eq(dataset)]
        sns.stripplot(data=panel, x="d", y="ari", hue="Forest", hue_order=["Axis-aligned", "Oblique"],
                      palette=["#1B9E77", "#D95F02"], dodge=True, jitter=0.14, alpha=0.75, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Ambient dimension")
        ax.set_ylim(-0.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is not axes[0] and ax.get_legend() is not None:
            ax.get_legend().remove()
    axes[0].set_ylabel("Adjusted Rand index")
    axes[0].legend(title=None, frameon=False, loc="lower left")
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure5.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

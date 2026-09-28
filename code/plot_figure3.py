#!/usr/bin/env python3
"""Plot Figure 3 robustness sweeps from the result tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SERIES = {
    "ari_geo_mle": ("AutoG2M2", "#F7BA14", "o"),
    "ari_gf_avg": ("Forest average-linkage", "#D89000", "s"),
    "ari_gf_comp": ("Forest complete-linkage", "#A86B00", "D"),
    "ari_kw": ("RBF-KPCA-Ward (whitened)", "#3E35BF", "^"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure3"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def draw_panel(ax, path: Path, x_name: str, title: str, x_label: str) -> None:
    frame = pd.read_csv(path)
    grouped = frame.groupby(x_name)
    if x_name == "ratio":
        x_values = ["1:1:1", "3:1:1", "10:1:1"]
    else:
        x_values = list(grouped.groups)
    for column, (label, color, marker) in SERIES.items():
        med = grouped[column].median().reindex(x_values)
        q25 = grouped[column].quantile(0.25).reindex(x_values)
        q75 = grouped[column].quantile(0.75).reindex(x_values)
        x = range(len(x_values)) if x_name == "ratio" else x_values
        ax.plot(x, med, color=color, marker=marker, linewidth=1.8, label=label)
        ax.fill_between(x, q25, q75, color=color, alpha=0.13)
    if x_name == "ratio":
        ax.set_xticks(range(len(x_values)), [str(value) for value in x_values], rotation=0)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panels = (
        ("subspace_snr_sweep.csv", "snr", "Signal-to-noise ratio", "SNR"),
        ("subspace_imbalance_sweep.csv", "ratio", "Class imbalance", "Class ratio"),
        ("subspace_k_sweep.csv", "k", "Number of clusters", "Number of clusters"),
        ("subspace_p_sweep.csv", "p", "Informative dimensions", "Informative dims per cluster"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharey=True)
    for ax, (name, x_name, title, x_label) in zip(axes.flat, panels):
        draw_panel(ax, args.results_dir / name, x_name, title, x_label)
        ax.set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("Adjusted Rand index")
    axes[1, 0].set_ylabel("Adjusted Rand index")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.05))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"figure3.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

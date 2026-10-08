#!/usr/bin/env python3
"""Plot PCA views of the Figure 2 generators at d=16 and d=256.

Run: python plot_figureS1.py --seed 4 --output-dir figures
The simulation functions match run_autog2m2_figure2_final.py; no clustering is run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


COLORS = ("#0072B2", "#D55E00", "#009E73")
DIMENSIONS = (16, 256)
N_PER_CLUSTER = 200
N_CLUSTERS = 3
INFORMATIVE_PER_CLUSTER = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def make_gaussian_subspace(
    *, n_per: int, d: int, k: int, p: int, snr: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    blocks: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for cluster in range(k):
        values = rng.normal(scale=1.0, size=(n_per, d))
        start = (cluster * p) % d
        indices = np.arange(start, start + p) % d
        center = rng.normal(scale=snr, size=p)
        values[:, indices] += center
        blocks.append(values)
        labels.append(np.full(n_per, cluster, dtype=int))
    values = StandardScaler().fit_transform(np.vstack(blocks))
    return np.asarray(values, dtype=np.float64), np.concatenate(labels)


def make_heavy_tail_subspace(
    *, n_per: int, d: int, k: int, p: int, df: float, snr: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = np.empty((k * n_per, d), dtype=np.float64)
    labels = np.empty(k * n_per, dtype=int)
    for cluster in range(k):
        start = cluster * n_per
        stop = (cluster + 1) * n_per
        block = rng.standard_t(df, size=(n_per, d))
        indices = np.arange(cluster * p, (cluster + 1) * p) % d
        center = rng.normal(scale=snr, size=p)
        block[:, indices] += center
        values[start:stop] = block
        labels[start:stop] = cluster
    return StandardScaler().fit_transform(values), labels


def swiss_roll_realization(
    *, n_per: int, d: int, k: int, roll_noise: float, embed_noise: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    original, intrinsic = make_swiss_roll(
        n_samples=n_per * k, noise=roll_noise, random_state=seed
    )
    original = original[np.argsort(intrinsic)]
    labels = np.repeat(np.arange(k, dtype=int), n_per)
    embedded = original @ rng.normal(size=(3, d))
    embedded += embed_noise * rng.normal(size=embedded.shape)
    return original, StandardScaler().fit_transform(embedded), labels


def pca_coordinates(values: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, svd_solver="full").fit_transform(values)


def scatter_2d(ax, coordinates: np.ndarray, labels: np.ndarray, title: str) -> None:
    for cluster, color in enumerate(COLORS):
        selected = labels == cluster
        ax.scatter(
            coordinates[selected, 0], coordinates[selected, 1],
            s=9, alpha=0.62, color=color, linewidths=0,
            label=f"Cluster {cluster + 1}",
        )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)


def main() -> int:
    args = parse_args()
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 3, "ps.fonttype": 3})
    fig, axes = plt.subplots(3, 2, figsize=(8.2, 9.2))
    for column, dimension in enumerate(DIMENSIONS):
        common = dict(
            n_per=N_PER_CLUSTER, d=dimension, k=N_CLUSTERS, seed=args.seed
        )
        gaussian = make_gaussian_subspace(
            **common, p=INFORMATIVE_PER_CLUSTER, snr=3.0
        )
        heavy_tail = make_heavy_tail_subspace(
            **common, p=INFORMATIVE_PER_CLUSTER, df=5.0, snr=4.0
        )
        _, embedded, labels = swiss_roll_realization(
            **common, roll_noise=0.03, embed_noise=0.05
        )
        datasets = (
            ("Gaussian subspace", gaussian),
            ("Heavy-tailed subspace", heavy_tail),
            ("Swiss roll", (embedded, labels)),
        )
        for row, (title, (values, labels)) in enumerate(datasets):
            scatter_2d(
                axes[row, column], pca_coordinates(values), labels,
                f"{title}, $d={dimension}$",
            )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels, loc="lower center", ncol=3, frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = args.output_dir / f"figureS1.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute the affinity matrices and neighborhood purities for Figure 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from geoforest_subspace import (
    make_forest_cfg,
    make_subspace_clusters,
    urf_similarity_from_cfg,
)


def rbf_similarity_from_dist(distances: np.ndarray) -> np.ndarray:
    median_distance = np.median(distances[distances > 0])
    affinity = np.exp(-(distances**2) / (median_distance**2 + 1e-12))
    np.fill_diagonal(affinity, 0.0)
    return affinity


def topk_directed_indices(affinity: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), affinity.shape[0] - 1)
    return np.argpartition(-affinity, kth=k - 1, axis=1)[:, :k]


def symmetrized_topk_affinity(affinity: np.ndarray, k: int) -> np.ndarray:
    neighbor_indices = topk_directed_indices(affinity, k)
    sparse_affinity = np.zeros_like(affinity, dtype=np.float32)
    rows = np.arange(affinity.shape[0])[:, None]
    sparse_affinity[rows, neighbor_indices] = affinity[
        rows, neighbor_indices
    ].astype(np.float32)
    sparse_affinity = np.maximum(sparse_affinity, sparse_affinity.T)
    np.fill_diagonal(sparse_affinity, 0.0)
    return sparse_affinity


def neighbor_purity(neighbor_indices: np.ndarray, labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    return (labels[neighbor_indices] == labels[:, None]).mean(axis=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figure1"))
    args = parser.parse_args()

    x, labels = make_subspace_clusters(
        n_per=200,
        d=1024,
        k=3,
        subspace_dim=5,
        snr=3.0,
        seed=0,
    )
    x = StandardScaler().fit_transform(x).astype(np.float32)

    n_samples = x.shape[0]
    n_neighbors = 15
    order = np.argsort(labels)

    euclidean_distances = pairwise_distances(x, metric="euclidean")
    euclidean_affinity = rbf_similarity_from_dist(euclidean_distances)
    euclidean_topk = symmetrized_topk_affinity(euclidean_affinity, n_neighbors)
    euclidean_purity = neighbor_purity(
        topk_directed_indices(euclidean_affinity, n_neighbors), labels
    )

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=0,
    ).fit(x)
    umap_graph = umap_model.graph_
    umap_affinity = umap_graph.maximum(umap_graph.T).toarray().astype(np.float32)
    np.fill_diagonal(umap_affinity, 0.0)
    umap_topk = symmetrized_topk_affinity(umap_affinity, n_neighbors)
    umap_purity = neighbor_purity(
        topk_directed_indices(umap_affinity, n_neighbors), labels
    )

    forest_class, forest_config = make_forest_cfg(
        n_samples, x.shape[1], oblique=False, random_state=0
    )
    forest_config["n_estimators"] = 300
    forest_config["n_jobs"] = -1
    geoforest_affinity = urf_similarity_from_cfg(
        x, forest_class, forest_config
    ).astype(np.float32)
    np.fill_diagonal(geoforest_affinity, 0.0)
    geoforest_topk = symmetrized_topk_affinity(geoforest_affinity, n_neighbors)
    geoforest_purity = neighbor_purity(
        topk_directed_indices(geoforest_affinity, n_neighbors), labels
    )

    purity = pd.DataFrame(
        {
            "method": (
                ["Euclidean kNN"] * n_samples
                + ["UMAP graph"] * n_samples
                + ["GeoForest kNN"] * n_samples
            ),
            "purity": np.concatenate(
                [euclidean_purity, umap_purity, geoforest_purity]
            ),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    purity.to_csv(args.output_dir / "fig1_neighbor_purity_raw.csv", index=False)
    np.savez_compressed(
        args.output_dir / "figure1_affinities.npz",
        euclidean=euclidean_topk,
        umap=umap_topk,
        geoforest=geoforest_topk,
        order=order,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

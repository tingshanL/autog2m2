#!/usr/bin/env python3
"""Run the fixed-K and automatic-K experiments used for Figure 2."""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree, shortest_path
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import kneighbors_graph
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DIMENSIONS = {
    "gaussian_subspace": (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "heavy_tail_subspace": (8, 16, 32, 64, 128, 256, 512, 1024, 2048),
    "swiss_roll": (8, 16, 32, 64, 128, 256),
}
SEEDS = tuple(range(20))
METHODS = (
    "autog2m2",
    "pca",
    "umap",
    "ae_tanh",
    "autogmm_spectral",
    "mclust",
    "ae_relu",
)
DEFAULT_METHODS = METHODS[:-1]
RESULT_COLUMNS = (
    "dataset",
    "dimension",
    "seed",
    "method",
    "stage",
    "status",
    "error",
    "ari",
    "n_samples",
    "true_components",
    "embedding_dim",
    "embedding_rank",
    "selected_components",
    "occupied_components",
    "selected_covariance",
    "bic",
    "selected_spectral_head",
    "k_min",
    "k_max",
    "selection_criterion",
    "embedding_seconds",
    "backend_seconds",
    "total_seconds",
    "forest_seconds",
    "graph_seconds",
    "shortest_path_seconds",
    "mds_seconds",
    "spectral_seconds",
    "ae_activation",
    "ae_iterations",
    "ae_converged",
    "log_likelihood",
    "degrees_of_freedom",
    "icl",
)


@dataclass(frozen=True)
class Config:
    n_per_cluster: int = 200
    true_components: int = 3
    informative_dimensions: int = 5
    gaussian_snr: float = 3.0
    heavy_tail_df: float = 5.0
    heavy_tail_snr: float = 4.0
    swiss_roll_noise: float = 0.03
    swiss_embedding_noise: float = 0.05
    embedding_dim: int = 2
    forest_trees: int = 1400
    forest_leaf_fraction: float = 0.015
    graph_neighbors: int = 50
    graph_connectivity_growth: float = 1.5
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    ae_hidden_cap: int = 128
    ae_hidden_floor: int = 64
    ae_l2_alpha: float = 1e-2
    ae_noise_sd: float = 0.05
    ae_learning_rate: float = 1e-3
    ae_batch_size: int = 64
    ae_max_epochs: int = 1000
    ae_validation_fraction: float = 0.20
    ae_patience: int = 20
    ae_tolerance: float = 1e-4


def make_dataset(dataset: str, dimension: int, seed: int, config: Config):
    n_per = config.n_per_cluster
    k = config.true_components
    p = config.informative_dimensions
    if dataset == "gaussian_subspace":
        rng = np.random.RandomState(seed)
        blocks = []
        labels = []
        for cluster in range(k):
            block = rng.normal(size=(n_per, dimension))
            indices = np.arange(cluster * p, (cluster + 1) * p) % dimension
            block[:, indices] += rng.normal(scale=config.gaussian_snr, size=p)
            blocks.append(block)
            labels.append(np.full(n_per, cluster, dtype=int))
        X = np.vstack(blocks)
        y = np.concatenate(labels)
    elif dataset == "heavy_tail_subspace":
        rng = np.random.default_rng(seed)
        X = np.empty((k * n_per, dimension), dtype=float)
        y = np.empty(k * n_per, dtype=int)
        for cluster in range(k):
            block = rng.standard_t(config.heavy_tail_df, size=(n_per, dimension))
            indices = np.arange(cluster * p, (cluster + 1) * p) % dimension
            block[:, indices] += rng.normal(scale=config.heavy_tail_snr, size=p)
            section = slice(cluster * n_per, (cluster + 1) * n_per)
            X[section] = block
            y[section] = cluster
    elif dataset == "swiss_roll":
        rng = np.random.default_rng(seed)
        X3, intrinsic = make_swiss_roll(
            n_samples=k * n_per,
            noise=config.swiss_roll_noise,
            random_state=seed,
        )
        X3 = X3[np.argsort(intrinsic)]
        y = np.repeat(np.arange(k, dtype=int), n_per)
        X = X3 @ rng.normal(size=(3, dimension))
        X += config.swiss_embedding_noise * rng.normal(size=X.shape)
    else:
        raise ValueError(dataset)
    return StandardScaler().fit_transform(X), y


def forest_proximity(X: np.ndarray, seed: int, config: Config):
    from treeple import UnsupervisedRandomForest

    leaf = max(3, int(config.forest_leaf_fraction * len(X)))
    started = time.perf_counter()
    forest = UnsupervisedRandomForest(
        n_estimators=config.forest_trees,
        criterion="twomeans",
        max_depth=None,
        min_samples_leaf=leaf,
        min_samples_split=max(2 * leaf, 6),
        max_features="sqrt",
        bootstrap=False,
        n_jobs=1,
        random_state=seed,
    )
    forest.fit(X)
    similarity = np.asarray(forest.compute_similarity_matrix(X), dtype=float)
    similarity = np.clip(0.5 * (similarity + similarity.T), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity, time.perf_counter() - started


def topk_graph(similarity: np.ndarray, neighbors: int) -> csr_matrix:
    n = len(similarity)
    k = min(max(1, neighbors), n - 1)
    selected = np.zeros((n, n), dtype=bool)
    indices = np.arange(n)
    for row in range(n):
        scores = similarity[row].copy()
        scores[row] = -np.inf
        selected[row, np.lexsort((indices, -scores))[:k]] = True
    selected |= selected.T
    rows, columns = np.nonzero(selected)
    lengths = np.maximum(1.0 - similarity[rows, columns], np.finfo(float).eps)
    return csr_matrix((lengths, (rows, columns)), shape=(n, n))


def geodesic_distances(similarity: np.ndarray, config: Config):
    neighbors = min(config.graph_neighbors, len(similarity) - 1)
    initial = neighbors
    graph_started = time.perf_counter()
    while True:
        graph = topk_graph(similarity, neighbors)
        components = int(connected_components(graph, directed=False)[0])
        if components == 1:
            break
        neighbors = min(
            len(similarity) - 1,
            max(neighbors + 1, int(np.ceil(neighbors * config.graph_connectivity_growth))),
        )
    graph_seconds = time.perf_counter() - graph_started
    started = time.perf_counter()
    distances = shortest_path(graph, directed=False, unweighted=False)
    np.fill_diagonal(distances, 0.0)
    return distances, {
        "graph_seconds": graph_seconds,
        "shortest_path_seconds": time.perf_counter() - started,
        "graph_k_initial": initial,
        "graph_k_final": neighbors,
    }


def classical_mds(
    distances: np.ndarray,
    dimensions: int = 2,
    relative_tolerance: float = 1e-10,
):
    started = time.perf_counter()
    squared = distances * distances
    gram = -0.5 * (
        squared
        - squared.mean(axis=1, keepdims=True)
        - squared.mean(axis=0, keepdims=True)
        + squared.mean()
    )
    gram = 0.5 * (gram + gram.T)
    n = len(gram)
    values, vectors = eigh(
        gram,
        subset_by_index=(n - dimensions, n - 1),
        check_finite=False,
        driver="evr",
    )
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    if np.any(values <= relative_tolerance * max(float(np.max(np.abs(values))), 1.0)):
        raise RuntimeError("Classical MDS did not return two positive dimensions")
    return vectors * np.sqrt(values), {"mds_seconds": time.perf_counter() - started}


def embed_autog2m2(X: np.ndarray, seed: int, config: Config):
    started = time.perf_counter()
    similarity, forest_seconds = forest_proximity(X, seed, config)
    distances, graph = geodesic_distances(similarity, config)
    coordinates, mds = classical_mds(distances, config.embedding_dim)
    return coordinates, {
        "embedding_seconds": time.perf_counter() - started,
        "forest_seconds": forest_seconds,
        **graph,
        **mds,
    }


def embed_pca(X: np.ndarray, seed: int, config: Config):
    started = time.perf_counter()
    coordinates = PCA(
        n_components=config.embedding_dim,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(X)
    return coordinates, {"embedding_seconds": time.perf_counter() - started}


def embed_umap(
    X: np.ndarray,
    seed: int,
    dimension_or_config: int | Config,
    config: Config | None = None,
):
    import umap

    if config is None:
        config = dimension_or_config
        dimension = config.embedding_dim
    else:
        dimension = int(dimension_or_config)
    started = time.perf_counter()
    coordinates = umap.UMAP(
        n_neighbors=config.umap_neighbors,
        min_dist=config.umap_min_dist,
        n_components=dimension,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(X)
    return np.asarray(coordinates), {"embedding_seconds": time.perf_counter() - started}


def embed_autoencoder(X: np.ndarray, seed: int, activation: str, config: Config):
    hidden = min(config.ae_hidden_cap, max(config.ae_hidden_floor, 2 * config.embedding_dim))
    noisy = X + np.random.RandomState(seed).normal(scale=config.ae_noise_sd, size=X.shape)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden, config.embedding_dim, hidden),
        activation=activation,
        solver="adam",
        alpha=config.ae_l2_alpha,
        batch_size=min(config.ae_batch_size, len(X)),
        learning_rate_init=config.ae_learning_rate,
        max_iter=config.ae_max_epochs,
        shuffle=True,
        random_state=seed,
        tol=config.ae_tolerance,
        early_stopping=True,
        validation_fraction=config.ae_validation_fraction,
        n_iter_no_change=config.ae_patience,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(noisy, X)
    transform = np.tanh if activation == "tanh" else lambda values: np.maximum(values, 0.0)
    hidden_values = transform(X @ model.coefs_[0] + model.intercepts_[0])
    coordinates = transform(hidden_values @ model.coefs_[1] + model.intercepts_[1])
    return coordinates, {
        "embedding_seconds": time.perf_counter() - started,
        "ae_activation": activation,
        "ae_iterations": int(model.n_iter_),
        "ae_converged": bool(model.n_iter_ < config.ae_max_epochs),
    }


def spectral_embeddings(X: np.ndarray, seed: int):
    started = time.perf_counter()
    n = len(X)
    neighbors = int(np.sqrt(n))
    adjacency = kneighbors_graph(
        X, n_neighbors=neighbors, mode="connectivity", include_self=True, n_jobs=1
    )
    adjacency = (0.5 * (adjacency + adjacency.T)).tolil()
    distances = kneighbors_graph(
        X, n_neighbors=neighbors, mode="distance", include_self=False, n_jobs=1
    )
    distances = (0.5 * (distances + distances.T)).tocsr()
    full = pdist(X, metric="euclidean")
    sigma = float(np.median(full))
    gamma = 1.0 / (2.0 * sigma * sigma)
    rbf = distances.copy()
    rbf.data = np.exp(-gamma * np.square(rbf.data))
    rbf = rbf.tolil()
    tree = minimum_spanning_tree(squareform(full))
    rows, columns = tree.nonzero()
    for row, column in zip(rows, columns):
        distance = float(tree[row, column])
        weight = max(float(np.exp(-gamma * distance * distance)), np.finfo(float).tiny)
        adjacency[row, column] = adjacency[column, row] = 1.0
        rbf[row, column] = rbf[column, row] = weight
    adjacency = adjacency.tocsr()
    rbf = rbf.tocsr()
    initial = np.random.RandomState(seed).normal(size=n)
    values, vectors = eigsh(adjacency, k=2, which="LA", v0=initial)
    order = np.argsort(values)[::-1]
    ase = vectors[:, order] * np.sqrt(np.clip(values[order], 0.0, None))
    lse = SpectralEmbedding(
        n_components=2,
        affinity="precomputed",
        eigen_solver="arpack",
        random_state=seed,
    ).fit_transform(rbf)
    return {"ase": ase, "lse": lse}, {"spectral_seconds": time.perf_counter() - started}


def fit_autogmm(coordinates: np.ndarray, seed: int, k_min: int, k_max: int):
    try:
        from autogmm import AutoGMM
        external_package = True
    except ImportError:
        from autogmm_Apr26 import AutoGMM
        external_package = False

    started = time.perf_counter()
    parameters = dict(
        min_components=k_min,
        max_components=k_max,
        init_agglomerative=True,
        agglom_linkages=["ward"],
        agglom_affinities=["mahalanobis"],
        n_init_kmeans=1,
        eigen_thres=True,
        reg_covar=1e-6,
        criterion="bic",
        random_state=seed,
        verbose=False,
        n_jobs=1,
        early_stop_delta=0.0,
    )
    if external_package:
        parameters["covariances"] = ["full", "diag", "tied", "spherical"]
    model = AutoGMM(**parameters)
    labels = np.asarray(model.fit_predict(np.asarray(coordinates, dtype=float)), dtype=int)
    fitted = model.best_model_
    return labels, {
        "backend_seconds": time.perf_counter() - started,
        "selected_components": int(getattr(model, "n_components_", fitted.n_components)),
        "selected_covariance": str(
            getattr(model, "covariance_type_", fitted.covariance_type)
        ),
        "bic": float(model.best_score_),
    }


_MCLUST = None


def fit_mclust(X: np.ndarray, components: range | tuple[int, ...]):
    global _MCLUST
    if _MCLUST is None:
        from mclust_utils import MclustRunner

        _MCLUST = MclustRunner()
    fitted = _MCLUST.fit(X, components)
    return fitted.labels, {
        "backend_seconds": fitted.elapsed_seconds,
        "selected_components": fitted.selected_components,
        "selected_covariance": fitted.selected_model,
        "bic": fitted.bic_max,
    }


def result_row(
    dataset: str,
    dimension: int,
    seed: int,
    method: str,
    stage: str,
    y: np.ndarray,
    labels: np.ndarray,
    fit: dict[str, Any],
    embedding: dict[str, Any] | None = None,
    selected_head: str | None = None,
):
    embedding = embedding or {}
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "dataset": dataset,
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "stage": stage,
            "status": "ok",
            "ari": adjusted_rand_score(y, labels),
            "n_samples": len(y),
            "true_components": 3,
            "embedding_dim": 2 if method != "mclust" else dimension,
            "selected_components": fit["selected_components"],
            "occupied_components": len(np.unique(labels)),
            "selected_covariance": fit["selected_covariance"],
            "bic": fit["bic"],
            "selected_spectral_head": selected_head,
            "k_min": 3 if stage == "fixed" else 1,
            "k_max": 3 if stage == "fixed" else 5,
            "selection_criterion": "bic",
            "backend_seconds": fit["backend_seconds"],
            "total_seconds": embedding.get("embedding_seconds", 0.0)
            + embedding.get("spectral_seconds", 0.0)
            + fit["backend_seconds"],
            **embedding,
        }
    )
    if method != "mclust":
        row["embedding_rank"] = 2
    return row


def error_row(
    dataset: str,
    dimension: int,
    seed: int,
    method: str,
    stage: str,
    error: str,
):
    row = {column: np.nan for column in RESULT_COLUMNS}
    row.update(
        {
            "dataset": dataset,
            "dimension": dimension,
            "seed": seed,
            "method": method,
            "stage": stage,
            "status": "error",
            "error": error,
            "n_samples": 600,
            "true_components": 3,
            "k_min": 3 if stage == "fixed" else 1,
            "k_max": 3 if stage == "fixed" else 5,
            "selection_criterion": "bic",
        }
    )
    return row


def stage_is_used(dataset: str, dimension: int, method: str, stage: str) -> bool:
    if stage == "fixed":
        return True
    if method == "mclust":
        return dataset == "heavy_tail_subspace" and dimension >= 512
    return True


def run_condition(payload: tuple[str, int, int, tuple[str, ...], tuple[str, ...]]):
    dataset, dimension, seed, methods, stages = payload
    config = Config()
    with threadpool_limits(limits=1):
        X, y = make_dataset(dataset, dimension, seed, config)
        rows = []
        for method in methods:
            active_stages = [stage for stage in stages if stage_is_used(dataset, dimension, method, stage)]
            if not active_stages:
                continue
            try:
                if method == "mclust":
                    for stage in active_stages:
                        components = (3,) if stage == "fixed" else range(1, 6)
                        labels, fit = fit_mclust(X, components)
                        rows.append(result_row(dataset, dimension, seed, method, stage, y, labels, fit))
                    continue

                if method == "autogmm_spectral":
                    embeddings, metadata = spectral_embeddings(X, seed)
                    fixed_candidates = []
                    for name in ("ase", "lse"):
                        labels, fit = fit_autogmm(embeddings[name], seed, 3, 3)
                        fixed_candidates.append((fit["bic"], name, labels, fit))
                    _, head, labels, fit = min(fixed_candidates, key=lambda item: (item[0], item[1]))
                    if "fixed" in active_stages:
                        rows.append(
                            result_row(
                                dataset, dimension, seed, method, "fixed", y, labels, fit, metadata, head
                            )
                        )
                    if "auto" in active_stages:
                        labels, fit = fit_autogmm(embeddings[head], seed, 1, 5)
                        rows.append(
                            result_row(
                                dataset, dimension, seed, method, "auto", y, labels, fit, metadata, head
                            )
                        )
                    continue

                if method == "autog2m2":
                    coordinates, metadata = embed_autog2m2(X, seed, config)
                elif method == "pca":
                    coordinates, metadata = embed_pca(X, seed, config)
                elif method == "umap":
                    coordinates, metadata = embed_umap(X, seed, config)
                elif method == "ae_tanh":
                    coordinates, metadata = embed_autoencoder(X, seed, "tanh", config)
                elif method == "ae_relu":
                    coordinates, metadata = embed_autoencoder(X, seed, "relu", config)
                else:
                    raise ValueError(method)
                for stage in active_stages:
                    limits = (3, 3) if stage == "fixed" else (1, 5)
                    labels, fit = fit_autogmm(coordinates, seed, *limits)
                    rows.append(
                        result_row(
                            dataset, dimension, seed, method, stage, y, labels, fit, metadata
                        )
                    )
            except Exception as exc:
                rows.extend(
                    error_row(dataset, dimension, seed, method, stage, str(exc))
                    for stage in active_stages
                )
        return rows


def summarize(frame: pd.DataFrame):
    usable = frame.loc[frame["status"].eq("ok") & frame["ari"].notna()]
    return (
        usable.groupby(["dataset", "dimension", "method"], as_index=False)["ari"]
        .agg(
            n_fits="size",
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["dataset", "dimension", "method"])
    )


def save_results(rows: list[dict[str, Any]], output_dir: Path):
    new = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    keys = ["dataset", "dimension", "seed", "method", "stage"]
    for stage, filename, summary_name in (
        ("fixed", "figure2_fixed_k_results.csv", "figure2_fixed_k_summary.csv"),
        ("auto", "figure2_auto_k_results.csv", "figure2_auto_k_summary.csv"),
    ):
        addition = new.loc[new["stage"].eq(stage)]
        path = output_dir / filename
        if path.exists():
            addition = pd.concat([pd.read_csv(path), addition], ignore_index=True)
        addition = addition.drop_duplicates(keys, keep="last").sort_values(keys)
        addition.to_csv(path, index=False)
        summarize(addition).to_csv(output_dir / summary_name, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DIMENSIONS, default=list(DIMENSIONS))
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    parser.add_argument("--stages", nargs="+", choices=("fixed", "auto"), default=["fixed", "auto"])
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure2"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = []
    for dataset in args.datasets:
        dimensions = args.dimensions or DIMENSIONS[dataset]
        tasks.extend(
            (dataset, dimension, seed, tuple(args.methods), tuple(args.stages))
            for dimension in dimensions
            for seed in args.seeds
        )

    rows = []
    if args.jobs == 1:
        for task in tasks:
            rows.extend(run_condition(task))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_condition, task) for task in tasks]
            for future in as_completed(futures):
                rows.extend(future.result())

    output_dir = args.results_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_results(rows, output_dir)
    print(f"Wrote {len(rows):,} rows to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

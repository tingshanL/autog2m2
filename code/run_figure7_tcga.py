#!/usr/bin/env python3
"""Run the TCGA feature-expansion experiments used for Figure 7."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import run_figure2 as figure2
from tcga_utils import load_tcga_inputs


DIMENSIONS = (256, 512, 1024, 2048, 4096)
SEEDS = tuple(range(20))
METHODS = ("autog2m2", "pca", "umap")
REGIMES = ("observed_genes", "independent_permutation", "structured_lowrank")
CORE_DIMENSION = 256
EMBEDDING_DIMENSION = 20
FOREST_TREES = 1000
FOREST_LEAF_FRACTION = 0.015
GRAPH_NEIGHBORS = 30
GRAPH_CONNECTIVITY_GROWTH = 1.5
STRUCTURED_FACTORS = 5
STRUCTURED_VARIANCE = 0.20


def rank_genes(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    variances = X.var(axis=0)
    core = np.argpartition(variances, -CORE_DIMENSION)[-CORE_DIMENSION:]
    remaining = np.setdiff1d(np.arange(X.shape[1]), core, assume_unique=True)
    donors = remaining[np.lexsort((remaining, -variances[remaining]))]
    return np.asarray(core, dtype=int), np.asarray(donors, dtype=int)


def make_matrix(
    X_raw: np.ndarray,
    core: np.ndarray,
    donors: np.ndarray,
    dimension: int,
    seed: int,
    regime: str,
) -> np.ndarray:
    added = dimension - CORE_DIMENSION
    blocks = [np.asarray(X_raw[:, core], dtype=float)]
    if added and regime == "observed_genes":
        blocks.append(np.asarray(X_raw[:, donors[:added]], dtype=float))
    elif added and regime == "independent_permutation":
        rng = np.random.RandomState(seed)
        nuisance = np.empty((len(X_raw), added), dtype=float)
        for column, source in enumerate(donors[:added]):
            nuisance[:, column] = X_raw[rng.permutation(len(X_raw)), source]
        blocks.append(nuisance)
    elif added and regime == "structured_lowrank":
        maximum = max(DIMENSIONS) - CORE_DIMENSION
        rng = np.random.RandomState(seed)
        factors = rng.standard_normal((len(X_raw), STRUCTURED_FACTORS))
        factors -= factors.mean(axis=0, keepdims=True)
        factors, _ = np.linalg.qr(factors, mode="reduced")
        factors *= np.sqrt(len(X_raw))
        loadings = rng.standard_normal((maximum, STRUCTURED_FACTORS))
        loadings /= np.linalg.norm(loadings, axis=1, keepdims=True)
        noise = rng.standard_normal((len(X_raw), maximum))
        noise -= noise.mean(axis=0, keepdims=True)
        noise -= factors @ (factors.T @ noise / len(X_raw))
        noise /= np.sqrt(np.mean(np.square(noise), axis=0, keepdims=True))
        nuisance = (
            np.sqrt(STRUCTURED_VARIANCE) * (factors @ loadings.T)
            + np.sqrt(1.0 - STRUCTURED_VARIANCE) * noise
        )
        blocks.append(nuisance[:, :added])
    elif added and regime not in REGIMES:
        raise ValueError(regime)
    return StandardScaler().fit_transform(np.concatenate(blocks, axis=1))


def top_k_affinity(similarity: np.ndarray, neighbors: int) -> csr_matrix:
    n = len(similarity)
    neighbors = min(neighbors, n - 1)
    rows = np.repeat(np.arange(n, dtype=np.int32), neighbors)
    columns = np.empty(n * neighbors, dtype=np.int32)
    values = np.empty(n * neighbors, dtype=np.float32)
    for row in range(n):
        indices = np.argpartition(similarity[row], -(neighbors + 1))[-(neighbors + 1):]
        indices = indices[indices != row]
        if len(indices) > neighbors:
            indices = indices[np.argsort(similarity[row, indices])[-neighbors:]]
        start = row * neighbors
        columns[start : start + neighbors] = indices
        values[start : start + neighbors] = np.maximum(
            similarity[row, indices], np.finfo(np.float32).tiny
        )
    affinity = coo_matrix((values, (rows, columns)), shape=(n, n)).tocsr()
    affinity = affinity.maximum(affinity.T)
    affinity.setdiag(1.0)
    return affinity


def length_graph(affinity: csr_matrix) -> csr_matrix:
    values = affinity.tocoo()
    keep = values.row != values.col
    lengths = np.maximum(1.0 - values.data[keep], np.finfo(float).eps)
    graph = csr_matrix(
        (lengths, (values.row[keep], values.col[keep])), shape=affinity.shape
    )
    return graph.maximum(graph.T)


def connected_graph(similarity: np.ndarray):
    neighbors = GRAPH_NEIGHBORS
    started = time.perf_counter()
    initial_components = None
    while True:
        graph = length_graph(top_k_affinity(similarity, neighbors))
        components = int(connected_components(graph, directed=False)[0])
        if initial_components is None:
            initial_components = components
        if components == 1:
            break
        neighbors = min(
            len(similarity) - 1,
            max(neighbors + 1, int(np.ceil(neighbors * GRAPH_CONNECTIVITY_GROWTH))),
        )
    return graph, {
        "graph_seconds": time.perf_counter() - started,
        "graph_components_initial": initial_components,
        "graph_k_initial": GRAPH_NEIGHBORS,
        "graph_k_final": neighbors,
    }


def embed_autog2m2(X: np.ndarray, seed: int):
    from treeple import UnsupervisedRandomForest

    leaf = max(3, int(FOREST_LEAF_FRACTION * len(X)))
    started = time.perf_counter()
    forest = UnsupervisedRandomForest(
        n_estimators=FOREST_TREES,
        criterion="twomeans",
        max_depth=None,
        min_samples_leaf=leaf,
        min_samples_split=max(2 * leaf, 6),
        max_features="sqrt",
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    ).fit(X)
    similarity = np.asarray(forest.compute_similarity_matrix(X), dtype=float)
    similarity = np.clip(0.5 * (similarity + similarity.T), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    forest_seconds = time.perf_counter() - started
    graph, graph_info = connected_graph(similarity)
    shortest_started = time.perf_counter()
    distances = shortest_path(graph, directed=False, unweighted=False)
    shortest_seconds = time.perf_counter() - shortest_started
    coordinates, mds_info = figure2.classical_mds(distances, EMBEDDING_DIMENSION)
    return coordinates, {
        "embedding_seconds": forest_seconds + graph_info["graph_seconds"]
        + shortest_seconds + mds_info["mds_seconds"],
        "forest_seconds": forest_seconds,
        "shortest_path_seconds": shortest_seconds,
        **graph_info,
        **mds_info,
    }


def embed_pca(X: np.ndarray, seed: int):
    started = time.perf_counter()
    model = PCA(
        n_components=EMBEDDING_DIMENSION,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    )
    coordinates = model.fit_transform(X)
    return coordinates, {
        "embedding_seconds": time.perf_counter() - started,
        "pca_explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
    }


def embed_umap(X: np.ndarray, seed: int):
    import umap

    started = time.perf_counter()
    coordinates = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=EMBEDDING_DIMENSION,
        metric="euclidean",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(X)
    return np.asarray(coordinates), {"embedding_seconds": time.perf_counter() - started}


def fit_backend(coordinates: np.ndarray, seed: int, k_min: int, k_max: int):
    try:
        from autogmm import AutoGMM
        external_package = True
    except ImportError:
        from autogmm_Apr26 import AutoGMM
        external_package = False

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
    started = time.perf_counter()
    model = AutoGMM(**parameters)
    labels = np.asarray(model.fit_predict(coordinates), dtype=int)
    fitted = model.best_model_
    return labels, {
        "backend_seconds": time.perf_counter() - started,
        "selected_components": int(getattr(model, "n_components_", fitted.n_components)),
        "selected_covariance": str(
            getattr(model, "covariance_type_", fitted.covariance_type)
        ),
        "bic": float(model.best_score_),
    }


def neighbor_purity(coordinates: np.ndarray, y: np.ndarray) -> float:
    distances = pairwise_distances(coordinates, n_jobs=1)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind="stable")[:, :30]
    return float(np.mean(y[neighbors] == y[:, None]))


def run_condition(payload):
    X_raw, y, core, donors, regime, dimension, seed = payload
    with threadpool_limits(limits=1):
        X = make_matrix(X_raw, core, donors, dimension, seed, regime)
        automatic_rows = []
        fixed_rows = []
        for method in METHODS:
            started = time.perf_counter()
            try:
                if method == "autog2m2":
                    coordinates, embedding = embed_autog2m2(X, seed)
                elif method == "pca":
                    coordinates, embedding = embed_pca(X, seed)
                else:
                    coordinates, embedding = embed_umap(X, seed)
                purity = neighbor_purity(coordinates, y)
                automatic_labels, automatic = fit_backend(coordinates, seed, 1, 10)
                fixed_labels, fixed = fit_backend(coordinates, seed, 5, 5)
                automatic_rows.append({
                    "data_regime": regime, "dimension": dimension, "seed": seed,
                    "method": method, "status": "ok",
                    "ari": adjusted_rand_score(y, automatic_labels),
                    "selected_components": automatic["selected_components"],
                    "occupied_components": len(np.unique(automatic_labels)),
                    "selected_covariance": automatic["selected_covariance"],
                    "bic": automatic["bic"], "neighbor_purity_at_30": purity,
                    "embedding_dim": EMBEDDING_DIMENSION,
                    "backend_seconds": automatic["backend_seconds"],
                    "total_seconds": time.perf_counter() - started, **embedding,
                })
                fixed_rows.append({
                    "data_regime": regime, "dimension": dimension, "seed": seed,
                    "method": method, "status": "ok",
                    "automatic_ari": adjusted_rand_score(y, automatic_labels),
                    "automatic_selected_components": automatic["selected_components"],
                    "automatic_selected_covariance": automatic["selected_covariance"],
                    "automatic_bic": automatic["bic"],
                    "neighbor_purity_at_30": purity, "fixed_k": 5,
                    "fixed_k_ari": adjusted_rand_score(y, fixed_labels),
                    "fixed_k_selected_components": fixed["selected_components"],
                    "fixed_k_occupied_components": len(np.unique(fixed_labels)),
                    "fixed_k_selected_covariance": fixed["selected_covariance"],
                    "fixed_k_bic": fixed["bic"],
                    "backend_seconds": fixed["backend_seconds"],
                })
            except Exception as exc:
                automatic_rows.append({
                    "data_regime": regime, "dimension": dimension, "seed": seed,
                    "method": method, "status": "error", "error": str(exc),
                })
                fixed_rows.append({
                    "data_regime": regime, "dimension": dimension, "seed": seed,
                    "method": method, "status": "error", "error": str(exc),
                    "fixed_k": 5,
                })
        return automatic_rows, fixed_rows


def update_table(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if path.exists():
        frame = pd.concat([pd.read_csv(path), frame], ignore_index=True, sort=False)
    keys = ["data_regime", "dimension", "seed", "method"]
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(path, index=False)
    return frame


def write_summaries(automatic: pd.DataFrame, fixed: pd.DataFrame, output_dir: Path):
    automatic_ok = automatic.loc[automatic["status"].eq("ok")]
    automatic_summary = (
        automatic_ok.groupby(["data_regime", "dimension", "method"], as_index=False)["ari"]
        .agg(n_fits="size", ari_median="median",
             ari_q25=lambda values: values.quantile(0.25),
             ari_q75=lambda values: values.quantile(0.75))
    )
    automatic_summary.to_csv(output_dir / "figure7_all_regimes_summary.csv", index=False)
    fixed_ok = fixed.loc[fixed["status"].eq("ok")]
    fixed_summary = (
        fixed_ok.groupby(["data_regime", "dimension", "method"], as_index=False)["fixed_k_ari"]
        .agg(n_fits="size", ari_median="median",
             ari_q25=lambda values: values.quantile(0.25),
             ari_q75=lambda values: values.quantile(0.75))
    )
    fixed_summary.to_csv(output_dir / "figure7_fixed_k5_summary.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=REGIMES, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tcga/data.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/tcga/labels.csv"))
    parser.add_argument("--mode", choices=("smoke", "final"), default="final")
    parser.add_argument("--jobs", "--outer-jobs", dest="jobs", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figure7"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    X_raw, y, _, _ = load_tcga_inputs(args.data, args.labels)
    core, donors = rank_genes(X_raw)
    dimensions = (512,) if args.mode == "smoke" else DIMENSIONS
    seeds = (0,) if args.mode == "smoke" else SEEDS
    tasks = [
        (X_raw, y, core, donors, args.regime, dimension, seed)
        for dimension in dimensions for seed in seeds
    ]
    automatic_rows = []
    fixed_rows = []
    if args.jobs == 1:
        outputs = map(run_condition, tasks)
        for automatic, fixed in outputs:
            automatic_rows.extend(automatic)
            fixed_rows.extend(fixed)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_condition, task) for task in tasks]
            for future in as_completed(futures):
                automatic, fixed = future.result()
                automatic_rows.extend(automatic)
                fixed_rows.extend(fixed)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    automatic = update_table(output_dir / "figure7_all_regimes_results.csv", automatic_rows)
    fixed = update_table(output_dir / "figure7_fixed_k5_results.csv", fixed_rows)
    write_summaries(automatic, fixed, output_dir)
    print(f"Wrote {len(automatic_rows)} automatic-K and {len(fixed_rows)} fixed-K rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

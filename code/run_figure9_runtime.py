#!/usr/bin/env python3
"""Run the matched runtime experiment for Figure 9."""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from run_figure2 import Config, classical_mds, fit_autogmm, geodesic_distances


DEFAULT_N_VALUES = (100, 500, 1000, 5000)
DEFAULT_SEEDS = tuple(range(5))
METHODS = ("autog2m2", "umap", "pca")

DIMENSION = 50
EMBEDDING_DIM = 10
K_MIN = 1
K_MAX = 10
SCALE_EMBEDDING = True

FOREST_TREES = 300
FOREST_MIN_LEAF = 2
FOREST_MIN_SPLIT = 4
FOREST_MAX_FEATURES = "sqrt"
GRAPH_NEIGHBORS = 30

UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

RESULT_COLUMNS = (
    "method",
    "n_samples",
    "dimension",
    "seed",
    "status",
    "error",
    "embedding_dim",
    "k_min",
    "k_max",
    "head_seconds",
    "backend_seconds",
    "total_seconds",
    "forest_fit_seconds",
    "similarity_seconds",
    "graph_seconds",
    "shortest_path_seconds",
    "mds_seconds",
    "selected_components",
    "selected_covariance",
    "bic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-values", nargs="+", type=int, default=list(DEFAULT_N_VALUES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--results-dir", type=Path, default=Path("results/figure9"))
    return parser.parse_args()


def make_isotropic_gaussian(n_samples: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    data = rng.normal(size=(n_samples, dimension))
    return StandardScaler().fit_transform(data)


def embed_autog2m2(data: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    from treeple import UnsupervisedRandomForest

    started = time.perf_counter()
    forest_started = time.perf_counter()
    forest = UnsupervisedRandomForest(
        n_estimators=FOREST_TREES,
        criterion="twomeans",
        max_depth=None,
        min_samples_leaf=FOREST_MIN_LEAF,
        min_samples_split=FOREST_MIN_SPLIT,
        max_features=FOREST_MAX_FEATURES,
        bootstrap=True,
        n_jobs=1,
        random_state=seed,
    )
    forest.fit(data)
    forest_seconds = time.perf_counter() - forest_started

    similarity_started = time.perf_counter()
    similarity = np.asarray(forest.compute_similarity_matrix(data), dtype=float)
    similarity = np.clip(0.5 * (similarity + similarity.T), 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    similarity_seconds = time.perf_counter() - similarity_started

    config = Config(
        embedding_dim=EMBEDDING_DIM,
        graph_neighbors=GRAPH_NEIGHBORS,
    )
    distances, graph_metadata = geodesic_distances(similarity, config)
    coordinates, mds_metadata = classical_mds(distances, EMBEDDING_DIM)
    if SCALE_EMBEDDING:
        coordinates = StandardScaler().fit_transform(coordinates)

    return coordinates, {
        "head_seconds": time.perf_counter() - started,
        "forest_fit_seconds": forest_seconds,
        "similarity_seconds": similarity_seconds,
        "graph_seconds": graph_metadata["graph_seconds"],
        "shortest_path_seconds": graph_metadata["shortest_path_seconds"],
        "mds_seconds": mds_metadata["mds_seconds"],
    }


def embed_umap(data: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    import umap

    started = time.perf_counter()
    coordinates = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=EMBEDDING_DIM,
        metric=UMAP_METRIC,
        random_state=seed,
        n_jobs=1,
    ).fit_transform(data)
    if SCALE_EMBEDDING:
        coordinates = StandardScaler().fit_transform(coordinates)
    return coordinates, {"head_seconds": time.perf_counter() - started}


def embed_pca(data: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, float]]:
    started = time.perf_counter()
    coordinates = PCA(
        n_components=EMBEDDING_DIM,
        whiten=False,
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(data)
    if SCALE_EMBEDDING:
        coordinates = StandardScaler().fit_transform(coordinates)
    return coordinates, {"head_seconds": time.perf_counter() - started}


def run_task(payload: tuple[str, int, int]) -> dict[str, Any]:
    method, n_samples, seed = payload
    base = {
        "method": method,
        "n_samples": n_samples,
        "dimension": DIMENSION,
        "seed": seed,
        "embedding_dim": EMBEDDING_DIM,
        "k_min": K_MIN,
        "k_max": K_MAX,
    }
    task_started = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            data = make_isotropic_gaussian(n_samples, DIMENSION, seed)
            if method == "autog2m2":
                coordinates, timing = embed_autog2m2(data, seed)
            elif method == "umap":
                coordinates, timing = embed_umap(data, seed)
            elif method == "pca":
                coordinates, timing = embed_pca(data, seed)
            else:
                raise ValueError(method)

            _, fit = fit_autogmm(coordinates, seed, K_MIN, K_MAX)

        return {
            **base,
            "status": "ok",
            "error": np.nan,
            **timing,
            "backend_seconds": fit["backend_seconds"],
            "total_seconds": timing["head_seconds"] + fit["backend_seconds"],
            "selected_components": fit["selected_components"],
            "selected_covariance": fit["selected_covariance"],
            "bic": fit["bic"],
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": str(exc),
            "total_seconds": time.perf_counter() - task_started,
        }


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RESULT_COLUMNS)
    frame = pd.read_csv(path)
    missing = set(RESULT_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    return frame.loc[:, RESULT_COLUMNS]


def save_results(frame: pd.DataFrame, path: Path) -> None:
    keys = ["method", "n_samples", "dimension", "seed"]
    frame = frame.drop_duplicates(keys, keep="last").sort_values(keys)
    frame.to_csv(path, index=False)


def add_result(frame: pd.DataFrame, row: dict[str, Any]) -> pd.DataFrame:
    addition = pd.DataFrame(
        [{column: row.get(column, np.nan) for column in RESULT_COLUMNS}],
        columns=RESULT_COLUMNS,
    )
    if frame.empty:
        return addition
    return pd.concat([frame, addition], ignore_index=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    usable = frame.loc[frame["status"].eq("ok")].copy()
    return (
        usable.groupby(["method", "n_samples", "dimension"], as_index=False)
        .agg(
            n_fits=("seed", "nunique"),
            head_median=("head_seconds", "median"),
            head_q25=("head_seconds", lambda x: x.quantile(0.25)),
            head_q75=("head_seconds", lambda x: x.quantile(0.75)),
            backend_median=("backend_seconds", "median"),
            backend_q25=("backend_seconds", lambda x: x.quantile(0.25)),
            backend_q75=("backend_seconds", lambda x: x.quantile(0.75)),
            total_median=("total_seconds", "median"),
            total_q25=("total_seconds", lambda x: x.quantile(0.25)),
            total_q75=("total_seconds", lambda x: x.quantile(0.75)),
        )
        .sort_values(["method", "n_samples"])
    )


def main() -> int:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "figure9_runtime_results.csv"
    summary_path = args.results_dir / "figure9_runtime_summary.csv"

    results = load_results(results_path)
    done = set(
        zip(
            results["method"],
            results["n_samples"],
            results["dimension"],
            results["seed"],
        )
    )
    tasks = [
        (method, n_samples, seed)
        for n_samples in args.n_values
        for seed in args.seeds
        for method in args.methods
        if (method, n_samples, DIMENSION, seed) not in done
    ]

    def record(row: dict[str, Any]) -> None:
        nonlocal results
        results = add_result(results, row)
        save_results(results, results_path)
        print(
            f"{row['method']} n={row['n_samples']} seed={row['seed']}: {row['status']}",
            flush=True,
        )

    if args.jobs == 1:
        for task in tasks:
            record(run_task(task))
    else:
        parallel_tasks = [
            task for task in tasks if not (task[0] == "autog2m2" and task[1] >= 5000)
        ]
        serial_tasks = [task for task in tasks if task not in parallel_tasks]
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_task, task) for task in parallel_tasks]
            for future in as_completed(futures):
                record(future.result())
        for task in serial_tasks:
            record(run_task(task))

    results = load_results(results_path)
    summarize(results).to_csv(summary_path, index=False)
    successful = int(results["status"].eq("ok").sum())
    failed = int(results["status"].eq("error").sum())
    print(f"Wrote {len(results):,} rows to {results_path}")
    print(f"Successful fits: {successful}; failed fits: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

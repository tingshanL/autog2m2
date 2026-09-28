#!/usr/bin/env python3
"""Run and plot the two anisotropic experiments in manuscript Figure 6."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from autogmm_Apr26 import AutoGMM
from mclust_utils import MclustRunner


LEFT_METHODS = (
    "autogmm_euclidean_ward",
    "autogmm_mahalanobis_ward",
    "autogmm_full",
    "mclust",
)
RIGHT_METHODS = ("autogmm_full", "mclust")
LEFT_LABELS = {
    "autogmm_euclidean_ward": "AutoGMM\n(Euc-Ward only)",
    "autogmm_mahalanobis_ward": "AutoGMM\n(Mah-Ward only)",
    "autogmm_full": "AutoGMM\n(Full)",
    "mclust": "mclust",
}
RIGHT_LABELS = {
    "autogmm_full": "AutoGMM\n(Full)",
    "mclust": "mclust",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figure6"))
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-reps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--phase",
        choices=("all", "initialization", "dimension"),
        default="all",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two repetitions at dimensions 2 and 4 in a separate output folder.",
    )
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".csv",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
        newline="",
    ) as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
    os.replace(temporary, path)


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def anisotropic_shared_covariance(
    seed: int,
    rep: int,
    d: int = 10,
    n_per: int = 150,
    mu: float = 3.0,
    cov_small: float = 1.0,
    cov_large: float = 50.0,
    fill_var: float = 1.0,
    rotate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Two Gaussian components with a shared anisotropic covariance."""

    rng = np.random.default_rng(seed + rep)
    eigenvalues = np.concatenate(
        ([cov_small, cov_large], np.full(max(d - 2, 0), fill_var))
    )
    if rotate:
        basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
        if np.linalg.det(basis) < 0:
            basis[:, 0] *= -1
    else:
        basis = np.eye(d)
    covariance = (basis * eigenvalues) @ basis.T
    mean_left = np.zeros(d)
    mean_right = np.zeros(d)
    mean_left[0] = -mu
    mean_right[0] = mu
    left = rng.multivariate_normal(mean_left, covariance, size=n_per)
    right = rng.multivariate_normal(mean_right, covariance, size=n_per)
    X = np.vstack((left, right))
    y = np.repeat((0, 1), n_per)
    return X, y


def lifted_anisotropic_blobs(
    random_state: int,
    expand_dim: int,
    n_samples: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift the notebook's transformed two-dimensional blobs to 2*expand_dim."""

    X, y = make_blobs(
        n_samples=n_samples,
        random_state=random_state,
        centers=3,
    )
    transform = np.asarray([[0.6, -0.6], [-0.4, 0.8]])
    blocks = [X @ transform]
    for index in range(expand_dim - 1):
        nuisance, _ = make_blobs(
            n_samples=n_samples,
            random_state=index * random_state,
        )
        blocks.append(nuisance @ transform)
    return np.concatenate(blocks, axis=1), y


def fit_autogmm(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    method: str,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "min_components": 1,
        "max_components": 5,
        "random_state": random_state,
        "agglom_linkages": ["ward"],
        "n_jobs": 1,
    }
    if method == "autogmm_euclidean_ward":
        parameters.update(
            n_init_kmeans=0,
            agglom_affinities=["euclidean"],
        )
    elif method == "autogmm_mahalanobis_ward":
        parameters.update(
            n_init_kmeans=0,
            agglom_affinities=["mahalanobis"],
        )
    elif method == "autogmm_full":
        parameters.update(n_init_kmeans=1)
    else:
        raise ValueError(f"Unknown AutoGMM method: {method}")

    started = time.perf_counter()
    fitted = AutoGMM(**parameters)
    labels = fitted.fit_predict(X)
    elapsed = time.perf_counter() - started
    best_model = getattr(fitted, "best_model_", None)
    return {
        "method": method,
        "ari": float(adjusted_rand_score(y, labels)),
        "selected_components": int(np.unique(labels).size),
        "selected_covariance": getattr(best_model, "covariance_type", None),
        "elapsed_seconds": float(elapsed),
        "status": "ok",
        "error_type": None,
        "error_message": None,
    }


def safe_autogmm(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    method: str,
) -> dict[str, Any]:
    try:
        return fit_autogmm(X, y, random_state, method)
    except Exception as exc:
        return {
            "method": method,
            "ari": np.nan,
            "selected_components": np.nan,
            "selected_covariance": None,
            "elapsed_seconds": np.nan,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def left_autogmm_rep(seed: int, rep: int) -> list[dict[str, Any]]:
    X, y = anisotropic_shared_covariance(seed, rep)
    rows = []
    for method in LEFT_METHODS[:-1]:
        row = safe_autogmm(X, y, seed + rep, method)
        row.update(phase="initialization", rep=rep, dimension=X.shape[1])
        rows.append(row)
    return rows


def right_autogmm_rep(seed: int, expand_dim: int, rep: int) -> dict[str, Any]:
    X, y = lifted_anisotropic_blobs(seed + rep, expand_dim)
    row = safe_autogmm(X, y, seed + rep, "autogmm_full")
    row.update(
        phase="dimension",
        rep=rep,
        expand_dim=expand_dim,
        dimension=X.shape[1],
    )
    return row


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def completed_keys(rows: list[dict[str, Any]], columns: list[str]) -> set[tuple[Any, ...]]:
    return {
        tuple(row[column] for column in columns)
        for row in rows
        if row.get("status") == "ok"
    }


def replace_row(
    rows: list[dict[str, Any]],
    columns: list[str],
    row: dict[str, Any],
) -> None:
    """Replace an earlier attempt for the same experimental condition."""

    key = tuple(row[column] for column in columns)
    rows[:] = [
        existing
        for existing in rows
        if tuple(existing[column] for column in columns) != key
    ]
    rows.append(row)


def chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def run_initialization(
    path: Path,
    seed: int,
    n_reps: int,
    jobs: int,
    batch_size: int,
) -> None:
    rows = read_rows(path)
    done = completed_keys(rows, ["rep", "method"])
    mclust = MclustRunner()
    pending = [
        rep
        for rep in range(n_reps)
        if any((rep, method) not in done for method in LEFT_METHODS)
    ]
    for batch in chunks(pending, batch_size):
        auto_batches = Parallel(n_jobs=jobs)(
            delayed(left_autogmm_rep)(seed, rep) for rep in batch
        )
        for rep, auto_rows in zip(batch, auto_batches):
            for row in auto_rows:
                key = (rep, row["method"])
                if key not in done:
                    replace_row(rows, ["rep", "method"], row)
                    if row["status"] == "ok":
                        done.add(key)
            if (rep, "mclust") not in done:
                X, y = anisotropic_shared_covariance(seed, rep)
                try:
                    # The submitted notebook used G=1,...,4 in this panel.
                    fitted = mclust.fit(X, range(1, 5))
                    row = {
                        "phase": "initialization",
                        "rep": rep,
                        "dimension": X.shape[1],
                        "method": "mclust",
                        "ari": float(adjusted_rand_score(y, fitted.labels)),
                        "selected_components": fitted.selected_components,
                        "selected_covariance": fitted.selected_model,
                        "elapsed_seconds": fitted.elapsed_seconds,
                        "status": "ok",
                        "error_type": None,
                        "error_message": None,
                    }
                except Exception as exc:
                    row = {
                        "phase": "initialization",
                        "rep": rep,
                        "dimension": X.shape[1],
                        "method": "mclust",
                        "ari": np.nan,
                        "selected_components": np.nan,
                        "selected_covariance": None,
                        "elapsed_seconds": np.nan,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                replace_row(rows, ["rep", "method"], row)
                if row["status"] == "ok":
                    done.add((rep, "mclust"))
        frame = pd.DataFrame(rows).sort_values(["rep", "method"])
        atomic_write_csv(frame, path)
        print(f"Initialization panel: {len(done)}/{n_reps * len(LEFT_METHODS)} fits")


def run_dimension(
    path: Path,
    seed: int,
    n_reps: int,
    expand_dims: tuple[int, ...],
    jobs: int,
    batch_size: int,
) -> None:
    rows = read_rows(path)
    done = completed_keys(rows, ["expand_dim", "rep", "method"])
    mclust = MclustRunner()
    total = len(expand_dims) * n_reps * len(RIGHT_METHODS)
    for expand_dim in expand_dims:
        pending = [
            rep
            for rep in range(n_reps)
            if any((expand_dim, rep, method) not in done for method in RIGHT_METHODS)
        ]
        for batch in chunks(pending, batch_size):
            auto_rows = Parallel(n_jobs=jobs)(
                delayed(right_autogmm_rep)(seed, expand_dim, rep) for rep in batch
            )
            for rep, auto_row in zip(batch, auto_rows):
                auto_key = (expand_dim, rep, "autogmm_full")
                if auto_key not in done:
                    replace_row(
                        rows,
                        ["expand_dim", "rep", "method"],
                        auto_row,
                    )
                    if auto_row["status"] == "ok":
                        done.add(auto_key)
                mclust_key = (expand_dim, rep, "mclust")
                if mclust_key not in done:
                    X, y = lifted_anisotropic_blobs(seed + rep, expand_dim)
                    try:
                        fitted = mclust.fit(X, range(1, 6))
                        mclust_row = {
                            "phase": "dimension",
                            "rep": rep,
                            "expand_dim": expand_dim,
                            "dimension": X.shape[1],
                            "method": "mclust",
                            "ari": float(adjusted_rand_score(y, fitted.labels)),
                            "selected_components": fitted.selected_components,
                            "selected_covariance": fitted.selected_model,
                            "elapsed_seconds": fitted.elapsed_seconds,
                            "status": "ok",
                            "error_type": None,
                            "error_message": None,
                        }
                    except Exception as exc:
                        mclust_row = {
                            "phase": "dimension",
                            "rep": rep,
                            "expand_dim": expand_dim,
                            "dimension": X.shape[1],
                            "method": "mclust",
                            "ari": np.nan,
                            "selected_components": np.nan,
                            "selected_covariance": None,
                            "elapsed_seconds": np.nan,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    replace_row(
                        rows,
                        ["expand_dim", "rep", "method"],
                        mclust_row,
                    )
                    if mclust_row["status"] == "ok":
                        done.add(mclust_key)
            frame = pd.DataFrame(rows).sort_values(["expand_dim", "rep", "method"])
            atomic_write_csv(frame, path)
            print(f"Dimension panel: {len(done)}/{total} fits")


def require_complete(
    frame: pd.DataFrame,
    expected: int,
    group_columns: list[str],
) -> pd.DataFrame:
    failures = frame.loc[~frame["status"].eq("ok")]
    if len(failures):
        raise RuntimeError(f"Cannot plot with failed fits:\n{failures.to_string(index=False)}")
    counts = frame.groupby(group_columns).size()
    if counts.empty or not counts.eq(expected).all():
        raise RuntimeError(f"Incomplete result groups:\n{counts.to_string()}")
    return frame


def plot_initialization(frame: pd.DataFrame, output_dir: Path, n_reps: int) -> None:
    frame = require_complete(frame, n_reps, ["method"])
    plot_frame = frame.copy()
    plot_frame["Method"] = plot_frame["method"].map(LEFT_LABELS)
    order = [LEFT_LABELS[method] for method in LEFT_METHODS]
    colors = ["#FA8072", "#B22222", "#8B0000", "#808000"]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.stripplot(
        data=plot_frame,
        x="Method",
        y="ari",
        order=order,
        hue="Method",
        hue_order=order,
        palette=colors,
        jitter=0.25,
        size=4,
        alpha=0.85,
        linewidth=0.3,
        legend=False,
        ax=ax,
    )
    sns.despine(ax=ax, top=True, right=True)
    ax.set_ylabel("ARI", fontsize=20)
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_yticks([0, 0.5, 1])
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"aniso_plus.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dimension(frame: pd.DataFrame, output_dir: Path, n_reps: int) -> None:
    frame = require_complete(frame, n_reps, ["dimension", "method"])
    summary = (
        frame.groupby(["dimension", "method"], as_index=False)["ari"]
        .agg(
            median="median",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        )
        .sort_values(["method", "dimension"])
    )
    summary.to_csv(output_dir / "figure6_dimension_summary.csv", index=False)
    colors = {"autogmm_full": "#8B0000", "mclust": "#808000"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in RIGHT_METHODS:
        values = summary.loc[summary["method"].eq(method)]
        x = values["dimension"].to_numpy()
        med = values["median"].to_numpy()
        q25 = values["q25"].to_numpy()
        q75 = values["q75"].to_numpy()
        ax.plot(
            x,
            med,
            color=colors[method],
            marker="o",
            linewidth=2,
            label=RIGHT_LABELS[method],
        )
        ax.fill_between(x, q25, q75, color=colors[method], alpha=0.18)
    sns.despine(ax=ax, top=True, right=True)
    ax.set_ylabel("ARI", fontsize=20)
    ax.set_xlabel("Dimension", fontsize=20)
    ax.set_xticks(sorted(frame["dimension"].unique()))
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=16)
    ax.legend(title=None, fontsize=16, frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"high_dim.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.jobs < 1 or args.batch_size < 1 or args.n_reps < 1:
        raise ValueError("jobs, batch-size, and n-reps must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    n_reps = 2 if args.smoke else args.n_reps
    expand_dims = (1, 2) if args.smoke else tuple(range(1, 11))
    if args.smoke:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "seed": args.seed,
        "n_reps": n_reps,
        "expand_dims": list(expand_dims),
        "left_data": {
            "d": 10,
            "n_per": 150,
            "mu": 3.0,
            "cov_small": 1.0,
            "cov_large": 50.0,
            "fill_var": 1.0,
            "rotate": True,
        },
        "left_component_ranges": {
            "autogmm": [1, 2, 3, 4, 5],
            "mclust": [1, 2, 3, 4],
        },
        "right_component_ranges": {
            "autogmm": [1, 2, 3, 4, 5],
            "mclust": [1, 2, 3, 4, 5],
        },
        "autogmm_backend": "bundled autogmm_Apr26.py",
        "python": platform.python_version(),
        "packages": {
            name: package_version(name)
            for name in (
                "joblib",
                "matplotlib",
                "numpy",
                "pandas",
                "rpy2",
                "scikit-learn",
                "seaborn",
            )
        },
    }
    config_path = output_dir / "figure6_config.json"
    if config_path.is_file():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        comparable_keys = (
            "seed",
            "n_reps",
            "expand_dims",
            "left_data",
            "left_component_ranges",
            "right_component_ranges",
        )
        if any(previous.get(key) != config.get(key) for key in comparable_keys):
            raise RuntimeError(
                f"Existing results use a different configuration: {config_path}"
            )
    write_json(config, config_path)

    left_path = output_dir / "figure6_anisotropic_initialization_results.csv"
    right_path = output_dir / "figure6_anisotropic_dimension_results.csv"
    if args.phase in ("all", "initialization"):
        run_initialization(
            left_path,
            args.seed,
            n_reps,
            args.jobs,
            args.batch_size,
        )
        plot_initialization(pd.read_csv(left_path), output_dir, n_reps)
    if args.phase in ("all", "dimension"):
        run_dimension(
            right_path,
            args.seed,
            n_reps,
            expand_dims,
            args.jobs,
            args.batch_size,
        )
        plot_dimension(pd.read_csv(right_path), output_dir, n_reps)
    print(f"Figure 6 outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

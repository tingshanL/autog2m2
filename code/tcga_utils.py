"""Shared input and clustering functions for the TCGA experiment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_SAMPLES = 801
EXPECTED_GENES = 20531
EXPECTED_CLASSES = ("PRAD", "LUAD", "BRCA", "KIRC", "COAD")


@dataclass(frozen=True)
class TCGAConfig:
    embedding_dim: int = 20
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1


def scientific_config() -> TCGAConfig:
    return TCGAConfig()


def _is_index_column(values: pd.Series, name: str) -> bool:
    lowered = str(name).strip().lower()
    if lowered.startswith("unnamed") or lowered in {
        "id",
        "index",
        "sample",
        "sample_id",
        "sampleid",
    }:
        return True
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        observed = numeric.to_numpy(dtype=float)
        return bool(
            np.array_equal(observed, np.arange(len(observed), dtype=float))
            or np.array_equal(observed, np.arange(1, len(observed) + 1, dtype=float))
        )
    return False


def load_tcga_inputs(
    data_path: Path, labels_path: Path
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    data = pd.read_csv(data_path)
    labels = pd.read_csv(labels_path)

    data_has_id = _is_index_column(data.iloc[:, 0], str(data.columns[0]))
    data_ids = data.iloc[:, 0].astype(str).to_numpy() if data_has_id else None
    features = data.iloc[:, 1:] if data_has_id else data
    X = features.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)

    labels_have_id = _is_index_column(labels.iloc[:, 0], str(labels.columns[0]))
    label_ids = labels.iloc[:, 0].astype(str).to_numpy() if labels_have_id else None
    candidates = [
        column
        for column in labels.columns
        if str(column).strip().lower()
        in {"class", "label", "labels", "cancer", "cancer_type", "type"}
    ]
    if len(candidates) == 1:
        label_column = candidates[0]
    else:
        remaining = list(labels.columns[1:] if labels_have_id else labels.columns)
        if len(remaining) != 1:
            raise ValueError("The label file must contain one class column")
        label_column = remaining[0]
    text_labels = labels[label_column].astype(str).str.strip().str.upper().to_numpy()

    alignment = "row_order"
    if data_ids is not None and label_ids is not None:
        lookup = dict(zip(label_ids, text_labels))
        text_labels = np.asarray([lookup[sample] for sample in data_ids])
        alignment = "sample_identifier"

    if X.shape != (EXPECTED_SAMPLES, EXPECTED_GENES):
        raise ValueError(f"Expected a {(EXPECTED_SAMPLES, EXPECTED_GENES)} expression matrix")
    if set(text_labels) != set(EXPECTED_CLASSES):
        raise ValueError(f"Expected classes: {', '.join(EXPECTED_CLASSES)}")

    encoding = {name: index for index, name in enumerate(EXPECTED_CLASSES)}
    y = np.asarray([encoding[label] for label in text_labels], dtype=int)
    info = {
        "alignment": alignment,
        "label_column": str(label_column),
        "class_order_for_integer_encoding": list(EXPECTED_CLASSES),
        "class_counts": {
            name: int(np.count_nonzero(text_labels == name)) for name in EXPECTED_CLASSES
        },
    }
    return X, y, [str(name) for name in features.columns], info


def backend_parameters(config: TCGAConfig, seed: int) -> dict[str, Any]:
    return {
        "min_components": 1,
        "max_components": 10,
        "init_agglomerative": True,
        "agglom_linkages": ["ward"],
        "agglom_affinities": ["mahalanobis"],
        "n_init_kmeans": 1,
        "eigen_thres": True,
        "reg_covar": 1e-6,
        "criterion": "bic",
        "random_state": seed,
        "verbose": False,
        "n_jobs": 1,
        "early_stop_delta": 0.0,
        "covariances": ["full", "diag", "tied", "spherical"],
    }


def fit_matched_backend(
    coordinates: np.ndarray, seed: int, config: TCGAConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        from autogmm import AutoGMM
        external_package = True
    except ImportError:
        from autogmm_Apr26 import AutoGMM
        external_package = False

    started = time.perf_counter()
    parameters = backend_parameters(config, seed)
    if not external_package:
        parameters.pop("covariances")
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

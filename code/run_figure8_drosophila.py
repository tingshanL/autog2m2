#!/usr/bin/env python3
"""Run and plot the Drosophila connectome analysis in manuscript Figure 8."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
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
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

from autogmm_Apr26 import AutoGMM
from mclust_utils import MclustRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding",
        type=Path,
        default=Path("data/drosophila/embedded_right.csv"),
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("data/drosophila/classes.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figure8"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def contingency(y_reference: np.ndarray, y_predicted: np.ndarray):
    reference_values = np.unique(y_reference)
    predicted_values = np.unique(y_predicted)
    counts = np.zeros((reference_values.size, predicted_values.size), dtype=int)
    for row, reference in enumerate(reference_values):
        mask = y_reference == reference
        for column, predicted in enumerate(predicted_values):
            counts[row, column] = np.sum(mask & (y_predicted == predicted))
    return counts, reference_values, predicted_values


def permute_to_reference(y_predicted: np.ndarray, y_reference: np.ndarray):
    counts, reference_values, predicted_values = contingency(
        y_reference, y_predicted
    )
    rows, columns = linear_sum_assignment(counts.max() - counts)
    mapping = {
        predicted_values[column]: reference_values[row]
        for row, column in zip(rows, columns)
        if row < len(reference_values) and column < len(predicted_values)
    }
    for predicted in predicted_values:
        if predicted not in mapping:
            column = np.where(predicted_values == predicted)[0][0]
            mapping[predicted] = reference_values[np.argmax(counts[:, column])]
    mapped = np.asarray([mapping[label] for label in y_predicted])
    return mapped, mapping


def permute_with_single_split(y_predicted: np.ndarray, y_reference: np.ndarray):
    counts, reference_values, predicted_values = contingency(
        y_reference, y_predicted
    )
    n_reference, n_predicted = counts.shape
    if n_predicted != n_reference + 1:
        raise RuntimeError(
            "The submitted Figure 8 layout requires mclust to select exactly "
            "one more cluster than the four reference classes."
        )
    best = None
    for row in range(n_reference):
        expanded = np.vstack([counts, counts[row]])
        row_labels = list(reference_values) + [reference_values[row]]
        assigned_rows, assigned_columns = linear_sum_assignment(
            expanded.max() - expanded
        )
        score = expanded[assigned_rows, assigned_columns].sum()
        if best is None or score > best[0]:
            split_predictions = [
                predicted_values[assigned_columns[index]]
                for index, assigned_row in enumerate(assigned_rows)
                if row_labels[assigned_row] == reference_values[row]
            ]
            best = (
                score,
                assigned_rows,
                assigned_columns,
                row_labels,
                reference_values[row],
                split_predictions,
            )
    assert best is not None
    _, assigned_rows, assigned_columns, row_labels, split_true, split_predictions = best
    mapping = {
        predicted_values[column]: row_labels[row]
        for row, column in zip(assigned_rows, assigned_columns)
    }
    mapped = np.asarray([mapping[label] for label in y_predicted])
    return mapped, mapping, (split_true, split_predictions)


def add_prediction_legend(
    ax,
    mapping,
    split_true,
    split_order,
    edge_styles,
    color_for,
    true_names,
) -> None:
    inverse = defaultdict(list)
    for predicted, reference in mapping.items():
        inverse[reference].append(predicted)
    handles = []
    labels = []
    for reference in sorted(inverse, key=lambda value: true_names[value]):
        if reference != split_true:
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    linestyle="None",
                    marker="o",
                    markersize=8,
                    markerfacecolor=color_for[reference],
                    markeredgecolor="none",
                )
            )
            labels.append(true_names[reference])
        else:
            for index, predicted in enumerate(split_order, 1):
                edge_color, edge_width = edge_styles[predicted]
                handles.append(
                    mlines.Line2D(
                        [],
                        [],
                        linestyle="None",
                        marker="o",
                        markersize=8,
                        markerfacecolor=color_for[reference],
                        markeredgecolor=edge_color,
                        markeredgewidth=edge_width,
                    )
                )
                labels.append(f"{true_names[reference]}-{index}")
    ax.legend(
        handles,
        labels,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        title="Predicted",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
        borderaxespad=0.2,
        handletextpad=0.6,
        columnspacing=1.0,
    )


def main() -> int:
    args = parse_args()
    embedding_path = args.embedding.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_csv(embedding_path).to_numpy(dtype=float)
    y = pd.read_csv(labels_path).iloc[:, 0].to_numpy()
    if X.shape != (213, 6) or y.shape != (213,):
        raise ValueError(f"Expected X=(213, 6) and y=(213,), observed {X.shape} and {y.shape}")
    if np.unique(y).size != 4:
        raise ValueError("Expected four Drosophila reference classes")

    from graspologic.embed import AdjacencySpectralEmbed

    started = time.perf_counter()
    autogmm = AutoGMM(
        min_components=1,
        max_components=5,
        agglom_linkages=["ward"],
        agglom_affinities=["mahalanobis"],
        random_state=args.seed,
        n_jobs=1,
    )
    labels_autogmm = autogmm.fit_predict(X)
    autogmm_seconds = time.perf_counter() - started
    ari_autogmm = float(adjusted_rand_score(y, labels_autogmm))

    mclust = MclustRunner().fit(X, range(1, 6))
    labels_mclust = mclust.labels
    ari_mclust = float(adjusted_rand_score(y, labels_mclust))

    # The submitted notebook visualized two coordinates from an ASE of the
    # squared-Euclidean matrix while clustering the original six coordinates.
    squared_distances = squareform(pdist(X, metric="sqeuclidean"))
    ase = AdjacencySpectralEmbed(n_elbows=3)
    transformed = np.asarray(ase.fit_transform(squared_distances))
    if transformed.ndim != 2 or transformed.shape[1] < 5:
        raise RuntimeError(f"ASE returned an unexpected shape: {transformed.shape}")
    plot_coordinates = transformed[:, 3:5]

    mapped_autogmm, map_autogmm = permute_to_reference(labels_autogmm, y)
    mapped_mclust, map_mclust, (split_true, split_predictions) = (
        permute_with_single_split(labels_mclust, y)
    )
    assignments = pd.DataFrame(
        {
            "ase_dimension_1": plot_coordinates[:, 0],
            "ase_dimension_2": plot_coordinates[:, 1],
            "reference": y,
            "autogmm_raw": labels_autogmm,
            "autogmm_mapped": mapped_autogmm,
            "mclust_raw": labels_mclust,
            "mclust_mapped": mapped_mclust,
        }
    )
    assignments.to_csv(output_dir / "figure8_assignments.csv", index=False)

    best_model = getattr(autogmm, "best_model_", None)
    results = pd.DataFrame(
        [
            {
                "method": "AutoGMM-Euclidean",
                "ari": ari_autogmm,
                "selected_components": int(np.unique(labels_autogmm).size),
                "selected_covariance": getattr(best_model, "covariance_type", None),
                "elapsed_seconds": autogmm_seconds,
            },
            {
                "method": "mclust",
                "ari": ari_mclust,
                "selected_components": mclust.selected_components,
                "selected_covariance": mclust.selected_model,
                "elapsed_seconds": mclust.elapsed_seconds,
            },
        ]
    )
    results.to_csv(output_dir / "figure8_results.csv", index=False)

    true_ids = np.unique(y)
    marker_values = ("o", "^", "s", "*")
    true_markers = dict(zip(true_ids, marker_values))
    true_names = dict(zip(true_ids, ("KC", "MBIN", "MBON", "PN")))
    palette = ("#00C853", "#008AA2", "#FF8F00", "#8D6E63")
    color_for = {value: palette[index] for index, value in enumerate(true_ids)}

    edge_styles = {
        predicted: ("none", 0.0) for predicted in np.unique(labels_mclust)
    }
    counts, reference_values, predicted_values = contingency(y, labels_mclust)
    split_row = np.where(reference_values == split_true)[0][0]
    overlaps = {
        predicted: counts[
            split_row, np.where(predicted_values == predicted)[0][0]
        ]
        for predicted in split_predictions
    }
    split_order = sorted(
        split_predictions,
        key=lambda value: overlaps[value],
        reverse=True,
    )
    for predicted, style in zip(split_order, (("#7F32A0", 2), ("#615F1C", 2))):
        edge_styles[predicted] = style

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.05)
    panels = (
        (
            f"AutoGMM (ARI={ari_autogmm:.2f})",
            "autogmm_raw",
            map_autogmm,
            None,
        ),
        (
            f"mclust (ARI={ari_mclust:.2f})",
            "mclust_raw",
            map_mclust,
            edge_styles,
        ),
    )
    for ax, (title, raw_column, mapping, panel_edges) in zip(axes, panels):
        for reference in true_ids:
            marker = true_markers[reference]
            marker_size = {"o": 30, "*": 90, "s": 40}.get(marker, 60)
            for predicted in np.unique(assignments[raw_column]):
                subset = assignments.loc[
                    assignments["reference"].eq(reference)
                    & assignments[raw_column].eq(predicted)
                ]
                if subset.empty:
                    continue
                face_color = color_for[mapping[predicted]]
                edge_color, edge_width = (
                    ("none", 0.0)
                    if panel_edges is None
                    else panel_edges.get(predicted, ("none", 0.0))
                )
                ax.scatter(
                    subset["ase_dimension_1"],
                    subset["ase_dimension_2"],
                    c=[face_color],
                    marker=marker,
                    s=marker_size,
                    alpha=0.9,
                    edgecolors=edge_color,
                    linewidth=edge_width,
                    zorder=2,
                )
        ax.set_title(title, fontsize=17)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Dimension 1", fontsize=15)
    axes[0].set_ylabel("Dimension 2", fontsize=15)

    true_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker=true_markers[value],
            linestyle="None",
            markersize=12 if true_markers[value] in {"^", "*"} else 8,
        )
        for value in true_ids
    ]
    fig.legend(
        true_handles,
        [true_names[value] for value in true_ids],
        frameon=False,
        loc="upper center",
        ncol=1,
        bbox_to_anchor=(0.95, 0.65),
        title="True classes",
        fontsize=13.5,
        title_fontsize=13,
    )
    add_prediction_legend(
        axes[0],
        map_autogmm,
        None,
        [],
        {},
        color_for,
        true_names,
    )
    add_prediction_legend(
        axes[1],
        map_mclust,
        split_true,
        split_order,
        edge_styles,
        color_for,
        true_names,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"real_droso.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    run_config = {
        "python": platform.python_version(),
        "seed": args.seed,
        "embedding_path": str(embedding_path),
        "labels_path": str(labels_path),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "autogmm_backend": "bundled autogmm_Apr26.py",
        "packages": {
            name: package_version(name)
            for name in (
                "graspologic",
                "matplotlib",
                "numpy",
                "pandas",
                "rpy2",
                "scikit-learn",
                "scipy",
            )
        },
    }
    (output_dir / "figure8_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Figure 8 outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

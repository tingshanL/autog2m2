# run_runtime_isotropic.py
"""
Runtime scaling on isotropic Gaussian data (checkpointed):
  - UMAP + GMM (BIC over K and covariance types)
  - AutoG2M2 (apply head) = URF apply -> sparse top-k leaf cooccurrence graph
                           -> SpectralEmbedding -> GMM(BIC)

Hard-coded: n_list = [100, 500, 1000, 5000, 10000], fixed d.
Saves incrementally to CSV after each (n, seed, method) finishes.
Automatically resumes by skipping rows already present in the CSV.

Requirements:
  pip install umap-learn
  treeple installed (already if your other scripts run)
  geoforest_subspace.py importable (for URF class + defaults)
"""

from __future__ import annotations

import os
import gc
import time
import warnings
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import SpectralEmbedding
from sklearn.exceptions import ConvergenceWarning

import umap
from scipy.sparse import csr_matrix

# We reuse the URF class (and criterion name) via your helper:
from geoforest_subspace import make_forest_cfg

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Graph is not fully connected")  # spectral embedding


# -----------------------------
# Hard-coded experiment config
# -----------------------------
OUT_DIR = "results/figure9"
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]                   # keep small for runtime
N_LIST = [100, 500, 1000, 5000, 10000]
D = 50

# --- UMAP params ---
UMAP_DIM = 10
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "euclidean"

# --- GMM(BIC) selection (same for both) ---
KMIN = 1
KMAX = 10
COV_TYPES = ("full", "diag", "tied", "spherical")
GMM_N_INIT = 3
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-6
SCALE_EMBEDDING = True

# --- AutoG2M2 apply-head params ---
GF_N_ESTIMATORS = 300
GF_N_JOBS = -1

# key: keep leaves reasonably small (protein-like), otherwise leaf-cooccurrence can explode
GF_MIN_SAMPLES_LEAF = 2
GF_MIN_SAMPLES_SPLIT = 4
GF_MAX_FEATURES = "sqrt"  # speeds up; common RF default

TOP_K = 30          # keep only top_k neighbors per node
MIN_COOCCUR = 2     # drop edges that co-occur in <2 trees
MAX_LEAF = 200      # cap leaf size to avoid O(m^2) blowups
SPEC_DIM = 10       # spectral embedding dimension

RESULT_COLUMNS = (
    "method",
    "n",
    "d",
    "seed",
    "t_head",
    "t_backend",
    "t_total",
    "t_fit",
    "t_apply",
    "t_graph",
    "t_spec",
    "umap_dim",
    "umap_neighbors",
    "umap_min_dist",
    "gf_estimators",
    "gf_min_leaf",
    "gf_min_split",
    "gf_max_features",
    "top_k",
    "min_cooccur",
    "max_leaf",
    "spec_dim",
    "kmax",
    "cov_types",
)

LEGACY_COLUMNS = (
    "method",
    "n",
    "d",
    "seed",
    "t_head",
    "t_backend",
    "t_total",
    "umap_dim",
    "umap_neighbors",
    "umap_min_dist",
    "kmax",
    "cov_types",
)


def make_isotropic_gaussian(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X


def gmm_bic_select_labels(Z: np.ndarray, seed: int) -> np.ndarray:
    best_bic = np.inf
    best_gm = None

    for cov in COV_TYPES:
        for k in range(KMIN, KMAX + 1):
            gm = GaussianMixture(
                n_components=k,
                covariance_type=cov,
                reg_covar=GMM_REG_COVAR,
                random_state=seed,
                n_init=GMM_N_INIT,
                max_iter=GMM_MAX_ITER,
                init_params="kmeans",
            )
            try:
                gm.fit(Z)
                bic = gm.bic(Z)
            except Exception:
                continue

            if bic < best_bic:
                best_bic = bic
                best_gm = gm

    if best_gm is None:
        return np.zeros(Z.shape[0], dtype=int)
    return best_gm.predict(Z)


def runtime_umap_gmm(X: np.ndarray, seed: int) -> dict:
    t0 = time.perf_counter()
    Z = umap.UMAP(
        n_neighbors=UMAP_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        n_components=min(UMAP_DIM, X.shape[1]),
        metric=UMAP_METRIC,
        random_state=seed,  # deterministic (UMAP will use n_jobs=1)
        n_jobs=1,
    ).fit_transform(X)
    t_head = time.perf_counter() - t0

    if SCALE_EMBEDDING:
        Z = StandardScaler().fit_transform(Z)

    t1 = time.perf_counter()
    _ = gmm_bic_select_labels(Z, seed)
    t_backend = time.perf_counter() - t1

    return {"t_head": t_head, "t_backend": t_backend, "t_total": t_head + t_backend}


def build_topk_graph_from_leaves(
    leaves: np.ndarray,
    top_k: int,
    min_cooccur: int,
    max_leaf: int,
    seed: int,
) -> csr_matrix:
    """
    Build a sparse adjacency from leaf co-occurrence.
    Weight w_ij = (#trees where i & j share a leaf) / n_trees, then keep top_k per node.

    This avoids any dense n×n similarity matrix.
    """
    rng = np.random.default_rng(seed)
    n, n_trees = leaves.shape
    counts = [defaultdict(int) for _ in range(n)]

    for t in range(n_trees):
        leaf_ids = leaves[:, t]
        order = np.argsort(leaf_ids, kind="mergesort")
        leaf_sorted = leaf_ids[order]

        # group boundaries where leaf id changes
        cuts = np.flatnonzero(leaf_sorted[1:] != leaf_sorted[:-1]) + 1
        bounds = np.concatenate(([0], cuts, [n]))

        for a, b in zip(bounds[:-1], bounds[1:]):
            members = order[a:b]
            m = members.size
            if m < 2:
                continue

            if m > max_leaf:
                members = rng.choice(members, size=max_leaf, replace=False)
                m = members.size
                if m < 2:
                    continue

            # update pair counts within this leaf
            for i, j in combinations(members, 2):
                i = int(i); j = int(j)
                counts[i][j] += 1
                counts[j][i] += 1

    # keep top-k neighbors per node
    rows, cols, data = [], [], []
    denom = float(n_trees)

    for i, di in enumerate(counts):
        if not di:
            continue
        js = np.fromiter(di.keys(), dtype=np.int32)
        cs = np.fromiter(di.values(), dtype=np.int32)

        keep = cs >= min_cooccur
        js, cs = js[keep], cs[keep]
        if js.size == 0:
            continue

        k_use = min(top_k, js.size)
        idx = np.argpartition(-cs, k_use - 1)[:k_use]

        js_k = js[idx]
        ws_k = (cs[idx].astype(np.float32) / denom)

        rows.extend([i] * k_use)
        cols.extend(js_k.tolist())
        data.extend(ws_k.tolist())

    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    A = A.maximum(A.T)
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A


def runtime_autog2m2_apply_head(X: np.ndarray, seed: int) -> dict:
    n, d = X.shape

    # forest class + defaults from your helper
    rf_cls, rf_cfg = make_forest_cfg(n, d, oblique=False, random_state=seed)

    # override to protein-like settings to keep leaves small
    rf_cfg["n_estimators"] = GF_N_ESTIMATORS
    rf_cfg["n_jobs"] = GF_N_JOBS
    rf_cfg["min_samples_leaf"] = GF_MIN_SAMPLES_LEAF
    rf_cfg["min_samples_split"] = GF_MIN_SAMPLES_SPLIT
    rf_cfg["max_features"] = GF_MAX_FEATURES

    # (1) fit forest
    t0 = time.perf_counter()
    rf = rf_cls(**rf_cfg).fit(X)
    t_fit = time.perf_counter() - t0

    # (2) apply -> leaf ids
    t1 = time.perf_counter()
    leaves = rf.apply(X)  # (n, n_trees)
    t_apply = time.perf_counter() - t1

    # (3) build sparse top-k graph from leaf co-occurrence
    t2 = time.perf_counter()
    A = build_topk_graph_from_leaves(
        leaves=leaves,
        top_k=TOP_K,
        min_cooccur=MIN_COOCCUR,
        max_leaf=MAX_LEAF,
        seed=seed,
    )
    t_graph = time.perf_counter() - t2

    del leaves
    gc.collect()

    # (4) spectral embedding
    t3 = time.perf_counter()
    Z = SpectralEmbedding(
        n_components=min(SPEC_DIM, n - 1),
        affinity="precomputed",
        random_state=seed,
        n_jobs=1,
    ).fit_transform(A)
    t_spec = time.perf_counter() - t3

    if SCALE_EMBEDDING:
        Z = StandardScaler().fit_transform(Z)

    # (5) backend: GMM(BIC)
    t4 = time.perf_counter()
    _ = gmm_bic_select_labels(Z, seed)
    t_gmm = time.perf_counter() - t4

    t_head = t_fit + t_apply + t_graph + t_spec
    return {
        "t_head": t_head,
        "t_backend": t_gmm,
        "t_total": t_head + t_gmm,
        "t_fit": t_fit,
        "t_apply": t_apply,
        "t_graph": t_graph,
        "t_spec": t_spec,
    }


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the current runtime-result schema, including legacy rows."""
    if list(frame.columns) == list(RESULT_COLUMNS):
        return frame
    if list(frame.columns) != list(LEGACY_COLUMNS):
        raise ValueError(
            "Runtime results have an unrecognized column layout: "
            + ", ".join(frame.columns)
        )

    normalized = pd.DataFrame(index=frame.index, columns=RESULT_COLUMNS)
    common = ("method", "n", "d", "seed", "t_head", "t_backend", "t_total")
    normalized.loc[:, list(common)] = frame.loc[:, list(common)]

    umap_rows = frame["method"].eq("UMAP+GMM(BIC)")
    normalized.loc[umap_rows, "umap_dim"] = frame.loc[umap_rows, "umap_dim"]
    normalized.loc[umap_rows, "umap_neighbors"] = frame.loc[
        umap_rows, "umap_neighbors"
    ]
    normalized.loc[umap_rows, "umap_min_dist"] = frame.loc[
        umap_rows, "umap_min_dist"
    ]
    normalized.loc[umap_rows, "kmax"] = frame.loc[umap_rows, "kmax"]
    normalized.loc[umap_rows, "cov_types"] = frame.loc[umap_rows, "cov_types"]

    auto_rows = frame["method"].eq("AutoG2M2(apply head)+GMM(BIC)")
    normalized.loc[auto_rows, "t_fit"] = frame.loc[auto_rows, "umap_dim"]
    normalized.loc[auto_rows, "t_apply"] = frame.loc[auto_rows, "umap_neighbors"]
    normalized.loc[auto_rows, "t_graph"] = frame.loc[auto_rows, "umap_min_dist"]
    normalized.loc[auto_rows, "t_spec"] = frame.loc[auto_rows, "kmax"]

    auto_parameters = frame.loc[auto_rows, "cov_types"].astype(str).str.split(",")
    parameter_columns = (
        "gf_estimators",
        "gf_min_leaf",
        "gf_min_split",
        "gf_max_features",
        "top_k",
        "min_cooccur",
        "max_leaf",
        "spec_dim",
        "kmax",
        "cov_types",
    )
    for row_index, values in auto_parameters.items():
        if len(values) != len(parameter_columns):
            raise ValueError(f"Cannot read legacy AutoG2M2 row {row_index}")
        normalized.loc[row_index, list(parameter_columns)] = values

    return normalized


def load_results(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return normalize_results(pd.read_csv(csv_path))


def load_done_keys(csv_path: str) -> set[tuple]:
    frame = load_results(csv_path)
    return set(zip(frame["method"], frame["n"], frame["d"], frame["seed"]))


def append_row(csv_path: str, row: dict) -> None:
    existing = load_results(csv_path)
    new_row = pd.DataFrame(
        [{column: row.get(column, np.nan) for column in RESULT_COLUMNS}],
        columns=RESULT_COLUMNS,
    )
    combined = new_row if existing.empty else pd.concat(
        [existing, new_row], ignore_index=True
    )
    combined.to_csv(csv_path, index=False)


def main():
    csv_path = os.path.join(
        OUT_DIR, f"runtime_isotropic_apply_d{D}_k{KMAX}.cleaned.csv"
    )
    done = load_done_keys(csv_path)

    for seed in tqdm(SEEDS, desc="seed"):
        for n in tqdm(N_LIST, desc="n", leave=False):
            X = make_isotropic_gaussian(n, D, seed)

            # ---- UMAP+GMM ----
            key = ("UMAP+GMM(BIC)", n, D, seed)
            if key not in done:
                out = runtime_umap_gmm(X, seed)
                append_row(csv_path, {
                    "method": "UMAP+GMM(BIC)",
                    "n": n, "d": D, "seed": seed,
                    **out,
                    "umap_dim": UMAP_DIM,
                    "umap_neighbors": UMAP_NEIGHBORS,
                    "umap_min_dist": UMAP_MIN_DIST,
                    "kmax": KMAX,
                    "cov_types": "|".join(COV_TYPES),  # avoid comma-parsing issues
                })
                done.add(key)

            # ---- AutoG2M2 apply head ----
            key = ("AutoG2M2(apply head)+GMM(BIC)", n, D, seed)
            if key not in done:
                out = runtime_autog2m2_apply_head(X, seed)
                append_row(csv_path, {
                    "method": "AutoG2M2(apply head)+GMM(BIC)",
                    "n": n, "d": D, "seed": seed,
                    **out,
                    "gf_estimators": GF_N_ESTIMATORS,
                    "gf_min_leaf": GF_MIN_SAMPLES_LEAF,
                    "gf_min_split": GF_MIN_SAMPLES_SPLIT,
                    "gf_max_features": GF_MAX_FEATURES,
                    "top_k": TOP_K,
                    "min_cooccur": MIN_COOCCUR,
                    "max_leaf": MAX_LEAF,
                    "spec_dim": SPEC_DIM,
                    "kmax": KMAX,
                    "cov_types": "|".join(COV_TYPES),
                })
                done.add(key)

            del X
            gc.collect()

    print(f"\nSaved (checkpointed): {csv_path}")
    df = pd.read_csv(csv_path)
    cols = [c for c in ["t_fit","t_apply","t_graph","t_spec","t_head","t_backend","t_total"] if c in df.columns]
    print(df.groupby(["method", "n"])[cols].median().round(3).reset_index())


if __name__ == "__main__":
    main()

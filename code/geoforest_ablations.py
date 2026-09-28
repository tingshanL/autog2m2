"""
run_autogmm_mclust_sparcl_sims.py

Run 3 AutoGMM variants (Euclidean, RBF/spectral head, GeoForest-MDS-Ward)
plus mclust and sparcl on:
  - Gaussian subspace clusters
  - Heavy-tailed (Student-t) subspace clusters
  - High-dimensional swiss-roll manifold clusters

Dependencies (Python side):
    pip install autogmm treeple rpy2 scikit-learn numpy pandas

R side (must be installed in your R env):
    install.packages(c("mclust", "sparcl"))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import OAS
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll

from autogmm_Apr26 import AutoGMM
from treeple import (
    UnsupervisedRandomForest as URF,
    UnsupervisedObliqueRandomForest as OURF,
)
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import KernelPCA
from sklearn.covariance import OAS
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def _median_gamma(X: np.ndarray) -> float:
    D = pairwise_distances(X)
    iu = np.triu_indices_from(D, 1)
    tri = D[iu]
    med = np.median(tri)
    if med <= 0:
        return 1.0
    return 1.0 / (2.0 * med ** 2)


# # R side: mclust + sparcl via rpy2
# import rpy2.robjects as ro
# from rpy2.robjects import numpy2ri
# from rpy2.robjects.packages import importr


# # # activate numpy <-> R automatic conversion
# # numpy2ri.activate()
# base = importr("base")
# mclust = importr("mclust")
# sparcl = importr("sparcl")


# ---------------------------------------------------------------------
#  Data generators
# ---------------------------------------------------------------------
def make_subspace_clusters(
    n_per: int = 200,
    d: int = 50,
    k: int = 3,
    subspace_dim: int = 5,
    snr: float = 3.0,
    seed: int | None = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian subspace clusters in high dimension (your original sim).

    Each cluster r lives in its own subspace of dimension `subspace_dim`
    supported on coordinates:

        idx_r = [r * subspace_dim, ..., (r+1) * subspace_dim - 1] mod d

    within R^d. All coordinates get N(0,1) noise; we add a mean shift of
    magnitude ~snr only on the cluster's subspace coordinates.
    """
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for r in range(k):
        Xr = rng.normal(scale=1.0, size=(n_per, d))
        start = (r * subspace_dim) % d
        idx = np.arange(start, start + subspace_dim) % d
        center = rng.normal(scale=snr, size=subspace_dim)
        Xr[:, idx] += center
        X_list.append(Xr)
        y_list.append(np.full(n_per, r, dtype=int))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    X = StandardScaler().fit_transform(X)
    return X, y


def make_heavy_tail_subspace_clusters(
    n_per: int = 200,
    d: int = 50,
    k: int = 3,
    subspace_dim: int = 5,
    df: float = 3.0,
    snr: float = 3.0,
    standardize: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-Gaussian subspace clusters in high dimension.

    Each cluster lies in its own `subspace_dim`-dimensional subspace
    inside R^d, like `make_subspace_clusters`, but the base noise is
    heavy-tailed Student-t(df) instead of Gaussian.
    """
    rng = default_rng(seed)
    n = k * n_per
    X = np.empty((n, d), dtype=float)
    y = np.empty(n, dtype=int)

    for r in range(k):
        start = r * n_per
        stop = (r + 1) * n_per

        # Heavy-tailed base noise
        Xr = rng.standard_t(df, size=(n_per, d))

        # Cluster-specific signal subspace
        idx = np.arange(r * subspace_dim, (r + 1) * subspace_dim) % d
        center = rng.normal(scale=snr, size=subspace_dim)
        Xr[:, idx] += center

        X[start:stop] = Xr
        y[start:stop] = r

    if standardize:
        X = StandardScaler().fit_transform(X)

    return X, y


def make_swiss_roll_clusters_highdim(
    n_per: int = 200,
    d: int = 50,
    k: int = 3,
    roll_noise: float = 0.05,
    embed_noise: float = 0.1,
    standardize: bool = True,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nonlinear manifold clusters: swiss-roll segments embedded in R^d.

    1. Sample a 3D swiss roll with intrinsic parameter t.
    2. Sort by t and slice into k contiguous segments -> k clusters.
    3. Embed 3D coords into R^d via a random linear map + small noise.
    """
    rng = default_rng(seed)
    n = n_per * k

    X3, t = make_swiss_roll(n_samples=n, noise=roll_noise, random_state=seed)
    t = (t - t.min()) / (t.max() - t.min() + 1e-12)

    order = np.argsort(t)
    X3 = X3[order]
    # Contiguous segments along the roll
    y = np.repeat(np.arange(k, dtype=int), n_per)

    # Random linear embedding: R^3 -> R^d
    R = rng.normal(size=(3, d))
    Xd = X3 @ R
    Xd += embed_noise * rng.normal(size=Xd.shape)

    if standardize:
        Xd = StandardScaler().fit_transform(Xd)

    return Xd, y


# ---------------------------------------------------------------------
#  GeoForest: similarity + Geo-MDS-Ward (GF-MDS-Ward)
# ---------------------------------------------------------------------
@dataclass
class ForestConfig:
    n_estimators: int = 1400
    criterion: str = "twomeans"
    max_depth: int | None = None
    min_samples_leaf: int = 9
    min_samples_split: int = 18
    max_features: str | None = None
    feature_combinations: float | None = None  # OURF only
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 0


def make_forest_cfg(
    n: int,
    d: int,
    oblique: bool = False,
    random_state: int = 0,
    max_features=None,
):
    """
    Dimension-aware hyperparameter schedule.

    Returns
    -------
    rf_cls : class
        URF or OURF.
    rf_cfg : dict
        Keyword args for the constructor.
    """
    mleaf = max(3, int(0.015 * n))
    msplt = max(2 * mleaf, 6)

    if oblique:
        # Oblique forest (OURF): DO NOT pass None for feature_combinations.
        # Use a sensible default, e.g. 1.5 (average #features in each oblique split).
        mleaf = max(5, int(0.015 * n))
        rf_cls = OURF
        rf_cfg = dict(
            n_estimators=1400,
            criterion="twomeans",
            max_depth=None,
            min_samples_leaf=mleaf,
            min_samples_split=2 * mleaf,
            max_features=max_features,          # this is fine, treeple can handle None here
            feature_combinations=1.5,   # <-- key change: real number, not None
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        rf_cls = URF
        rf_cfg = dict(
            n_estimators=1400,
            criterion="twomeans",
            max_depth=None,
            min_samples_leaf=mleaf,
            min_samples_split=msplt,
            max_features=None,
            bootstrap=True,
            n_jobs=-1,
            random_state=random_state,
        )
    return rf_cls, rf_cfg



def urf_similarity_from_cfg(
    X: np.ndarray,
    rf_cls: Callable,
    rf_cfg: Dict,
) -> np.ndarray:
    """Fit forest and return symmetric similarity with diag=1."""
    rf = rf_cls(**rf_cfg).fit(X)
    S = rf.compute_similarity_matrix(X).astype(float)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 1.0)
    return S


def classical_mds_from_dist(
    D: np.ndarray,
    rel_tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical MDS from a (possibly non-metric) distance matrix.

    Returns
    -------
    Z : array, shape (n_samples, n_pos)
        Coordinates for positive eigenvalues.
    w_pos : array, shape (n_pos,)
        Corresponding positive eigenvalues (ascending).
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)  # ascending
    thr = rel_tol * (np.max(np.abs(w)) + 1e-12)
    pos = w > thr
    w_pos = w[pos]
    V_pos = V[:, pos]
    Z = V_pos * np.sqrt(np.maximum(w_pos, 0.0))
    return Z, w_pos


def geo_mds_ward_labels(
    X: np.ndarray,
    k: int,
    whiten: bool = True,
    oblique: bool = False,
    random_state: int = 0,
) -> np.ndarray:
    """
    GeoForest-MDS-Ward pipeline (GF-MDS-Ward).

    1. Fit URF (or OURF) and compute similarity S.
    2. Convert to dissimilarity D = 1 - S.
    3. Classical MDS on D.
    4. Keep top-q dimensions (q <= k-1) by eigenvalue.
    5. Optional OAS whitening in the MDS space.
    6. Ward clustering in the whitened space.
    """
    n, d = X.shape
    rf_cls, rf_cfg = make_forest_cfg(n, d, oblique=oblique, random_state=random_state)
    S = urf_similarity_from_cfg(X, rf_cls, rf_cfg)
    D = 1.0 - S

    Z_all, w_pos = classical_mds_from_dist(D)
    if Z_all.shape[1] == 0:
        # Fallback: use PCA-ish embedding (not expected often).
        warnings.warn("MDS produced no positive eigenvalues; falling back to raw X.")
        Zw = StandardScaler().fit_transform(X)
    else:
        q = min(k - 1, Z_all.shape[1])
        if q < Z_all.shape[1]:
            # select top-q eigen-dims by eigenvalue
            idx = np.argsort(w_pos)[-q:]
            Zq = Z_all[:, idx]
        else:
            Zq = Z_all

        if whiten and q > 1:
            oas = OAS().fit(Zq)
            L = np.linalg.cholesky(oas.covariance_)
            Zw = np.linalg.solve(L, (Zq - oas.location_).T).T
        else:
            Zw = Zq

    ward = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = ward.fit_predict(Zw)
    return labels


# ---------------------------------------------------------------------
#  AutoGMM variants
# ---------------------------------------------------------------------
def run_autogmm_euclid(
    X: np.ndarray,
    random_state: int = 0,
    min_components: int = 1,
    max_components: int = 10,
) -> np.ndarray:
    """Plain AutoGMM on Euclidean features."""
    model = AutoGMM(
        min_components=min_components,
        max_components=max_components,
        criterion="bic",
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    return np.asarray(labels, dtype=int)


def _median_gamma(X: np.ndarray) -> float:
    D = pairwise_distances(X)
    iu = np.triu_indices_from(D, 1)
    tri = D[iu]
    med = np.median(tri)
    if med <= 0:
        return 1.0
    return 1.0 / (2.0 * med**2)


def run_autogmm_rbf(
    X: np.ndarray,
    random_state: int = 0,
    n_components: int = 32,
    min_components: int = 1,
    max_components: int = 10,
) -> np.ndarray:
    """
    AutoGMM on an RBF KernelPCA embedding (spectral head).

    This approximates the "fixed kernel / spectral head" version.
    """
    gamma = _median_gamma(X)
    n_components = min(n_components, X.shape[1])
    kpca = KernelPCA(
        n_components=n_components,
        kernel="rbf",
        gamma=gamma,
        random_state=random_state,
    )
    Z = kpca.fit_transform(X)
    model = AutoGMM(
        min_components=min_components,
        max_components=max_components,
        criterion="bic",
        random_state=random_state,
        n_jobs=-1,
    )
    labels = model.fit_predict(Z)
    return np.asarray(labels, dtype=int)


def run_autogmm_geo_gf_mds_ward(
    X: np.ndarray,
    k_true: int,
    random_state: int = 0,
) -> np.ndarray:
    """
    Geo-AutoGMM via the GF-MDS-Ward pipeline.

    Note: This uses URF only (no oblique) for the main experiments.
    """
    labels = geo_mds_ward_labels(
        X,
        k=k_true,
        whiten=True,
        oblique=False,
        random_state=random_state,
    )
    return labels.astype(int)


# ---------------------------------------------------------------------
#  R baselines: mclust and sparcl
# ---------------------------------------------------------------------
# def run_mclust(
#     X: np.ndarray,
#     random_state: int = 0,
# ) -> np.ndarray:
#     """
#     Run mclust::Mclust on X.

#     Lets Mclust choose (G, covariance model) by BIC.
#     """
#     base.set_seed(random_state)
#     # rpy2: convert numpy -> R matrix
#     # use local converter so numpy arrays are handled
#     with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
#         r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])
#         fit = mclust.Mclust(r_X)
#     labels = np.array(fit.rx2("classification"), dtype=int) - 1  # 1-based -> 0-based
#     return labels


# def run_sparcl(
#     X: np.ndarray,
#     k_true: int,
#     random_state: int = 0,
#     nperms: int = 10,
# ) -> np.ndarray:
#     """
#     Run sparcl::KMeansSparseCluster with tuning via KMeansSparseCluster.permute.
#     """
#     base.set_seed(random_state)
#     with ro.conversion.localconverter(ro.default_converter + numpy2ri.converter):
#         r_X = ro.r.matrix(X, nrow=X.shape[0], ncol=X.shape[1])

#         # Choose tuning parameter (L1 bound) by permutation.
#         perm_out = sparcl.KMeansSparseCluster_permute(
#             r_X,
#             K=k_true,
#             nperms=nperms,
#             silent=True,
#         )
#         bestw = perm_out.rx2("bestw")

#         out = sparcl.KMeansSparseCluster(
#             r_X,
#             K=k_true,
#             wbounds=bestw,
#             nstart=10,
#             silent=True,
#         )
#     # R: sparse_km[[1]]$Cs  -> rpy2: out[0].rx2("Cs")
#     clusters = np.array(out[0].rx2("Cs"), dtype=int) - 1
#     return clusters


# ---------------------------------------------------------------------
#  Harness
# ---------------------------------------------------------------------
def evaluate_methods_on_dataset(
    X: np.ndarray,
    y_true: np.ndarray,
    seed: int = 0,
) -> Dict[str, float]:
    """Run all methods on a single (X, y) and return ARIs."""
    k_true = len(np.unique(y_true))

    aris: Dict[str, float] = {}

    # 1) AutoGMM variants
    aris["AutoGMM_Euclid"] = adjusted_rand_score(
        y_true, run_autogmm_euclid(X, random_state=seed)
    )
    aris["AutoGMM_RBF"] = adjusted_rand_score(
        y_true, run_autogmm_rbf(X, random_state=seed)
    )
    aris["AutoGMM_Geo_GF_MDS_Ward"] = adjusted_rand_score(
        y_true, run_autogmm_geo_gf_mds_ward(X, k_true=k_true, random_state=seed)
    )

    # # 2) R baselines
    # try:
    #     aris["mclust"] = adjusted_rand_score(y_true, run_mclust(X, random_state=seed))
    # except Exception as e:
    #     warnings.warn(f"mclust failed: {e}")
    #     aris["mclust"] = np.nan

    # try:
    #     aris["sparcl"] = adjusted_rand_score(
    #         y_true, run_sparcl(X, k_true=k_true, random_state=seed)
    #     )
    # except Exception as e:
    #     warnings.warn(f"sparcl failed: {e}")
    #     aris["sparcl"] = np.nan

    return aris


###
# ablations
def run_geo_head_ablation_once(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int,
) -> dict:
    """
    Ablation 1: compare different 'heads' on the same data (and same forest).

    Heads:
      - Euclid_Ward
      - RBF_KPCA_Ward
      - GF_Agglo_avg / GF_Agglo_comp
      - Geo_MDS_Ward_no_whiten
      - Geo_MDS_Ward_whiten  (canonical)
    """
    n, d = X.shape
    rng = np.random.RandomState(seed)
    out = dict(seed=seed, n=n, d=d)

    # -------- Euclid Ward baseline --------
    Xe = StandardScaler().fit_transform(X)
    ward_e = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y_euclid = ward_e.fit_predict(Xe)
    out["ARI_Euclid_Ward"] = adjusted_rand_score(y, y_euclid)

    # -------- RBF + KPCA + Ward --------
    gamma = _median_gamma(X)
    n_kpca = min(32, d)
    kpca = KernelPCA(
        n_components=n_kpca,
        kernel="rbf",
        gamma=gamma,
        random_state=seed,
    )
    Z_rbf = kpca.fit_transform(X)
    Zr = StandardScaler().fit_transform(Z_rbf)
    ward_r = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y_rbf = ward_r.fit_predict(Zr)
    out["ARI_RBF_KPCA_Ward"] = adjusted_rand_score(y, y_rbf)

    # -------- Forest similarity (shared across geo heads) --------
    rf_cls, rf_cfg = make_forest_cfg(n, d, oblique=False, random_state=seed)
    S = urf_similarity_from_cfg(X, rf_cls, rf_cfg)
    D = 1.0 - S

    # GF-Agglo (average / complete)
    agg_avg = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    y_gf_avg = agg_avg.fit_predict(D)
    out["ARI_GF_Agglo_avg"] = adjusted_rand_score(y, y_gf_avg)

    agg_comp = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="complete",
    )
    y_gf_comp = agg_comp.fit_predict(D)
    out["ARI_GF_Agglo_comp"] = adjusted_rand_score(y, y_gf_comp)

    # Classical MDS on D
    Z_all, w_pos = classical_mds_from_dist(D)
    if Z_all.shape[1] == 0:
        # Degenerate case; just bail out to NaNs for geo heads
        out["ARI_Geo_MDS_Ward_nowhite"] = np.nan
        out["ARI_Geo_MDS_Ward_white"] = np.nan
        return out

    # take top-q dims (q <= k-1)
    q = min(k - 1, Z_all.shape[1])
    idx = np.argsort(w_pos)[-q:]
    Zq = Z_all[:, idx]

    # Geo-MDS-Ward (no whitening)
    ward_g_now = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y_g_now = ward_g_now.fit_predict(Zq)
    out["ARI_Geo_MDS_Ward_nowhite"] = adjusted_rand_score(y, y_g_now)

    # Geo-MDS-Ward (with OAS whitening) -- canonical
    if q > 1:
        oas = OAS().fit(Zq)
        L = np.linalg.cholesky(oas.covariance_)
        Zw = np.linalg.solve(L, (Zq - oas.location_).T).T
    else:
        Zw = Zq

    ward_g_w = AgglomerativeClustering(n_clusters=k, linkage="ward")
    y_g_w = ward_g_w.fit_predict(Zw)
    out["ARI_Geo_MDS_Ward_white"] = adjusted_rand_score(y, y_g_w)

    return out


def run_ablation1_geo_heads(
    seeds: list[int] | None = None,
    d_subspace: int = 64,
    d_swiss: int = 64,
    n_per: int = 200,
    k: int = 3,
    output_path: str | Path = "results/figure4/ablation1_geo_heads.csv",
) -> pd.DataFrame:
    """
    Ablation 1: compare different geo heads on
      - Gaussian subspace (d_subspace)
      - Swiss-roll high-D (d_swiss)
    """
    if seeds is None:
        seeds = list(range(20))

    rows = []

    for seed in tqdm(seeds):
        # --- Gaussian subspace ---
        X_sub, y_sub = make_subspace_clusters(
            n_per=n_per,
            d=d_subspace,
            k=k,
            subspace_dim=5,
            snr=3.0,
            seed=seed,
        )
        res_sub = run_geo_head_ablation_once(X_sub, y_sub, k=k, seed=seed)
        res_sub["dataset"] = "subspace_gaussian"
        rows.append(res_sub)

        # --- Gaussian heavy tail ---
        X_ht, y_ht = make_heavy_tail_subspace_clusters(
                n_per=n_per,
                d=d_subspace,
                k=k,
                subspace_dim=5,
                df=5.0,
                snr=4.0,
                standardize=True,
                seed=seed,
            )
        res_ht = run_geo_head_ablation_once(X_ht, y_ht, k=k, seed=seed)
        res_ht["dataset"] = "heavy_tail"
        rows.append(res_ht)

        # --- Swiss-roll manifold ---
        X_sw, y_sw = make_swiss_roll_clusters_highdim(
            n_per=n_per,
            d=d_swiss,
            k=k,
            roll_noise=0.03,
            embed_noise=0.05,
            standardize=True,
            seed=seed,
        )
        res_sw = run_geo_head_ablation_once(X_sw, y_sw, k=k, seed=seed)
        res_sw["dataset"] = "swiss_roll"
        rows.append(res_sw)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

def make_rotated_subspace_clusters(
    n_per: int = 200,
    d: int = 64,
    k: int = 3,
    subspace_dim: int = 5,
    snr: float = 3.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gaussian subspace clusters rotated by a random orthonormal matrix.

    This is used to test rotation invariance: axis-aligned URF vs oblique OURF.
    """
    rng = np.random.RandomState(seed)
    X, y = make_subspace_clusters(
        n_per=n_per,
        d=d,
        k=k,
        subspace_dim=subspace_dim,
        snr=snr,
        seed=seed,
    )

    # Random orthonormal rotation R in R^{d x d}
    A = rng.normal(size=(d, d))
    U, _, Vt = np.linalg.svd(A, full_matrices=True)
    R = U @ Vt  # orthonormal
    X_rot = X @ R

    return X_rot, y
def run_urf_vs_ourf_once(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int,
) -> dict:
    """
    Ablation 2: compare Geo-MDS-Ward using URF vs OURF.
    """
    out = dict(seed=seed)

    # Axis-aligned URF
    y_urf = geo_mds_ward_labels(
        X,
        k=k,
        whiten=True,
        oblique=False,
        random_state=seed,
    )
    out["ARI_Geo_URF"] = adjusted_rand_score(y, y_urf)

    # Oblique OURF
    y_ourf = geo_mds_ward_labels(
        X,
        k=k,
        whiten=True,
        oblique=True,
        random_state=seed,
    )
    out["ARI_Geo_OURF"] = adjusted_rand_score(y, y_ourf)

    return out

def run_ablation2_urf_vs_ourf(
    seeds: list[int] | None = None,
    d_list: list[int] | None = None,
    n_per: int = 200,
    k: int = 3,
    output_path: str | Path = "results/figure5/ablation2_urf_vs_ourf.csv",
) -> pd.DataFrame:
    """
    Ablation 2: URF vs OURF on
      - subspace_gaussian (axis-aligned)
      - subspace_gaussian_rotated
      - swiss_roll_highdim
    """
    if seeds is None:
        seeds = list(range(20))
    if d_list is None:
        d_list = [64, 256]

    rows = []

    for seed in tqdm(seeds):
        for d in d_list:
            # Axis-aligned subspace
            X_sub, y_sub = make_subspace_clusters(
                n_per=n_per,
                d=d,
                k=k,
                subspace_dim=5,
                snr=3.0,
                seed=seed,
            )
            res_sub = run_urf_vs_ourf_once(X_sub, y_sub, k=k, seed=seed)
            res_sub.update({"dataset": "subspace_gaussian", "d": d})
            rows.append(res_sub)

            # Rotated subspace
            X_rot, y_rot = make_rotated_subspace_clusters(
                n_per=n_per,
                d=d,
                k=k,
                subspace_dim=5,
                snr=3.0,
                seed=seed,
            )
            res_rot = run_urf_vs_ourf_once(X_rot, y_rot, k=k, seed=seed)
            res_rot.update({"dataset": "subspace_gaussian_rotated", "d": d})
            rows.append(res_rot)

        # Swiss-roll once per seed (fix d=64 here)
        d_sw = 64
        X_sw, y_sw = make_swiss_roll_clusters_highdim(
            n_per=n_per,
            d=d_sw,
            k=k,
            roll_noise=0.03,
            embed_noise=0.05,
            standardize=True,
            seed=seed,
        )
        res_sw = run_urf_vs_ourf_once(X_sw, y_sw, k=k, seed=seed)
        res_sw.update({"dataset": "swiss_roll_highdim", "d": d_sw})
        rows.append(res_sw)

    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df_ab1 = run_ablation1_geo_heads()
    print(df_ab1.groupby(["dataset"])[[
        "ARI_Euclid_Ward",
        "ARI_RBF_KPCA_Ward",
        "ARI_GF_Agglo_avg",
        "ARI_GF_Agglo_comp",
        "ARI_Geo_MDS_Ward_nowhite",
        "ARI_Geo_MDS_Ward_white",
    ]].mean())
    
    df_ab2 = run_ablation2_urf_vs_ourf()
    print("Ablation 2 summary:")
    print(df_ab2.groupby(["dataset", "d"])[["ARI_Geo_URF", "ARI_Geo_OURF"]].mean())

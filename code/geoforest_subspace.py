
# geoforest_subspace.py
# GeoForest clustering on subspace datasets.
# Methods:
#   - GF-Agglo (forest similarity S -> D=1-S -> agglomerative)
#   - GF-EmbedWard (Geo-MLE): S -> D -> classical MDS -> top-q MDS coords -> (optional) OAS-whiten -> Ward
#   - Kernel+Ward baseline: RBF -> KPCA -> OAS-whiten -> Ward


import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, roc_auc_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.covariance import OAS
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import seaborn as sns

from treeple import UnsupervisedRandomForest as URF
from treeple import UnsupervisedObliqueRandomForest as OURF



# ----------------------
# Data: subspace clusters
# ----------------------
def make_subspace_clusters(n_per=200, d=50, k=3, subspace_dim=5, snr=3.0, seed=1):
    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    for r in range(k):
        Xr = rng.normal(scale=1.0, size=(n_per, d))
        start = (r * subspace_dim) % d
        idx = np.arange(start, start + subspace_dim) % d
        center = rng.normal(scale=snr, size=subspace_dim)
        Xr[:, idx] += center
        X_list.append(Xr); y_list.append(np.full(n_per, r))
    X = np.vstack(X_list); y = np.concatenate(y_list)
    return StandardScaler().fit_transform(X), y


# ----------------------
# Forest configuration
# ----------------------
@dataclass
class ForestCfg:
    oblique: bool = False
    n_estimators: int = 1400
    criterion: str = "twomeans"
    max_depth: Optional[int] = None
    min_samples_leaf: int = 9
    min_samples_split: int = 18
    max_features: Optional[str] = None  # None (all) is good for sparse subspace signal
    feature_combinations: Optional[float] = None  # OURF only; e.g., 4.0
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 0


def make_forest_cfg(n: int, d: int, oblique: bool = False, random_state: int = 0) -> Tuple[object, dict]:
    """Dimension-aware schedule; returns (rf_class, kwargs)."""
    mleaf = max(3, int(0.015 * n))          # ~1.5% of n
    msplt = max(2 * mleaf, 6)
    leaf_scale = 1.0
    if oblique:
        mleaf = max(5, int(0.015 * n * leaf_scale))
        rf_cls = OURF
        rf_cfg = dict(
            n_estimators=1400, criterion="twomeans", max_depth=None,
            min_samples_leaf=mleaf, min_samples_split=2*mleaf,
            max_features=None,                   # let oblique sample from all dims
            feature_combinations=1.5,
            bootstrap=True, n_jobs=-1, random_state=random_state
        )
    else:
        rf_cls = URF
        rf_cfg = dict(
            n_estimators=1400, criterion="twomeans",
            max_depth=None, min_samples_leaf=mleaf, min_samples_split=msplt,
            max_features=None, bootstrap=True, n_jobs=-1, random_state=random_state
        )
    return rf_cls, rf_cfg


# ----------------------
# Similarity & distances
# ----------------------
def urf_similarity_from_cfg(X: np.ndarray, rf_cls, rf_cfg) -> np.ndarray:
    """Fit forest and return symmetric similarity with diag=1."""
    rf = rf_cls(**rf_cfg).fit(X)
    S = rf.compute_similarity_matrix(X).astype(float)
    S = 0.5 * (S + S.T)
    # Do not clip unless out-of-range; TreePle returns [0,1]
    np.fill_diagonal(S, 1.0)
    return S


def auc_from_similarity(S: np.ndarray, y: np.ndarray) -> float:
    iu = np.triu_indices(S.shape[0], 1)
    labels = (y[iu[0]] == y[iu[1]]).astype(int)
    return float(roc_auc_score(labels, S[iu]))


# ----------------------
# Classical MDS on distances
# ----------------------
def classical_mds_from_dist(D: np.ndarray, return_eigs: bool = False, rel_tol: float = 1e-10):
    """Return coords from positive eigenpairs of B = -1/2 J D^2 J. Eigenvalues ascending."""
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    w, V = np.linalg.eigh(B)  # ascending
    thr = rel_tol * (np.max(np.abs(w)) + 1e-300)
    pos = w > thr
    Z = V[:, pos] * np.sqrt(np.maximum(w[pos], 0.0))
    return (Z, w) if return_eigs else Z


# ----------------------
# Geo-MLE (GF-EmbedWard): top-q MDS coords -> (optional) whiten -> Ward
# ----------------------
def geo_mle_topq_mds_ward(S: np.ndarray, K: int, whiten: bool = True, rel_tol: float = 1e-10):
    """From similarity S (diag=1) to labels via MDS(D) top-q coords and Ward."""
    # Distance and MDS
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    Z_full, w = classical_mds_from_dist(D, return_eigs=True, rel_tol=rel_tol)

    # Count true positive mass
    thr = rel_tol * (np.max(np.abs(w)) + 1e-300)
    pos_mask = w > thr
    r_pos = int(pos_mask.sum())

    if r_pos == 0 or Z_full.shape[1] == 0:
        return None, 0, None, dict(r_pos=0, pos_share=0.0)

    # q-floor: at most K-1 and at most r_pos, but at least 1
    q = min(K - 1, r_pos)
    q = max(q, 1)

    # Keep TOP-q positive coords (largest eigenvalues are at the end)
    Zm = Z_full[:, -q:]

    # Optionally whiten to equalize axis scales for Ward
    if whiten:
        Zc = Zm - Zm.mean(axis=0, keepdims=True)
        oas = OAS(store_precision=True).fit(Zc)
        wv, Vv = np.linalg.eigh(oas.covariance_)
        Winv_half = Vv @ np.diag(1.0 / np.sqrt(np.maximum(wv, 1e-12))) @ Vv.T
        Z_use = Zc @ Winv_half
    else:
        Z_use = Zm

    labels = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Z_use)

    pos_share = float(np.sum(w[pos_mask]) / (np.sum(np.abs(w)) + 1e-300))
    return labels, q, Z_use, dict(r_pos=r_pos, pos_share=pos_share)


# ----------------------
# GF-Agglo (GeoForest-Agglo)
# ----------------------
def gf_agglo_labels(S: np.ndarray, K: int, linkage: str = "complete"):
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage=linkage).fit_predict(D)
    return y


# ----------------------
# Kernel+Ward baseline
# ----------------------
def choose_gamma_from_median(X: np.ndarray) -> float:
    D = pairwise_distances(X)
    iu = np.triu_indices_from(D, 1)
    sigma = np.median(D[iu]) + 1e-12
    return 1.0 / (2.0 * sigma * sigma)


def kernel_ward_mahal_labels(X: np.ndarray, K: int, n_components: int = 32, seed: int = 0):
    nc = min(n_components, X.shape[1])
    gamma = choose_gamma_from_median(X)
    Z = KernelPCA(n_components=nc, kernel="rbf", gamma=gamma, random_state=seed,
                  fit_inverse_transform=False, eigen_solver="auto").fit_transform(X)
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(Z)
    # OAS-whiten + Ward
    Zc = Z - Z.mean(axis=0, keepdims=True)
    oas = OAS(store_precision=True).fit(Zc)
    w, V = np.linalg.eigh(oas.covariance_)
    Winv_half = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    Zw = Zc @ Winv_half
    y = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(Zw)
    return y, Zw


# ----------------------
# Experiment runners
# ----------------------
def run_once_subspace(seed: int, d: int, n_per: int = 200, k: int = 3,
                      subspace_dim: int = 5, snr: float = 3.0,
                      oblique: bool = False, whiten_geo_mle: bool = True):
    """Run all arms on one dataset realization; return dict of metrics."""
    X, y = make_subspace_clusters(n_per=n_per, d=d, k=k, subspace_dim=subspace_dim, snr=snr, seed=seed)
    n = X.shape[0]

    # Forest & similarity
    rf_cls, rf_cfg = make_forest_cfg(n, d, oblique=oblique, random_state=seed)
    S = urf_similarity_from_cfg(X, rf_cls, rf_cfg)

    # Arms
    # GF-Agglo
    y_avg = gf_agglo_labels(S, k, linkage="average")
    y_cmp = gf_agglo_labels(S, k, linkage="complete")

    # Geo-MLE (top-q MDS -> Ward)
    y_mle, q, Zw, meta = geo_mle_topq_mds_ward(S, k, whiten=whiten_geo_mle)

    # Kernel+Ward
    y_kw, Zw_kw = kernel_ward_mahal_labels(X, k, n_components=min(32, d), seed=seed)

    # Metrics
    D = 1.0 - S; np.fill_diagonal(D, 0.0)
    auc_S = auc_from_similarity(S, y)
    out = dict(
        seed=seed, d=d,
        auc_S=auc_S,
        ari_gf_avg=adjusted_rand_score(y, y_avg),
        ari_gf_comp=adjusted_rand_score(y, y_cmp),
        ari_gf_best=max(adjusted_rand_score(y, y_avg), adjusted_rand_score(y, y_cmp)),
        ari_geo_mle=(adjusted_rand_score(y, y_mle) if y_mle is not None else np.nan),
        q_geo_mle=(q if y_mle is not None else 0),
        rpos_mds=meta.get("r_pos", 0) if meta else 0,
        pos_share=meta.get("pos_share", 0.0) if meta else 0.0,
        ari_kw=adjusted_rand_score(y, y_kw),
    )
    return out


def run_dim_sweep(dims: List[int], seeds: List[int], n_per: int = 200, k: int = 3,
                  subspace_dim: int = 5, snr: float = 3.0,
                  oblique: bool = False, whiten_geo_mle: bool = True) -> pd.DataFrame:
    rows = []
    for d in tqdm(dims):
        for s in tqdm(seeds):
            rows.append(run_once_subspace(seed=s, d=d, n_per=n_per, k=k,
                                          subspace_dim=subspace_dim, snr=snr,
                                          oblique=oblique, whiten_geo_mle=whiten_geo_mle))
    return pd.DataFrame(rows)




# # ----------------------
# # Script entry point
# # ----------------------

# dims = [2**i for i in range(3, 12)]  # 8..512
# seeds = list(range(20))

# df = run_dim_sweep(dims=dims, seeds=seeds, n_per=200, k=3,
#                     subspace_dim=5, snr=3.0, oblique=False, whiten_geo_mle=True)
# df.to_csv("subspace_3x5_dim_sweep.csv", index=False)

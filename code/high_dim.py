import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator
import matplotlib.patches as mpatches
import numpy as np

# from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.exceptions import ConvergenceWarning

from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns


from autogmm_Apr26 import AutoGMM, KernelAutoGMM

import time
import numpy as np
import pandas as pd

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter, pandas2ri
from rpy2.robjects.vectors import IntVector, StrVector
from sklearn.metrics import adjusted_rand_score

mclust = importr("mclust")
MODELNAMES_VVV = StrVector(["VVV", "EEE", "VVI", "VII"])

def run_mclust(
    X: pd.DataFrame,
    constrain_cov: bool = True,
    n_components: int | list[int] | range | None = None,
    y_true: np.ndarray | None = None,
):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # --- build R kwargs ---
    kwargs: dict[str, ro.Vector | bool] = {}
    if constrain_cov:
        kwargs["modelNames"] = MODELNAMES_VVV
    if n_components is not None:
        G = [int(g) for g in (n_components if isinstance(n_components, (list, range, np.ndarray)) else [n_components])]
        kwargs["G"] = IntVector(G)
    kwargs["verbose"] = False

    # --- convert input ONLY ---
    with localconverter(default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(X)

    # --- run Mclust OUTSIDE the converter so we keep an R ListVector ---
    t0 = time.perf_counter()
    res = mclust.Mclust(r_df, **kwargs)
    runtime = time.perf_counter() - t0

    # --- extract and convert outputs ---
    cls   = res.rx2("classification")
    bic_r = res.rx2("bic")
    mod_r = res.rx2("modelName")

    with localconverter(default_converter + pandas2ri.converter):
        labels = np.asarray(ro.conversion.rpy2py(cls), dtype=int)
        bic    = np.asarray(ro.conversion.rpy2py(bic_r), dtype=float)
        model  = ro.conversion.rpy2py(mod_r)

    model_name = model if isinstance(model, str) else [str(m) for m in model]

    if y_true is not None:
        ari = adjusted_rand_score(y_true, labels)
        return labels, runtime, bic, model_name, ari
    return labels, runtime, bic, model_name


import numpy as np

def extreme_aniso(
    n_total=200,
    d=2,
    delta=4.0,
    cond=50.0,
    seed=0,
    style="one-large",          # "one-large" -> match data2; "geometric" -> original
    align_mean="small",         # "small" (match data2) or "large"
    axis_aligned_2d=False       # True -> exactly axis-aligned in 2D (like data2)
):
    """
    Two-component GMM with shared full covariance (Σ).
    - style="one-large": one dominant eigenvalue = cond, others = 1  (data2-like)
    - style="geometric": geometric spectrum from 1 down to 1/cond     (original)
    - align_mean="small": means separated along a small-variance eigenvector (data2-like)
    - axis_aligned_2d=True: in 2D, Σ = diag([1, cond]) and separation along e1

    delta is interpreted in *standard deviation units* along the chosen separation direction.
    """
    rng = np.random.default_rng(seed)

    # --- 1) Eigen spectrum ---
    if style == "one-large":
        eigs = np.ones(d)
        # Make one direction very large variance (dominant axis)
        # For exact data2 in 2D with axis_aligned_2d=True:
        if axis_aligned_2d and d == 2:
            # large variance on y, small on x -> Σ = diag([1, cond])
            eigs[0] = 1.0
            eigs[1] = float(cond)
            Q = np.eye(d)  # axis aligned
        else:
            # Random orientation, one large eigenvalue
            eigs[0] = float(cond)
            Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    elif style == "geometric":
        lam_max, lam_min = 1.0, 1.0 / float(cond)
        eigs = np.geomspace(lam_max, lam_min, num=d)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    else:
        raise ValueError("style must be 'one-large' or 'geometric'")

    # --- 2) Covariance ---
    Sigma = (Q * eigs) @ Q.T  # Q diag(eigs) Q^T

    # --- 3) Choose separation direction ---
    if axis_aligned_2d and d == 2 and style == "one-large":
        # Σ = diag([1, cond]); separate along small-variance axis e1
        v_sep = np.array([1.0, 0.0])
        lam_sep = 1.0
    else:
        # Index of large eigen (argmax) and some small eigen index (not the large one)
        idx_large = int(np.argmax(eigs))
        small_idxs = [i for i in range(d) if i != idx_large]
        idx_small = small_idxs[0] if small_idxs else idx_large

        if align_mean == "small":
            idx = idx_small
        elif align_mean == "large":
            idx = idx_large
        else:
            raise ValueError("align_mean must be 'small' or 'large'")

        v_sep = Q[:, idx] / np.linalg.norm(Q[:, idx])
        lam_sep = float(eigs[idx])

    # --- 4) Means (delta SD along v_sep) ---
    mu1 = -0.5 * delta * np.sqrt(lam_sep) * v_sep
    mu2 =  0.5 * delta * np.sqrt(lam_sep) * v_sep

    # --- 5) Sample ---
    n_per = n_total // 2
    X1 = rng.multivariate_normal(mu1, Sigma, size=n_per)
    X2 = rng.multivariate_normal(mu2, Sigma, size=n_per)
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n_per, dtype=int), np.ones(n_per, dtype=int)])
    return X, y


def one_rep(dim_level, base_seed, rep_id, k1=1, k2=5, n_total=200,
            delta=4.0, cond=300.0):
    """
    Run one repetition at dimension d = 2*dim_level.
    Returns a dict of method -> ARI.
    """
    d = 2 * dim_level
    seed = base_seed + rep_id
    X, y = extreme_aniso(n_total=n_total, d=d, delta=delta, cond=cond, seed=seed)

    if k1 is None or k2 is None:
        k_true = len(np.unique(y))
        k1_, k2_ = k_true, k_true
    else:
        k1_, k2_ = k1, k2

    res = {}

    # 1) sklearn GMM (baseline)
    # pred = GaussianMixture(n_components=min(k2_, len(np.unique(y))), random_state=seed).fit_predict(X)
    # res["sklearn-GM"] = adjusted_rand_score(y, pred)

    # 2) AutoGMM (KMeans-only)
    pred = AutoGMM(min_components=k1_, max_components=k2_,
                   init_agglomerative=False, n_init_kmeans=1,
                   random_state=seed).fit_predict(X)
    res["AutoGMM (KMeans)"] = adjusted_rand_score(y, pred)

    # 3) AutoGMM (Ward–Mahalanobis-only)
    pred = AutoGMM(min_components=k1_, max_components=k2_,
                   n_init_kmeans=0,
                   agglom_linkages=['ward'],
                   agglom_affinities=['mahalanobis'],
                   random_state=seed).fit_predict(X)
    res["AutoGMM (Ward–Mah)"] = adjusted_rand_score(y, pred)

    # 4) AutoGMM (Ward–Euclidean-only)
    pred = AutoGMM(min_components=k1_, max_components=k2_,
                   n_init_kmeans=0,
                   agglom_linkages=['ward'],
                   agglom_affinities=['euclidean'],
                   random_state=seed).fit_predict(X)
    res["AutoGMM (Ward–Euc)"] = adjusted_rand_score(y, pred)

    # 5) AutoGMM (Full combo)
    pred = AutoGMM(min_components=k1_, max_components=k2_,
                   n_init_kmeans=1,
                   agglom_linkages=['ward'],
                   agglom_affinities=['euclidean', 'mahalanobis'],
                   random_state=seed).fit_predict(X)
    res["AutoGMM (Full)"] = adjusted_rand_score(y, pred)

    # _, _, _, _, res["mclust (4-cov)"] = run_mclust(X=pd.DataFrame(X), y_true=y, 
    #                                     n_components=range(k1_, k2_+1))
    # _, _, _, _, res["mclust (14-cov)"] = run_mclust(X=pd.DataFrame(X), constrain_cov=False, 
    #                                         y_true=y, n_components=range(k1_, k2_+1))


    return d, res


n_reps = 50
base_seed = 0
dim_levels = range(1, 11)   # d in {2,4,...,20}
k1, k2 = 1, 5               # set (None, None) if you want known-K per dataset

rows = []

for dim_level in tqdm(dim_levels):
    # 1) Parallel block for sklearn/AutoGMM methods
    out = Parallel(n_jobs=-1)(
        delayed(one_rep)(dim_level, base_seed, rep_id, k1, k2)
        for rep_id in range(n_reps)
    )
    for d, res in out:
        for method, ari in res.items():
            rows.append({"Dim": d, "Method": method, "ARI": float(ari)})

    # 2) Sequential pass for mclust (outside parallel)
    d = 2 * dim_level
    for rep_id in range(n_reps):
        seed = base_seed + rep_id
        X, y = extreme_aniso(n_total=200, d=d, delta=4.0, cond=300.0, seed=seed)

        # Determine K range (unknown-K uses [k1..k2], known-K uses the true K)
        if k1 is None or k2 is None:
            k_true = len(np.unique(y))
            k_range = range(k_true, k_true + 1)
        else:
            k_range = range(k1, k2 + 1)

        # mclust with restricted (4) vs full (14) covariance families
        _, _, _, _, ari_mclust_4  = run_mclust(X=pd.DataFrame(X), y_true=y, n_components=k_range, constrain_cov=True)
        _, _, _, _, ari_mclust_14 = run_mclust(X=pd.DataFrame(X), y_true=y, n_components=k_range, constrain_cov=False)

        rows.append({"Dim": d, "Method": "mclust (4-cov)",  "ARI": float(ari_mclust_4)})
        rows.append({"Dim": d, "Method": "mclust (14-cov)", "ARI": float(ari_mclust_14)})
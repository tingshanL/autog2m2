# geoforest_sweeps.py
# Requires geoforest_subspace.py to be in PYTHONPATH / same folder.

from pathlib import Path

import numpy as np, pandas as pd
from tqdm import tqdm

from geoforest_subspace import (
    make_subspace_clusters, make_forest_cfg, urf_similarity_from_cfg,
    gf_agglo_labels, geo_mle_topq_mds_ward, kernel_ward_mahal_labels
)

# ---------- helpers ----------

def run_all_methods_on_Xy(X, y, oblique=False, whiten_geo_mle=True, seed=0):
    n, d = X.shape
    k = len(np.unique(y))
    rf_cls, rf_cfg = make_forest_cfg(n, d, oblique=oblique, random_state=seed)
    S = urf_similarity_from_cfg(X, rf_cls, rf_cfg)

    # GF-Agglo
    y_avg = gf_agglo_labels(S, k, linkage="average")
    y_cmp = gf_agglo_labels(S, k, linkage="complete")

    # Geo-MLE
    y_mle, q, Zw, meta = geo_mle_topq_mds_ward(S, k, whiten=whiten_geo_mle)

    # Kernel-Ward
    y_kw, _ = kernel_ward_mahal_labels(X, k, n_components=min(32, d), seed=seed)

    from sklearn.metrics import adjusted_rand_score
    return dict(
        ari_gf_avg=adjusted_rand_score(y, y_avg),
        ari_gf_comp=adjusted_rand_score(y, y_cmp),
        ari_geo_mle=(adjusted_rand_score(y, y_mle) if y_mle is not None else np.nan),
        q_geo_mle=(q if y_mle is not None else 0),
        rpos_mds=meta.get("r_pos", 0) if meta else 0,
        pos_share=meta.get("pos_share", 0.0) if meta else 0.0,
        ari_kw=adjusted_rand_score(y, y_kw),
    )

def make_subspace_clusters_imbalanced(ratios, d=256, p=5, snr=3.0, seed=0):
    """ratios: list like [1,1,1] or [3,1,1] or [10,1,1]; n_total = 600 fixed."""
    n_total = 600
    weights = np.array(ratios, dtype=float)
    weights = weights / weights.sum()
    counts = np.floor(n_total * weights).astype(int)
    # fix rounding to exact n_total
    while counts.sum() < n_total:
        counts[np.argmax(weights - counts / n_total)] += 1

    rng = np.random.RandomState(seed)
    X_list, y_list = [], []
    k = len(counts)
    for r, n_per in enumerate(counts):
        Xr = rng.normal(scale=1.0, size=(n_per, d))
        start = (r * p) % d
        idx = np.arange(start, start + p) % d
        center = rng.normal(scale=snr, size=p)
        Xr[:, idx] += center
        X_list.append(Xr)
        y_list.append(np.full(n_per, r))
    X = np.vstack(X_list); y = np.concatenate(y_list)

    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X), y

# ---------- sweeps ----------

def snr_sweep(seeds=range(20), d=256, k=3, p=5, snrs=(1,2,3,4,6),
              oblique=False, out_path="results/figure3/subspace_snr_sweep.csv"):
    rows = []
    for snr in snrs:
        for s in tqdm(seeds, desc=f"SNR={snr}"):
            X, y = make_subspace_clusters(n_per=600//k, d=d, k=k, subspace_dim=p, snr=snr, seed=s)
            rec = dict(seed=s, d=d, k=k, p=p, snr=snr, ratio="1:1:1")
            rec.update(run_all_methods_on_Xy(X, y, oblique=oblique, seed=s))
            rows.append(rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def imbalance_sweep(seeds=range(20), d=256, p=5, snr=3.0, ratios=((1,1,1),(3,1,1),(10,1,1)),
                    oblique=False, out_path="results/figure3/subspace_imbalance_sweep.csv"):
    rows = []
    for ratio in ratios:
        for s in tqdm(seeds, desc=f"ratio={ratio}"):
            X, y = make_subspace_clusters_imbalanced(list(ratio), d=d, p=p, snr=snr, seed=s)
            k = len(np.unique(y))
            rec = dict(seed=s, d=d, k=k, p=p, snr=snr, ratio=":".join(map(str,ratio)))
            rec.update(run_all_methods_on_Xy(X, y, oblique=oblique, seed=s))
            rows.append(rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def p_sweep(seeds=range(20), d=256, k=3, ps=(2,5,10), snr=3.0,
            oblique=False, out_path="results/figure3/subspace_p_sweep.csv"):
    rows = []
    for p in ps:
        for s in tqdm(seeds, desc=f"p={p}"):
            X, y = make_subspace_clusters(n_per=600//k, d=d, k=k, subspace_dim=p, snr=snr, seed=s)
            rec = dict(seed=s, d=d, k=k, p=p, snr=snr, ratio="1:1:1")
            rec.update(run_all_methods_on_Xy(X, y, oblique=oblique, seed=s))
            rows.append(rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

def k_sweep(seeds=range(20), d=256, ks=(5,10), p=5, snr=3.0,
            oblique=False, out_path="results/figure3/subspace_k_sweep.csv"):
    rows = []
    for k in ks:
        for s in tqdm(seeds, desc=f"k={k}"):
            X, y = make_subspace_clusters(n_per=600//k, d=d, k=k, subspace_dim=p, snr=snr, seed=s)
            rec = dict(seed=s, d=d, k=k, p=p, snr=snr, ratio="1:1:1")
            rec.update(run_all_methods_on_Xy(X, y, oblique=oblique, seed=s))
            rows.append(rec)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    snr_sweep()
    imbalance_sweep()
    p_sweep()
    k_sweep()

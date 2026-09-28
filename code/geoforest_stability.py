# geoforest_stability.py

import numpy as np, pandas as pd
from tqdm import tqdm
from geoforest_subspace import (
    make_subspace_clusters, make_forest_cfg, urf_similarity_from_cfg,
    gf_agglo_labels, geo_mle_topq_mds_ward
)

def run_once_with_cfg(X, y, rf_cls, rf_cfg):
    k = len(np.unique(y))
    S = urf_similarity_from_cfg(X, rf_cls, rf_cfg)
    y_cmp = gf_agglo_labels(S, k, linkage="complete")
    y_mle, q, Zw, meta = geo_mle_topq_mds_ward(S, k, whiten=True)
    from sklearn.metrics import adjusted_rand_score
    return dict(
        ari_gf_comp=adjusted_rand_score(y, y_cmp),
        ari_geo_mle=(adjusted_rand_score(y, y_mle) if y_mle is not None else np.nan),
        q_geo_mle=(q if y_mle is not None else 0)
    )

def run_stability_sweep(d=256, k=3, p=5, snr=3.0, seeds=range(20),
                        est_scales=(0.5, 1.0, 1.5), leaf_scales=(0.5, 1.0, 1.5),
                        max_features_list=(None,)):
    rows = []
    for s in seeds:
        X, y = make_subspace_clusters(n_per=600//k, d=d, k=k, subspace_dim=p, snr=snr, seed=s)
        n = X.shape[0]
        # base cfg
        rf_cls, rf_base = make_forest_cfg(n, d, oblique=False, random_state=s)
        base_leaf = rf_base["min_samples_leaf"]
        base_ests = rf_base["n_estimators"]
        for mf in max_features_list:
            for es in est_scales:
                for ls in leaf_scales:
                    rf_cfg = rf_base.copy()
                    rf_cfg["n_estimators"]   = max(100, int(base_ests * es))
                    rf_cfg["min_samples_leaf"] = max(3, int(base_leaf * ls))
                    rf_cfg["min_samples_split"] = max(2*rf_cfg["min_samples_leaf"], 6)
                    if mf is not None:
                        rf_cfg["max_features"] = mf
                    rec = dict(seed=s, d=d, k=k, p=p, snr=snr,
                               est_scale=es, leaf_scale=ls, max_features=rf_cfg.get("max_features"))
                    rec.update(run_once_with_cfg(X, y, rf_cls, rf_cfg))
                    rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv("subspace_stability.csv", index=False)
    return "subspace_stability.csv"

if __name__ == "__main__":
    run_stability_sweep()

# geoforest_oblique.py

from pathlib import Path

import numpy as np, pandas as pd
from tqdm import tqdm
from geoforest_subspace import run_dim_sweep  # already in your file

dims  = [2**i for i in range(3, 12)]  # 8..2048 if supported
seeds = list(range(20))

df_axis = run_dim_sweep(dims=dims, seeds=seeds, n_per=200, k=3, subspace_dim=5, snr=3.0,
                        oblique=False, whiten_geo_mle=True)
df_axis["forest"] = "axis"

df_obliq = run_dim_sweep(dims=dims, seeds=seeds, n_per=200, k=3, subspace_dim=5, snr=3.0,
                         oblique=True,  whiten_geo_mle=True)
df_obliq["forest"] = "oblique"

df = pd.concat([df_axis, df_obliq], ignore_index=True)
output_path = Path("results/figure5/subspace_oblique.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

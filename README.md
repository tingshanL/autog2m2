# AutoG2M2 reproducibility

Code, result tables, and plotting scripts for *AutoG2M2: Geodesic-Forest
Neighborhood Learning for Stable Gaussian-Mixture Clustering in Data-Poor
High Dimensions*.

## Contents

```text
code/       experiment and plotting scripts
data/       Drosophila inputs and TCGA download instructions
results/    seed-level results and summaries
figures/    figures generated from the included results
```

`code/autogmm_Apr26.py` is retained because several experiments import this
implementation.

## Setup

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The mclust comparisons also require R and mclust 6.1.3:

```bash
Rscript -e 'install.packages("mclust", repos="https://cloud.r-project.org")'
```

Run all commands from the repository root.

## Generate the figures

The included result tables can be plotted without rerunning the experiments:

```bash
python code/plot_all.py
```

Each plotting script can also be run separately. Figures are written to
`figures/` as PDF and PNG files.

## Recompute the experiments

Figure 1:

```bash
python code/run_figure1.py
```

Figure 2, including fixed K=3 and BIC selection over K=1,...,5:

```bash
python code/run_figure2.py --jobs 2
python code/run_figure2.py --methods ae_relu --jobs 2
```

All representation-based Figure 2 methods use an embedding dimension of two.
Gaussian and heavy-tailed subspace experiments extend through d=2048;
Swiss-roll experiments extend through d=256.

The remaining experiments use the following scripts:

| Figure | Experiment script |
|---|---|
| 3 | `geoforest_subspace.py`, `geoforest_sweeps.py` |
| 4 | `geoforest_ablations.py` |
| 5 | `geoforest_ablations.py`, `geoforest_oblique.py` |
| 6 | `run_figure6_anisotropic.py` |
| 7 | `run_figure7_tcga.py` |
| 8 | `run_figure8_drosophila.py` |
| 9 | `run_runtime_isotropic.py` |

The runners write to the corresponding subdirectories of `results/`. Most
runners support additional options listed by `--help`.

## Data

The Drosophila inputs are included in `data/drosophila/`. TCGA RNA-seq data
are not redistributed; download and file-placement instructions are provided
in `data/tcga/README.md`.

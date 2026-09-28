# TCGA Pan-Cancer RNA-seq input files

Download the UCI Gene Expression Cancer RNA-Seq dataset (Dataset 401):

https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

Place the sample-by-gene matrix and matching class labels at:

```text
data/tcga/data.csv
data/tcga/labels.csv
```

The analysis removes an optional first sample-ID column, selects the top genes
by sample variance for each feature budget, and standardizes each selected gene
across samples.

Use the file arguments shown by:

```bash
python code/run_figure7_tcga.py --help
```

Then run each data regime with the same expression and label files:

```bash
python code/run_figure7_tcga.py --regime observed_genes --mode final --outer-jobs 2 \
  --data data/tcga/data.csv --labels data/tcga/labels.csv --output-dir results/figure7
python code/run_figure7_tcga.py --regime independent_permutation --mode final --outer-jobs 2 \
  --data data/tcga/data.csv --labels data/tcga/labels.csv --output-dir results/figure7
python code/run_figure7_tcga.py --regime structured_lowrank --mode final --outer-jobs 2 \
  --data data/tcga/data.csv --labels data/tcga/labels.csv --output-dir results/figure7
```

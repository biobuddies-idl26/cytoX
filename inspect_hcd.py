# quick look at the full HCD h5ad (obs columns, label counts, matrix type)
# without loading the 41 GB matrix. backed="r" keeps X on disk.

import argparse

import anndata as ad
import numpy as np

RAW = "/Volumes/My Passport for Mac/10M_PBMC_12donor_90cytokines_h5ad_20260203/20260203_Parse_10M_PBMC_cytokines.h5ad"

p = argparse.ArgumentParser()
p.add_argument("--raw", default=RAW)
args = p.parse_args()

adata = ad.read_h5ad(args.raw, backed="r")  # slow on a usb hdd, few minutes
print(adata)
print(adata.obs.head())

for c in adata.obs.columns:
    print(c, adata.obs[c].nunique())

# only print counts for label-like columns
for c in adata.obs.columns:
    if adata.obs[c].nunique() <= 200:
        print(f"\n[{c}]")
        print(adata.obs[c].value_counts().to_string())

x = adata.X[:200]
vals = x.data if hasattr(x, "data") else np.asarray(x).ravel()
print("\nX type:", type(x), "| max:", vals.max(), "| integers:", bool(np.all(vals % 1 == 0)))

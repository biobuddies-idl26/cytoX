# pull a balanced subset of the HCD for scGen: up to N cells per (cytokine, cell type),
# so every cell type has PBS controls and treated cells. reads from the backed file,
# so only the selected rows get loaded.
#
# python subset_hcd.py --cytokines PBS IFN-beta IFN-gamma IL-4 IL-15 IL-1-beta IL-32-beta
# add --all-genes to keep every gene (needed for collecTRI, HVGs still flagged in var)

import argparse
import os

import anndata as ad
import numpy as np
import scanpy as sc

RAW = "/Volumes/My Passport for Mac/10M_PBMC_12donor_90cytokines_h5ad_20260203/20260203_Parse_10M_PBMC_cytokines.h5ad"

p = argparse.ArgumentParser()
p.add_argument("--raw", default=RAW)
p.add_argument("--cytokines", nargs="+", required=True, help="include PBS")
p.add_argument("--n", type=int, default=500)
p.add_argument("--hvg", type=int, default=7000)
p.add_argument("--all-genes", action="store_true")
p.add_argument("--out", default="data/hcd_subset.h5ad")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

adata = ad.read_h5ad(args.raw, backed="r")

missing = set(args.cytokines) - set(adata.obs["cytokine"].astype(str))
if missing:
    raise SystemExit(f"not in the data: {sorted(missing)}")

obs = adata.obs[["cytokine", "cell_type"]]
obs = obs[obs["cytokine"].astype(str).isin(args.cytokines)]
rng = np.random.default_rng(args.seed)
idx = []
for _, g in obs.groupby(["cytokine", "cell_type"], observed=True):
    pos = adata.obs_names.get_indexer(g.index)
    idx.extend(rng.choice(pos, size=min(args.n, len(pos)), replace=False))
idx = np.sort(np.array(idx))  # sorted = mostly sequential reads off the drive
print(f"{len(idx):,} cells selected, reading...")

sub = adata[idx].to_memory()
print(sub.obs.groupby(["cell_type", "cytokine"], observed=True).size().unstack(fill_value=0).to_string())

# hvgs on raw counts (seurat_v3), then log-normalize. keep counts for later.
sub.layers["counts"] = sub.X.copy()
sc.pp.highly_variable_genes(sub, n_top_genes=args.hvg, flavor="seurat_v3", layer="counts")
sc.pp.normalize_total(sub, target_sum=1e4)
sc.pp.log1p(sub)
if not args.all_genes:
    sub = sub[:, sub.var.highly_variable].copy()

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
sub.write_h5ad(args.out, compression="gzip")
print(f"saved {sub.n_obs:,} x {sub.n_vars:,} -> {args.out}")

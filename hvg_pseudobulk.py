# reduce pseudobulked train / val h5ads to the top-N HVGs.
# HVGs are chosen on the TRAINING set only, then the same gene list is applied to val,
# so validation never influences feature selection and both files share identical columns.
#
# python hvg_pseudobulk.py --train data/pb_train.h5ad --val data/pb_val.h5ad --n 7000
# optional: --batch-key donor   (rank genes within each donor, then combine)

import argparse
import os

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

p = argparse.ArgumentParser()
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--n", type=int, default=7000)
p.add_argument("--counts-layer", default=None, help="layer with raw summed counts (default: auto, 'counts' if present)")
p.add_argument("--batch-key", default=None, help="obs column, e.g. donor")
p.add_argument("--suffix", default="_hvg")
args = p.parse_args()

train = ad.read_h5ad(args.train)
val = ad.read_h5ad(args.val)
if list(train.var_names) != list(val.var_names):
    shared = train.var_names.intersection(val.var_names)
    print(f"gene sets differ, using the {len(shared):,} shared genes")
    train, val = train[:, shared].copy(), val[:, shared].copy()


def is_counts(X):
    v = X[: min(200, X.shape[0])]
    v = v.data if sp.issparse(v) else np.asarray(v).ravel()
    return v.size > 0 and v.min() >= 0 and np.allclose(v, np.round(v))


layer = args.counts_layer or ("counts" if "counts" in train.layers else None)
src = train.layers[layer] if layer else train.X
if is_counts(src):
    # seurat_v3 models raw counts; summed pseudobulk counts are still counts
    flavor = "seurat_v3"
    print(f"raw counts found ({'layer ' + layer if layer else 'X'}) -> seurat_v3")
else:
    # X looks log-normalized -> dispersion-based selection
    flavor, layer = "seurat", None
    print("X looks log-normalized -> flavor='seurat'")

# genes that are all zero in train break the dispersion fit
sc.pp.filter_genes(train, min_cells=1)
n = min(args.n, train.n_vars)
sc.pp.highly_variable_genes(train, n_top_genes=n, flavor=flavor, layer=layer, batch_key=args.batch_key)
genes = train.var_names[train.var["highly_variable"]]
print(f"train: {train.n_obs:,} samples, {len(genes):,} HVGs selected")

for path, a in [(args.train, train), (args.val, val)]:
    out = a[:, genes].copy()
    root, ext = os.path.splitext(path)
    out.write_h5ad(root + args.suffix + ext, compression="gzip")
    print(f"{out.n_obs:,} x {out.n_vars:,} -> {root + args.suffix + ext}")

np.savetxt(os.path.splitext(args.train)[0] + args.suffix + "_genes.txt", np.asarray(genes), fmt="%s")
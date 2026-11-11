# per-cell TF activity for the whole HCD subset, saved as a cells x TFs h5ad.
# this is Model B's input: scGen trains on these vectors instead of gene expression.
#
# scored on the scGen HVGs by default, so Model A (genes -> scGen -> TF) and Model B
# (TF -> scGen) are scored with the same genes and the same network and share one truth.
#   python tf_scores.py                     # real collectri -> data/tf_scores_collectri.h5ad
#   python tf_scores.py --net shuffled      # ablation: targets shuffled -> data/tf_scores_shuffled.h5ad
#   python tf_scores.py --genes all         # every gene instead of the HVGs

import argparse

import anndata as ad
import pandas as pd

from collecTRI import get_net, filter_net, tf_scores

p = argparse.ArgumentParser()
p.add_argument("--net", choices=["collectri", "shuffled"], default="collectri")
p.add_argument("--genes", choices=["hvg", "all"], default="hvg")
p.add_argument("--data", default="data/hcd_subset_allgenes.h5ad")
p.add_argument("--hvg-from", default="data/hcd_subset.h5ad", help="gene list scGen trains on")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default=None)
args = p.parse_args()

adata = ad.read_h5ad(args.data)
if args.genes == "hvg":
    hvg = ad.read_h5ad(args.hvg_from, backed="r").var_names
    adata = adata[:, adata.var_names.intersection(hvg)].copy()

net = filter_net(get_net(args.net, args.seed), adata.var_names)
print(f"{adata.n_obs:,} cells, {adata.n_vars:,} genes, {net['source'].nunique()} TFs ({args.net})")

s = tf_scores(adata, net)
obs = adata.obs[["cytokine", "cell_type", "donor"]].copy()
out = ad.AnnData(X=s.values.astype("float32"), obs=obs, var=pd.DataFrame(index=s.columns))
out.uns["net"] = args.net
out.uns["genes"] = args.genes

path = args.out or f"data/tf_scores_{args.net}{'' if args.genes == 'hvg' else '_allgenes'}.h5ad"
out.write_h5ad(path)
print(f"saved {path}: {out.shape}")

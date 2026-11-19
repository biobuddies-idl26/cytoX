#build Model B input --> per-cell CollecTRI ULM TF activity scores stored in obsm["score_ulm"]
#uses the repo-root collecTRI.py (not /scgen) --> needs decoupler installed
#
#python cytoX_attention/make_tf_input.py --data data/hcd_subset.h5ad --out data/hcd_subset_tf.h5ad --zscore
#then: python cytoX_attention/train.py --data data/hcd_subset_tf.h5ad --space tf --use-obsm score_ulm --holdout "CD14 Mono"

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #repo root --> collecTRI.py

from collecTRI import filter_net, load_collectri, tf_scores  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/hcd_subset.h5ad")
p.add_argument("--out", default="data/hcd_subset_tf.h5ad")
p.add_argument("--layer", default=None, help="log-normalized layer to score (default X); use the all-genes copy if present")
p.add_argument("--tmin", type=int, default=5)
p.add_argument("--zscore", action="store_true", help="z-score each TF --> mean / std saved in uns for un-scaling")
args = p.parse_args()

adata = ad.read_h5ad(args.data)
net = filter_net(load_collectri(), adata.var_names, tmin=args.tmin)  #same filtered net for real and predicted data
scores = tf_scores(adata, net, layer=args.layer, tmin=args.tmin)
scores = pd.DataFrame(np.asarray(scores), index=adata.obs_names, columns=scores.columns).fillna(0.0)
if args.zscore:
    mu, sd = scores.mean(0), scores.std(0).replace(0, 1.0)
    scores = (scores - mu) / sd
    adata.uns["score_ulm_mean"] = mu.values
    adata.uns["score_ulm_std"] = sd.values
adata.uns["score_ulm_tfs"] = np.array(scores.columns)
adata.obsm["score_ulm"] = scores
adata.write_h5ad(args.out)
print(f"wrote {args.out}: {adata.n_obs:,} cells x {scores.shape[1]} TFs")

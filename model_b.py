# Model B: scGen trained on CollecTRI TF activity instead of gene expression.
# same held-out setup as scgen_baseline.py (Model A), but the VAE sees TF vectors
# and predicts the held-out cell type's TF activity directly.
#
# also scores the naive baseline in TF space: the held-out type's predicted delta TF
# is the average delta TF of the other cell types.
#
# python model_b.py --cytokine IFN-beta --holdout "CD14 Mono"
# python model_b.py --cytokine IFN-beta --holdout "CD14 Mono" --scores data/tf_scores_shuffled.h5ad
# needs data/tf_scores_<net>.h5ad from tf_scores.py

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd
import scgen
import torch
from scgen._scgenvae import SCGENVAE

from collecTRI import delta_tf, compare

# same scvi-tools key fix as scgen_baseline.py
_inference = SCGENVAE.inference


def _inference_fixed(self, x):
    out = _inference(self, x)
    out.setdefault("qzm", out["qz_m"])
    out.setdefault("qzv", out["qz_v"])
    return out


SCGENVAE.inference = _inference_fixed

RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]

p = argparse.ArgumentParser()
p.add_argument("--cytokine", required=True)
p.add_argument("--holdout", required=True)
p.add_argument("--scores", default="data/tf_scores_collectri.h5ad")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--outdir", default="data/model_b_runs")
args = p.parse_args()

adata = ad.read_h5ad(args.scores)
net = adata.uns.get("net", "collectri")
adata = adata[adata.obs["cytokine"].isin(["PBS", args.cytokine])].copy()
adata = adata[~adata.obs["cell_type"].isin(RARE)].copy()
adata.obs_names_make_unique()
for k in ["cell_type", "cytokine", "donor"]:
    adata.obs[k] = adata.obs[k].astype(str)
if args.holdout not in set(adata.obs["cell_type"]):
    raise SystemExit(f"no cell type '{args.holdout}'. options: {sorted(adata.obs['cell_type'].unique())}")
adata.obs["condition"] = np.where(adata.obs["cytokine"] == "PBS", "control", "stimulated")
X = np.asarray(adata.X, dtype=np.float32)
adata.X = X

ct = adata.obs["cell_type"].values
is_ho = ct == args.holdout
is_stim = (adata.obs["condition"] == "stimulated").values
train = adata[~(is_ho & is_stim)].copy()
print(f"{net} | {adata.n_vars} TFs | train: {train.n_obs:,} cells, held out: {int((is_ho & is_stim).sum()):,} stim {args.holdout}")

scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(train)
model.train(max_epochs=args.epochs, batch_size=32, early_stopping=True, early_stopping_patience=25)
pred, _ = model.predict(ctrl_key="control", stim_key="stimulated", celltype_to_predict=args.holdout)

# decoded held-out controls, same reason as in scgen_baseline.py
z = model.get_latent_representation(adata[is_ho & ~is_stim])
recon = model.module.generative(torch.tensor(z))["px"].detach().cpu().numpy()


def frame(x, obs, cond, prefix):
    df = pd.DataFrame(np.asarray(x), columns=adata.var_names, index=[f"{prefix}_{i}" for i in range(len(x))])
    o = pd.DataFrame({"cell_type": obs["cell_type"].values, "donor": obs["donor"].values, "cond": cond}, index=df.index)
    return df, o


# real delta TF for the held-out type, per donor then averaged (same as hcd_tf.py)
real = adata[is_ho]
s_real, o_real = frame(real.X, real.obs, np.where(real.obs["condition"] == "stimulated", "stim", "ctrl"), "real")
d_real = delta_tf(s_real, o_real, "cond", "ctrl", "stim", "cell_type", donor_key="donor")

ho_ctrl_obs = adata.obs[is_ho & ~is_stim]
s_pred, o_pred = frame(pred.X, pred.obs, "pred", "pred")
s_rec, o_rec = frame(recon, ho_ctrl_obs, "recon", "recon")
s_ctrl, o_ctrl = frame(X[is_ho & ~is_stim], ho_ctrl_obs, "ctrl", "ctrl")
s = pd.concat([s_pred, s_rec, s_ctrl])
o = pd.concat([o_pred, o_rec, o_ctrl])
d_b = delta_tf(s, o, "cond", "recon", "pred", "cell_type", donor_key="donor")
d_b_real = delta_tf(s, o, "cond", "ctrl", "pred", "cell_type", donor_key="donor")

# naive: mean delta TF over the training cell types
others = adata[~is_ho]
s_o, o_o = frame(others.X, others.obs, np.where(others.obs["condition"] == "stimulated", "stim", "ctrl"), "o")
d_naive = delta_tf(s_o, o_o, "cond", "ctrl", "stim", "cell_type", donor_key="donor").mean().to_frame(args.holdout).T

rows = []
for name, d in [("modelB", d_b), ("modelB_vs_real_ctrl", d_b_real), ("naive_tf", d_naive)]:
    d.index = [args.holdout]
    r = compare(d, d_real).iloc[0].to_dict()
    r.update(cytokine=args.cytokine, holdout=args.holdout, model=name, net=net)
    rows.append(r)
res = pd.DataFrame(rows)[["cytokine", "holdout", "net", "model", "pearson", "mse", "sign_agree", "n_tfs"]]
print(f"\n{args.cytokine} | held out: {args.holdout} | {net}")
print(res[["model", "pearson", "mse", "sign_agree", "n_tfs"]].round(3).to_string(index=False))

os.makedirs(args.outdir, exist_ok=True)
res.to_csv(os.path.join(args.outdir, "results.csv"), index=False)
tag = f"{args.cytokine}__{args.holdout.replace(' ', '_').replace('/', '-')}"
pd.concat([d_real.assign(kind="real"), d_b.assign(kind="modelB"), d_naive.assign(kind="naive_tf")]).to_csv(
    os.path.join(args.outdir, tag + "_delta_tf.csv"))

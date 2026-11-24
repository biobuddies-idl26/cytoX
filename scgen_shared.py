# Shared scGen ablation derived from the original scgen_baseline.py.
# One VAE per held-out cell type, one prediction per cytokine.
# python scgen_shared.py --holdout "CD14 Mono"

import argparse
import os

import anndata as ad
import numpy as np
import scgen
import torch
import wandb
from scgen._scgenvae import SCGENVAE

# newer scvi-tools reads "qzm"/"qzv" from inference(), scgen still returns "qz_m"/"qz_v",
# which crashes predict(). alias the keys.
_inference = SCGENVAE.inference


def _inference_fixed(self, x):
    out = _inference(self, x)
    out.setdefault("qzm", out["qz_m"])
    out.setdefault("qzv", out["qz_v"])
    return out


SCGENVAE.inference = _inference_fixed

# too few cells per condition to train/evaluate on
RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]

p = argparse.ArgumentParser()
p.add_argument("--cytokines", nargs="+", default=['IFN-beta', 'IFN-gamma', 'IL-1-beta', 'IL-15', 'IL-32-beta', 'IL-4'])
p.add_argument("--holdout", required=True)
p.add_argument("--data", default="data/hcd_subset.h5ad")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--keep-rare", action="store_true")
p.add_argument("--outdir", default="data/scgen_shared_runs")
p.add_argument("--n-layers", type=int, default=2)
p.add_argument("--n-hidden", type=int, default=800)
p.add_argument("--n-latent", type=int, default=100)
p.add_argument("--wandb-project", default=None, help="log to W&B if set")
p.add_argument("--wandb-group", default=None, help="e.g. L4, groups all tasks of one config")
args = p.parse_args()

CYTOKINES = list(dict.fromkeys(args.cytokines))
if "PBS" in CYTOKINES:
    raise SystemExit("--cytokines must contain only treatment names; PBS is included automatically")

adata = ad.read_h5ad(args.data)
adata = adata[adata.obs["cytokine"].isin(["PBS"] + CYTOKINES)].copy()
if not args.keep_rare:
    adata = adata[~adata.obs["cell_type"].isin(RARE)].copy()
if args.holdout not in set(adata.obs["cell_type"]):
    raise SystemExit(f"no cell type '{args.holdout}'. options: {sorted(adata.obs['cell_type'].unique())}")

# Same data, rare-cell filter and log-normalized X as scgen_baseline.py.
# Preserve individual treatment labels so predict() can select one cytokine.
adata.obs["condition"] = adata.obs["cytokine"].astype(str)
adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
del adata.layers["counts"]
ct = adata.obs["cell_type"].values
conditions = adata.obs["condition"].values
is_ho = ct == args.holdout
is_stim = conditions != "PBS"

# Verify that the original per-cytokine evaluation has both groups for
# every included cell type. Do not silently change the naive baseline.
for cytokine in CYTOKINES:
    pair_mask = np.isin(conditions, ["PBS", cytokine])
    if not np.any(is_ho & (conditions == cytokine)):
        raise SystemExit(f"Missing held-out cells: {args.holdout} / {cytokine}")
    for cell_type in np.unique(ct[pair_mask]):
        for condition in ("PBS", cytokine):
            if not np.any((ct == cell_type) & (conditions == condition)):
                raise SystemExit(f"Missing group: {cell_type} / {condition}")

# The ONLY scientific change: use all selected cytokines to train ONE VAE
# per held-out cell type. Exclude that cell type's response to EVERY cytokine.
# No donor-based filtering or grouping is introduced.
train = adata[~(is_ho & is_stim)].copy()
train.X = (train.X.toarray() if hasattr(train.X, "toarray") else np.asarray(train.X)).astype(np.float32, copy=False)
assert not ((train.obs["cell_type"] == args.holdout) & (train.obs["condition"] != "PBS")).any()
print(f"train: {train.n_obs:,} cells, held out: {int((is_ho & is_stim).sum()):,} treated {args.holdout}")
print(train.obs.groupby(["cell_type", "condition"], observed=True).size().unstack(fill_value=0))

# Avoid duplicate metrics if this run is accidentally launched twice.
if os.path.exists(os.path.join(args.outdir, "results.csv")):
    raise SystemExit("results.csv already exists; choose a new --outdir")

# initiate wandb run
run = None
if args.wandb_project:
    run = wandb.init(
        entity="scgen",                      # your team's W&B entity name
        project=args.wandb_project,
        group=args.wandb_group,              # e.g. all 4-layer tasks together
        name=f"shared | {args.holdout}",
        config={
            "cytokines": CYTOKINES,
            "holdout": args.holdout,
            "n_layers": args.n_layers,
            "n_hidden": args.n_hidden,                 
            "n_latent": args.n_latent,                 
            "max_epochs": args.epochs,
            "batch_size": 32,
            "n_genes": train.n_vars,
            "n_train_cells": train.n_obs,
            "early_stopping_monitor": "validation_loss", 
        },
    )


# initiate model setup
scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(train, n_layers=args.n_layers, n_hidden=args.n_hidden, n_latent=args.n_latent)
print(model.module)   # logs the actual architecture so you can confirm it
# stop on the weighted training objective (validation_loss), as in the scGen paper.
# elbo_validation counts the unweighted KL, which rises even while the model improves.
model.train(max_epochs=args.epochs, batch_size=32, early_stopping=True,
            early_stopping_patience=25, early_stopping_monitor="validation_loss")
# log loss curves per epoch
if run:
    hist = model.history                      # dict: metric name -> DataFrame indexed by epoch
    n_ep = max(len(v) for v in hist.values())
    for ep in range(n_ep):
        row = {k: float(v.iloc[ep, 0]) for k, v in hist.items() if ep < len(v)}
        run.log(row, step=ep)

# Save one model, not six copies of the same weights.
# The original runs are CPU runs; moving to CPU also supports the original
# direct decoder calls if training used a GPU.
model.to_device("cpu")
os.makedirs(args.outdir, exist_ok=True)
shared_tag = "shared__" + args.holdout.replace(" ", "_").replace("/", "-")
model.save(os.path.join(args.outdir, shared_tag), overwrite=True)

for cytokine in CYTOKINES:
    args.cytokine = cytokine
    # Recreate the original PBS-versus-one-cytokine evaluation matrix.
    pair = adata[adata.obs["cytokine"].isin(["PBS", cytokine])].copy()
    X = pair.X.toarray() if hasattr(pair.X, "toarray") else np.asarray(pair.X)
    pair.X = X.astype(np.float32)
    ct = pair.obs["cell_type"].values
    is_ho = ct == args.holdout
    is_stim = (pair.obs["condition"] == cytokine).values

    # Same upstream method AND celltype_to_predict argument as original.
    # predict() uses only the two selected conditions for latent arithmetic.
    pred, delta = model.predict(ctrl_key="PBS", stim_key=cytokine, celltype_to_predict=args.holdout)

    # decoded (reconstructed) held-out controls. decoder output is smoother than real
    # cells, so pred - real ctrl mixes that smoothing in with the cytokine effect.
    # pred - recon ctrl cancels it and is the fair delta for scGen.
    z = model.get_latent_representation(pair[is_ho & ~is_stim])
    recon = model.module.generative(torch.tensor(z))["px"].detach().cpu().numpy()

    ctrl = X[is_ho & ~is_stim].mean(0)
    real = X[is_ho & is_stim].mean(0)
    others = [c for c in np.unique(ct) if c != args.holdout]
    shift = np.mean([X[(ct == c) & is_stim].mean(0) - X[(ct == c) & ~is_stim].mean(0) for c in others], axis=0)
    # each entry: (predicted mean, control mean it is compared against)
    preds = {
        "scGen": (np.asarray(pred.X).mean(0), recon.mean(0)),
        "scGen_vs_real_ctrl": (np.asarray(pred.X).mean(0), ctrl),
        "naive": (ctrl + shift, ctrl),
    }

    # score delta (pred - ctrl) vs (real - ctrl). most genes barely move, so also
    # report r on the 100 genes with the largest real change.
    real_d = real - ctrl
    top = np.argsort(-np.abs(real_d))[:100]

    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, "results.csv")
    header = not os.path.exists(csv)
    print(f"\n{args.cytokine} | held out: {args.holdout}")
    print(f"{'model':<20}{'r_all':>8}{'mse':>10}{'r_top100':>10}")
    with open(csv, "a") as f:
        if header:
            f.write("cytokine,holdout,model,r_all,mse,r_top100\n")
        for name, (m, base) in preds.items():
            d = m - base
            r = np.corrcoef(d, real_d)[0, 1]
            mse = np.mean((d - real_d) ** 2)
            r_top = np.corrcoef(d[top], real_d[top])[0, 1]
            print(f"{name:<20}{r:>8.3f}{mse:>10.5f}{r_top:>10.3f}")
            f.write(f'"{args.cytokine}","{args.holdout}",{name},{r:.4f},{mse:.6f},{r_top:.4f}\n')
            if run:
                run.summary[f"{cytokine}/{name}/r_all"] = r
                run.summary[f"{cytokine}/{name}/mse"] = mse
                run.summary[f"{cytokine}/{name}/r_top100"] = r_top


    tag = f"{args.cytokine}__{args.holdout.replace(' ', '_').replace('/', '-')}"
    # Keep the original exported prediction condition label.
    pred.obs["condition"] = "control"
    pred.write_h5ad(os.path.join(args.outdir, tag + "_pred.h5ad"))
    ad.AnnData(X=recon, obs=pred.obs.copy(), var=pred.var.copy()).write_h5ad(os.path.join(args.outdir, tag + "_recon.h5ad"))
    np.save(os.path.join(args.outdir, tag + "_latent_shift.npy"), delta)
    del X, pair, pred, recon
if run:
    run.finish()

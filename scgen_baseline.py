# baseline: original scGen package for one cytokine with one cell type held out,
# compared to naive gene-space vector arithmetic (mean shift over training cell
# types added to the held-out control). scores the predicted change vs real change.
#
# python scgen_baseline.py --cytokine IFN-beta --holdout "CD14 Mono"
# needs scgen from github (pypi version is broken):
#   pip install "git+https://github.com/theislab/scgen.git"

import argparse
import os

import anndata as ad
import numpy as np
import scgen
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
p.add_argument("--cytokine", required=True)
p.add_argument("--holdout", required=True)
p.add_argument("--data", default="data/hcd_subset.h5ad")
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--keep-rare", action="store_true")
p.add_argument("--outdir", default="data/scgen_runs")
args = p.parse_args()

adata = ad.read_h5ad(args.data)
adata = adata[adata.obs["cytokine"].isin(["PBS", args.cytokine])].copy()
if not args.keep_rare:
    adata = adata[~adata.obs["cell_type"].isin(RARE)].copy()
if args.holdout not in set(adata.obs["cell_type"]):
    raise SystemExit(f"no cell type '{args.holdout}'. options: {sorted(adata.obs['cell_type'].unique())}")

adata.obs["condition"] = np.where(adata.obs["cytokine"] == "PBS", "control", "stimulated")
adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
del adata.layers["counts"]  # scgen trains on log-normalized X
X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
adata.X = X.astype(np.float32)

ct = adata.obs["cell_type"].values
is_ho = ct == args.holdout
is_stim = (adata.obs["condition"] == "stimulated").values
train = adata[~(is_ho & is_stim)].copy()
print(f"train: {train.n_obs:,} cells, held out: {int((is_ho & is_stim).sum()):,} stim {args.holdout}")

scgen.SCGEN.setup_anndata(train, batch_key="condition", labels_key="cell_type")
model = scgen.SCGEN(train)
model.train(max_epochs=args.epochs, batch_size=32, early_stopping=True, early_stopping_patience=25)
pred, _ = model.predict(ctrl_key="control", stim_key="stimulated", celltype_to_predict=args.holdout)

ctrl = X[is_ho & ~is_stim].mean(0)
real = X[is_ho & is_stim].mean(0)
others = [c for c in np.unique(ct) if c != args.holdout]
shift = np.mean([X[(ct == c) & is_stim].mean(0) - X[(ct == c) & ~is_stim].mean(0) for c in others], axis=0)
preds = {"scGen": np.asarray(pred.X).mean(0), "naive": ctrl + shift}

# score delta (pred - ctrl) vs (real - ctrl). most genes barely move, so also
# report r on the 100 genes with the largest real change.
real_d = real - ctrl
top = np.argsort(-np.abs(real_d))[:100]

os.makedirs(args.outdir, exist_ok=True)
csv = os.path.join(args.outdir, "results.csv")
header = not os.path.exists(csv)
print(f"\n{args.cytokine} | held out: {args.holdout}")
print(f"{'model':<8}{'r_all':>8}{'mse':>10}{'r_top100':>10}")
with open(csv, "a") as f:
    if header:
        f.write("cytokine,holdout,model,r_all,mse,r_top100\n")
    for name, m in preds.items():
        d = m - ctrl
        r = np.corrcoef(d, real_d)[0, 1]
        mse = np.mean((d - real_d) ** 2)
        r_top = np.corrcoef(d[top], real_d[top])[0, 1]
        print(f"{name:<8}{r:>8.3f}{mse:>10.5f}{r_top:>10.3f}")
        f.write(f'"{args.cytokine}","{args.holdout}",{name},{r:.4f},{mse:.6f},{r_top:.4f}\n')

tag = f"{args.cytokine}__{args.holdout.replace(' ', '_').replace('/', '-')}"
model.save(os.path.join(args.outdir, tag), overwrite=True)
pred.write_h5ad(os.path.join(args.outdir, tag + "_pred.h5ad"))

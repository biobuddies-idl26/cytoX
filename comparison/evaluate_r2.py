"""Score scGen reconstructions with BOTH metric families, plus the naive baseline.

Works on the PBMC reproduction (../data/reconstructed/scGen/pbmc.h5ad) and on HCD
reconstructions (hcd_<cytokine>.h5ad), so every row in the output is comparable.

Models scored for each held-out cell type:
  scGen  - the trained model's predicted stimulated cells
  naive  - held-out control mean + average (stim - ctrl) shift of the OTHER cell types
           (same definition as cytoX/scgen_baseline.py)

Metrics:
  Team (cytoX/scgen_baseline.py), computed on the change from control:
    delta_r_all     Pearson r of (pred - ctrl) vs (real - ctrl), all genes
    delta_mse       mean squared error of the same deltas
    delta_r_top100  same r on the 100 genes with the largest |real - ctrl|
  Paper (scGen Fig. 2, calc_R2), computed on mean expression:
    r2_meanexpr_all         R^2 of mean(pred) vs mean(real), all genes
    r2_meanexpr_top100deg   same on top 100 Wilcoxon DEGs (real_stim vs ctrl/pred)
    (scGen: bootstrapped 100x on 80% resamples, as in the paper; naive: point estimate)

Usage (from code/, scgen-repro env):
  python evaluate_r2.py ../data/reconstructed/scGen/pbmc.h5ad --label pbmc_kang --cytokine IFN-beta
  python evaluate_r2.py ../data/reconstructed/scGen/hcd_*.h5ad
Writes/updates ../results/scgen_scores.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

N_BOOT = 100
SEED = 0


def dense(x):
    return x.A if sparse.issparse(x) else np.asarray(x)


def r2(x, y):
    return stats.linregress(x, y).rvalue ** 2


def boot_r2(pred, real, rng, genes=None):
    if genes is not None:
        pred, real = pred[:, genes], real[:, genes]
    vals = np.zeros(N_BOOT)
    for i in range(N_BOOT):
        p = rng.choice(pred.shape[0], int(0.8 * pred.shape[0]))
        q = rng.choice(real.shape[0], int(0.8 * real.shape[0]))
        vals[i] = r2(pred[p].mean(0), real[q].mean(0))
    return vals.mean(), vals.std()


def delta_metrics(pred_mean, ctrl_mean, real_mean, top):
    d, real_d = pred_mean - ctrl_mean, real_mean - ctrl_mean
    return dict(delta_r_all=np.corrcoef(d, real_d)[0, 1],
                delta_mse=np.mean((d - real_d) ** 2),
                delta_r_top100=np.corrcoef(d[top], real_d[top])[0, 1])


def score_file(path, label, cytokine, ct_key="cell_type", cond_key="condition"):
    adata = sc.read(path)
    cond = adata.obs[cond_key].astype(str)
    cts = adata.obs[ct_key].astype(str)
    rng = np.random.RandomState(SEED)

    means = {}
    for ct in cts.unique():
        for tag in ("ctrl", "real_stim"):
            sel = (cond == f"{ct}_{tag}").values
            if sel.any():
                means[(ct, tag)] = dense(adata[sel].X).mean(0)

    rows = []
    for ct in sorted(cts.unique()):
        if not (cond == f"{ct}_pred_stim").any():
            continue  # no model trained for this cell type; used only for the naive shift
        cell = adata[(cts == ct).values].copy()
        cell.obs[cond_key] = cell.obs[cond_key].astype(str).astype("category")
        pred = dense(cell[(cell.obs[cond_key] == f"{ct}_pred_stim").values].X)
        real = dense(cell[(cell.obs[cond_key] == f"{ct}_real_stim").values].X)
        ctrl_m, real_m = means[(ct, "ctrl")], means[(ct, "real_stim")]

        others = [c for c in cts.unique() if c != ct and (c, "ctrl") in means and (c, "real_stim") in means]
        shift = np.mean([means[(c, "real_stim")] - means[(c, "ctrl")] for c in others], axis=0)
        naive_m = ctrl_m + shift

        top_delta = np.argsort(-np.abs(real_m - ctrl_m))[:100]
        sc.tl.rank_genes_groups(cell, groupby=cond_key, n_genes=100, method="wilcoxon")
        degs = [cell.var_names.get_loc(g) for g in cell.uns["rank_genes_groups"]["names"][f"{ct}_real_stim"]]

        base = dict(dataset=label, cytokine=cytokine, held_out=ct, n_real=real.shape[0], n_other_cell_types=len(others))
        a_all, a_sd = boot_r2(pred, real, rng)
        a_top, a_top_sd = boot_r2(pred, real, rng, degs)
        rows.append(dict(base, model="scGen", **delta_metrics(pred.mean(0), ctrl_m, real_m, top_delta),
                         r2_meanexpr_all=a_all, r2_meanexpr_all_sd=a_sd,
                         r2_meanexpr_top100deg=a_top, r2_meanexpr_top100deg_sd=a_top_sd))
        rows.append(dict(base, model="naive", **delta_metrics(naive_m, ctrl_m, real_m, top_delta),
                         r2_meanexpr_all=r2(naive_m, real_m), r2_meanexpr_all_sd=np.nan,
                         r2_meanexpr_top100deg=r2(naive_m[degs], real_m[degs]), r2_meanexpr_top100deg_sd=np.nan))
        for r in rows[-2:]:
            print(f"  {label:>16} | {ct:>14} | {r['model']:<6} delta r all={r['delta_r_all']:.3f} "
                  f"mse={r['delta_mse']:.5f} r top100={r['delta_r_top100']:.3f} | "
                  f"paper R2 all={r['r2_meanexpr_all']:.3f} top100DEG={r['r2_meanexpr_top100deg']:.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--label", default=None, help="dataset label (default: from file name)")
    ap.add_argument("--cytokine", default=None, help="default: parsed from hcd_<cytokine>.h5ad")
    ap.add_argument("--cell-type-key", default="cell_type")
    ap.add_argument("--out", default="../results/scgen_scores.csv")
    args = ap.parse_args()

    rows = []
    for f in args.files:
        stem = os.path.splitext(os.path.basename(f))[0]
        label = args.label or stem
        cyt = args.cytokine or (stem[4:] if stem.startswith("hcd_") else stem)
        print(f"Scoring {f}")
        rows += score_file(f, label, cyt, ct_key=args.cell_type_key)

    df = pd.DataFrame(rows).round(5)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if os.path.exists(args.out):
        old = pd.read_csv(args.out)
        old = old[~old["dataset"].isin(df["dataset"].unique())]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

# TF activity on the HCD subset with collecTRI.py.
#
# truth: real delta TF (cytokine - PBS) per cell type, donor-averaged, for every
#   cytokine in the subset. scored on all genes and on the scGen HVGs, so we can
#   see how much the HVG restriction costs (same check as check_kang.py).
#   python hcd_tf.py truth
#
# pred: for each scGen prediction in data/scgen_runs/*_pred.h5ad, score the
#   predicted cells and compare their delta TF to the real one for that held-out
#   cell type. everything is scored on the prediction's genes with the same net.
#   the predicted delta is pred - decoded ctrl (*_recon.h5ad), not pred - real ctrl:
#   decoded cells are smoother than real ones and that alone shifts ULM scores
#   (on IFN-beta/CD14 Mono the decoder shift by itself had r = -0.33 with the real delta).
#   python hcd_tf.py pred

import argparse
import glob
import os

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from collecTRI import load_collectri, filter_net, tf_scores, delta_tf, compare

RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]  # same exclusion as scgen_baseline.py

# canonical pathway TFs, should go up in most cell types if the scoring works
EXPECTED_UP = {
    "IFN-beta": ["STAT1", "STAT2", "IRF9"],
    "IFN-gamma": ["STAT1", "IRF1"],
    "IL-4": ["STAT6"],
    "IL-15": ["STAT5A", "STAT5B"],
    "IL-1-beta": ["NFKB1", "RELA"],
}

p = argparse.ArgumentParser()
p.add_argument("mode", choices=["truth", "pred"])
p.add_argument("--data", default="data/hcd_subset_allgenes.h5ad")
p.add_argument("--runs", default="data/scgen_runs")
args = p.parse_args()

adata = ad.read_h5ad(args.data)
adata = adata[~adata.obs["cell_type"].isin(RARE)].copy()
adata.obs_names_make_unique()  # delta_tf indexes obs by name
adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)
adata.obs["cytokine"] = adata.obs["cytokine"].astype(str)
net = load_collectri()
print(adata.shape, "|", net["source"].nunique(), "TFs in collectri")

if args.mode == "truth":
    sc.pp.filter_genes(adata, min_cells=10)
    hvg = adata.var_names[adata.var["highly_variable"]]
    cytokines = sorted(set(adata.obs["cytokine"]) - {"PBS"})

    results = {}
    for name, genes in [("all", adata.var_names), ("hvg", hvg)]:
        sub = adata[:, genes]
        n = filter_net(net, genes)
        print(f"[{name}] {len(genes)} genes -> {n['source'].nunique()} TFs with >=5 targets")
        s = tf_scores(sub, n)
        per_cyto = []
        for cyto in cytokines:
            keep = sub.obs["cytokine"].isin(["PBS", cyto]).values
            d = delta_tf(s[keep], sub.obs[keep], "cytokine", "PBS", cyto, "cell_type", donor_key="donor")
            d.insert(0, "cytokine", cyto)
            per_cyto.append(d.reset_index())
        out = pd.concat(per_cyto).set_index(["cytokine", "cell_type"])
        out.to_csv(f"data/hcd_delta_tf_{name}.csv")
        results[name] = out

    d = results["all"]
    for cyto in cytokines:
        dc_ = d.loc[cyto]
        print(f"\n{cyto}: top 5 TFs by mean delta across cell types")
        print(dc_.mean().sort_values(ascending=False).head(5).round(2).to_string())
        tfs = [t for t in EXPECTED_UP.get(cyto, []) if t in dc_.columns]
        if tfs:
            frac = (dc_[tfs] > 0).mean().round(2).to_dict()
            print(f"expected up, fraction of cell types with delta > 0: {frac}")

    print("\nhvg vs all-genes delta TF (how much the HVG restriction costs), mean over cell types:")
    rows = {}
    for cyto in cytokines:
        rows[cyto] = compare(results["hvg"].loc[cyto], results["all"].loc[cyto]).mean()
    print(pd.DataFrame(rows).T.round(3))

else:
    files = sorted(glob.glob(os.path.join(args.runs, "*_pred.h5ad")))
    if not files:
        raise SystemExit(f"no *_pred.h5ad in {args.runs}, run scgen_baseline.py first")

    rows = []
    for f in files:
        pred = ad.read_h5ad(f)
        cyto = os.path.basename(f).split("__")[0]
        holdout = str(pred.obs["cell_type"].iloc[0])
        genes = pred.var_names
        n = filter_net(net, genes)

        # real ctrl + real stim for the held-out type, on the prediction's genes
        real = adata[(adata.obs["cell_type"] == holdout) & adata.obs["cytokine"].isin(["PBS", cyto]), genes].copy()
        real.obs["cond"] = np.where(real.obs["cytokine"] == "PBS", "ctrl", "stim")
        rf = f.replace("_pred.h5ad", "_recon.h5ad")
        if not os.path.exists(rf):
            print(f"skipping {f}: no {os.path.basename(rf)}, rerun scgen_baseline.py for it")
            continue
        decoded = []
        for tag, path in [("pred", f), ("recon", rf)]:
            x = pred if tag == "pred" else ad.read_h5ad(path)
            dd = ad.AnnData(X=np.asarray(x.X), obs=x.obs[["cell_type", "donor"]].copy(), var=pd.DataFrame(index=genes))
            dd.obs["cond"] = tag
            dd.obs_names = [f"{tag}_{i}" for i in range(dd.n_obs)]
            decoded.append(dd)
        both = ad.concat([real] + decoded, join="inner")
        both.obs = both.obs[["cell_type", "donor", "cond"]]

        s = tf_scores(both, n)
        d_real = delta_tf(s, both.obs, "cond", "ctrl", "stim", "cell_type", donor_key="donor")
        d_pred = delta_tf(s, both.obs, "cond", "recon", "pred", "cell_type", donor_key="donor")
        r = compare(d_pred, d_real).iloc[0].to_dict()
        r.update(cytokine=cyto, holdout=holdout)
        rows.append(r)
        print(f"{cyto} | {holdout}: pearson {r['pearson']:.3f}, sign agree {r['sign_agree']:.2f}, {int(r['n_tfs'])} TFs")

    out = pd.DataFrame(rows)[["cytokine", "holdout", "pearson", "mse", "sign_agree", "n_tfs"]]
    out.to_csv(os.path.join(args.runs, "tf_results.csv"), index=False)
    print(f"\nsaved {os.path.join(args.runs, 'tf_results.csv')}")

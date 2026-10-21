import os

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import decoupler as dc


def load_collectri(path="data/collectri_human.csv"):
    # cache the network so we don't hit omnipath every run
    if os.path.exists(path):
        return pd.read_csv(path)
    net = dc.op.collectri(organism="human")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    net.to_csv(path, index=False)
    return net


def filter_net(net, genes, tmin=5):
    # keep only edges whose target is in our gene set, then drop TFs with < tmin targets.
    # use the SAME gene list for real and predicted data so the TF set matches.
    net = net[net["target"].isin(set(genes))]
    counts = net.groupby("source")["target"].nunique()
    keep = counts[counts >= tmin].index
    return net[net["source"].isin(keep)].reset_index(drop=True)


def tf_scores(adata, net, layer=None, tmin=5, bsize=5000):
    # per-cell TF activity with ulm. input must be log-normalized, not counts.
    # bsize = cells densified at a time; decoupler's default (250k) runs out of memory on big data
    X = adata.layers[layer] if layer else adata.X
    if sp.issparse(X):
        X = sp.csr_matrix(X)  # kang h5ad is CSC; row slicing needs CSR
    sample = X[:200].toarray() if sp.issparse(X) else np.asarray(X[:200])
    if np.allclose(sample, np.round(sample)) and sample.max() > 50:
        raise ValueError("looks like raw counts - normalize_total + log1p first")

    # run on a bare copy so we don't write into the caller's adata.obsm
    tmp = ad.AnnData(X=X, obs=pd.DataFrame(index=adata.obs_names),
                     var=pd.DataFrame(index=adata.var_names))
    dc.mt.ulm(tmp, net, tmin=tmin, empty=False, bsize=bsize)
    return tmp.obsm["score_ulm"]


def delta_tf(scores, obs, cond_key, ctrl, stim, group_key, donor_key=None):
    # mean(stim) - mean(ctrl) per group. if donor_key is given, compute per donor
    # first and then average across donors so one big donor doesn't dominate.
    df = scores.copy()
    df["_cond"] = obs.loc[df.index, cond_key].values
    df["_group"] = obs.loc[df.index, group_key].values
    keys = ["_group", "_cond"]
    if donor_key is not None:
        df["_donor"] = obs.loc[df.index, donor_key].values
        keys = ["_group", "_donor", "_cond"]

    means = df.groupby(keys, observed=True).mean()
    d = means.xs(stim, level="_cond") - means.xs(ctrl, level="_cond")
    d = d.dropna(how="all")
    if donor_key is not None:
        d = d.groupby(level="_group").mean()
    d.index.name = group_key
    return d


def compare(pred, true):
    # pred/true: groups x TFs delta tables. returns one row per group.
    tfs = pred.columns.intersection(true.columns)
    groups = pred.index.intersection(true.index)
    rows = []
    for g in groups:
        p, t = pred.loc[g, tfs].values, true.loc[g, tfs].values
        ok = ~(np.isnan(p) | np.isnan(t))
        p, t = p[ok], t[ok]
        rows.append({
            "group": g,
            "pearson": np.corrcoef(p, t)[0, 1],
            "mse": np.mean((p - t) ** 2),
            "sign_agree": np.mean(np.sign(p) == np.sign(t)),
            "n_tfs": ok.sum(),
        })
    return pd.DataFrame(rows).set_index("group")

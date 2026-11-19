#data handling --> load h5ad, hold out a cell type's treated cells, balance, train/val split
from typing import List, Optional

import anndata as ad
import numpy as np
import scipy.sparse as sp

#too few cells per condition to train / evaluate on (same list as scgen_baseline.py)
RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]


def dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def load_adata(
    path: str,
    cytokines: Optional[List[str]] = None,
    ctrl: str = "PBS",
    cytokine_key: str = "cytokine",
    cell_type_key: str = "cell_type",
    keep_rare: bool = False,
    use_obsm: Optional[str] = None,
) -> ad.AnnData:
    #read --> keep PBS + chosen cytokines --> drop rare types --> dense float32 X
    #use_obsm (e.g. score_ulm) --> X becomes TF activity scores for Model B
    adata = ad.read_h5ad(path)
    adata.obs[cytokine_key] = adata.obs[cytokine_key].astype(str)
    adata.obs[cell_type_key] = adata.obs[cell_type_key].astype(str)
    if cytokines:
        adata = adata[adata.obs[cytokine_key].isin([ctrl] + list(cytokines))].copy()
    if not keep_rare:
        adata = adata[~adata.obs[cell_type_key].isin(RARE)].copy()
    if use_obsm is not None:
        import pandas as pd

        S = adata.obsm[use_obsm]
        names = list(S.columns) if isinstance(S, pd.DataFrame) else [f"f{i}" for i in range(S.shape[1])]
        X = S.values if isinstance(S, pd.DataFrame) else S
        adata = ad.AnnData(X=dense(X), obs=adata.obs.copy(), var=pd.DataFrame(index=names))
    else:
        #rebuild from X only --> drops counts layers (newer anndata aliases layers[None] to X, so no del loop)
        adata = ad.AnnData(X=dense(adata.X), obs=adata.obs.copy(), var=adata.var.copy())
    return adata


def split_holdout(adata: ad.AnnData, holdout: str, ctrl: str = "PBS", cytokine_key: str = "cytokine", cell_type_key: str = "cell_type"):
    #extractor equivalent --> train = everything except the held-out type's treated cells (all cytokines)
    #held-out type's PBS cells stay in training, as in scGen
    ct = adata.obs[cell_type_key].values
    cy = adata.obs[cytokine_key].values
    if holdout not in set(ct):
        raise ValueError(f"no cell type '{holdout}'. options: {sorted(set(ct))}")
    is_ho = ct == holdout
    is_stim = cy != ctrl
    train = adata[~(is_ho & is_stim)].copy()
    ho_ctrl = adata[is_ho & ~is_stim].copy()
    ho_stim = adata[is_ho & is_stim].copy()
    return train, ho_ctrl, ho_stim


def balance(adata: ad.AnnData, key: str, seed: int = 0) -> ad.AnnData:
    #balancer equivalent --> upsample every group in obs[key] to the size of the largest group
    rng = np.random.default_rng(seed)
    labels = adata.obs[key].values
    groups = np.unique(labels)
    n_max = max((labels == g).sum() for g in groups)
    idx = np.concatenate([rng.choice(np.nonzero(labels == g)[0], n_max) for g in groups])
    out = adata[idx].copy()
    out.obs_names_make_unique()  #upsampling repeats cells --> unique names
    return out


def train_val_split(n: int, val_frac: float = 0.1, seed: int = 0):
    #random 90 / 10 split of cell indices
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]

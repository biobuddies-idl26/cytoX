"""scGen on Human Cytokine Dictionary data, using the ORIGINAL scGen code and settings.

This mirrors train_scGen.py (test_train_whole_data_one_celltype_out + reconstruct_whole_data)
with identical hyperparameters, so HCD results are directly comparable to the PBMC reproduction.

Run in the scgen-repro env, from code/:
  python -u hcd_scgen.py train IFN_beta CD4_Memory     # one model: hold out stimulated CD4_Memory
  python -u hcd_scgen.py reconstruct IFN_beta          # needs every cell type's model for IFN_beta
Inputs come from prepare_hcd.py (../data/hcd/<cytokine>/).
"""
import os
import sys

import anndata
import numpy as np
import pandas as pd
from scipy import sparse

import scgen

# Same as the PBMC reproduction (train_scGen.py __main__)
HP = dict(z_dimension=100, alpha=0.00005, dropout_rate=0.2, learning_rate=0.001)
N_EPOCHS = int(os.environ.get("HCD_EPOCHS", 300))   # override only for quick tests
BATCH_SIZE = 32
CT_KEY, COND_KEY, CTRL, STIM = "cell_type", "condition", "control", "stimulated"
DATA_DIR = "../data/hcd"


def load(cytokine, split):
    folder = os.path.join(DATA_DIR, cytokine)
    X = sparse.load_npz(os.path.join(folder, f"X_{split}.npz")).tocsr()
    obs = pd.read_csv(os.path.join(folder, f"obs_{split}.csv"), index_col=0)
    obs.index = obs.index.astype(str)
    genes = pd.read_csv(os.path.join(folder, "genes.csv"))["gene"].astype(str).values
    adata = anndata.AnnData(X, obs=obs)
    adata.var_names = genes
    return adata


def model_path(cytokine, cell_type):
    return f"../models/scGen/hcd/{cytokine}/{cell_type}/scgen"


def train(cytokine, cell_type):
    train_ad, valid_ad = load(cytokine, "train"), load(cytokine, "valid")
    hide_t = (train_ad.obs[CT_KEY] == cell_type) & (train_ad.obs[COND_KEY] == STIM)
    hide_v = (valid_ad.obs[CT_KEY] == cell_type) & (valid_ad.obs[COND_KEY] == STIM)
    net_train, net_valid = train_ad[~hide_t.values], valid_ad[~hide_v.values]
    print(f"{cytokine} | holding out stimulated {cell_type}: "
          f"{hide_t.sum()} train cells hidden, training on {net_train.n_obs}")

    os.makedirs(os.path.dirname(model_path(cytokine, cell_type)), exist_ok=True)  # original code saves before mkdir
    net = scgen.VAEArith(x_dimension=net_train.X.shape[1], model_path=model_path(cytokine, cell_type), **HP)
    net.train(net_train, use_validation=True, valid_data=net_valid, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE)
    net.sess.close()
    print(f"network_{cytokine}_{cell_type} has been trained!")


def trained_holdouts(cytokine):
    tasks = pd.read_csv(os.path.join(DATA_DIR, "tasks.tsv"), sep="\t", header=None, names=["cyt", "ct"])
    return set(tasks.loc[tasks["cyt"] == cytokine, "ct"])


def _piece(X, cell_type, tag, var_names):
    X = X.A if sparse.issparse(X) else np.asarray(X)
    ad = anndata.AnnData(X, obs=pd.DataFrame({COND_KEY: [f"{cell_type}_{tag}"] * len(X),
                                              CT_KEY: [cell_type] * len(X)}))
    ad.var_names = var_names
    return ad


def reconstruct(cytokine):
    """Same layout as reconstruct_whole_data: <ct>_ctrl / <ct>_pred_stim / <ct>_real_stim.
    ctrl and real_stim are written for EVERY cell type (the naive baseline needs them);
    pred_stim only for cell types that had a model trained."""
    train_ad = load(cytokine, "train")
    trained = trained_holdouts(cytokine)
    pieces = []
    for cell_type in sorted(train_ad.obs[CT_KEY].unique()):
        is_ct = train_ad.obs[CT_KEY] == cell_type
        ctrl = train_ad[(is_ct & (train_ad.obs[COND_KEY] == CTRL)).values]
        real = train_ad[(is_ct & (train_ad.obs[COND_KEY] == STIM)).values]
        pieces += [_piece(ctrl.X, cell_type, "ctrl", train_ad.var_names),
                   _piece(real.X, cell_type, "real_stim", train_ad.var_names)]
        if cell_type not in trained:
            continue

        net = scgen.VAEArith(x_dimension=train_ad.X.shape[1], model_path=model_path(cytokine, cell_type), **HP)
        net.restore_model()
        net_train = train_ad[~(is_ct & (train_ad.obs[COND_KEY] == STIM)).values]
        pred, _ = net.predict(adata=net_train, conditions={"ctrl": CTRL, "stim": STIM},
                              cell_type_key=CT_KEY, condition_key=COND_KEY, celltype_to_predict=cell_type)
        net.sess.close()
        pieces.append(_piece(pred, cell_type, "pred_stim", train_ad.var_names))
        print(f"Reconstructed {cytokine} / {cell_type}: {ctrl.n_obs} ctrl, {len(pred)} pred, {real.n_obs} real")

    out = pieces[0].concatenate(*pieces[1:])
    os.makedirs("../data/reconstructed/scGen", exist_ok=True)
    path = f"../data/reconstructed/scGen/hcd_{cytokine}.h5ad"
    out.write_h5ad(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    step = sys.argv[1]
    if step == "train":
        train(sys.argv[2], sys.argv[3])
    elif step == "reconstruct":
        reconstruct(sys.argv[2])
    else:
        raise SystemExit("usage: hcd_scgen.py train <cytokine> <cell_type> | reconstruct <cytokine>")

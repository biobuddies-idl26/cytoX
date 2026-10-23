"""Turn the cytoX HCD subset (subset_hcd.py output) into datasets the original scGen code can train on.

For each chosen cytokine this builds a PBMC-style dataset, matching cytoX/scgen_baseline.py:
  PBS cells   -> condition = "control"
  cytokine    -> condition = "stimulated"
  all cell types except rare ones (pDC, HSPC, ILC, Plasmablast), split 90/10 train/valid
  (the original PBMC data also came as a separate train/valid pair).

Run in a MODERN env (hcd-prep), not scgen-repro: the TF1 env's anndata 0.6 cannot read
h5ad files written by current anndata, so outputs are plain sparse-matrix + CSV files.

Usage (from code/):
  python prepare_hcd.py --input ../data/hcd_subset.h5ad --list
  python prepare_hcd.py --input ../data/hcd_subset.h5ad --cytokines IFN-beta --holdouts "CD14 Mono"
  python prepare_hcd.py --input ../data/hcd_subset.h5ad --cytokines IFN-beta IL-4      # every cell type held out in turn
Outputs: ../data/hcd/<cytokine>/ and ../data/hcd/tasks.tsv (one line per training job).
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

OUT_DIR = "../data/hcd"
RARE = ["pDC", "HSPC", "ILC", "Plasmablast"]  # same exclusion as cytoX/scgen_baseline.py


def safe(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def looks_like_counts(X):
    sample = X[:2000]
    vals = sample.data if sparse.issparse(sample) else np.asarray(sample).ravel()
    vals = vals[:200000]
    # log-normalized values are essentially never all whole numbers; raw counts always are
    return vals.size > 0 and np.allclose(vals, np.round(vals)) and vals.max() > 1


def save_split(adata, folder, split):
    X = adata.X if sparse.issparse(adata.X) else sparse.csr_matrix(adata.X)
    sparse.save_npz(os.path.join(folder, f"X_{split}.npz"), sparse.csr_matrix(X, dtype=np.float32))
    adata.obs[["cell_type", "condition", "donor", "orig_cell_type", "orig_cytokine"]].to_csv(
        os.path.join(folder, f"obs_{split}.csv"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cell-type-col", default="cell_type")
    ap.add_argument("--cytokine-col", default="cytokine")
    ap.add_argument("--donor-col", default="donor", help="optional; ignored if the column doesn't exist")
    ap.add_argument("--control", default="PBS")
    ap.add_argument("--cytokines", nargs="*", default=[])
    ap.add_argument("--holdouts", nargs="*", default=None,
                    help="cell types to hold out (one model each); default: every non-rare cell type")
    ap.add_argument("--exclude", nargs="*", default=RARE, help="cell types to drop entirely")
    ap.add_argument("--cells-per-group", type=int, default=5000,
                    help="cap per (cell type x control/stimulated); the cytoX subset is already 500 per group")
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print available values and exit")
    args = ap.parse_args()

    full = sc.read_h5ad(args.input, backed="r")
    obs = full.obs
    for col in (args.cell_type_col, args.cytokine_col):
        if col not in obs.columns:
            raise SystemExit(f"Column '{col}' not in obs. Available: {list(obs.columns)}")
    has_donor = args.donor_col in obs.columns

    if args.list:
        print(f"{full.n_obs} cells x {full.n_vars} genes | obs columns: {list(obs.columns)}")
        print(obs.groupby([args.cell_type_col, args.cytokine_col], observed=True).size().unstack(fill_value=0))
        return
    if not args.cytokines:
        raise SystemExit("Pass --cytokines (run with --list first to see the names).")
    if not has_donor:
        print(f"Note: no '{args.donor_col}' column; donor info won't be carried through.")

    cyto = obs[args.cytokine_col].astype(str)
    missing = [c for c in args.cytokines + [args.control] if c not in set(cyto)]
    if missing:
        raise SystemExit(f"Not found in '{args.cytokine_col}': {missing}")

    all_cts = sorted(set(obs[args.cell_type_col].astype(str)) - set(args.exclude))
    holdouts = all_cts if args.holdouts is None else args.holdouts
    bad = [h for h in holdouts if h not in all_cts]
    if bad:
        raise SystemExit(f"Unknown or excluded hold-out cell types: {bad}. Options: {all_cts}")

    rng = np.random.RandomState(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)
    tasks, manifest = [], {}

    for cyt in args.cytokines:
        ct_col = obs[args.cell_type_col].astype(str)
        mask = (cyto.isin([args.control, cyt]) & ct_col.isin(all_cts)).values
        ad = full[mask].to_memory()
        ad.obs = pd.DataFrame({
            "cell_type": [safe(c) for c in ad.obs[args.cell_type_col].astype(str)],
            "condition": np.where(ad.obs[args.cytokine_col].astype(str) == args.control, "control", "stimulated"),
            "donor": ad.obs[args.donor_col].astype(str).values if has_donor else "NA",
            "orig_cell_type": ad.obs[args.cell_type_col].astype(str).values,
            "orig_cytokine": ad.obs[args.cytokine_col].astype(str).values,
        }, index=ad.obs_names)

        if looks_like_counts(ad.X):
            print(f"  {cyt}: .X looks like raw counts -> normalize_total(1e4) + log1p")
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)

        keep = []
        for _, idx in ad.obs.groupby(["cell_type", "condition"], observed=True).indices.items():
            keep.extend(rng.choice(idx, min(len(idx), args.cells_per_group), replace=False))
        ad = ad[np.sort(keep)].copy()

        is_valid = np.zeros(ad.n_obs, dtype=bool)
        for _, idx in ad.obs.groupby(["cell_type", "condition"], observed=True).indices.items():
            n_val = max(1, int(round(len(idx) * args.valid_frac)))
            is_valid[rng.choice(idx, n_val, replace=False)] = True

        folder = os.path.join(OUT_DIR, safe(cyt))
        os.makedirs(folder, exist_ok=True)
        save_split(ad[~is_valid], folder, "train")
        save_split(ad[is_valid], folder, "valid")
        pd.Series(ad.var_names).to_csv(os.path.join(folder, "genes.csv"), index=False, header=["gene"])

        counts = ad.obs.groupby(["cell_type", "condition"], observed=True).size().unstack(fill_value=0)
        print(f"== {cyt} -> {folder}  ({(~is_valid).sum()} train / {is_valid.sum()} valid)\n{counts}\n")
        manifest[safe(cyt)] = {"cytokine": cyt, "holdouts": holdouts, "counts": counts.to_dict()}
        tasks += [f"{safe(cyt)}\t{safe(h)}" for h in holdouts]

    with open(os.path.join(OUT_DIR, "tasks.tsv"), "w") as f:
        f.write("\n".join(tasks) + "\n")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump({"input": args.input, "control": args.control, "excluded": args.exclude,
                   "cell_types": all_cts, "seed": args.seed, "cytokines": manifest}, f, indent=2)
    print(f"Wrote {len(tasks)} training tasks to {OUT_DIR}/tasks.tsv")


if __name__ == "__main__":
    main()

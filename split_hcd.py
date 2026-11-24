# split hcd_subset.h5ad (single cells, 7k HVGs, log-normalized) into train / test
# for one model that has never seen the held-out cell types treated (scGen out-of-sample design).
#
#   train : every cell of the remaining cell types + the held-out types' PBS (control) cells.
#           the model carves its own random 10% validation set out of this at runtime
#           (CytoXAttention.fit: val_frac=0.1, patience=25, like scGen's train_size=0.9).
#   test  : all cells of the held-out cell types (PBS = model input, cytokines = ground truth)
#
# python split_hcd.py --data data/hcd_subset.h5ad --holdout "CD14 Mono" NK
# add --also data/hcd_subset_allgenes.h5ad to apply the SAME split to other files
# (e.g. the all-gene copy for TF scoring / Model B), matched by cell name

import argparse
import os

import anndata as ad
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/hcd_subset.h5ad")
p.add_argument("--holdout", nargs="+", required=True, help="cell types whose treated cells are never seen in training")
p.add_argument("--ctrl", default="PBS")
p.add_argument("--cell-type-key", default="cell_type")
p.add_argument("--cytokine-key", default="cytokine")
p.add_argument("--outdir", default="data/split")
p.add_argument("--also", nargs="*", default=[], help="other h5ads with the same cells to split identically")
args = p.parse_args()

adata = ad.read_h5ad(args.data)
if not adata.obs_names.is_unique:
    raise SystemExit("cell names are not unique, so the split can't be matched across files")
ct = adata.obs[args.cell_type_key].astype(str).values
cy = adata.obs[args.cytokine_key].astype(str).values

missing = set(args.holdout) - set(ct)
if missing:
    raise SystemExit(f"not in the data: {sorted(missing)}. options: {sorted(set(ct))}")

is_ho = np.isin(ct, args.holdout)
is_ctrl = cy == args.ctrl
is_train = ~is_ho | is_ctrl  # held-out types keep only their PBS cells
is_test = is_ho              # PBS = input, treated = ground truth
assert not (is_train & is_ho & ~is_ctrl).any(), "held-out treated cells leaked into train"

splits = [("train", is_train), ("test", is_test)]
os.makedirs(args.outdir, exist_ok=True)
for name, mask in splits:
    sub = adata[mask].copy()
    path = os.path.join(args.outdir, f"{name}.h5ad")
    sub.write_h5ad(path, compression="gzip")
    np.savetxt(os.path.join(args.outdir, f"{name}_cells.txt"), np.asarray(sub.obs_names), fmt="%s")
    print(f"{name:<5} {sub.n_obs:>9,} cells -> {path}")
    for t in sorted(set(ct[mask])):
        m = mask & (ct == t)
        print(f"      {t:<20} PBS {int((m & is_ctrl).sum()):>7,} | treated {int((m & ~is_ctrl).sum()):>7,}")

# same cells -> same split in every other file (all-gene copy, TF scores, ...)
for other in args.also:
    o = ad.read_h5ad(other, backed="r")
    stem = os.path.splitext(os.path.basename(other))[0]
    for name, mask in splits:
        cells = adata.obs_names[mask]
        missing = cells.difference(o.obs_names)
        if len(missing):
            raise SystemExit(f"{other}: {len(missing):,} {name} cells not found (e.g. {missing[0]})")
        path = os.path.join(args.outdir, f"{stem}_{name}.h5ad")
        o[o.obs_names.get_indexer(cells)].to_memory().write_h5ad(path, compression="gzip")
        print(f"{stem}_{name}: {len(cells):,} cells -> {path}")
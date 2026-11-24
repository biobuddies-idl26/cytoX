#train the attention VAE with one cell type held out --> predict every cytokine --> score vs baselines
#
#python cytoX_attention/train.py --holdout "CD14 Mono" --encoder hybrid --shift learned
#one model, two unseen cell types, pre-split files from split_hcd.py:
#python cytoX_attention/train.py --train data/split/train.h5ad --test data/split/test.h5ad --holdout "CD14 Mono" NK
#python cytoX_attention/train.py --holdout "CD14 Mono" --space tf --use-obsm score_ulm --data data/hcd_subset_tf.h5ad

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  #run as a script --> package importable

from cytoX_attention.data import load_adata, split_holdout  # noqa: E402
from cytoX_attention.metrics import evaluate, naive_shift, top_deg, unchanged  # noqa: E402
from cytoX_attention.model import CytoXAttention  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/hcd_subset.h5ad", help="single file, split here (ignored with --train/--test)")
    p.add_argument("--train", default=None, help="pre-split training cells (split_hcd.py)")
    p.add_argument("--test", default=None, help="pre-split held-out cells (split_hcd.py)")
    p.add_argument("--holdout", nargs="+", default=None,
                   help="held-out cell type(s), one model for all; default with --test: every type in the test file")
    p.add_argument("--cytokines", nargs="+", default=None, help="default: every non-PBS cytokine in the data")
    p.add_argument("--ctrl", default="PBS")
    p.add_argument("--space", choices=["genes", "tf"], default="genes")
    p.add_argument("--use-obsm", default=None, help="obsm key to use as X (e.g. score_ulm for Model B)")
    p.add_argument("--encoder", choices=["mlp", "attention", "hybrid"], default="hybrid")
    p.add_argument("--shift", choices=["mean", "cosine", "learned"], default="learned")
    p.add_argument("--shift-query", choices=["celltype", "cell"], default="celltype")
    p.add_argument("--decoded-weight", type=float, default=0.0, help=">0 --> also fit shift attention on decoded means")
    p.add_argument("--shift-epochs", type=int, default=200)
    p.add_argument("--n-latent", type=int, default=100)
    p.add_argument("--n-tokens", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-attn-layers", type=int, default=2)
    p.add_argument("--n-hidden", type=int, default=800)
    p.add_argument("--n-enc-layers", type=int, default=2, help="hidden layers in the scGen MLP encoder path (mlp / hybrid)")
    p.add_argument("--n-dec-layers", type=int, default=2, help="hidden layers in the MLP decoder")
    p.add_argument("--ff-mult", type=int, default=2, help="attention feed-forward width = ff_mult * d_model")
    p.add_argument("--dropout", type=float, default=0.2, help="dropout in MLP layers")
    p.add_argument("--attn-dropout", type=float, default=0.1, help="dropout in attention blocks")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-rare", action="store_true")
    p.add_argument("--outdir", default="data/cytox_attention_runs")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args()


def load_split(args):
    #--> train (model input), ho_ctrl / ho_stim (held-out PBS input / treated ground truth), holdout list
    kw = dict(cytokines=args.cytokines, ctrl=args.ctrl, keep_rare=args.keep_rare, use_obsm=args.use_obsm)
    if args.train or args.test:
        if not (args.train and args.test):
            raise SystemExit("--train and --test go together")
        train, test = load_adata(args.train, **kw), load_adata(args.test, **kw)
        if list(train.var_names) != list(test.var_names):
            raise SystemExit("train and test have different features")
        holdouts = args.holdout or sorted(set(test.obs["cell_type"]))
        missing = sorted(set(holdouts) - set(test.obs["cell_type"]))
        if missing:
            raise SystemExit(f"{missing} not in --test. options: {sorted(set(test.obs['cell_type']))}")
        leak = train.obs["cell_type"].isin(holdouts) & (train.obs["cytokine"] != args.ctrl)
        if leak.any():
            raise SystemExit(f"{int(leak.sum()):,} treated held-out cells found in --train")
        in_ho = test.obs["cell_type"].isin(holdouts).values
        is_ctrl = (test.obs["cytokine"] == args.ctrl).values
        ho_ctrl, ho_stim = test[in_ho & is_ctrl].copy(), test[in_ho & ~is_ctrl].copy()
    else:
        if not args.holdout:
            raise SystemExit("--holdout is required with --data")
        holdouts = args.holdout
        adata = load_adata(args.data, **kw)
        train, ho_ctrl, ho_stim = split_holdout(adata, holdouts, args.ctrl)
    return train, ho_ctrl, ho_stim, holdouts


def main():
    args = parse()
    train, ho_ctrl, ho_stim, holdouts = load_split(args)
    cytokines = args.cytokines or sorted(set(train.obs["cytokine"]) - {args.ctrl})
    print(f"train: {train.n_obs:,} cells | held out: {', '.join(holdouts)} "
          f"({ho_ctrl.n_obs:,} PBS input, {ho_stim.n_obs:,} treated) | features: {train.n_vars}")

    #stage 1 --> VAE (random 10% of train held out for early stopping), stage 2 --> shift attention
    model = CytoXAttention(
        train, cytokines, ctrl=args.ctrl, shift_mode=args.shift, shift_query=args.shift_query,
        encoder_type=args.encoder, space=args.space, n_latent=args.n_latent, n_tokens=args.n_tokens,
        d_model=args.d_model, n_heads=args.n_heads, n_attn_layers=args.n_attn_layers, n_hidden=args.n_hidden,
        n_enc_layers=args.n_enc_layers, n_dec_layers=args.n_dec_layers, ff_mult=args.ff_mult,
        dropout=args.dropout, attn_dropout=args.attn_dropout,
    )
    model.fit(max_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, patience=args.patience, seed=args.seed)
    info = model.fit_shift(decoded_weight=args.decoded_weight, epochs=args.shift_epochs, seed=args.seed, verbose=True)
    print(f"shift attention ({args.shift}): {info}")

    #baselines only see training cell types --> another held-out type's treated cells can never leak in
    Xtr = np.asarray(train.X)
    ct_tr = train.obs["cell_type"].values
    cy_tr = train.obs["cytokine"].values
    others = ~np.isin(ct_tr, holdouts)
    ho_tag = "+".join(h.replace(" ", "_").replace("/", "-") for h in holdouts)
    tag = f"{ho_tag}__{args.encoder}_{args.shift}_s{args.seed}"
    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, "results.csv")
    header = not os.path.exists(csv)
    alphas, scores = {}, []
    with open(csv, "a") as f:
        if header:
            f.write("cytokine,holdout,model,encoder,shift,seed,R2,r_delta,mse_delta,r_all,mse,r_top100\n")
        for k in holdouts:
            ctrl_k = ho_ctrl[ho_ctrl.obs["cell_type"] == k]
            stim_k = ho_stim[ho_stim.obs["cell_type"] == k]
            Xc = np.asarray(ctrl_k.X)
            if len(Xc) == 0:
                print(f"skip {k}: no PBS cells to predict from")
                continue
            #pool = training types + this held-out type's PBS cells --> input for naive / unchanged
            Xp = np.vstack([Xtr[others], Xc])
            ctp = np.concatenate([ct_tr[others], [k] * len(Xc)])
            cyp = np.concatenate([cy_tr[others], [args.ctrl] * len(Xc)])
            for c in cytokines:
                Xs = np.asarray(stim_k[stim_k.obs["cytokine"] == c].X)
                if len(Xs) == 0:
                    print(f"skip {k} / {c}: no held-out treated cells to score")
                    continue
                pred, _, alpha = model.predict(ctrl_k, k, c)
                alphas[(k, c)] = alpha
                deg = top_deg(Xs, Xc, 100)  #same DEGs for every model --> fair comparison
                preds = {
                    "cytoX_attention": np.asarray(pred.X).mean(0),
                    "naive": naive_shift(Xp, ctp, cyp, k, c, args.ctrl),
                    "unchanged": unchanged(Xp, ctp, cyp, k, args.ctrl),
                }
                print(f"\n{c} | held out: {k}")
                print(f"{'model':<16}{'R2':>8}{'r_delta':>9}{'mse_d':>10}{'r_all':>8}{'r_top100':>10}")
                for name, m in preds.items():
                    s = evaluate(m, Xs, Xc, deg_idx=deg)
                    scores.append({"model": name, **s})
                    print(f"{name:<16}{s['R2']:>8.3f}{s['r_delta']:>9.3f}{s['mse_delta']:>10.5f}{s['r_all']:>8.3f}{s['r_top100']:>10.3f}")
                    f.write(f'"{c}","{k}",{name},{args.encoder},{args.shift},{args.seed},'
                            f"{s['R2']:.4f},{s['r_delta']:.4f},{s['mse_delta']:.6f},{s['r_all']:.4f},{s['mse']:.6f},{s['r_top100']:.4f}\n")
                if not args.no_save:
                    kt = k.replace(" ", "_").replace("/", "-")
                    pred.write_h5ad(os.path.join(args.outdir, f"{tag}__{kt}__{c}_pred.h5ad"))
                top = sorted(alpha.items(), key=lambda kv: -kv[1])[:3]
                print("  shift attention top cell types: " + ", ".join(f"{t} {w:.2f}" for t, w in top))

    #report format --> mean +/- std over (cell type, cytokine) test pairs
    if scores:
        print(f"\nsummary over test pairs ({', '.join(holdouts)})")
        print(f"{'model':<16}{'n':>4}{'R2':>16}{'r_delta':>16}{'mse_delta':>20}")
        for name in dict.fromkeys(s["model"] for s in scores):
            rows = [s for s in scores if s["model"] == name]
            ms = lambda key: f"{np.nanmean([r[key] for r in rows]):.3f} +/- {np.nanstd([r[key] for r in rows]):.3f}"  # noqa: E731
            mse = [r["mse_delta"] for r in rows]
            print(f"{name:<16}{len(rows):>4}{ms('R2'):>16}{ms('r_delta'):>16}{np.mean(mse):>11.5f} +/- {np.std(mse):.5f}")

    if not args.no_save:
        model.save(os.path.join(args.outdir, tag))
        np.save(os.path.join(args.outdir, f"{tag}__alpha.npy"), alphas, allow_pickle=True)
        if args.encoder != "mlp":
            maps = model.attention_maps(ho_ctrl)
            np.savez(os.path.join(args.outdir, f"{tag}__attention.npz"), token_names=_token_names(model, train), **maps)


def _token_names(model, adata):
    #tf space --> tokens are TF names; genes space --> tokens are module_0..module_K
    tok = model.vae.encoder.tokenizer
    if hasattr(tok, "assignment"):
        return np.array([f"module_{k}" for k in range(tok.n_tokens)])
    return np.array(adata.var_names)


if __name__ == "__main__":
    main()

#train the attention VAE with one cell type held out --> predict every cytokine --> score vs baselines
#
#python cytoX_attention/train.py --holdout "CD14 Mono" --encoder hybrid --shift learned
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
    p.add_argument("--data", default="data/hcd_subset.h5ad")
    p.add_argument("--holdout", required=True)
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


def main():
    args = parse()
    adata = load_adata(args.data, args.cytokines, args.ctrl, keep_rare=args.keep_rare, use_obsm=args.use_obsm)
    cytokines = args.cytokines or sorted(set(adata.obs["cytokine"]) - {args.ctrl})
    train, ho_ctrl, ho_stim = split_holdout(adata, args.holdout, args.ctrl)
    print(f"train: {train.n_obs:,} cells | held out: {ho_stim.n_obs:,} treated {args.holdout} | features: {adata.n_vars}")

    #stage 1 --> VAE, stage 2 --> shift attention on training cell types
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

    X = np.asarray(adata.X)
    ct = adata.obs["cell_type"].values
    cy = adata.obs["cytokine"].values
    Xc = np.asarray(ho_ctrl.X)
    tag = f"{args.holdout.replace(' ', '_').replace('/', '-')}__{args.encoder}_{args.shift}_s{args.seed}"
    os.makedirs(args.outdir, exist_ok=True)
    csv = os.path.join(args.outdir, "results.csv")
    header = not os.path.exists(csv)
    alphas = {}
    with open(csv, "a") as f:
        if header:
            f.write("cytokine,holdout,model,encoder,shift,seed,R2,r_delta,mse_delta,r_all,mse,r_top100\n")
        for c in cytokines:
            Xs = np.asarray(ho_stim[ho_stim.obs["cytokine"] == c].X)
            if len(Xs) == 0:
                print(f"skip {c}: no held-out treated cells to score")
                continue
            pred, _, alpha = model.predict(ho_ctrl, args.holdout, c)
            alphas[c] = alpha
            deg = top_deg(Xs, Xc, 100)  #same DEGs for every model --> fair comparison
            preds = {
                "cytoX_attention": np.asarray(pred.X).mean(0),
                "naive": naive_shift(X, ct, cy, args.holdout, c, args.ctrl),
                "unchanged": unchanged(X, ct, cy, args.holdout, args.ctrl),
            }
            print(f"\n{c} | held out: {args.holdout}")
            print(f"{'model':<16}{'R2':>8}{'r_delta':>9}{'mse_d':>10}{'r_all':>8}{'r_top100':>10}")
            for name, m in preds.items():
                s = evaluate(m, Xs, Xc, deg_idx=deg)
                print(f"{name:<16}{s['R2']:>8.3f}{s['r_delta']:>9.3f}{s['mse_delta']:>10.5f}{s['r_all']:>8.3f}{s['r_top100']:>10.3f}")
                f.write(f'"{c}","{args.holdout}",{name},{args.encoder},{args.shift},{args.seed},'
                        f"{s['R2']:.4f},{s['r_delta']:.4f},{s['mse_delta']:.6f},{s['r_all']:.4f},{s['mse']:.6f},{s['r_top100']:.4f}\n")
            if not args.no_save:
                pred.write_h5ad(os.path.join(args.outdir, f"{tag}__{c}_pred.h5ad"))
            top = sorted(alpha.items(), key=lambda kv: -kv[1])[:3]
            print("  shift attention top cell types: " + ", ".join(f"{t} {w:.2f}" for t, w in top))

    if not args.no_save:
        model.save(os.path.join(args.outdir, tag))
        np.save(os.path.join(args.outdir, f"{tag}__alpha.npy"), alphas, allow_pickle=True)
        if args.encoder != "mlp":
            maps = model.attention_maps(ho_ctrl)
            np.savez(os.path.join(args.outdir, f"{tag}__attention.npz"), token_names=_token_names(model, adata), **maps)


def _token_names(model, adata):
    #tf space --> tokens are TF names; genes space --> tokens are module_0..module_K
    tok = model.vae.encoder.tokenizer
    if hasattr(tok, "assignment"):
        return np.array([f"module_{k}" for k in range(tok.n_tokens)])
    return np.array(adata.var_names)


if __name__ == "__main__":
    main()

#smoke tests on tiny synthetic data --> check the code paths run, not model quality
import os
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from cytoX_attention import AttentionVAE, CytoXAttention, ShiftAttention  # noqa: E402
from cytoX_attention.data import balance, split_holdout  # noqa: E402
from cytoX_attention.metrics import evaluate, top_deg  # noqa: E402
from cytoX_attention.shift_attention import compute_celltype_deltas  # noqa: E402

TYPES = ["A", "B", "C", "D"]
CYTOS = ["IFN", "IL4"]
SMALL = dict(n_latent=8, n_hidden=32, n_tokens=8, d_model=16, n_heads=2, n_attn_layers=1)


def make_adata(n_genes=60, n_per=25, seed=0):
    #cell type --> baseline profile, cytokine --> shared shift + type-specific scaling
    rng = np.random.default_rng(seed)
    base = {t: rng.normal(1, 0.5, n_genes) for t in TYPES}
    shift = {c: rng.normal(0, 0.8, n_genes) * (rng.random(n_genes) < 0.2) for c in CYTOS}
    X, ct, cy = [], [], []
    for i, t in enumerate(TYPES):
        for c in ["PBS"] + CYTOS:
            mean = base[t] + (0 if c == "PBS" else shift[c] * (1 + 0.3 * i))
            X.append(np.clip(mean + rng.normal(0, 0.3, (n_per, n_genes)), 0, None))
            ct += [t] * n_per
            cy += [c] * n_per
    obs = pd.DataFrame({"cell_type": ct, "cytokine": cy}, index=[f"c{i}" for i in range(len(ct))])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return ad.AnnData(X=np.vstack(X).astype(np.float32), obs=obs, var=var)


@pytest.mark.parametrize("encoder", ["mlp", "attention", "hybrid"])
@pytest.mark.parametrize("space", ["genes", "tf"])
def test_forward_backward(encoder, space):
    vae = AttentionVAE(30, encoder_type=encoder, space=space, **SMALL)
    x = torch.rand(6, 30)
    x_hat, mu, logvar = vae(x)
    loss, _, _ = vae.loss(x, x_hat, mu, logvar)
    loss.backward()
    assert x_hat.shape == x.shape and mu.shape == (6, 8)
    if encoder != "mlp":
        A = vae.get_attention(x)
        T = (8 if space == "genes" else 30) + 1  #+1 --> [CLS]
        assert A.shape == (6, 1, 2, T, T)


def test_layer_options():
    #separate encoder / decoder MLP depth + attention feed-forward width
    vae = AttentionVAE(30, encoder_type="hybrid", space="genes", n_enc_layers=3, n_dec_layers=1, ff_mult=4, **SMALL)
    n_lin = lambda seq: sum(isinstance(m, torch.nn.Linear) for m in seq)  # noqa: E731
    assert n_lin(vae.encoder.trunk) == 3 and n_lin(vae.decoder.trunk) == 1
    assert vae.encoder.blocks[0].ff[0].out_features == 4 * SMALL["d_model"]
    x = torch.rand(6, 30)
    x_hat, mu, logvar = vae(x)
    vae.loss(x, x_hat, mu, logvar)[0].backward()
    #n_layers shortcut --> sets both depths
    vae2 = AttentionVAE(30, encoder_type="mlp", n_layers=3, **SMALL)
    assert n_lin(vae2.encoder.trunk) == 3 and n_lin(vae2.decoder.trunk) == 3


def test_loss_matches_scgen_formula():
    vae = AttentionVAE(10, encoder_type="mlp", **SMALL)
    x, x_hat = torch.rand(4, 10), torch.rand(4, 10)
    mu, logvar = torch.randn(4, 8), torch.randn(4, 8)
    loss, _, _ = vae.loss(x, x_hat, mu, logvar)
    #scGen: kl(Normal(mu, sigma), Normal(0, 1)) summed --> 0.5 * rl + 0.5 * kl_weight * kld --> mean
    q = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
    kld = torch.distributions.kl_divergence(q, torch.distributions.Normal(0.0, 1.0)).sum(1)
    ref = (0.5 * ((x - x_hat) ** 2).sum(1) + 0.5 * vae.kl_weight * kld).mean()
    assert torch.allclose(loss, ref, atol=1e-6)


def test_learned_shift_starts_as_mean():
    M, V = torch.randn(5, 8), torch.randn(5, 8)
    km = torch.tensor([True, True, False, True, True])
    q = torch.randn(1, 8)
    s_mean, _ = ShiftAttention(8, 2, mode="mean")(q, 0, M, V, km)
    s_learn, a = ShiftAttention(8, 2, mode="learned")(q, 1, M, V, km)
    assert torch.allclose(s_mean, s_learn, atol=1e-6)
    assert torch.allclose(s_mean, V[km].mean(0, keepdim=True), atol=1e-6)
    assert a[0, 2] == 0  #masked key gets no weight


def test_holdout_never_a_key_and_predict():
    adata = make_adata()
    train, ho_ctrl, ho_stim = split_holdout(adata, "B")
    assert not ((train.obs["cell_type"] == "B") & (train.obs["cytokine"] != "PBS")).any()
    assert ho_ctrl.n_obs == 25 and ho_stim.n_obs == 50
    for shift in ["mean", "cosine", "learned"]:
        m = CytoXAttention(train, CYTOS, shift_mode=shift, encoder_type="hybrid", device="cpu", **SMALL)
        m.fit(max_epochs=2, verbose=False)
        m.fit_shift(epochs=3, decoded_weight=0.1 if shift == "learned" else 0.0)
        types = m.stats["types"]
        assert not m.stats["mask"][:, types.index("B")].any()  #B has no observed delta
        pred, delta, alpha = m.predict(ho_ctrl, "B", "IFN")
        assert pred.shape == ho_ctrl.shape and delta.shape == (8,)
        assert alpha["B"] == 0 and abs(sum(alpha.values()) - 1) < 1e-5


def test_cell_query_and_save_load(tmp_path):
    adata = make_adata()
    train, ho_ctrl, _ = split_holdout(adata, "C")
    m = CytoXAttention(train, CYTOS, shift_mode="learned", shift_query="cell", encoder_type="attention", space="tf", device="cpu", **SMALL)
    m.fit(max_epochs=1, verbose=False)
    m.fit_shift(epochs=2, decoded_weight=0.1)
    p1, _, _ = m.predict(ho_ctrl, "C", "IL4")
    m.save(str(tmp_path / "m"))
    m2 = CytoXAttention.load(str(tmp_path / "m"), train, device="cpu")
    p2, _, _ = m2.predict(ho_ctrl, "C", "IL4")
    assert np.allclose(p1.X, p2.X, atol=1e-5)
    maps = m.attention_maps(ho_ctrl)
    assert maps["cls_to_tokens"].shape == (adata.n_vars,)


def test_deltas_balance_metrics():
    adata = make_adata()
    Z = np.asarray(adata.X)[:, :8]
    st = compute_celltype_deltas(Z, adata.obs["cell_type"].values, adata.obs["cytokine"].values, CYTOS)
    assert st["deltas"].shape == (2, 4, 8) and st["mask"].all()
    b = balance(adata[:60], "cytokine")
    assert b.obs["cytokine"].value_counts().nunique() == 1
    Xs = np.asarray(adata[(adata.obs.cell_type == "A") & (adata.obs.cytokine == "IFN")].X)
    Xc = np.asarray(adata[(adata.obs.cell_type == "A") & (adata.obs.cytokine == "PBS")].X)
    s = evaluate(Xs.mean(0), Xs, Xc, deg_idx=top_deg(Xs, Xc, 20))
    assert np.isclose(s["r_delta"], 1.0) and np.isclose(s["mse_delta"], 0.0)


def test_train_cli(tmp_path):
    path = tmp_path / "toy.h5ad"
    make_adata().write_h5ad(path)
    configs = [("mlp", "mean", []), ("attention", "cosine", []),
               ("hybrid", "learned", ["--n-enc-layers", "3", "--n-dec-layers", "1", "--ff-mult", "4", "--dropout", "0.1"])]
    for enc, sh, extra in configs:
        cmd = [sys.executable, os.path.join(ROOT, "cytoX_attention", "train.py"), "--data", str(path), "--holdout", "A",
               "--encoder", enc, "--shift", sh, "--epochs", "2", "--shift-epochs", "2", "--n-latent", "8",
               "--n-hidden", "32", "--n-tokens", "8", "--d-model", "16", "--n-heads", "2", "--n-attn-layers", "1",
               "--outdir", str(tmp_path / "out")] + extra
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
    res = pd.read_csv(tmp_path / "out" / "results.csv")
    assert len(res) == 3 * 2 * 3  #3 configs x 2 cytokines x 3 models
    assert any(f.endswith("__attention.npz") for f in os.listdir(tmp_path / "out"))

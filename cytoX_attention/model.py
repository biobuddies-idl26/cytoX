#high-level wrapper --> train VAE --> fit shift attention --> predict held-out cell type per cytokine
import json
import os
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import torch

from .data import train_val_split
from .shift_attention import ShiftAttention, compute_celltype_deltas, fit_loo
from .vae import AttentionVAE


class CytoXAttention:
    def __init__(
        self,
        adata: ad.AnnData,
        cytokines: List[str],
        ctrl: str = "PBS",
        cytokine_key: str = "cytokine",
        cell_type_key: str = "cell_type",
        shift_mode: str = "learned",
        shift_query: str = "celltype",
        device: Optional[str] = None,
        **vae_kwargs,
    ):
        #adata = training cells only (held-out type's treated cells already removed)
        self.adata = adata
        self.cytokines = list(cytokines)
        self.ctrl = ctrl
        self.cytokine_key = cytokine_key
        self.cell_type_key = cell_type_key
        self.shift_query = shift_query
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae_kwargs = dict(vae_kwargs)
        self.vae = AttentionVAE(adata.n_vars, **vae_kwargs).to(self.device)
        self.shift = ShiftAttention(self.vae_kwargs.get("n_latent", 100), len(self.cytokines), mode=shift_mode)
        self.stats = None
        self.history: Dict[str, list] = {"train": [], "val": []}

    def _X(self, adata=None) -> torch.Tensor:
        return torch.from_numpy(np.asarray((adata if adata is not None else self.adata).X, dtype=np.float32))

    #stage 1: VAE
    def fit(
        self,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        patience: int = 25,
        val_frac: float = 0.1,
        seed: int = 0,
        verbose: bool = True,
    ):
        #Adam --> early stopping on validation loss --> restore best weights
        torch.manual_seed(seed)
        np.random.seed(seed)
        X = self._X()
        tr, va = train_val_split(len(X), val_frac, seed)
        Xtr, Xva = X[tr], X[va]
        opt = torch.optim.Adam(self.vae.parameters(), lr=lr, weight_decay=weight_decay)
        g = torch.Generator().manual_seed(seed)
        best, best_state, wait = float("inf"), None, 0
        for ep in range(max_epochs):
            self.vae.train()
            perm = torch.randperm(len(Xtr), generator=g)
            tot, n = 0.0, 0
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:  #BatchNorm needs > 1 cell
                    continue
                x = Xtr[idx].to(self.device)
                x_hat, mu, logvar = self.vae(x)
                loss, _, _ = self.vae.loss(x, x_hat, mu, logvar)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
                n += len(idx)
            val = self._eval_loss(Xva if len(Xva) else Xtr, batch_size=1024)
            self.history["train"].append(tot / max(n, 1))
            self.history["val"].append(val)
            if verbose:
                print(f"epoch {ep + 1:>4}  train {tot / max(n, 1):.4f}  val {val:.4f}")
            if val < best - 1e-6:
                best, wait = val, 0
                best_state = {k: v.detach().clone() for k, v in self.vae.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"early stop at epoch {ep + 1}")
                    break
        if best_state is not None:
            self.vae.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval_loss(self, X, batch_size=1024):
        self.vae.eval()
        tot = 0.0
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(self.device)
            mu, logvar, _ = self.vae.encode(x)
            x_hat = self.vae.decode(mu)
            tot += self.vae.loss(x, x_hat, mu, logvar)[0].item() * len(x)
        return tot / len(X)

    #stage 2: shift attention
    def latent(self, adata=None) -> torch.Tensor:
        return self.vae.get_latent(self._X(adata))

    def fit_shift(self, decoded_weight: float = 0.0, epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-3, seed: int = 0, verbose: bool = False):
        #encode training cells --> per-type ctrl means + deltas --> LOO fit of the shift attention
        Z = self.latent().numpy()
        ct = self.adata.obs[self.cell_type_key].values
        cy = self.adata.obs[self.cytokine_key].values
        self.stats = compute_celltype_deltas(Z, ct, cy, self.cytokines, self.ctrl)
        types = self.stats["types"]
        ctrl_latents = {j: torch.from_numpy(Z[(ct == t) & (cy == self.ctrl)]) for j, t in enumerate(types)}
        stim_means = None
        if decoded_weight > 0:
            X = np.asarray(self.adata.X)
            stim_means = torch.zeros(len(self.cytokines), len(types), X.shape[1])
            for ci, c in enumerate(self.cytokines):
                for j, t in enumerate(types):
                    sel = (ct == t) & (cy == c)
                    if sel.any():
                        stim_means[ci, j] = torch.from_numpy(X[sel].mean(0))
        dec = self.vae.decoder.cpu() if decoded_weight > 0 else None
        info = fit_loo(
            self.shift, self.stats, ctrl_latents=ctrl_latents, query=self.shift_query, decoder=dec,
            stim_means=stim_means, decoded_weight=decoded_weight, epochs=epochs, lr=lr,
            weight_decay=weight_decay, seed=seed, verbose=verbose,
        )
        for p in self.vae.parameters():
            p.requires_grad_(True)
        self.vae.to(self.device)
        return info

    #prediction
    @torch.no_grad()
    def predict(self, ctrl_adata: ad.AnnData, holdout: str, cytokine: str):
        #held-out control cells --> z --> z + attention-weighted shift --> decode
        if self.stats is None:
            self.fit_shift()
        ci = self.cytokines.index(cytokine)
        types = self.stats["types"]
        key_mask = self.stats["mask"][ci].clone()
        if holdout in types:
            key_mask[types.index(holdout)] = False  #held-out type is never its own key
        if not key_mask.any():
            raise ValueError(f"no training cell type has '{cytokine}' cells")
        z = self.latent(ctrl_adata)
        if self.shift_query == "cell":
            query = z
        else:
            query = z.mean(0, keepdim=True)
        shift, alpha = self.shift(query, ci, self.stats["M"], self.stats["deltas"][ci], key_mask)
        x_pred = self.vae.decode_latent(z + shift).numpy()
        pred = ad.AnnData(X=x_pred, obs=ctrl_adata.obs.copy(), var=ctrl_adata.var.copy())
        pred.obs[self.cytokine_key] = cytokine
        alpha_by_type = dict(zip(types, alpha.mean(0).numpy().tolist()))
        return pred, shift.mean(0).numpy(), alpha_by_type

    @torch.no_grad()
    def decoded_control(self, ctrl_adata: ad.AnnData) -> np.ndarray:
        #decoded (smoothed) controls --> reference for TF deltas, see report section 4.2
        return self.vae.decode_latent(self.latent(ctrl_adata)).numpy()

    #interpretation
    @torch.no_grad()
    def attention_maps(self, adata: ad.AnnData, max_cells: int = 2000, seed: int = 0):
        #mean attention over cells --> (layers, heads, T+1, T+1), token 0 = [CLS]
        #cls_to_tokens --> how much [CLS] reads each token (gene module or TF), averaged over layers + heads
        X = self._X(adata)
        if len(X) > max_cells:
            X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))[:max_cells]]
        A = self.vae.get_attention(X).mean(0)
        out = {"attn": A.numpy(), "cls_to_tokens": A[:, :, 0, 1:].mean((0, 1)).numpy()}
        tok = getattr(self.vae.encoder, "tokenizer", None)
        if tok is not None and hasattr(tok, "assignment"):
            out["gene_module_assignment"] = tok.assignment().detach().cpu().numpy()  #(K, G)
        return out

    #persistence
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({"vae": self.vae.state_dict(), "shift": self.shift.state_dict(), "tau": self.shift.tau, "stats": self.stats}, os.path.join(path, "model.pt"))
        cfg = dict(cytokines=self.cytokines, ctrl=self.ctrl, cytokine_key=self.cytokine_key, cell_type_key=self.cell_type_key,
                   shift_mode=self.shift.mode, shift_query=self.shift_query, vae_kwargs=self.vae_kwargs, var_names=list(self.adata.var_names))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def load(cls, path: str, adata: ad.AnnData, device: Optional[str] = None):
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        cfg.pop("var_names")
        vae_kwargs = cfg.pop("vae_kwargs")
        m = cls(adata, device=device, **cfg, **vae_kwargs)
        ckpt = torch.load(os.path.join(path, "model.pt"), map_location=m.device, weights_only=False)
        m.vae.load_state_dict(ckpt["vae"])
        m.shift.load_state_dict(ckpt["shift"])
        m.shift.tau = ckpt["tau"]
        m.stats = ckpt["stats"]
        return m

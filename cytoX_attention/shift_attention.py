#cell-type attention over latent shifts
#scGen: one shift per cytokine = mean over training cell types
#here: shift for held-out type k = sum_t alpha_c(k, t) * delta_{t,c} --> similar cell types count more
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn


def compute_celltype_deltas(Z: np.ndarray, cell_type: np.ndarray, cytokine: np.ndarray, cytokines: List[str], ctrl: str = "PBS"):
    #latents of training cells --> per cell type: control mean m_t and delta_{t,c} = mean z(t,c) - m_t
    #averaging inside each cell type first --> every cell type gets equal weight (same goal as scGen's balancer)
    types = sorted(t for t in np.unique(cell_type) if ((cell_type == t) & (cytokine == ctrl)).any())
    T, C, D = len(types), len(cytokines), Z.shape[1]
    M = np.zeros((T, D), dtype=np.float32)
    deltas = np.zeros((C, T, D), dtype=np.float32)
    mask = np.zeros((C, T), dtype=bool)  #True --> delta_{t,c} observed in training
    for ti, t in enumerate(types):
        M[ti] = Z[(cell_type == t) & (cytokine == ctrl)].mean(0)
        for ci, c in enumerate(cytokines):
            sel = (cell_type == t) & (cytokine == c)
            if sel.any():
                deltas[ci, ti] = Z[sel].mean(0) - M[ti]
                mask[ci, ti] = True
    return {"types": types, "M": torch.from_numpy(M), "deltas": torch.from_numpy(deltas), "mask": torch.from_numpy(mask)}


class ShiftAttention(nn.Module):
    #mode = mean    --> uniform weights (scGen)
    #mode = cosine  --> softmax(cos(query, m_t) / tau), no learned weights, tau picked by LOO
    #mode = learned --> softmax((W_q [query ; e_c]) . (W_k m_t) / sqrt(d)), W_q starts at 0 --> starts as mean
    def __init__(self, n_latent: int, n_cytokines: int, mode: str = "learned", d_attn: int = 32, tau: float = 0.1):
        super().__init__()
        if mode not in ("mean", "cosine", "learned"):
            raise ValueError(f"mode must be mean / cosine / learned, got {mode}")
        self.mode = mode
        self.tau = tau
        self.d_attn = d_attn
        if mode == "learned":
            self.cyto_emb = nn.Embedding(n_cytokines, n_latent)
            self.W_q = nn.Linear(2 * n_latent, d_attn, bias=False)
            self.W_k = nn.Linear(n_latent, d_attn, bias=False)
            nn.init.zeros_(self.W_q.weight)  #zero query --> all scores equal --> uniform alpha == scGen mean

    def scores(self, query: torch.Tensor, c_idx: int, keys: torch.Tensor) -> torch.Tensor:
        #query (Q, D), keys (T, D) --> (Q, T)
        if self.mode == "mean":
            return torch.zeros(query.shape[0], keys.shape[0], device=query.device)
        if self.mode == "cosine":
            qn = nn.functional.normalize(query, dim=-1)
            kn = nn.functional.normalize(keys, dim=-1)
            return qn @ kn.T / self.tau
        e = self.cyto_emb.weight[c_idx].expand(query.shape[0], -1)
        q = self.W_q(torch.cat([query, e], dim=-1))
        k = self.W_k(keys)
        return q @ k.T / self.d_attn ** 0.5

    def forward(self, query, c_idx, keys, values, key_mask):
        #key_mask (T,) bool --> only cell types with an observed delta for this cytokine (never the held-out type)
        s = self.scores(query, c_idx, keys).masked_fill(~key_mask.unsqueeze(0), float("-inf"))
        alpha = torch.softmax(s, dim=-1)
        return alpha @ values, alpha  #shift (Q, D), alpha (Q, T)


def _episodes(stats):
    #leave-one-cell-type-out: every (cytokine c, training type j) with delta observed and at least 1 other key
    mask = stats["mask"]
    eps = []
    for c in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[c, j] and mask[c].sum() > 1:
                km = mask[c].clone()
                km[j] = False
                eps.append((c, j, km))
    return eps


def loo_error(shift: ShiftAttention, stats, query_fn) -> float:
    #mean latent MSE of predicting delta_{j,c} from the other training cell types
    eps = _episodes(stats)
    errs = []
    with torch.no_grad():
        for c, j, km in eps:
            pred, _ = shift(query_fn(j), c, stats["M"], stats["deltas"][c], km)
            errs.append(((pred.mean(0) - stats["deltas"][c, j]) ** 2).mean().item())
    return float(np.mean(errs)) if errs else float("nan")


def fit_loo(
    shift: ShiftAttention,
    stats: Dict,
    ctrl_latents: Optional[Dict[int, torch.Tensor]] = None,
    query: str = "celltype",
    decoder: Optional[nn.Module] = None,
    stim_means: Optional[torch.Tensor] = None,
    decoded_weight: float = 0.0,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-3,
    tau_grid=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    seed: int = 0,
    verbose: bool = False,
):
    #fit the shift attention on training cell types only, VAE frozen
    #query = celltype --> query is m_j; query = cell --> query is each control cell z_i of type j
    #decoded_weight > 0 --> also match decode(z_ctrl_j + shift).mean to observed stim mean x_{j,c}
    g = torch.Generator().manual_seed(seed)

    def query_fn(j):
        if query == "cell":
            z = ctrl_latents[j]
            if len(z) > 128:  #subsample cells per episode --> keeps episodes cheap
                z = z[torch.randperm(len(z), generator=g)[:128]]
            return z
        return stats["M"][j:j + 1]

    if shift.mode == "mean":
        return {"loo_mse": loo_error(shift, stats, query_fn)}

    if shift.mode == "cosine":
        #pick tau with the lowest LOO error
        best = None
        for tau in tau_grid:
            shift.tau = tau
            err = loo_error(shift, stats, query_fn)
            if verbose:
                print(f"  cosine tau={tau:<6} loo_mse={err:.5f}")
            if best is None or err < best[1]:
                best = (tau, err)
        shift.tau = best[0]
        return {"tau": best[0], "loo_mse": best[1]}

    eps = _episodes(stats)
    if not eps:
        return {"loo_mse": float("nan")}
    opt = torch.optim.Adam(shift.parameters(), lr=lr, weight_decay=weight_decay)
    use_dec = decoder is not None and stim_means is not None and decoded_weight > 0
    if use_dec:
        decoder.eval()
        for p in decoder.parameters():
            p.requires_grad_(False)
    init = loo_error(shift, stats, query_fn)
    for ep in range(epochs):
        order = torch.randperm(len(eps), generator=g)
        total = 0.0
        for i in order.tolist():
            c, j, km = eps[i]
            q = query_fn(j)
            pred, _ = shift(q, c, stats["M"], stats["deltas"][c], km)
            #latent target --> delta_{j,c} actually observed for type j
            loss = ((pred.mean(0) - stats["deltas"][c, j]) ** 2).mean()
            if use_dec:
                #decoded target --> held-out-style prediction for type j matches its observed stim mean
                if query == "cell":
                    zc, s = q, pred  #per-cell shift for the sampled control cells
                else:
                    zc = ctrl_latents[j] if ctrl_latents is not None else stats["M"][j:j + 1]
                    s = pred  #(1, D) --> broadcast over control cells
                x_hat = decoder(zc + s).mean(0)
                loss = loss + decoded_weight * ((x_hat - stim_means[c, j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        if verbose and (ep % 50 == 0 or ep == epochs - 1):
            print(f"  shift epoch {ep:>4} loss={total / len(eps):.5f}")
    return {"loo_mse_init": init, "loo_mse": loo_error(shift, stats, query_fn)}

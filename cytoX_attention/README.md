# cytoX_attention

A VAE in the style of scGen, with attention added in two places, for predicting cytokine responses in cell types the model never saw treated. It is plain PyTorch and does not import `/scgen` or `scvi-tools`.

## What changed relative to scGen

| | scGen (`/scgen`) | cytoX_attention |
|---|---|---|
| Encoder | MLP: x → 800 → 800 → (μ, logσ²) | `mlp` (same as scGen), `attention` (transformer over feature tokens) or `hybrid` (both, added together) |
| Decoder | MLP: z → 800 → 800 → x | unchanged |
| Loss | `0.5·SSE + 0.5·5e-5·KL` | unchanged |
| Latent shift | one δ per cytokine, the mean over training cell types | δ̂ₖ,c = Σₜ αc(k,t)·δt,c, with attention over the training cell types (`mean`, `cosine` or `learned`) |
| Cytokines | one model per perturbation | one model for PBS + all cytokines, with a shift per (cell type, cytokine) |

### (a) Feature-token attention in the encoder
- **Gene space (Model A).** The 7000 HVGs are pooled into `K` soft gene-module tokens: `token_k = Σ_g A[k,g]·x_g·E_g`. `A` is a learned gene → module assignment (`attention_maps()["gene_module_assignment"]`). Attention then runs over `K` tokens instead of 7000.
- **TF space (Model B).** Each CollecTRI TF becomes one token: `token_f = x_f·e_f + b_f`. The attention weights therefore point at named TFs.
- The tokens go through a `[CLS]` token, pre-norm transformer blocks, and pooling over `[CLS ; mean]` to give (μ, logσ²).
- `attention_maps()` returns the mean per-head attention and `cls_to_tokens`, which measures how much the pooled representation reads from each module or TF. These outputs feed the TF attention and masking experiments in the report (§7).

### (b) Cell-type attention over latent shifts
After VAE training, each training cell type `t` gets a control latent mean `m_t` and a shift `δ_{t,c}` for each cytokine. For the held-out type `k`, the query is its control latent mean (or each control cell with `--shift-query cell`). The keys are the `m_t` and the values are the `δ_{t,c}`.
- `mean`: uniform weights. This is scGen's arithmetic.
- `cosine`: `softmax(cos(m_k, m_t)/τ)`, with no learned weights. `τ` is chosen by leave-one-cell-type-out on the training types.
- `learned`: `softmax(W_q[m_k ; e_c] · W_k m_t / √d)`, with a cytokine embedding `e_c`. `W_q` is initialised to zero, so before training this equals `mean`. It is trained episodically with the VAE frozen: hold out each training cell type in turn, predict its δ from the remaining types, and minimise latent MSE. `--decoded-weight > 0` adds a decoded-mean MSE term.

The held-out type can never be a key, because its treated cells are not in training.

## Usage
```bash
# Model A (gene space): attention encoder + learned shift attention
python cytoX_attention/train.py --data data/hcd_subset.h5ad --holdout "CD14 Mono" --encoder hybrid --shift learned

# scGen re-implementation (ablation reference)
python cytoX_attention/train.py --holdout "CD14 Mono" --encoder mlp --shift mean

# Model B (TF space)
python cytoX_attention/make_tf_input.py --data data/hcd_subset.h5ad --out data/hcd_subset_tf.h5ad --zscore
python cytoX_attention/train.py --data data/hcd_subset_tf.h5ad --use-obsm score_ulm --space tf --holdout "CD14 Mono"
```
`results.csv` gets one row per (cytokine, model), where the model is `cytoX_attention`, `naive` or `unchanged`. The columns are the report metrics `R2`, `r_delta` and `mse_delta` (on the top-100 Wilcoxon DEGs), plus `r_all`, `mse` and `r_top100`, which use the same definitions as `scgen_baseline.py`. Each run also saves the model, the predicted h5ad for each cytokine, the shift weights α (`__alpha.npy`) and the encoder attention maps (`__attention.npz`).

Python API:
```python
from cytoX_attention import CytoXAttention
from cytoX_attention.data import load_adata, split_holdout
adata = load_adata("data/hcd_subset.h5ad")
train, ho_ctrl, ho_stim = split_holdout(adata, "CD14 Mono")
m = CytoXAttention(train, cytokines=sorted(set(adata.obs.cytokine) - {"PBS"}), encoder_type="hybrid", shift_mode="learned")
m.fit(max_epochs=100); m.fit_shift()
pred, delta, alpha = m.predict(ho_ctrl, "CD14 Mono", "IFN-beta")
```

## Ablation grid
Run `{mlp, attention, hybrid} × {mean, cosine, learned}` on the same held-out pairs and over several `--seed` values. The `mlp + mean` cell isolates what attention adds on top of scGen.

## Tests
```bash
pytest cytoX_attention/tests -q
```

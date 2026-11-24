# Shared scGen and similarity-guided perturbation transfer

## Setup

Run the following commands from the root of the modified `cytoX-hcd-scgen-shared` repository with the scGen Python environment activated. Use a CUDA-enabled PyTorch installation for GPU runs.

Required scripts: `subset_hcd.py`, `scgen_baseline.py`, `scgen_shared.py`, `compare_shared.py`, `ablation_similarity_transfer_v2.py`, and `plot_final_model_scores.py`.

All experiments use the same six cytokines, 7,000 highly variable genes (HVGs), and the CD14 Mono cell-type holdout. Donors are pooled; no donor split is introduced.

## 1. Subset the HCD dataset

Prepare the original input dataset with up to 500 cells per cytokine–cell-type combination, 7,000 HVGs, and the original CP10K + log1p normalization. Use the same H5AD for every model.

Set `RAW` to the location of the original HCD dataset.

```bash
RAW="/path/to/20260203_Parse_10M_PBMC_cytokines.h5ad"

python -u subset_hcd.py \
    --raw "$RAW" \
    --cytokines PBS IFN-beta IFN-gamma IL-4 IL-15 IL-1-beta IL-32-beta \
    --n 500 \
    --hvg 7000 \
    --seed 0 \
    --out data/hcd_subset.h5ad
```

## 2. Train the original cytokine-specific scGen models

Train six independent VAEs, one for each cytokine, with CD14 Mono as the held-out cell type.

The GPU-compatible script (`scgen_baseline_gpu.py`) uses the GPU for training and moves the trained model to the CPU before prediction and reconstruction.

```bash
mkdir -p data/original_CD14_Mono/logs

for CYT in IFN-beta IFN-gamma IL-1-beta IL-15 IL-32-beta IL-4; do

    CUDA_VISIBLE_DEVICES=0 python -u scgen_baseline_gpu.py \
        --data data/hcd_subset.h5ad \
        --cytokine "$CYT" \
        --holdout "CD14 Mono" \
        --epochs 100 \
        --n-layers 2 \
        --n-hidden 800 \
        --n-latent 100 \
        --outdir data/original_CD14_Mono \
        2>&1 | tee "data/original_CD14_Mono/logs/${CYT}.log"

done
```

Results: `data/original_CD14_Mono/results.csv`

This step can be skipped if using the historical cytokine-specific results included in the repository.

## 3. Train the shared scGen baseline (Z100)

Test whether a single VAE trained jointly on PBS and all six cytokines can replace the six independent VAEs while retaining the original scGen prediction procedure. The shared model excludes all treated CD14 Mono cells.

Run **either** the CPU or GPU command, not both. Both use the same output directory.

### CPU

```bash
CUDA_VISIBLE_DEVICES="" python -u scgen_shared.py \
    --data data/hcd_subset.h5ad \
    --holdout "CD14 Mono" \
    --epochs 100 \
    --n-layers 2 \
    --n-hidden 800 \
    --n-latent 100 \
    --outdir data/shared_CD14_Mono
```

### GPU

```bash
CUDA_VISIBLE_DEVICES=0 python -u scgen_shared.py \
    --data data/hcd_subset.h5ad \
    --holdout "CD14 Mono" \
    --epochs 100 \
    --n-layers 2 \
    --n-hidden 800 \
    --n-latent 100 \
    --outdir data/shared_CD14_Mono
```

Output: one shared VAE, six cytokine predictions, and `data/shared_CD14_Mono/results.csv`.

## 4. Compare the shared Z100 model with the historical cytokine-specific models

Compare against the original L2/H800/Z100 results bundled in `analysis/results_L2_vs_L4_H800_Z100.csv`. The comparison checks that the original naive scores match before writing the results.

```bash
python compare_shared.py data/shared_CD14_Mono --historical
```

Output: `data/shared_CD14_Mono/comparison.csv`.

## 5. Ablate latent dimensionality (Z200 and Z400)

Test whether latent capacity limits the shared model. One Z100 VAE may need to represent cell identity, baseline heterogeneity and six cytokine perturbations, whereas each original Z100 VAE is trained on PBS and a single cytokine. Increase latent dimensionality while keeping all other training settings unchanged.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scgen_shared.py \
    --data data/hcd_subset.h5ad \
    --holdout "CD14 Mono" \
    --epochs 100 \
    --n-layers 2 \
    --n-hidden 800 \
    --n-latent 200 \
    --outdir data/shared_CD14_Mono_Z200

CUDA_VISIBLE_DEVICES=0 python -u scgen_shared.py \
    --data data/hcd_subset.h5ad \
    --holdout "CD14 Mono" \
    --epochs 100 \
    --n-layers 2 \
    --n-hidden 800 \
    --n-latent 400 \
    --outdir data/shared_CD14_Mono_Z400
```

Compare both models against the original cytokine-specific Z100 reference:

```bash
python compare_shared.py data/shared_CD14_Mono_Z200 --historical
python compare_shared.py data/shared_CD14_Mono_Z400 --historical
```

## 6. Compare Z100, Z200 and Z400 together

Summarize the three shared latent-dimensionality configurations alongside the historical original scGen and original naive baselines. Retain both real-PBS and reconstructed-PBS evaluations for scGen. The naive baseline uses real PBS only.

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd

out = Path('data/latent_dimension_comparison')
out.mkdir(parents=True, exist_ok=True)

runs = {
    'Shared Z100': 'data/shared_CD14_Mono',
    'Shared Z200': 'data/shared_CD14_Mono_Z200',
    'Shared Z400': 'data/shared_CD14_Mono_Z400',
}
metrics = ['r_all', 'r_top100', 'mse']
keys = ['cytokine', 'holdout', 'model']
parts = []

for name, folder in runs.items():
    df = pd.read_csv(Path(folder) / 'comparison.csv')
    assert len(df) == 18 and not df.duplicated(keys).any()
    assert set(df['holdout']) == {'CD14 Mono'}
    assert df['cytokine'].nunique() == 6

    shared = df.loc[df.model != 'naive', keys +
                    [m + '_shared' for m in metrics]].copy()
    shared = shared.rename(columns={m + '_shared': m for m in metrics})
    parts.append(shared.assign(architecture=name))

    if name == 'Shared Z100':
        original = df[keys + [m + '_original' for m in metrics]].copy()
        original = original.rename(columns={m + '_original': m for m in metrics})
        original['architecture'] = original.model.map(
            {'naive': 'Original naive', 'scGen': 'Original scGen Z100',
             'scGen_vs_real_ctrl': 'Original scGen Z100'})
        parts.insert(0, original)

scores = pd.concat(parts, ignore_index=True)
scores['reference'] = scores.model.map({
    'scGen': 'reconstructed_PBS',
    'scGen_vs_real_ctrl': 'real_PBS',
    'naive': 'real_PBS',
})
scores.to_csv(out / 'scores_by_cytokine.csv', index=False)
summary = scores.groupby(['reference', 'architecture'], sort=False)[metrics].mean()
summary.to_csv(out / 'summary.csv')
print(summary.to_string(float_format=lambda x: f'{x:.6f}'))

# Plot latent dimensionality comparison
import matplotlib.pyplot as plt

order = [
    'Original naive',
    'Original scGen Z100',
    'Shared Z100',
    'Shared Z200',
    'Shared Z400',
]

colors = [
    '#999999',
    '#4477AA',
    '#88CCEE',
    '#228833',
    '#AA3377',
]

references = {
    'real_PBS': 'Real PBS',
    'reconstructed_PBS': 'Reconstructed PBS',
}

metric_labels = {
    'r_all': 'Pearson r | all genes',
    'r_top100': 'Pearson r | top 100 genes',
    'mse': 'Mean squared error',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
})

fig, axes = plt.subplots(
    2, 3, figsize=(15, 7),
    sharex='col',
    constrained_layout=True,
)

for i, (ref, ref_label) in enumerate(references.items()):

    plot_df = summary.loc[ref].reindex(order).dropna().copy()
    
    plot_order = plot_df.index.tolist()
    plot_colors = [colors[order.index(m)] for m in plot_order]

    for j, metric in enumerate(metrics):
        ax = axes[i, j]
        values = plot_df[metric]

        bars = ax.barh(
            plot_order, values,
            color=plot_colors,
            height=0.65,
        )

        ax.invert_yaxis()

        ax.bar_label(
            bars,
            fmt='%.5f' if metric == 'mse' else '%.3f',
            padding=4,
            fontsize=10,
        )

        ax.set_title(
            f'{ref_label}\n{metric_labels[metric]}',
            fontweight='bold',
            fontsize=12,
        )

        ax.grid(axis='x', alpha=0.2)
        ax.set_axisbelow(True)

        if metric == 'mse':
            ax.set_xlim(0, summary[metric].max() * 1.25)
            ax.set_xlabel('MSE (lower is better)')
        else:
            ax.set_xlim(min(0, summary[metric].min() * 1.2), 1.12)
            ax.set_xlabel('Pearson r (higher is better)')

        if j != 0:
            ax.tick_params(axis='y', labelleft=False)

fig.suptitle(
    'Effect of latent dimensionality on prediction performance',
    fontsize=17,
    fontweight='bold',
)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(
        out / f'latent_dimension_comparison.{ext}',
        dpi=600,
        bbox_inches='tight',
    )

plt.close(fig)
PY
```

Outputs: `data/latent_dimension_comparison/scores_by_cytokine.csv` and `summary.csv`.

## 7. Ablate the latent-shift estimation strategy

Use the frozen shared Z200 checkpoint to test whether similarity-guided
source-cell selection and proportional weighting improve perturbation
transfer. No VAE retraining is required.

### Source-cell selection and weighting

The target cell type is the held-out cell type whose cytokine response
we aim to predict (CD14 Mono). The remaining 13 cell types are referred
to as source cell types.

Source cell types are ranked by cosine similarity between their mean
PBS latent representations (latent centroids) and that of the target
cell type.

For each source cell type, a cytokine-specific latent shift is calculated
by subtracting its mean PBS latent representation from its mean
stimulated latent representation.

**K** denotes the number of most similar source cell types selected
to estimate the perturbation:

- **K1:** Use the most similar source cell type.
- **K2:** Combine the shifts of the two most similar source cell types.
- **K3–K10:** Incorporate additional source cell types in descending
  order of similarity.

Two weighting strategies are evaluated:

- **Equal weighting:** Each selected source contributes equally (1/K).
- **Similarity-proportional weighting:** Each selected source contributes
  according to its normalized positive cosine similarity.

The combined latent shift is added to the PBS latent representations
of the target cells and decoded to predict stimulated gene expression.

### Ablation experiments

The analysis evaluates:

- Top-K source selection versus randomly selected subsets of the same size.
- All 78 possible pairs of the 13 source cell types.
- Association between source-pair similarity and prediction accuracy
  using Spearman correlation.
- Equal versus proportional weighting across K = 1–10.
- Alignment of the combined perturbation vector with the equal-weighted
  K2 latent shift.
- A strong learning-free K1 baseline that selects the closest source
  using cosine similarity between mean PBS gene-expression profiles
  and transfers its perturbation directly in gene space, without a VAE.

```bash
CUDA_VISIBLE_DEVICES=0 python -u ablation_similarity_transfer_v2.py \
    --data data/hcd_subset.h5ad \
    --shared-z100 data/shared_CD14_Mono \
    --shared-z200 data/shared_CD14_Mono_Z200 \
    --holdout "CD14 Mono" \
    --k-max 10 \
    --random-subsets 24 \
    --seed 42 \
    --device cuda \
    --outdir data/similarity_transfer_FINAL
```

For CPU inference, replace `CUDA_VISIBLE_DEVICES=0` with
`CUDA_VISIBLE_DEVICES=""` and `--device cuda` with `--device cpu`.

Outputs: `data/similarity_transfer_FINAL/results.csv`, `summary.csv`,
source-selection and weighting tables, pair-similarity and Spearman
correlation tables, and publication figures under `figures/`.

## 8. Generate the final model-comparison figures (K2)

Compare the original naive, strong learning-free naive K1, original cytokine-specific scGen Z100, shared scGen Z100/Z200, matched all-source Z200, and shared Z200 K2 with equal and proportional weighting. Use real PBS as the common reference.

```bash
python plot_final_model_scores.py \
    --results data/similarity_transfer_FINAL/results.csv \
    --outdir data/final_model_comparison_K2
```

Outputs: overall and cytokine-specific scores and figures in PNG, PDF and SVG formats under `data/final_model_comparison_K2/`.

For subsequent held-out cell-type experiments, fix the proposed model to L2/H800/Z200, latent-space cosine similarity, K = 2 and equal weights. The strong naive K1 remains the independent learning-free baseline.
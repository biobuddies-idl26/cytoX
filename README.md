# Shared scGen with learned cytokine perturbation models

## Setup

Run all commands from the repository root with the scGen Python environment activated. GPU training requires CUDA-enabled PyTorch.

Required scripts: `subset_hcd.py`, `scgen_baseline.py`, `scgen_baseline_gpu.py`, `scgen_shared.py`, and `train_nonlinear_shared_scgen.py`.

Use the **simplified fixed-epoch version** of the perturbation training script (previously named `train_nonlinear_shared_scgen_simple.py`), saved in the repository as `train_nonlinear_shared_scgen.py`.

All experiments use the same six cytokines, 7,000 highly variable genes (HVGs), and CD14 Mono cell-type holdout. PBS is the control condition. Donors are pooled; no donor split is introduced. Treated CD14 Mono cells are excluded from training and used for final evaluation.

## Overview

The experiments address three questions:

- Can a shared VAE replace independent cytokine-specific VAEs while retaining cytokine-specific latent shifts?
- Does increasing the shared latent dimension from 100 to 200 help represent multiple cytokine responses?
- Can directly learning the perturbation transformation improve prediction over the original scGen centroid-based arithmetic?

The final comparison includes eight configurations: the original naive baseline, original cytokine-specific scGen Z100, shared scGen Z100, shared scGen Z200, and four learned perturbation heads (global, affine, nonlinear and hybrid).

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

Results: `data/original_CD14_Mono/results.csv`. Use a fresh output directory if retraining to avoid duplicate result rows.

This step is optional when using the historical cytokine-specific Z100 scores bundled in the repository. The final comparison imports these historical scores by default; the freshly trained scores above do not automatically replace them.

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

## 4. Train the shared scGen model (Z200)

Test whether the shared representation requires greater latent capacity to accommodate cell identity, baseline heterogeneity and multiple cytokine responses. Increase the latent dimension to 200 while retaining the original scGen perturbation arithmetic and other VAE settings.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scgen_shared.py \
    --data data/hcd_subset.h5ad \
    --holdout "CD14 Mono" \
    --epochs 100 \
    --n-layers 2 \
    --n-hidden 800 \
    --n-latent 200 \
    --outdir data/shared_CD14_Mono_Z200
```

Output: the shared Z200 VAE checkpoint and `data/shared_CD14_Mono_Z200/results.csv`.

The trained Z200 encoder and decoder are reused and frozen in the next step.

## 5. Train and compare the four learned perturbation models

Test whether a trainable perturbation head can improve on the fixed
latent translation used by scGen.

The shared Z200 encoder and decoder remain frozen. Each perturbation
model learns its own 32-dimensional embedding for each of the six cytokines.

### Prediction workflow

1. Encode the untreated PBS cell into a 200-dimensional latent representation.
2. Retrieve the learned 32-dimensional embedding of the selected cytokine.
3. Estimate a 200-dimensional perturbation vector using the trained head.
4. Add the estimated perturbation vector to the original latent representation.
5. Decode the modified latent representation into predicted expression
   across 7,000 genes.

The four models differ only in how the perturbation vector is estimated.

### Perturbation-head architectures

| Head | Perturbation estimation | Architecture | Trainable parameters |
|---|---|---|---:|
| Global | Cytokine-specific shift, independent of the initial cellular state | Linear: 32 → 200 | 6,792 |
| Affine | Linear combination of the initial cellular state and cytokine embedding | Linear: 232 → 200 | 46,792 |
| Nonlinear | Nonlinear interaction between the initial cellular state and cytokine embedding | MLP: 232 → 256 → 256 → 200 | 177,032 |
| Hybrid | Global cytokine-specific shift plus a nonlinear state-dependent correction | Global linear branch + nonlinear MLP | 183,632 |

The nonlinear and hybrid models use GELU activation functions and
dropout (0.1) in their hidden layers.

### Scientific questions

| Comparison | Question addressed |
|---|---|
| Default shared scGen vs. learned global | Does directly optimizing a cytokine-specific shift improve upon centroid subtraction? |
| Global vs. affine | Does incorporating the initial cellular state improve perturbation estimation? |
| Affine vs. nonlinear | Are nonlinear interactions between cellular state and cytokine identity beneficial? |
| Nonlinear vs. hybrid | Does explicitly combining a global response with a state-dependent correction improve generalization? |

### Training strategy

All four perturbation heads are trained independently using the same
frozen shared Z200 VAE.

- 13 source cell types.
- Six cytokine treatments.
- 78 cell-type–cytokine training groups.
- Up to 128 randomly sampled PBS cells per group per optimization step.
- 100 training epochs per model.
- AdamW optimizer with a learning rate of 0.001.
- Mean squared error across all 7,000 genes.
- No epoch selection, validation split, or refitting.
- No additional encoder or decoder training.
- No explicit cell-type labels as model inputs.
- No similarity-based source selection during prediction.

The training objective minimizes the difference between the mean
predicted treated expression and the experimentally observed treated
expression for each source cell-type–cytokine group.

Individual PBS and treated cells are not artificially paired.
Although the decoder is frozen, gradients propagate through its
operations to update the perturbation head and cytokine embeddings.

Treated CD14 Mono cells remain excluded from training and are used
only for final evaluation.

### Training

Each head is initialized once and trained independently for exactly
100 epochs on all 13 source cell types (78 cell-type–cytokine groups).
There is no epoch selection, validation split, refitting, or VAE retraining.

For each source group, up to 128 PBS cells are randomly sampled from
the precomputed latent representations. The perturbation head estimates
their latent shifts, and the frozen decoder generates predicted
treated gene expression.

The training objective is to minimize the mean squared error (MSE)
between the predicted and experimentally observed treated group means
across all 7,000 genes.

No individual PBS–treated cell pairing is assumed.

Only the perturbation head and its cytokine embeddings are updated.
Gradients pass through the frozen decoder without changing its parameters.

### Run

The following saved inputs are required:

- `analysis/results_L2_vs_L4_H800_Z100.csv` — historical original scGen Z100 benchmark.
- `data/shared_CD14_Mono/results.csv` — shared scGen Z100 scores.
- `data/shared_CD14_Mono_Z200/results.csv` — default shared scGen Z200 scores.
- `data/shared_CD14_Mono_Z200/shared__CD14_Mono/model.pt` — frozen shared Z200 VAE.

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_nonlinear_shared_scgen.py \
    --data data/hcd_subset.h5ad \
    --run data/shared_CD14_Mono_Z200 \
    --holdout "CD14 Mono" \
    --embedding-dim 32 \
    --hidden 256 \
    --epochs 100 \
    --batch-size 128 \
    --lr 0.001 \
    --seed 42 \
    --device cuda \
    --heads global affine nonlinear hybrid \
    --outdir data/nonlinear_shared_Z200_E32_fixed100
```

For CPU training, replace `CUDA_VISIBLE_DEVICES=0` with `CUDA_VISIBLE_DEVICES=""` and `--device cuda` with `--device cpu`. Use a new or empty output directory for each run.

### Evaluation and outputs

The script automatically compares the following eight models:

| Model | Operation |
|---|---|
| Original naive | Recalculate the mean gene-space perturbation across all sources |
| Original cytokine-specific scGen Z100 | Import historical scores |
| Shared scGen Z100 | Import saved scores |
| Shared scGen Z200 (default) | Import saved scores |
| Learned global shift | Train once |
| Learned affine shift | Train once |
| Learned nonlinear shift | Train once |
| Learned hybrid shift | Train once |

The three scGen baseline scores are imported, and the original naive scores are recalculated and checked against the saved benchmarks.

All models are evaluated against experimentally observed expression changes in CD14 Mono using Pearson correlation across all genes, Pearson correlation across the top 100 genes ranked by absolute observed change, and MSE. Real-PBS evaluation is available for all eight models; reconstructed-PBS evaluation is available only for the seven VAE-based models.

Results are written to `data/nonlinear_shared_Z200_E32_fixed100/`, including:

- `results.csv` and `summary.csv`: per-cytokine and aggregate scores.
- `training_history.csv`: the four fixed-epoch training curves.
- `head_*.pt` and `cytokine_embeddings_*.csv`: trained heads and embeddings.
- `learned_shift_summary.csv`, `hybrid_shift_components.csv`, and `hybrid_global_shifts.csv`: estimated latent perturbations.
- `paired_comparisons.csv` and `paired_comparison_summary.csv`: benchmark comparisons.
- `validation.csv` and `run_config.json`: consistency checks and run settings.

The script automatically generates the overall comparison plots, per-cytokine heatmaps, and training curves in `figures/` (PNG, PDF and SVG). No separate plotting command is required.

At inference, the learned heads require the frozen shared VAE, trained head and cytokine embedding.

## 6. Visualize cellular states and cytokine-induced latent perturbations

Investigate how cellular states are represented in the shared latent
space and compare the perturbations predicted by shared scGen Z200
and the learned hybrid model.

Both scripts use previously trained models or saved predictions.
No additional training is required.

### Models included

| Model | Perturbation estimation |
|---|---|
| Shared scGen Z200 | Fixed cytokine-specific latent shift calculated by centroid subtraction |
| Learned hybrid | Learned global cytokine shift plus a nonlinear state-dependent correction |

Both models use the same frozen 200-dimensional latent representation,
allowing direct comparison of their predicted cellular states.

### 6.1. Visualize latent states and perturbation vectors

Install UMAP if needed:

```bash
python -m pip install umap-learn
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python -u visualize_all_latent_shifts.py \
    --data data/hcd_subset.h5ad \
    --shared-z200 data/shared_CD14_Mono_Z200 \
    --heads-dir data/nonlinear_shared_Z200_E32_fixed100 \
    --holdout "CD14 Mono" \
    --device cuda \
    --outdir data/all_latent_shifts_Z200_hybrid_v2
```

Use a new or empty output directory for each run.

#### Visualization strategy

Two separate UMAP representations are generated using all 200 latent dimensions:

- **Reference UMAP:** Fitted on reference cells, excluding treated
  CD14 Mono cells. Predicted and experimentally observed target cells
  are subsequently projected into this space.
- **Target-only UMAP:** Fitted exclusively on experimentally measured
  CD14 Mono cells, including PBS and all six cytokine conditions.
  This provides a more detailed view of target-cell perturbations
  without displaying other cell populations.

All predictions are generated before constructing the target-only UMAP.
The experimentally observed treated target cells are used for
visualization and evaluation only, not for perturbation estimation.

#### Reference UMAP figures

Located under `Z200/figures/umap/`.

| Figure | Description |
|---|---|
| `01_cell_state.png` | Cellular states colored by cell type and cytokine treatment. |
| `02_states_by_treatment.png` | Cellular distributions under PBS and each cytokine using identical coordinates. |
| `03_source_observed_arrows.png` | Experimentally observed PBS-to-treated centroid movements for each source cell type. |
| `04_target_all_model_arrows.png` | Shared scGen and hybrid predicted target movements compared with the observed response. |
| `05_source_predictions_default.png` | Shared scGen predictions compared with observed source-cell responses. |
| `05_source_predictions_hybrid.png` | Hybrid predictions compared with observed source-cell responses. |

Source-cell predictions are in-sample diagnostics rather than
independent evaluations of model generalization.

#### Target-only UMAP figures

Located under `Z200/figures/umap/target_only_umap/`.

A dedicated figure is generated for each cytokine, showing:

- Untreated PBS cells and their centroid.
- Experimentally observed treated cells and their centroid.
- Shared scGen predicted centroid.
- Hybrid predicted centroid.
- Arrows connecting the starting PBS centroid to each endpoint.

Additional summary figures are generated under `overview/`.

| Figure | Description |
|---|---|
| `01_cell_state.png` | Target cellular states colored by treatment status and cytokine identity. |
| `02_states_by_treatment.png` | Target-cell distributions under PBS and each cytokine. |
| `03_observed_target_arrows.png` | Experimentally observed target centroid movements. |
| `04_target_all_model_arrows.png` | Shared scGen and hybrid predicted movements compared with observed movements. |
| `05_target_predictions_default.png` | Shared scGen predicted cell distributions and centroid movements. |
| `05_target_predictions_hybrid.png` | Hybrid predicted cell distributions and centroid movements. |

All overview panels use the same target-only UMAP coordinates and
axis limits.

#### Perturbation direction, similarity and magnitude

Located under `Z200/figures/`.

| Figure | Description |
|---|---|
| `04_UMAP_shift_directions.png` | UMAP of observed source perturbation directions, colored by cell type or cytokine. |
| `05_latent_shift_directions.png` | Signed components of normalized perturbation vectors across all 200 latent dimensions. |
| `06_shift_cosine_similarity.png` | Pairwise cosine similarity between observed perturbation directions. |
| `07_cell_type_shift_similarity_per_cytokine.png` | Similarity of responses between cell types exposed to the same cytokine. |
| `08_cytokine_shift_similarity_per_cell_type.png` | Similarity of responses to different cytokines within the same cell type. |
| `09_latent_shift_magnitudes.png` | Magnitude of each experimentally observed cell-type–cytokine perturbation. |
| `10_full_latent_prediction_metrics.png` | Quantitative comparison of shared scGen and hybrid target predictions. |

Figures 05, 06, 07 and 08 are generated in two versions:

- **Non-clustered:** Fixed cell-type and cytokine ordering to facilitate
  comparisons across panels.
- **Clustered:** Hierarchical clustering using cosine distance and
  average linkage to reveal groups with similar perturbation directions.

Clustered figures have the suffix `_clustered`. Both versions use
identical similarity values and color scales.

#### Understanding the perturbation arrows

- **Observed perturbations:** Calculated by subtracting the mean PBS
  latent representation from the mean experimentally measured treated
  latent representation.
- **Predicted perturbations:** Calculated by applying the corresponding
  perturbation model to untreated cells.

Solid arrows represent predicted movements, whereas dashed arrows
represent experimentally observed movements in prediction comparisons.

Shared scGen applies the same latent perturbation vector to every
cell for a given cytokine. The hybrid model can generate different
perturbation vectors for individual cells depending on their initial
latent states.

#### Quantitative latent-space evaluation

| Metric | Interpretation | Ideal value |
|---|---|---|
| Cosine similarity | Agreement between predicted and observed perturbation directions | 1 |
| Relative centroid error | Prediction error relative to the observed perturbation magnitude | 0 |
| Shift magnitude ratio | Predicted magnitude divided by observed magnitude | 1 |
| Within-cell-type shift RMS | Variation of predicted perturbation vectors among individual cells | Model-dependent |

All metrics are calculated using the original 200-dimensional latent
vectors rather than their two-dimensional UMAP coordinates.

### 6.2. Visualize individual-cell perturbation predictions

Centroid-based visualization shows whether the models predict the
average movement of a cellular population. However, predicting the
correct centroid does not necessarily imply that the predicted cells
reproduce the experimentally observed treated-cell distribution.

This additional analysis visualizes the predicted latent position of
every individual PBS cell using shared scGen and hybrid.

Run:

```bash
python -u visualize_target_cell_predictions.py \
    --input-dir data/all_latent_shifts_Z200_hybrid_v2
```

The script reuses the saved target-only UMAP coordinates and individual
cell predictions generated in Step 6.1. No GPU, model loading,
retraining or additional UMAP fitting is required.

#### Generated figures

One figure per cytokine containing three comparable panels:

| Panel | Content |
|---|---|
| A — Experimentally observed | PBS and experimentally observed treated cells |
| B — Shared scGen Z200 | PBS, observed treated cells and all shared scGen predicted cells |
| C — Learned hybrid | PBS, observed treated cells and all hybrid predicted cells |

All three panels use identical UMAP coordinates and axis limits.

These figures allow us to investigate whether the predicted populations
capture the location, spread and heterogeneity of experimentally
observed treated populations, beyond their mean centroid positions.

Optionally, visualize individual PBS-to-prediction movements:

```bash
python -u visualize_target_cell_predictions.py \
    --input-dir data/all_latent_shifts_Z200_hybrid_v2 \
    --cell-arrows 50
```

This displays 50 individual computational cell movements per model,
while retaining all predicted cells in the figures.

Results are saved under:

`Z200/figures/umap/target_only_umap/cell_level_comparison/`

### Interpretation

UMAP is a nonlinear projection. Arrow lengths, angles and distances
do not necessarily preserve the geometry of the original latent space.

All predicted cellular states are visualized directly in latent space,
before decoding into gene expression.

Experimentally measured PBS and treated cells are unpaired. Therefore,
overlap between predicted and observed distributions does not establish
accurate individual-cell responses.

The latent-space analyses complement, rather than replace, the
gene-expression prediction metrics described in Step 5.

### Output directory

```text
data/all_latent_shifts_Z200_hybrid_v2/
├── FIGURE_GUIDE.txt
├── run_config.json
├── all_latent_prediction_metrics.csv
└── Z200/
    ├── figures/
    │   ├── umap/
    │   │   ├── Reference UMAP figures
    │   │   └── target_only_umap/
    │   │       ├── Individual cytokine centroid figures
    │   │       ├── overview/
    │   │       └── cell_level_comparison/
    │   ├── Perturbation direction figures
    │   ├── Clustered and non-clustered similarity heatmaps
    │   └── Quantitative prediction figures
    ├── latent_embeddings.npz
    ├── latent_centroids.csv
    ├── latent_prediction_metrics.csv
    ├── predicted_and_observed_mean_shifts.csv
    ├── target_predicted_latent_states.npz
    ├── target_only_umap_coordinates.csv
    ├── target_only_umap_predictions.npz
    └── Additional numerical results
```

All figures are generated as PNG files. Full-dimensional latent
representations, perturbation vectors, predicted cellular states,
UMAP coordinates and quantitative metrics are retained for
subsequent analysis.

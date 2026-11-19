#!/usr/bin/env bash
# Launch the complete HCD pseudobulk comparison.
# Run from any directory; paths are interpreted relative to the repository root.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DATA_PATH="${1:-data/hcd_subset.h5ad}"
OUT_DIR="${2:-data/pseudobulk_runs_full}"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "ERROR: HCD dataset not found: $DATA_PATH" >&2
  echo "Usage: bash run_pseudobulk_full.sh [path/to/hcd_subset.h5ad] [output_directory]" >&2
  exit 1
fi

if [[ ! -f "pseudobulk_experiments.py" ]]; then
  echo "ERROR: pseudobulk_experiments.py must be in the repository root." >&2
  exit 1
fi

run_experiments() {
  python pseudobulk_experiments.py \
    --data "$DATA_PATH" \
    --strategies sum_counts mean_log pooled_donor \
    --thresholds 5 10 20 \
    --mean-log-min-cells 10 \
    --pooled-min-cells 10 \
    --min-eval-donors 3 \
    --epochs 100 \
    --batch-size 32 \
    --patience 25 \
    --target-sum 10000 \
    --seed 0 \
    --outdir "$OUT_DIR" \
    --wandb-project cytoX \
    --wandb-entity scgen \
    --wandb-mode online \
    "$@"
}

# Avoid expanding an empty Bash array under macOS's default Bash 3.2 + nounset.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  run_experiments --dry-run
else
  run_experiments
fi
